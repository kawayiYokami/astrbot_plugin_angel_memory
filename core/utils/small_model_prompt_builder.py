"""
小模型提示词构建器

负责构建用于小模型的记忆整理提示词。
"""

from typing import List, Dict, Tuple
from datetime import datetime
from ..session_memory import MemoryItem
from ..config import MemoryConstants
from .memory_formatter import MemoryFormatter


class SmallModelPromptBuilder:
    """小模型提示词构建器"""

    @staticmethod
    def format_relative_time(timestamp: float) -> str:
        """
        格式化相对时间

        Args:
            timestamp: Unix时间戳

        Returns:
            相对时间字符串，如"刚刚"、"2分钟前"等
        """
        if not timestamp:
            return ""

        now = datetime.now().timestamp()
        diff = now - timestamp

        if diff < MemoryConstants.TIME_MINUTE:
            return "刚刚"
        if diff < MemoryConstants.TIME_HOUR:
            return f"{int(diff / MemoryConstants.TIME_MINUTE)}分钟前"
        if diff < MemoryConstants.TIME_DAY:
            return f"{int(diff / MemoryConstants.TIME_HOUR)}小时前"
        return f"{int(diff / MemoryConstants.TIME_DAY)}天前"

    @staticmethod
    def extract_text_from_content(content) -> str:
        """
        从content中提取文本

        Args:
            content: 消息内容，可能是字符串、列表或字典

        Returns:
            提取的文本内容
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
        return ""

    @staticmethod
    def format_chat_records(chat_records: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        格式化对话历史为可读文本，并生成对话参与者清单。

        Args:
            chat_records: 对话记录列表

        Returns:
            一个元组，包含：
            - 格式化后的对话文本字符串
            - 对话参与者清单（去重后）
        """
        formatted_lines = []
        participants = {}  # 使用字典去重，key为sender_id

        for msg in chat_records:
            role = msg.get("role")

            # 过滤掉toolcall相关消息
            if msg.get("is_structured_toolcall"):
                continue  # 跳过结构化toolcall消息

            # 检查内容中是否包含toolcall关键词
            content = msg.get("content", "")
            if isinstance(content, str):
                toolcall_keywords = ["调用", "tool_call", "function", "工具调用结果："]
                if any(keyword in content for keyword in toolcall_keywords):
                    continue  # 跳过包含toolcall关键词的消息

            if role == "user":
                sender_name = msg.get("sender_name", "成员")
                sender_id = msg.get("sender_id", "Unknown")

                # 构建参与者清单
                if sender_id != "Unknown" and sender_id not in participants:
                    participants[sender_id] = {"id": sender_id, "name": sender_name}

                # 格式化对话文本
                timestamp = msg.get("timestamp", 0)
                time_str = SmallModelPromptBuilder.format_relative_time(timestamp)
                header = f"[群友: {sender_name}/{sender_id}]（{time_str}）: "
                content = msg.get("content", [])
                text = SmallModelPromptBuilder.extract_text_from_content(content)
                formatted_lines.append(f"{header}{text}")

            else:  # assistant
                content = msg.get("content", "")
                text = SmallModelPromptBuilder.extract_text_from_content(content)
                formatted_lines.append(f"[助理]: {text}")

        # 将去重后的参与者字典转换为列表
        user_list = list(participants.values())

        # 返回包含两项内容的元组
        return "\n".join(formatted_lines), user_list

    @staticmethod
    def format_note_query(chat_records: List[Dict]) -> str:
        """
        格式化笔记查询文本，仅提取is_processed=false的最新对话记录内容。

        Args:
            chat_records: 对话记录列表

        Returns:
            格式化后的查询文本字符串（纯文本内容拼接）
        """
        # 过滤出未处理的记录
        unprocessed_records = [
            msg for msg in chat_records if not msg.get("is_processed", True)
        ]

        # 如果没有未处理的记录，返回空字符串
        if not unprocessed_records:
            return ""

        # 提取最新的未处理记录内容
        return SmallModelPromptBuilder.extract_text_from_content(
            unprocessed_records[-1].get("content", [])
        ).strip()

    @staticmethod
    def build_memory_prompt(
        formatted_query: str,
        memories: List[MemoryItem],
        user_list: List[Dict],
        candidate_notes: List[Dict] = None,
        secretary_decision: Dict = None,
        core_topic: str = None,
        system_prompt: str = None,
        memory_config=None,
    ) -> str:
        """
        构建用于小模型的记忆整理提示词

        Args:
            formatted_query: 已格式化的对话历史字符串
            memories: 记忆列表
            user_list: 对话参与者清单
            candidate_notes: 候选笔记列表（可选）
            secretary_decision: 秘书决策信息，包含AI人格和别名（可选）
            core_topic: 当前对话的核心话题（可选）
            system_prompt: 系统提示词（可选，如果不提供则需要在方法内获取）

        Returns:
            完整的提示词字符串
        """
        from .note_context_builder import NoteContextBuilder

        # 获取系统提示词
        if system_prompt is None:
            # 延迟导入以避免循环依赖
            from ...llm_memory import CognitiveService

            system_prompt = CognitiveService.get_prompt(memory_config)

        # 1. 构建参与者信息（使用列表推导式）
        participants_section = "# 对话参与者\n" + (
            "\n".join(
                f"- {p.get('name', '未知')}: {p.get('id', '未知')}" for p in user_list
            )
            if user_list
            else "暂无详细信息\n"
        )

        # 2. 构建新的规则
        rules_section = (
            '\n重要规则： 在生成任何判断或记忆时，你必须使用"对话参与者"部分提供的具体用户昵称来指代人物。'
            '绝对禁止使用"用户"、"提问者"等模糊代词。'
        )

        # 3. 构建AI身份信息
        ai_identity_section = ""
        if secretary_decision:
            persona_name = secretary_decision.get("persona_name", "")
            alias = secretary_decision.get("alias", "")
            if persona_name or alias:
                ai_info_parts = []
                if persona_name:
                    ai_info_parts.append(f"助理的名字是：{persona_name}")
                if alias:
                    ai_info_parts.append(f"昵称有：{alias}")
                ai_identity_section = (
                    "\n# AI身份信息\n" + "\n".join(ai_info_parts) + "\n"
                )

        # 4. 构建核心话题信息
        topic_section = ""
        if core_topic and core_topic.strip():
            topic_section = f"\n# 当前对话核心话题\n{core_topic.strip()}\n"

        # 5. 构建候选笔记清单
        notes_section = ""
        if candidate_notes:
            notes_section = NoteContextBuilder.build_candidate_list_for_prompt(
                candidate_notes
            )
            if topic_section:
                # 添加话题上下文说明，帮助LLM理解候选笔记的来源
                notes_section = f"\n# 基于上述核心话题检索到的相关笔记{notes_section}"

        # 6. 组装新的 system_prompt
        final_system_prompt = f"{system_prompt}\n\n{participants_section}{rules_section}{ai_identity_section}{topic_section}{notes_section}"

        # 7. 构建完整提示词（简化记忆处理）
        parts = [final_system_prompt]
        if formatted_query:
            parts.append(f"\n\n对话历史：\n{formatted_query}")
        if memories:
            parts.append(
                f"\n\n你回忆起了：\n{MemoryFormatter.format_memories_for_display(memories)}"
            )
        parts.append("\n\n请按照任务进行处理")

        return "".join(parts)

    @staticmethod
    def build_post_hoc_analysis_prompt(
        historical_query: str,
        main_llm_response: str,
        raw_memories: List,
        raw_notes: List,
        core_topic: str = "",
        memory_id_mapping: Dict[str, str] = None,
        note_id_mapping: Dict[str, str] = None,
        config=None,
    ) -> str:
        """
        构建用于事后反思的提示词

        Args:
            historical_query: 历史对话内容
            main_llm_response: 主LLM的回答内容
            raw_memories: 原始记忆列表
            raw_notes: 原始笔记列表
            core_topic: 核心话题（可选）
            config: 配置对象（可选）

        Returns:
            完整的反思提示词字符串
        """
        # 读取反思指南 - 使用 PathManager 统一管理路径
        from ...llm_memory.utils.path_manager import PathManager

        guide_path = PathManager.get_prompt_path()
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_content = f.read()

        # 构建反思上下文
        reflection_context = f"""
### 历史对话内容

{historical_query}

### 你做出的回答

{main_llm_response}

### 现在请你反思

根据以上对话和你做出的回答，请分析以下"相关记忆"和"相关笔记"哪些是真正有用的，哪些是无用的。并判断是否可以根据这次成功的回答，总结出任何新的、有价值的记忆。

#### 召回的相关记忆

"""

        # 添加记忆信息
        if raw_memories:
            for i, memory in enumerate(raw_memories):
                # 使用短ID显示（如果有映射表）
                short_id = (
                    memory_id_mapping.get(memory.id, memory.id)
                    if memory_id_mapping
                    else memory.id
                )
                reflection_context += f"""
- **id**: `{short_id}`
- **type**: `{memory.memory_type.value if hasattr(memory.memory_type, "value") else memory.memory_type}`
- **judgment**: `{memory.judgment}`
"""
        else:
            reflection_context += "\n无相关记忆\n"

        # 添加笔记信息
        reflection_context += "\n#### 你以前做的相关笔记\n"

        if raw_notes:
            for i, note in enumerate(raw_notes):
                # 使用短ID显示（如果有映射表）
                note_id = note.get("id", "N/A")
                short_id = (
                    note_id_mapping.get(note_id, note_id)
                    if note_id_mapping
                    else note_id
                )
                reflection_context += f"""
- **id**: `{short_id}`
- **content**: `{note.get("content", "")[:200]}...`
"""
        else:
            reflection_context += "\n无相关笔记\n"

        # 组合完整提示词
        full_prompt = f"""{guide_content}

---

{reflection_context}

---

请按照指南要求，先用自然语言详细描述你的反思过程，然后输出JSON格式的结果。
"""

        return full_prompt
