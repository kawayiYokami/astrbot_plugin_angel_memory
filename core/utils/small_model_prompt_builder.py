"""
小模型提示词构建器

负责构建用于小模型的记忆整理提示词。
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from ..session_memory import MemoryItem
from .memory_formatter import MemoryFormatter


class SmallModelPromptBuilder:
    """小模型提示词构建器"""

    # 记忆创建时间统一按 UTC+8 显示，与用户所在时区一致
    MEMORY_TIMEZONE = timezone(timedelta(hours=8))

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
        把对话记录按时间顺序拼成纯文本，并生成对话参与者清单。

        天使之心模式下 prompt 已带「[名字 (ID: 号)] (时间): 」前缀，
        原生模式下 prompt 就是原文；两种都原样照搬，不再补署名行、缩进、
        时间行，也不渲染引用。这里只做过滤、排序与参与者收集。

        Args:
            chat_records: 对话记录列表

        Returns:
            一个元组，包含：
            - 拼接后的对话文本字符串
            - 对话参与者清单（去重后）
        """
        usable: List[Dict] = []
        for msg in chat_records or []:
            if not isinstance(msg, dict):
                continue

            # 过滤掉toolcall相关消息
            if msg.get("is_structured_toolcall"):
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                toolcall_keywords = ["调用", "tool_call", "function", "工具调用结果："]
                if any(keyword in content for keyword in toolcall_keywords):
                    continue

            usable.append(msg)

        # 严格按发送时间升序：sorted 是稳定排序，时间戳相同就保持传入顺序
        usable.sort(key=lambda item: float(item.get("timestamp", 0) or 0))

        lines: List[str] = []
        participants: Dict[str, Dict] = {}
        for msg in usable:
            text = SmallModelPromptBuilder.extract_text_from_content(
                msg.get("content", "")
            )
            if not text.strip():
                continue
            lines.append(text)

            if str(msg.get("role", "") or "").strip() == "assistant":
                continue
            sender_id = str(msg.get("sender_id", "") or "").strip()
            sender_name = str(msg.get("sender_name", "") or "").strip()
            if (
                sender_id
                and sender_id.lower() != "unknown"
                and sender_id not in participants
            ):
                participants[sender_id] = {
                    "id": sender_id,
                    "name": sender_name or "群友",
                }

        return "\n".join(lines), list(participants.values())

    @staticmethod
    def _format_memory_created_at(value) -> str:
        """把 created_at 转成 YYYY-MM-DD HH:MM（UTC+8）。

        取不到有效时间时返回空串：不输出 1970-01-01，
        否则模型会误以为这条记忆比所有东西都旧。
        """
        try:
            seconds = float(value or 0)
        except (TypeError, ValueError):
            return ""
        if seconds <= 0:
            return ""
        moment = datetime.fromtimestamp(seconds, tz=SmallModelPromptBuilder.MEMORY_TIMEZONE)
        return moment.strftime("%Y-%m-%d %H:%M")

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
        # 获取系统提示词
        if system_prompt is None:
            # p-memory 模式：直接取提示词装配器，不再经过已下线的 CognitiveService
            from ..prompts.prompt_assembler import PromptAssembler

            system_prompt = PromptAssembler.build_memory_system_guide()

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

        # 5. 组装新的 system_prompt (已移除废弃的笔记清单)
        final_system_prompt = f"{system_prompt}\n\n{participants_section}{rules_section}{ai_identity_section}{topic_section}"

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
    def build_post_hoc_analysis_prompts(
        historical_query: str,
        main_llm_response: str,
        raw_memories: List,
        core_topic: str = "",
        memory_id_mapping: Dict[str, str] = None,
        config=None,
        persona_name: str = "",
    ) -> Tuple[str, str]:
        """
        构建用于事后反思的提示词，返回 (system_prompt, user_prompt)

        system_prompt 固定为记忆系统指南，便于命中提示词缓存；
        user_prompt 只放本次对话数据与输出要求。

        Args:
            historical_query: 历史对话内容
            main_llm_response: 主LLM的回答内容
            raw_memories: 原始记忆列表
            core_topic: 核心话题（可选）
            memory_id_mapping: 记忆ID映射表（可选）
            config: 配置对象（可选）

        Returns:
            (system_prompt, user_prompt)
        """
        from ...llm_memory.prompts.prompt_assembler import PromptAssembler

        system_prompt = PromptAssembler.build_memory_system_guide()

        # 构建数据上下文（仅包含实际数据，不包含指令）
        sections = ["# 本次对话数据"]
        ai_name = str(persona_name or "").strip()
        if ai_name:
            sections += ["", "## 你自己", f"你叫 {ai_name}。"]
            # 只有天使之心会在正文前套发言人前缀，原生模式下每段都没有前缀，这句会误导
            if "[群友" in str(historical_query or ""):
                sections.append(
                    "下面「历史对话内容」里没有「[群友: …]」前缀的段落，就是你自己说过的话。"
                )
        sections += ["", "## 历史对话内容", historical_query]
        reply_text = str(main_llm_response or "").strip()
        if reply_text:
            sections += ["", "## 你的最新回复", reply_text]
        else:
            sections += [
                "",
                "## 你的最新回复",
                "本轮没有回复记录：发送现场和 LLM 响应钩子都没有取到文本。"
                "这是记录缺失，不代表你没有说话，也不代表本轮没有回复；"
                "不要据此推断自己没开口，也不要为补全这一点编造回复内容。",
            ]

        sections += ["", "## 召回的相关记忆"]
        data_context = "\n".join(sections) + "\n"

        # 添加记忆信息
        if raw_memories:
            for memory in raw_memories:
                short_id = (
                    memory_id_mapping.get(memory.id, memory.id)
                    if memory_id_mapping
                    else memory.id
                )
                created_at = SmallModelPromptBuilder._format_memory_created_at(
                    getattr(memory, "created_at", None)
                )
                stamp = f" {created_at}" if created_at else ""
                data_context += f"id:{short_id}{stamp} - {memory.judgment}\n"
        else:
            data_context += "\n无相关记忆\n"

        user_prompt = f"""{data_context}

---

现在请开始分析并输出JSON结果。
"""

        return system_prompt, user_prompt
