"""
小模型提示词构建器

负责构建用于小模型的记忆整理提示词。
"""

from typing import List, Dict, Tuple
from datetime import datetime
from ...llm_memory import CognitiveService
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

            if role == "user":
                sender_name = msg.get('sender_name', '成员')
                sender_id = msg.get('sender_id', 'Unknown')

                # 构建参与者清单
                if sender_id != 'Unknown' and sender_id not in participants:
                    participants[sender_id] = {"id": sender_id, "name": sender_name}

                # 格式化对话文本
                timestamp = msg.get('timestamp', 0)
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
        unprocessed_records = [msg for msg in chat_records if not msg.get("is_processed", True)]

        # 如果没有未处理的记录，返回空字符串
        if not unprocessed_records:
            return ""

        # 提取最新的未处理记录内容
        return SmallModelPromptBuilder.extract_text_from_content(
            unprocessed_records[-1].get("content", [])
        ).strip()

    @staticmethod
    def build_memory_prompt(formatted_query: str, memories: List[MemoryItem], user_list: List[Dict], candidate_notes: List[Dict] = None, secretary_decision: Dict = None, core_topic: str = None) -> str:
        """
        构建用于小模型的记忆整理提示词

        Args:
            formatted_query: 已格式化的对话历史字符串
            memories: 记忆列表
            user_list: 对话参与者清单
            candidate_notes: 候选笔记列表（可选）
            secretary_decision: 秘书决策信息，包含AI人格和别名（可选）
            core_topic: 当前对话的核心话题（可选）

        Returns:
            完整的提示词字符串
        """
        from .note_context_builder import NoteContextBuilder

        # 获取系统提示词
        system_prompt = CognitiveService.get_prompt()

        # 1. 构建参与者信息（使用列表推导式）
        participants_section = "# 对话参与者\n" + (
            "\n".join(f"- {p.get('name', '未知')}: {p.get('id', '未知')}" for p in user_list)
            if user_list else "暂无详细信息\n"
        )

        # 2. 构建新的规则
        rules_section = (
            '\n重要规则： 在生成任何判断或记忆时，你必须使用"对话参与者"部分提供的具体用户昵称来指代人物。'
            '绝对禁止使用"用户"、"提问者"等模糊代词。'
        )

        # 3. 构建AI身份信息
        ai_identity_section = ""
        if secretary_decision:
            persona_name = secretary_decision.get('persona_name', '')
            alias = secretary_decision.get('alias', '')
            if persona_name or alias:
                ai_info_parts = []
                if persona_name:
                    ai_info_parts.append(f"助理的名字是：{persona_name}")
                if alias:
                    ai_info_parts.append(f"昵称有：{alias}")
                ai_identity_section = "\n# AI身份信息\n" + "\n".join(ai_info_parts) + "\n"

        # 4. 构建核心话题信息
        topic_section = ""
        if core_topic and core_topic.strip():
            topic_section = f"\n# 当前对话核心话题\n{core_topic.strip()}\n"

        # 5. 构建候选笔记清单
        notes_section = ""
        if candidate_notes:
            notes_section = NoteContextBuilder.build_candidate_list_for_prompt(candidate_notes)
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
            parts.append(f"\n\n你回忆起了：\n{MemoryFormatter.format_memories_for_display(memories)}")
        parts.append("\n\n请按照任务进行处理")

        return "".join(parts)
