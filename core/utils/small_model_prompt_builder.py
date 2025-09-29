"""
小模型提示词构建器

负责构建用于小模型的记忆整理提示词。
"""

from typing import List, Dict, Tuple
from datetime import datetime
from ...llm_memory import CognitiveService
from ..session_memory import MemoryItem
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

        if diff < 60:
            return "刚刚"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes}分钟前"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours}小时前"
        else:
            days = int(diff / 86400)
            return f"{days}天前"

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
        elif isinstance(content, list):
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
    def build_memory_prompt(formatted_query: str, memories: List[MemoryItem], user_list: List[Dict]) -> str:
        """
        构建用于小模型的记忆整理提示词

        Args:
            formatted_query: 已格式化的对话历史字符串
            memories: 记忆列表
            user_list: 对话参与者清单

        Returns:
            完整的提示词字符串
        """
        # 获取系统提示词
        system_prompt = CognitiveService.get_prompt()

        # 1. 构建参与者信息
        participants_section = "# 对话参与者\n"
        if user_list:
            for p in user_list:
                participants_section += f"- {p.get('name', '未知')}: {p.get('id', '未知')}\n"
        else:
            participants_section += "暂无详细信息\n"

        # 2. 构建新的规则
        rules_section = (
            "\n重要规则： 在生成任何判断或记忆时，你必须使用“对话参与者”部分提供的具体用户昵称来指代人物。"
            "绝对禁止使用“用户”、“提问者”等模糊代词。"
        )

        # 3. 组装新的 system_prompt
        final_system_prompt = f"{system_prompt}\n\n{participants_section}{rules_section}"

        # 格式化记忆
        memory_context = ""
        if memories:
            memory_context = "\n\n你回忆起了：\n"
            formatted_display = MemoryFormatter.format_memories_for_display(memories)
            memory_context += formatted_display

        # 构建完整提示词
        parts = [final_system_prompt]

        if formatted_query:
            parts.append(f"\n\n对话历史：\n{formatted_query}")

        if memory_context:
            parts.append(memory_context)

        parts.append("\n\n请按照任务进行处理")

        return "".join(parts)
