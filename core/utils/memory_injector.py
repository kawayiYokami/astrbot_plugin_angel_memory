"""
记忆注入器

负责格式化记忆并注入到大模型的提示词中。
"""

from typing import List, Dict, Any
from ..session_memory import MemoryItem
from .memory_formatter import MemoryFormatter
from .memory_id_resolver import MemoryIDResolver


class MemoryInjector:
    """记忆注入器"""

    @staticmethod
    def format_memories_for_prompt(
        feedback_data: Dict[str, Any], session_memories: List[MemoryItem]
    ) -> str:
        """
        格式化记忆用于LLM提示词

        Args:
            feedback_data: 小模型返回的反馈数据
            session_memories: 会话记忆列表

        Returns:
            格式化后的记忆上下文
        """
        useful_memory_ids = feedback_data.get("useful_memory_ids", [])
        memory_actions_raw = feedback_data.get("memory_actions", [])

        memory_actions = MemoryIDResolver.normalize_memory_actions_format(
            memory_actions_raw
        )
        action_memories = [
            dict(action.get("memory", {}))
            for action in memory_actions
            if isinstance(action, dict) and isinstance(action.get("memory"), dict)
        ]

        # 使用记忆格式化器
        return MemoryFormatter.format_memories_for_prompt(
            memories=session_memories,
            useful_memory_ids=useful_memory_ids,
            action_memories=action_memories,
        )

    @staticmethod
    def format_session_memories_for_prompt(session_memories: List[MemoryItem]) -> str:
        """
        格式化会话工作记忆用于LLM提示词

        Args:
            session_memories: 会话工作记忆列表

        Returns:
            格式化后的记忆上下文
        """
        return MemoryFormatter.format_session_memories(memories=session_memories)

    @staticmethod
    def inject_into_system_prompt(system_prompt: str, memory_context: str) -> str:
        """
        将记忆上下文注入到系统提示词中

        Args:
            system_prompt: 原始系统提示词
            memory_context: 记忆上下文

        Returns:
            注入记忆后的系统提示词
        """
        if not memory_context:
            return system_prompt

        return f"{system_prompt}\n\n{memory_context}"
