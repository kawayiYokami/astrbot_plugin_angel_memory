"""
记忆注入器

负责格式化记忆并注入到大模型的提示词中。
"""

from typing import List, Dict, Any
from ..session_memory import MemoryItem
from .memory_formatter import MemoryFormatter


class MemoryInjector:
    """记忆注入器"""

    @staticmethod
    def format_memories_for_prompt(feedback_data: Dict[str, Any], session_memories: List[MemoryItem]) -> str:
        """
        格式化记忆用于LLM提示词

        Args:
            feedback_data: 小模型返回的反馈数据
            session_memories: 会话记忆列表

        Returns:
            格式化后的记忆上下文
        """
        useful_memory_ids = feedback_data.get('useful_memory_ids', [])
        new_memories_raw = feedback_data.get('new_memories', {})

        # 转换 new_memories 格式：从字典（按类型分组）转换为列表
        new_memories = []
        if isinstance(new_memories_raw, dict):
            for memory_type, memories in new_memories_raw.items():
                if isinstance(memories, list):
                    for memory in memories:
                        if isinstance(memory, dict):
                            # 添加类型字段
                            memory['type'] = memory_type
                            new_memories.append(memory)
        elif isinstance(new_memories_raw, list):
            # 如果已经是列表，直接使用
            new_memories = new_memories_raw

        # 使用新的记忆格式化器
        return MemoryFormatter.format_memories_for_prompt(
            memories=session_memories,
            useful_memory_ids=useful_memory_ids,
            new_memories=new_memories
        )

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