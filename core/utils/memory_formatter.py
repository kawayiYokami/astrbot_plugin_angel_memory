"""
记忆格式化器

负责将记忆格式化为统一的文本格式，便于阅读和理解。
"""

from typing import List, Dict, Any, Optional
from ..session_memory import MemoryItem
from ..config import MemoryConstants


class MemoryFormatter:
    """记忆格式化器"""

    # 记忆类型到中文的映射
    MEMORY_TYPE_NAMES = MemoryConstants.MEMORY_TYPE_NAMES

    @staticmethod
    def format_single_memory(memory: MemoryItem) -> str:
        """
        格式化单条记忆

        Args:
            memory: 记忆对象

        Returns:
            格式化后的记忆文本
        """
        # 处理 MemoryItem 对象的 memory_type 属性

        # 格式化论点
        judgment = memory.judgment.strip()

        # 格式化理由
        reasoning = ""
        if memory.reasoning and memory.reasoning.strip():
            reasoning = f"\n——因为{memory.reasoning.strip()}"

        return f"{judgment}{reasoning}"

    @staticmethod
    def format_memories_by_type(memories: List[MemoryItem]) -> Dict[str, List[str]]:
        """
        按类型分组格式化记忆

        Args:
            memories: 记忆列表

        Returns:
            按类型分组的格式化记忆字典
        """
        # 按类型分组（使用setdefault简化）
        grouped_memories = {}
        for memory in memories:
            # 处理 MemoryItem 对象的 memory_type 属性
            type_value = memory.memory_type
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(type_value, type_value)
            grouped_memories.setdefault(type_name, []).append(MemoryFormatter.format_single_memory(memory))
        return grouped_memories

    @staticmethod
    def format_memories_for_prompt(memories: List[MemoryItem],
                                  useful_memory_ids: Optional[List[str]] = None,
                                  new_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        格式化记忆用于提示词

        Args:
            memories: 记忆列表
            useful_memory_ids: 有用记忆ID列表
            new_memories: 新记忆列表

        Returns:
            格式化后的记忆文本
        """
        if not memories and not new_memories:
            return ""

        # 创建ID到记忆的映射
        memory_map = {memory.id: memory for memory in memories}
        memories_to_format = []

        # 添加有用的历史记忆（使用列表推导式）
        if useful_memory_ids:
            memories_to_format.extend(memory_map[mid] for mid in useful_memory_ids if mid in memory_map)

        # 添加新记忆（使用列表推导式）
        if new_memories:
            memories_to_format.extend(
                MemoryItem(
                    id=nm.get('id', 'new'),
                    memory_type=nm.get('type', 'knowledge'),
                    judgment=nm.get('judgment', ''),
                    reasoning=nm.get('reasoning', ''),
                    tags=nm.get('tags', []),
                    strength=nm.get('strength', 0.5)
                ) for nm in new_memories
            )

        if not memories_to_format:
            return ""

        # 按类型分组并格式化
        grouped_memories = MemoryFormatter.format_memories_by_type(memories_to_format)

        # 构建最终文本（使用extend简化）
        formatted_lines = ["[相关记忆]"]
        for memory_type, formatted_memories in grouped_memories.items():
            formatted_lines.append(f"\n[{memory_type}]")
            formatted_lines.extend(f"\n{memory_text}" for memory_text in formatted_memories)

        return "".join(formatted_lines)

    @staticmethod
    def format_fifo_memories(memories: List[MemoryItem]) -> str:
        """
        专门格式化来自FIFO短期记忆的列表，无过滤。

        Args:
            memories: 记忆列表

        Returns:
            格式化后的记忆文本
        """
        if not memories:
            return ""

        # 按类型分组并格式化
        grouped_memories = MemoryFormatter.format_memories_by_type(memories)

        # 构建最终文本
        formatted_lines = ["[相关记忆]"]
        for memory_type, formatted_memories in grouped_memories.items():
            formatted_lines.append(f"\n[{memory_type}]")
            formatted_lines.extend(f"\n{memory_text}" for memory_text in formatted_memories)

        return "".join(formatted_lines)

    @staticmethod
    def format_memories_for_display(memories: List[MemoryItem]) -> str:
        """
        格式化记忆用于显示

        Args:
            memories: 记忆列表

        Returns:
            格式化后的记忆文本
        """
        if not memories:
            return "暂无记忆"

        # 按类型分组（保留原始对象）
        from .memory_id_resolver import MemoryIDResolver
        grouped = {}
        for memory in memories:
            # 处理 MemoryItem 对象的 memory_type 属性
            type_value = memory.memory_type
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(type_value, type_value)
            grouped.setdefault(type_name, []).append(memory)

        display_lines = []
        for memory_type, memory_list in grouped.items():
            display_lines.append(f"\n=== {memory_type} ===")
            for i, memory in enumerate(memory_list, 1):
                # 生成短ID并格式化记忆
                short_id = MemoryIDResolver.generate_short_id(memory.id)
                judgment = memory.judgment.strip()
                reasoning = f"\n   ——因为{memory.reasoning.strip()}" if memory.reasoning and memory.reasoning.strip() else ""
                display_lines.append(f"\n{i}. [id:{short_id}]{judgment}{reasoning}")

        return "".join(display_lines)