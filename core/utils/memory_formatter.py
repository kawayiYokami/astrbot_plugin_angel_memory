"""
记忆格式化器

负责将记忆格式化为统一的文本格式，便于阅读和理解。
"""

from typing import List, Dict, Any, Optional
from ..session_memory import MemoryItem


class MemoryFormatter:
    """记忆格式化器"""

    # 记忆类型到中文的映射
    MEMORY_TYPE_NAMES = {
        'knowledge': '知识',
        'event': '事件',
        'skill': '技能',
        'emotional': '情感',
        'task': '任务',
        'meta': '元记忆',
        'sensory': '感官记忆'
    }

    @staticmethod
    def format_single_memory(memory: MemoryItem) -> str:
        """
        格式化单条记忆

        Args:
            memory: 记忆对象

        Returns:
            格式化后的记忆文本
        """
        # 获取记忆类型的中文名称
        type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(memory.type, memory.type)
        
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
        # 按类型分组
        grouped_memories = {}
        for memory in memories:
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(memory.type, memory.type)
            if type_name not in grouped_memories:
                grouped_memories[type_name] = []
            
            formatted_memory = MemoryFormatter.format_single_memory(memory)
            grouped_memories[type_name].append(formatted_memory)
        
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
        
        # 收集要格式化的记忆
        memories_to_format = []
        
        # 添加有用的历史记忆
        if useful_memory_ids:
            for memory_id in useful_memory_ids:
                if memory_id in memory_map:
                    memories_to_format.append(memory_map[memory_id])
        
        # 添加新记忆
        if new_memories:
            for new_memory in new_memories:
                # 创建临时记忆对象
                temp_memory = MemoryItem(
                    id=new_memory.get('id', 'new'),
                    type=new_memory.get('type', 'knowledge'),
                    judgment=new_memory.get('judgment', ''),
                    reasoning=new_memory.get('reasoning', ''),
                    tags=new_memory.get('tags', []),
                    strength=new_memory.get('strength', 0.5)
                )
                memories_to_format.append(temp_memory)

        if not memories_to_format:
            return ""

        # 按类型分组并格式化
        grouped_memories = MemoryFormatter.format_memories_by_type(memories_to_format)
        
        # 构建最终文本
        result_parts = ["[相关记忆]"]
        
        for memory_type, formatted_memories in grouped_memories.items():
            result_parts.append(f"\n[{memory_type}]")
            for memory_text in formatted_memories:
                result_parts.append(f"\n{memory_text}")
        
        return "".join(result_parts)

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
        grouped = {}
        for memory in memories:
            type_name = MemoryFormatter.MEMORY_TYPE_NAMES.get(memory.type, memory.type)
            if type_name not in grouped:
                grouped[type_name] = []
            grouped[type_name].append(memory)

        result_parts = []
        for memory_type, memory_list in grouped.items():
            result_parts.append(f"\n=== {memory_type} ===")
            for i, memory in enumerate(memory_list, 1):
                # 生成短ID（前6位）
                short_id = memory.id[:6] if len(memory.id) >= 6 else memory.id
                # 格式化单条记忆
                judgment = memory.judgment.strip()
                reasoning = ""
                if memory.reasoning and memory.reasoning.strip():
                    reasoning = f"\n   ——因为{memory.reasoning.strip()}"
                result_parts.append(f"\n{i}. [id:{short_id}]{judgment}{reasoning}")

        return "".join(result_parts)