"""
记忆类型处理器 - 统一的记忆存储和检索。

基于新的统一 BaseMemory 类，提供简化的记忆处理接口。
所有记忆类型使用相同的三元组结构（judgment, reasoning, tags）。
"""

from typing import List, Optional

from ..models.data_models import BaseMemory, MemoryType
from ...core.logger import get_logger


class MemoryHandler:
    """
    统一的记忆处理器。

    处理所有类型的记忆存储和检索，基于统一的三元组结构。
    """

    def __init__(self, memory_type: MemoryType, vector_store):
        """
        初始化记忆处理器。

        Args:
            memory_type: 记忆类型枚举
            vector_store: 向量存储实例
        """
        self.memory_type = memory_type
        self.store = vector_store
        self.logger = get_logger()

    def remember(self, judgment: str, reasoning: str, tags: List[str]) -> str:
        """
        记住一条记忆。

        Args:
            judgment: 论断
            reasoning: 解释
            tags: 标签列表

        Returns:
            创建的记忆ID
        """
        memory = BaseMemory(
            memory_type=self.memory_type,
            judgment=judgment,
            reasoning=reasoning,
            tags=tags
        )
        self.store.remember(memory)
        return memory.id

    def recall(self, query: str, limit: int = 10, include_consolidated: bool = True) -> List[BaseMemory]:
        """
        回忆相关记忆。

        Args:
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            include_consolidated: 是否包含已巩固的记忆

        Returns:
            相关的记忆列表
        """
        where_filter = {"memory_type": self.memory_type.value}
        if not include_consolidated:
            where_filter["is_consolidated"] = False
        return self.store.recall(query, limit, where_filter=where_filter)


class MemoryHandlerFactory:
    """记忆处理器工厂 - 创建和管理各种记忆处理器"""

    def __init__(self, vector_store):
        """
        初始化记忆处理器工厂。

        Args:
            vector_store: 向量存储实例
        """
        self.store = vector_store
        self.logger = get_logger()

        # 创建各种记忆处理器
        self.handlers = {
            "event": MemoryHandler(MemoryType.EVENT, vector_store),
            "knowledge": MemoryHandler(MemoryType.KNOWLEDGE, vector_store),
            "skill": MemoryHandler(MemoryType.SKILL, vector_store),
            "emotional": MemoryHandler(MemoryType.EMOTIONAL, vector_store),
            "task": MemoryHandler(MemoryType.TASK, vector_store)
        }

    def get_handler(self, memory_type: str) -> MemoryHandler:
        """
        获取指定类型的记忆处理器。

        Args:
            memory_type: 记忆类型（字符串key）

        Returns:
            对应的记忆处理器

        Raises:
            ValueError: 如果记忆类型不支持
        """
        if memory_type not in self.handlers:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
        return self.handlers[memory_type]