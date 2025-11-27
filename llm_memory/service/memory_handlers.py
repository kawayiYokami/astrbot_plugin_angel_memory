"""
记忆类型处理器 - 统一的记忆存储和检索。

基于新的统一 BaseMemory 类，提供简化的记忆处理接口。
所有记忆类型使用相同的三元组结构（judgment, reasoning, tags）。
"""

from typing import List, Optional # 导入 Optional

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
from ..models.data_models import BaseMemory, MemoryType


class MemoryHandler:
    """
    统一的记忆处理器。

    处理所有类型的记忆存储和检索，基于统一的三元组结构。
    """

    def __init__(self, memory_type: MemoryType, main_collection, vector_store):
        """
        初始化记忆处理器。

        Args:
            memory_type: 记忆类型枚举
            main_collection: 用于操作的主集合实例
            vector_store: 向量存储实例
        """
        self.memory_type = memory_type
        self.collection = main_collection
        self.store = vector_store
        self.logger = logger
        from ..config.system_config import system_config as global_system_config # 导入全局配置
        self.system_config = global_system_config


    async def remember(self, judgment: str, reasoning: str, tags: List[str], is_active: bool = False, strength: Optional[int] = None) -> BaseMemory:
        """
        记住一条记忆。

        Args:
            judgment: 论断
            reasoning: 解释
            tags: 标签列表
            is_active: 是否为主动记忆，主动记忆永不衰减。
            strength: 记忆强度（可选），如果未提供则使用被动记忆默认强度。

        Returns:
            创建的记忆对象
        """
        actual_strength = strength if strength is not None else self.system_config.default_passive_strength # 获取实际强度
        
        memory = BaseMemory(
            memory_type=self.memory_type,
            judgment=judgment,
            reasoning=reasoning,
            tags=tags,
            is_active=is_active,
            strength=actual_strength, # 将实际强度传递给 BaseMemory 构造函数
        )
        await self.store.remember(self.collection, memory)
        return memory

    async def recall(
        self,
        query: str,
        limit: int = 10,
        include_consolidated: bool = True,
        similarity_threshold: float = 0.6,
    ) -> List[BaseMemory]:
        """
        回忆相关记忆。

        Args:
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            include_consolidated: 是否包含已巩固的记忆
            similarity_threshold: 相似度阈值（0.0-1.0），低于此阈值的结果将被过滤

        Returns:
            相关的记忆列表
        """
        where_filter = {"memory_type": self.memory_type.value}
        if not include_consolidated:
            where_filter["is_consolidated"] = False
        return await self.store.recall(
            self.collection,
            query,
            limit,
            where_filter=where_filter,
            similarity_threshold=similarity_threshold,
        )


class MemoryHandlerFactory:
    """记忆处理器工厂 - 创建和管理各种记忆处理器"""

    def __init__(self, main_collection, vector_store):
        """
        初始化记忆处理器工厂。

        Args:
            main_collection: 用于操作的主集合实例
            vector_store: 向量存储实例
        """
        self.collection = main_collection
        self.store = vector_store
        self.logger = logger

        # 创建各种记忆处理器
        self.handlers = {
            "event": MemoryHandler(MemoryType.EVENT, self.collection, self.store),
            "knowledge": MemoryHandler(
                MemoryType.KNOWLEDGE, self.collection, self.store
            ),
            "skill": MemoryHandler(MemoryType.SKILL, self.collection, self.store),
            "emotional": MemoryHandler(
                MemoryType.EMOTIONAL, self.collection, self.store
            ),
            "task": MemoryHandler(MemoryType.TASK, self.collection, self.store),
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
