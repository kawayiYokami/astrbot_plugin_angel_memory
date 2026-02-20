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

    def __init__(
        self,
        memory_type: MemoryType,
        main_collection,
        vector_store,
        memory_sql_manager=None,
        memory_index_collection=None,
    ):
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
        self.memory_sql_manager = memory_sql_manager
        self.memory_index_collection = memory_index_collection
        self.logger = logger
        from ..config.system_config import system_config as global_system_config # 导入全局配置
        self.system_config = global_system_config


    async def remember(
        self,
        judgment: str,
        reasoning: str,
        tags: List[str],
        is_active: bool = False,
        strength: Optional[int] = None,
        memory_scope: str = "public",
    ) -> BaseMemory:
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
        
        # 新架构：中央 SQL 为真相源，向量层仅做轻量索引。
        if self.memory_sql_manager is not None:
            memory = await self.memory_sql_manager.remember(
                memory_type=self.memory_type.value,
                judgment=judgment,
                reasoning=reasoning,
                tags=tags,
                is_active=is_active,
                strength=actual_strength,
                memory_scope=memory_scope,
            )
            if self.memory_index_collection is not None:
                try:
                    vector_text = self.memory_sql_manager._build_vector_text(
                        judgment=memory.judgment,
                        tags=memory.tags,
                    )
                    if vector_text:
                        await self.store.upsert_memory_index_rows(
                            collection=self.memory_index_collection,
                            rows=[{"id": memory.id, "vector_text": vector_text}],
                        )
                except Exception as e:
                    self.logger.warning(f"写入轻量向量索引失败（不影响主流程）: {e}")
            return memory

        # 兼容旧路径（无中央 SQL 时）
        memory = BaseMemory(
            memory_type=self.memory_type,
            judgment=judgment,
            reasoning=reasoning,
            tags=tags,
            is_active=is_active,
            strength=actual_strength,
            memory_scope=memory_scope,
        )
        await self.store.remember(self.collection, memory)
        return memory

    async def recall(
        self,
        query: str,
        limit: int = 10,
        include_consolidated: bool = True,
        similarity_threshold: float = 0.6,
        memory_scope: Optional[str] = None,
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
        clauses = [{"memory_type": self.memory_type.value}]
        if not include_consolidated:
            clauses.append({"is_consolidated": False})

        scope = str(memory_scope or "").strip()
        if scope:
            if scope == "public":
                clauses.append({"memory_scope": "public"})
            else:
                clauses.append(
                    {"$or": [{"memory_scope": scope}, {"memory_scope": "public"}]}
                )

        # 新架构下 recall 统一走中央 SQL 的 tags 召回。
        if self.memory_sql_manager is not None:
            vector_scores = None
            if self.memory_index_collection is not None:
                try:
                    id_scores = await self.store.recall_memory_ids(
                        collection=self.memory_index_collection,
                        query=query,
                        limit=max(1, int(limit) * 4),
                        similarity_threshold=0.0,
                    )
                    if id_scores:
                        vector_scores = {mid: score for mid, score in id_scores}
                except Exception as e:
                    self.logger.warning(f"记忆 recall 读取向量分失败，降级为 FTS-only: {e}")
            recalled = await self.memory_sql_manager.recall_by_tags(
                query=query,
                limit=limit,
                memory_scope=memory_scope or "public",
                vector_scores=vector_scores,
            )
            filtered = [
                mem for mem in recalled
                if getattr(mem.memory_type, "value", str(mem.memory_type)) == self.memory_type.value
            ]
            return filtered

        where_filter = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        return await self.store.recall(
            self.collection,
            query,
            limit,
            where_filter=where_filter,
            similarity_threshold=similarity_threshold,
        )


class MemoryHandlerFactory:
    """记忆处理器工厂 - 创建和管理各种记忆处理器"""

    def __init__(
        self,
        main_collection,
        vector_store,
        memory_sql_manager=None,
        memory_index_collection=None,
    ):
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
            "event": MemoryHandler(
                MemoryType.EVENT,
                self.collection,
                self.store,
                memory_sql_manager,
                memory_index_collection,
            ),
            "knowledge": MemoryHandler(
                MemoryType.KNOWLEDGE,
                self.collection,
                self.store,
                memory_sql_manager,
                memory_index_collection,
            ),
            "skill": MemoryHandler(
                MemoryType.SKILL,
                self.collection,
                self.store,
                memory_sql_manager,
                memory_index_collection,
            ),
            "emotional": MemoryHandler(
                MemoryType.EMOTIONAL,
                self.collection,
                self.store,
                memory_sql_manager,
                memory_index_collection,
            ),
            "task": MemoryHandler(
                MemoryType.TASK,
                self.collection,
                self.store,
                memory_sql_manager,
                memory_index_collection,
            ),
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
