"""
认知服务 - 统一的多类型记忆管理。

这是一个更高层次的服务，统一管理五种不同类型的记忆存储和检索。
为每种记忆类型提供专门的接口，同时支持跨记忆类型的复杂查询。
"""

from typing import List, Optional, Dict, Any

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
from ..models.data_models import BaseMemory
from ..config.system_config import system_config
from .memory_handlers import MemoryHandlerFactory
from .memory_manager import MemoryManager
from .memory_decay_policy import MemoryDecayConfig


class CognitiveService:
    """
    认知服务类（Cognitive Service）- 统一的多类型记忆管理（中文核心概念）

    中文定义：统一管理五种记忆类型的存储和检索，为每种记忆提供专门的接口
    英文翻译：Unified management of five types of memory storage and retrieval, providing dedicated interfaces for each memory type

    功能特点：
    - 统一接口：为不同类型的记忆提供一致的访问方式
    - 协调管理：协调不同类型的知识存储和检索
    - 系统大脑：作为整个记忆系统的控制中心
    - 清醒睡眠：支持清醒模式（学习强化）和睡眠模式（巩固遗忘）
    """

    def __init__(
        self,
        vector_store,
        memory_sql_manager=None,
        decay_config: MemoryDecayConfig | None = None,
    ):
        """
        初始化认知服务。

        Args:
            vector_store: 一个已经初始化好的、共享的向量索引实例。
        """
        # 设置日志记录器
        self.logger = logger

        if not vector_store:
            raise ValueError("必须提供一个 VectorStore 实例。")
        self.vector_store = vector_store
        self.memory_sql_manager = memory_sql_manager

        # 当前中央库为真相源，向量后端仅维护轻量召回索引；旧主集合不再创建。
        self.main_collection = None
        # 轻量记忆索引集合（仅 id + vector_text + embedding）
        self.memory_index_collection = (
            self.vector_store.get_or_create_collection_with_dimension_check(
                "memory_index"
            )
        )

        # 创建记忆处理器工厂
        self.memory_handler_factory = MemoryHandlerFactory(
            self.main_collection,
            self.vector_store,
            self.memory_sql_manager,
            self.memory_index_collection,
        )

        # 创建记忆管理器，并传入具体的 collection 对象
        self.memory_manager = MemoryManager(
            self.main_collection,
            self.vector_store,
            memory_sql_manager=self.memory_sql_manager,
            memory_index_collection=self.memory_index_collection,
            decay_config=decay_config,
        )

        # 记录初始化状态以验证向量索引
        self.logger.info(
            f"认知服务初始化完成。向量索引: {self.vector_store.client}"
        )

    # ===== 存储管理接口 =====

    def set_storage_path(self, new_path: str):
        """
        设置记忆系统的新存储路径。

        这个方法会更新所有向量存储的路径，包括主存储和关联管理器使用的存储。

        注意：切换路径后，之前的数据仍在原路径中。如果需要迁移数据，
        请手动复制数据库文件到新路径。

        Args:
            new_path: 新的存储路径（可以是绝对路径或相对路径）
        """
        try:
            # 更新主存储路径
            self.vector_store.set_storage_path(new_path)

            # 更新系统配置中的路径 - 通过 PathManager 统一管理
            # 注意：这里需要通过 PluginContext 获取 PathManager 实例
            # 但当前方法没有 plugin_context 参数，所以暂时注释掉
            # 如果需要更新路径，应该通过外部调用 PathManager 的方法

            self.logger.info(f"记忆系统存储路径已更新到: {new_path}")
            self.logger.warning(
                "注意：system_config.index_dir 未更新，请通过 PathManager 更新路径"
            )

        except Exception as e:
            self.logger.error(f"更新存储路径失败: {e}")
            raise

    # ===== 记忆接口 =====

    async def remember(
        self,
        memory_type: str,
        judgment: str,
        reasoning: str,
        tags: List[str],
        is_active: bool = False,
        strength: Optional[int] = None,
        memory_scope: str = "public",
    ) -> str:
        """
        记住一条记忆。

        Args:
            memory_type: 记忆类型（event/knowledge/skill/emotional/task）
            judgment: 论断
            reasoning: 解释
            tags: 标签列表
            is_active: 是否为主动记忆，主动记忆永不衰减。
            strength: 记忆强度（可选），如果未提供则使用被动记忆默认强度。

        Returns:
            创建的记忆ID
        """
        handler = self.memory_handler_factory.get_handler(memory_type)
        memory = await handler.remember(
            judgment, reasoning, tags, is_active, strength, memory_scope
        )
        return memory.id

    async def recall(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        """
        回忆记忆。

        Args:
            memory_type: 记忆类型（event/knowledge/skill/emotional/task）
            query: 搜索查询
            limit: 返回数量限制

        Returns:
            记忆列表
        """
        handler = self.memory_handler_factory.get_handler(memory_type)
        return await handler.recall(
            query, limit, memory_scope=memory_scope
        )

    # ===== 高级记忆功能 =====

    async def comprehensive_recall(
        self,
        query: str,
        limit: int = None,
        event=None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        """执行统一记忆检索。"""
        return await self.memory_manager.comprehensive_recall(
            query=query,
            limit=limit,
            event=event,
            vector=vector,
            memory_scope=memory_scope,
        )

    async def consolidate_memories(self):
        """执行记忆巩固过程（睡眠模式）"""
        return await self.memory_manager.consolidate_memories()

    async def chained_recall(
        self,
        query: str,
        entities: List[str],
        per_type_limit: int = 7,
        final_limit: Optional[int] = None,
        vector: Optional[List[float]] = None,
        event=None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        """链式多通道回忆 - 基于关联网络的多轮回忆

        Args:
            query: 查询字符串
            entities: 核心实体列表
            per_type_limit: 每种记忆类型的限制
            final_limit: 最终返回数量限制
            vector: 预计算的查询向量（可选）
            event: 消息事件
        """
        return await self.memory_manager.chained_recall(
            query=query, 
            entities=entities, 
            per_type_limit=per_type_limit, 
            final_limit=final_limit, 
            vector=vector,
            event=event,
            memory_scope=memory_scope,
        )

    async def get_memories_by_ids(
        self,
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        """按 ID 批量读取长期记忆真相源。"""
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return []

        if self.memory_sql_manager is not None:
            memories = await self.memory_sql_manager.get_memories_by_ids(ids)
            scope = str(memory_scope or "").strip()
            memory_map = {
                str(memory.id): memory
                for memory in memories or []
                if getattr(memory, "id", None)
            }
            ordered: List[BaseMemory] = []
            seen = set()
            for memory_id in ids:
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                memory = memory_map.get(memory_id)
                if memory is None:
                    continue
                if scope:
                    mem_scope = str(
                        getattr(memory, "memory_scope", "public") or "public"
                    ).strip()
                    if scope == "public":
                        if mem_scope != "public":
                            continue
                    elif mem_scope not in {scope, "public"}:
                        continue
                ordered.append(memory)
            return ordered

        if self.main_collection is not None:
            try:
                result = self.main_collection.get(ids=ids, include=["metadatas"])
                result_ids = [str(mid) for mid in (result.get("ids") or [])]
                metadatas = result.get("metadatas") or []
                memories_by_id: Dict[str, BaseMemory] = {}
                for memory_id, metadata in zip(result_ids, metadatas):
                    if not isinstance(metadata, dict):
                        continue
                    memory = self.memory_manager._build_memory_from_metadata(
                        memory_id,
                        metadata,
                    )
                    if memory is not None:
                        memories_by_id[memory_id] = memory
                scope = str(memory_scope or "").strip()
                ordered = []
                for memory_id in ids:
                    memory = memories_by_id.get(memory_id)
                    if memory is None:
                        continue
                    if scope:
                        mem_scope = str(
                            getattr(memory, "memory_scope", "public") or "public"
                        ).strip()
                        if scope == "public":
                            if mem_scope != "public":
                                continue
                        elif mem_scope not in {scope, "public"}:
                            continue
                    ordered.append(memory)
                return ordered
            except Exception as e:
                self.logger.warning(f"按 ID 读取旧向量记忆失败: {e}", exc_info=True)

        return []

    async def feedback(
        self,
        useful_memory_ids: List[str] = None,
        recalled_memory_ids: List[str] = None,
        memory_actions: List[dict] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        """统一反馈接口 - 处理回忆后的反馈（核心工作流）"""
        memory_handlers = self.memory_handler_factory.handlers
        return await self.memory_manager.process_feedback(
            useful_memory_ids,
            recalled_memory_ids,
            memory_actions,
            memory_handlers,
            memory_scope=memory_scope,
        )

    # ===== 管理功能 =====

    def clear_all_memories(self):
        """
        清空所有记忆。

        这是一个危险操作，会永久删除所有存储的记忆。
        """
        if self.memory_sql_manager is not None:
            self.logger.warning("中央库模式不支持通过认知服务直接清空全部记忆。")
            return
        if self.main_collection is not None:
            self.vector_store.clear_collection(self.main_collection)
        self.logger.info("所有记忆向量索引已被清空。")

    @staticmethod
    def get_prompt(memory_config=None) -> str:
        """
        获取记忆系统使用指南的提示词。

        下游模块可以将此提示词加入到系统提示词中，
        AI就能知道如何维护记忆系统。

        Args:
            memory_config: 记忆配置对象（为了保持兼容性，但实际总是使用异步提示词）

        Returns:
            记忆系统使用指南的完整内容
        """
        from ..prompts.prompt_assembler import PromptAssembler

        try:
            return PromptAssembler.build_memory_system_guide()
        except FileNotFoundError:
            return "记忆系统提示词文件未找到，请检查文件是否存在。"
        except Exception as e:
            return f"读取记忆系统提示词失败: {str(e)}"
