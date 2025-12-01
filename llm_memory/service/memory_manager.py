"""
记忆管理器 - 处理记忆的高级功能。

这个模块包含了记忆系统的高级功能，如记忆巩固、强化、
链式回忆、记忆合并等复杂操作。
"""

from typing import List, Optional, Dict, Any
import logging
from collections import defaultdict

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)
from ..models.data_models import BaseMemory, MemoryType, ValidationError
from ..components.association_manager import AssociationManager, MemorySnapshot
from ..config.system_config import system_config

# 导入查询处理器（用于统一检索词预处理）
from ...core.utils.query_processor import get_query_processor


class MemoryManager:
    """
    记忆管理器类 - 处理记忆的高级功能。

    负责记忆的巩固、强化、链式回忆、记忆合并等复杂操作。
    这些功能不直接与特定记忆类型绑定，而是处理记忆的通用行为。
    """

    def __init__(
        self, main_collection, vector_store, association_manager: AssociationManager
    ):
        """
        初始化记忆管理器。

        Args:
            main_collection: 用于操作的主集合实例
            vector_store: 向量存储实例 (用于调用 recall, update_memory 等高级方法)
            association_manager: 关联管理器实例
        """
        self.collection = main_collection
        self.store = vector_store
        self.association_manager = association_manager
        self.logger = logger

        # 初始化查询处理器（用于统一检索词预处理）
        self.query_processor = get_query_processor()

    # ===== 记忆巩固和强化 =====

    async def consolidate_memories(self):
        """
        执行记忆清理过程（睡眠模式）。

        清理规则：
        - 删除被动记忆中强度 ≤ 0 的记忆
        - 主动记忆永不删除（不受强度影响）
        - 清理弱关联
        """
        self.logger.debug("开始记忆清理过程...")

        try:
            # 删除被动记忆中强度 ≤ 0 的记忆
            weak_results = self.collection.get(
                where={"$and": [{"strength": {"$lte": 0}}, {"is_active": False}]},
                include=["metadatas"]
            )
            deleted_count = 0
            if weak_results and weak_results["ids"]:
                self.collection.delete(ids=weak_results["ids"])
                deleted_count = len(weak_results["ids"])
                self.logger.debug(f"已删除 {deleted_count} 条被动记忆（强度≤0）")

            # 清理弱关联
            await self.association_manager.cleanup_weak_associations()

            self.logger.debug(f"记忆清理完成: 删除{deleted_count}条被动记忆")

        except Exception as e:
            self.logger.error(f"记忆清理过程失败: {str(e)}")
            raise

    async def reinforce_memories(self, memory_ids: List[str]):
        """
        强化记忆强度（清醒模式）。

        当记忆被成功使用时，增加其强度计数。

        Args:
            memory_ids: 要强化的记忆ID列表
        """
        for memory_id in memory_ids:
            try:
                # 根据 metadata 中的 id 查找记忆
                result = self.collection.get(where={"id": memory_id}, limit=1)
                if result and result["metadatas"] and result["ids"]:
                    current_meta = result["metadatas"][0]
                    current_strength = current_meta.get("strength", 1)
                    # 获取 ChromaDB 的文档 ID
                    chroma_doc_id = result["ids"][0]
                    # 使用 ChromaDB 文档 ID 进行更新
                    await self.store.update_memory(
                        self.collection, chroma_doc_id, {"strength": current_strength + 3}
                    )
            except Exception as e:
                self.logger.error(f"强化记忆 {memory_id} 失败: {str(e)}")

    # ===== 高级回忆功能 =====

    async def comprehensive_recall(
        self,
        query: str,
        limit: int = None,
        event=None,
        vector: Optional[List[float]] = None,
    ) -> List[BaseMemory]:
        """
        统一记忆检索 - 使用混合检索获取所有相关记忆。

        Args:
            query: 搜索查询字符串
            limit: 最大返回数量
            event: 消息事件（用于查询词预处理）
            vector: 可选的预计算向量，如果提供则直接使用

        Returns:
            记忆列表（相似度 >= 0.5）
        """
        # 预处理查询词
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(
                query, event
            )

        if limit is None:
            limit = system_config.fresh_recall_limit

        # 直接使用混合检索，相似度阈值0.5
        if vector is not None:
            memories = await self.store.recall_with_vector(
                collection=self.collection,
                vector=vector,
                query=processed_query,
                limit=limit,
                where_filter=None,
                similarity_threshold=0.5,
            )
        else:
            memories = await self.store.recall(
                collection=self.collection,
                query=processed_query,
                limit=limit,
                where_filter=None,
                similarity_threshold=0.5,
            )

        # 数据迁移：补充 is_active 字段，删除 is_consolidated 字段
        batch_updates = []
        for mem in memories:
            updates = {}

            # 如果缺少 is_active，添加默认值 False（被动记忆）
            if not hasattr(mem, "is_active") or mem.is_active is None:
                updates["is_active"] = False
                mem.is_active = False

            # 删除废弃的 is_consolidated 字段（如果存在）
            updates["is_consolidated"] = None

            if updates:
                batch_updates.append({"id": mem.id, "updates": updates})

        # 批量更新
        if batch_updates:
            await self.store.update_memory(self.collection, batch_updates)

        return memories

    async def chained_recall(
        self,
        query: str,
        entities: List[str],
        per_type_limit: int = 7,
        final_limit: int = 7,
        memory_handlers: Dict[str, Any] = None,
        event=None,
        vector: Optional[List[float]] = None,
    ) -> List[BaseMemory]:
        """
        链式回忆 - 混合检索 + 实体优先 + 类型分组

        Args:
            query: 搜索查询字符串
            entities: 核心实体列表
            per_type_limit: 每种类型最多召回数量（默认7）
            final_limit: 候选池大小（默认7，实际使用50）
            memory_handlers: 未使用（保留兼容性）
            event: 消息事件
            vector: 预计算向量

        Returns:
            所有相关记忆列表
        """
        # 预处理查询词
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(query, event)

        # 步骤1: 混合检索获取候选池（相似度≥0.5的所有记忆）
        candidate_pool = await self.comprehensive_recall(
            query=processed_query,
            limit=100,  # 足够大的限制，让相似度阈值0.5来过滤
            event=event,
            vector=vector,
        )

        # 步骤2: 从候选池提取实体记忆（每个实体最多3条）
        entity_memories = []
        seen_ids = set()

        for entity in entities:
            entity_count = 0
            for mem in candidate_pool:
                if entity in mem.tags and mem.id not in seen_ids:
                    entity_memories.append(mem)
                    seen_ids.add(mem.id)
                    entity_count += 1
                    if entity_count >= 3:  # 每个实体最多3条
                        break

        # 步骤3: 从候选池按类型提取记忆（每类最多7条）
        type_memories = defaultdict(list)

        for mem in candidate_pool:
            if mem.id in seen_ids:
                continue
            mem_type = mem.memory_type.value if hasattr(mem.memory_type, "value") else str(mem.memory_type)
            if len(type_memories[mem_type]) < per_type_limit:
                type_memories[mem_type].append(mem)
                seen_ids.add(mem.id)

        # 步骤4: 合并所有记忆
        all_memories = entity_memories + [mem for mems in type_memories.values() for mem in mems]

        # 步骤5: 被动记忆衰减（强度-1，最低为0）
        # 步骤5: 被动记忆衰减（强度-1，最低为0）
        batch_updates = []
        for mem in all_memories:
            if not mem.is_active:  # 只对被动记忆衰减
                new_strength = max(0, mem.strength - 1)
                if new_strength != mem.strength:
                    mem.strength = new_strength
                    batch_updates.append({"id": mem.id, "updates": {"strength": new_strength}})

        if batch_updates:
            await self.store.update_memory(self.collection, batch_updates)

        self.logger.debug(f"链式回忆: 实体记忆={len(entity_memories)}, 类型记忆={sum(len(v) for v in type_memories.values())}, 总计={len(all_memories)}")

        return all_memories


    def _build_memory_from_metadata(
        self, memory_id: str, metadata: dict, document: str
    ) -> Optional[BaseMemory]:
        """从元数据构建记忆对象的辅助方法"""
        memory_type_str = metadata.get("memory_type", "知识记忆")
        try:
            # 解析记忆类型
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.KNOWLEDGE  # 默认为知识记忆

            # 统一构建BaseMemory对象
            return BaseMemory(
                memory_type=memory_type,
                judgment=metadata.get("judgment", ""),
                reasoning=metadata.get("reasoning", ""),
                tags=BaseMemory._parse_tags(metadata.get("tags", [])),
                id=memory_id,
                strength=metadata.get("strength", 1),
                is_active=metadata.get("is_active", False),
                associations=BaseMemory._parse_associations(
                    metadata.get("associations", {})
                ),
            )
        except Exception as e:
            self.logger.error(f"构建记忆对象失败: {str(e)}")
            return None

    # ===== 记忆合并功能 =====

    async def merge_memories(
        self,
        memories_to_merge_ids: List[str],
        new_judgment: str,
        new_reasoning: str,
        new_tags: List[str],
    ) -> BaseMemory:
        """
        记忆合并方法 - LLM驱动的记忆抽象化（中文核心概念）

        中文定义：LLM在回忆过程中发现相似记忆时，自主决定合并，形成更抽象、更精炼的新记忆
        英文翻译：LLM-driven memory abstraction process where similar memories are autonomously merged into a more abstract, refined new memory

        合并规则：
        - 新记忆强度 = 所有被合并记忆强度之和
        - 新记忆类型默认为知识记忆
        - 删除所有被合并的旧记忆
        - 不记录合并来源，新记忆独立存在

        Args:
            memories_to_merge_ids: 被合并的旧记忆ID列表
            new_judgment: 新记忆的判断/论断
            new_reasoning: 新记忆的理由
            new_tags: 新记忆的标签列表

        Returns:
            新创建的合并记忆对象

        Raises:
            ValidationError: 如果输入参数无效或合并过程失败
        """
        if not memories_to_merge_ids or len(memories_to_merge_ids) < 2:
            raise ValidationError("至少需要2个记忆ID才能进行合并")

        # 步骤 1: 加载所有待合并记忆
        memories_to_merge = []
        total_strength = 0

        for memory_id in memories_to_merge_ids:
            # 从存储中获取记忆元数据
            try:
                memory_results = self.collection.get(ids=[memory_id])
                if not memory_results["metadatas"]:
                    self.logger.warning(f"记忆 {memory_id} 不存在，跳过合并")
                    continue

                metadata = memory_results["metadatas"][0]
                document = (
                    memory_results["documents"][0]
                    if memory_results["documents"]
                    else ""
                )

                # 使用统一的构建方法
                memory_obj = self._build_memory_from_metadata(
                    memory_id, metadata, document
                )

                if not memory_obj:
                    self.logger.warning(f"无法构建记忆对象: {memory_id}")
                    continue

                memories_to_merge.append(memory_obj)
                total_strength += memory_obj.strength

            except Exception as e:
                self.logger.error(f"加载记忆 {memory_id} 失败: {str(e)}")
                continue

        if len(memories_to_merge) < 2:
            raise ValidationError(f"只有 {len(memories_to_merge)} 个有效记忆，无法合并")

        # 步骤 2: 创建新记忆（默认为知识记忆）
        # 使用去重后的强度，避免相同ID重复累加
        unique_strength = sum(
            memory_obj.strength for memory_obj in set(memories_to_merge)
        )

        new_memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment=new_judgment,
            reasoning=new_reasoning,
            tags=new_tags,
            strength=unique_strength,  # 强度等于去重后记忆之和
            is_consolidated=False,  # 合并记忆作为新鲜记忆重新开始生命周期
        )

        # 步骤 3: 存储新记忆
        await self.store.remember(self.collection, new_memory)

        # 步骤 4: 删除所有旧记忆
        self.collection.delete(ids=memories_to_merge_ids)

        self.logger.debug(
            f"成功合并 {len(memories_to_merge)} 条记忆为新记忆 '{new_judgment}'，强度为 {total_strength}"
        )

        return new_memory

    # ===== 反馈处理 =====

    async def process_feedback(
        self,
        useful_memory_ids: List[str] = None,
        new_memories: List[dict] = None,
        merge_groups: List[List[str]] = None,
        memory_handlers: Dict[str, object] = None,
    ) -> List[BaseMemory]:
        """
        统一反馈接口 - 处理回忆后的反馈（核心工作流）

        工作流：观察 → 回忆 → 反馈 → 睡眠

        Args:
            useful_memory_ids: 有用的回忆ID列表
            new_memories: 新记忆列表，每个元素格式：
                {
                    "type": "knowledge/event/skill/emotional/task",
                    "judgment": "论断",
                    "reasoning": "解释",
                    "tags": ["标签1", "标签2"]
                }
            merge_groups: 要合并的记忆ID组列表，每组是要合并的ID列表
            memory_handlers: 记忆处理器字典，用于创建新记忆

        Returns:
            新创建的记忆对象列表

        功能：
        1. 标记有用回忆 → 强化 + 两两建立关联（双向、去重、累加）
        2. 批量创建新记忆 → 两两建立关联（初始强度1）
        3. 新记忆和有用回忆建立关联
        4. 合并重复记忆 → 继承并累加关联
        """
        useful_memory_ids = useful_memory_ids or []
        new_memories = new_memories or []
        merge_groups = merge_groups or []
        memory_handlers = memory_handlers or {}

        created_memories = []  # 新增：收集创建的记忆对象

        association_cache: Dict[str, MemorySnapshot] = {}
        if useful_memory_ids:
            association_cache = self.association_manager.preload_memories(
                useful_memory_ids
            )
        if useful_memory_ids:
            # 强化记忆强度
            await self.reinforce_memories(useful_memory_ids)

            # 有用回忆之间两两建立关联（双向、去重、累加）
            for i in range(len(useful_memory_ids)):
                for j in range(i + 1, len(useful_memory_ids)):
                    id1, id2 = useful_memory_ids[i], useful_memory_ids[j]
                    await self.association_manager._add_or_update_association(
                        id1, id2, strength_increase=1, cache=association_cache
                    )
                    await self.association_manager._add_or_update_association(
                        id2, id1, strength_increase=1, cache=association_cache
                    )
        # 2. 批量创建新记忆
        created_ids = []
        for mem_data in new_memories:
            mem_type = mem_data.get("type", "knowledge")

            # 检查必需字段
            judgment = mem_data.get("judgment")
            reasoning = mem_data.get("reasoning")

            if not judgment or not reasoning:
                self.logger.warning(
                    f"跳过缺少必需字段的记忆: judgment={bool(judgment)}, reasoning={bool(reasoning)}"
                )
                continue

            # tags 是可选的，默认为空列表
            tags = mem_data.get("tags", [])

            # 根据类型创建记忆，获取返回的对象
            new_memory_object = None
            handler = memory_handlers.get(mem_type)
            if handler:
                new_memory_object = await handler.remember(judgment, reasoning, tags)
            else:
                self.logger.warning(f"未找到记忆类型 {mem_type} 的处理器，跳过创建")

            if new_memory_object:
                created_ids.append(new_memory_object.id)
                created_memories.append(new_memory_object)  # 收集创建的对象

                # 任务记忆特殊状态管理：新任务记忆创建时，将之前最新的任务记忆转为已巩固状态
                if mem_type == "task":
                    await self._consolidate_previous_task_memory(new_memory_object.id)

        if created_ids:
            association_cache = self.association_manager.preload_memories(
                created_ids, association_cache
            )

        # 3. 新记忆之间两两建立关联（初始强度1）
        for i in range(len(created_ids)):
            for j in range(i + 1, len(created_ids)):
                id1, id2 = created_ids[i], created_ids[j]
                await self.association_manager._add_or_update_association(
                    id1, id2, strength_increase=1, cache=association_cache
                )
                await self.association_manager._add_or_update_association(
                    id2, id1, strength_increase=1, cache=association_cache
                )

        # 4. 新记忆和有用回忆之间建立关联
        for new_id in created_ids:
            for useful_id in useful_memory_ids:
                await self.association_manager._add_or_update_association(
                    new_id, useful_id, strength_increase=1, cache=association_cache
                )
                await self.association_manager._add_or_update_association(
                    useful_id, new_id, strength_increase=1, cache=association_cache
                )

        # 5. 合并重复记忆
        for group in merge_groups:
            # 验证格式：必须是列表且至少包含2个元素
            if not isinstance(group, list):
                self.logger.warning(f"跳过无效的merge_group（非列表）: {type(group)}")
                continue
            if len(group) < 2:
                self.logger.warning(f"跳过无效的merge_group（少于2个ID）: {group}")
                continue

            # 获取第一个记忆作为模板
            first_mem = self.collection.get(ids=[group[0]])
            if first_mem and first_mem["metadatas"]:
                metadata = first_mem["metadatas"][0]
                # 使用第一个记忆的数据作为合并后的内容
                await self.merge_memories(
                    memories_to_merge_ids=group,
                    new_judgment=metadata.get("judgment", "合并记忆"),
                    new_reasoning=metadata.get("reasoning", "合并多个相似记忆"),
                    new_tags=metadata.get("tags", "").split(", ")
                    if metadata.get("tags")
                    else [],
                )

        return created_memories  # 返回新创建的记忆对象列表

    async def _consolidate_previous_task_memory(self, current_task_id: str):
        """
        任务记忆特殊状态管理：创建新任务记忆时，将之前最新的任务记忆转为已巩固状态

        这样确保始终只有一个最新（新鲜的）任务记忆，其他都是已巩固的。

        Args:
            current_task_id: 当前新创建的任务记忆ID
        """
        try:
            # 查找除了当前记忆之外的所有新鲜任务记忆
            where_filter = {
                "$and": [{"memory_type": "任务记忆"}, {"is_consolidated": False}]
            }

            fresh_task_results = self.collection.get(where=where_filter)
            if not fresh_task_results or not fresh_task_results["ids"]:
                return  # 没有其他新鲜任务记忆

            # 过滤掉当前刚创建的任务记忆
            previous_task_ids = [
                mem_id
                for mem_id in fresh_task_results["ids"]
                if mem_id != current_task_id
            ]

            if not previous_task_ids:
                return  # 没有之前的任务记忆需要巩固

            # 将之前的新鲜任务记忆转为已巩固状态
            for task_id in previous_task_ids:
                await self.store.update_memory(
                    self.collection, task_id, {"is_consolidated": True}
                )

            self.logger.debug(
                f"已将 {len(previous_task_ids)} 个之前的任务记忆转为已巩固状态"
            )

        except Exception as e:
            self.logger.error(f"巩固之前任务记忆失败: {str(e)}")
