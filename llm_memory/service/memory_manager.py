"""
记忆管理器 - 处理记忆的高级功能。

这个模块包含了记忆系统的高级功能，如记忆巩固、强化、
链式回忆、记忆合并等复杂操作。
"""

from typing import List, Optional, Dict, Any
import random
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
        执行记忆巩固过程（睡眠模式）。

        这个方法模拟大脑的睡眠巩固，将新鲜记忆转变为已巩固记忆。

        设计原则：
        - 记忆永不丢失（过目不忘），只会通过merge_memories主动合并
        - 关联永不删除，"忘记"的本质是无法召回而非不存在
        - 在正确的情景下，弱关联的记忆仍然可以被回忆起来
        """
        self.logger.debug("开始记忆巩固过程...")

        try:
            # 将所有新鲜记忆转变为已巩固记忆
            fresh_results = self.collection.get(where={"is_consolidated": False})
            consolidated_count = 0
            if fresh_results and fresh_results["ids"]:
                for memory_id in fresh_results["ids"]:
                    await self.store.update_memory(
                        self.collection, memory_id, {"is_consolidated": True}
                    )
                    consolidated_count += 1
                self.logger.debug(f"已巩固 {consolidated_count} 条新鲜记忆")

            # 清理弱关联
            await self.association_manager.cleanup_weak_associations()

            self.logger.debug("记忆巩固过程完成（记忆和关联永不丢失）")

        except Exception as e:
            self.logger.error(f"记忆巩固过程失败: {str(e)}")
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
                # 获取当前记忆的强度
                current_meta = self.collection.get(ids=[memory_id])["metadatas"][0]
                if current_meta:
                    current_strength = current_meta.get("strength", 1)
                    # 强度 +1
                    await self.store.update_memory(
                        self.collection, memory_id, {"strength": current_strength + 1}
                    )
            except Exception as e:
                self.logger.error(f"强化记忆 {memory_id} 失败: {str(e)}")

    # ===== 高级回忆功能 =====

    async def comprehensive_recall(
        self,
        query: str,
        fresh_limit: int = None,
        consolidated_limit: int = None,
        event=None,
        vector: Optional[List[float]] = None,
    ) -> List[BaseMemory]:
        """
        实现双轨检索：同时从新鲜记忆和已巩固记忆中检索相关内容。

        Args:
            query: 搜索查询字符串
            fresh_limit: 新鲜记忆的最大返回数量
            consolidated_limit: 已巩固记忆的最大返回数量
            event: 消息事件（用于查询词预处理）
            vector: 可选的预计算向量，如果提供则直接使用

        Returns:
            合并后的记忆列表，新鲜记忆优先
        """
        # 预处理查询词（如果有查询处理器）
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(
                query, event
            )

        if fresh_limit is None:
            fresh_limit = system_config.fresh_recall_limit
        if consolidated_limit is None:
            consolidated_limit = system_config.consolidated_recall_limit

        # 如果提供了预计算向量，使用向量检索；否则使用文本检索
        if vector is not None:
            # 使用预计算向量检索新鲜记忆
            fresh_memories = await self.store.recall_with_vector(
                collection=self.collection,
                vector=vector,
                limit=fresh_limit,
                where_filter={"is_consolidated": False},
                query=processed_query,  # 传递查询文本
            )

            # 使用预计算向量检索已巩固记忆
            consolidated_memories = await self.store.recall_with_vector(
                collection=self.collection,
                vector=vector,
                limit=consolidated_limit,
                where_filter={"is_consolidated": True},
                query=processed_query,  # 传递查询文本
            )
        else:
            # 检索新鲜记忆
            fresh_memories = await self.store.recall(
                collection=self.collection,
                query=processed_query,
                limit=fresh_limit,
                where_filter={"is_consolidated": False},
            )

            # 检索已巩固记忆
            consolidated_memories = await self.store.recall(
                collection=self.collection,
                query=processed_query,
                limit=consolidated_limit,
                where_filter={"is_consolidated": True},
            )

        # 合并结果：新鲜记忆在前，已巩固记忆在后
        all_memories = fresh_memories + consolidated_memories

        # 去重（以防有重复的记忆）
        seen_ids = set()
        unique_memories = []
        for memory in all_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_memories.append(memory)

        return unique_memories

    async def chained_recall(
        self,
        query: str,
        entities: List[str],  # 新增: 核心实体列表
        per_type_limit: int = 7,
        final_limit: int = 7,
        memory_handlers: Dict[str, Any] = None,
        event=None,
        vector: Optional[List[float]] = None,
    ) -> Dict[str, List[BaseMemory]]:  # 返回值修改为字典
        """
        链式多通道回忆 - 基于关联网络的多轮回忆（中文核心概念）

        Args:
            query: 搜索查询字符串
            entities: 核心实体列表，用于优先检索相关记忆。
            per_type_limit: 每种类型第一轮最多召回数量（默认7）
            final_limit: 最终返回的记忆数量（默认7）
            memory_handlers: 记忆处理器字典，用于分类型召回
            event: 消息事件（用于查询词预处理）
            vector: 可选的预计算向量，如果提供则直接使用

        Returns:
            包含 "entity_memories" 和 "regular_memories" 的字典。
        """
        self.logger.debug(f"Chained Recall - Query: '{query}', Entities: {entities}")

        # 预处理查询词（如果有查询处理器）
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(
                query, event
            )

        # ==========================================
        #  新增：第零轮 - 实体优先通道 (按强度)
        # ==========================================
        entity_memories = []
        if entities:
            # 调用新的辅助方法来完成这个逻辑
            # 每个实体召回的记忆数量可以配置，这里暂定为3
            entity_memories = await self._recall_entities_by_strength(
                entities, per_entity_limit=3
            )
            self.logger.debug(f"实体优先通道召回了 {len(entity_memories)} 条记忆。")

        # 获取已经找到的实体记忆的ID，用于后续去重
        seen_ids = {mem.id for mem in entity_memories}

        if not memory_handlers:
            self.logger.warning("未提供记忆处理器，使用简单回忆")
            # 如果没有处理器，直接使用 comprehensive_recall
            # 此时 entity_memories 已经获取，这里获取的是常规记忆
            regular_memories_candidates = await self.comprehensive_recall(
                processed_query,
                fresh_limit=final_limit,
                consolidated_limit=final_limit,
                event=event,
                vector=vector,
            )
            # 从常规记忆中移除已在实体记忆中的ID
            regular_memories = [mem for mem in regular_memories_candidates if mem.id not in seen_ids]

            return {
                "entity_memories": entity_memories,
                "regular_memories": regular_memories[:final_limit],  # 确保不超过最终限制
            }

        # 存储所有召回的记忆，按类型分组
        recalled_by_type = {}
        all_recalled_ids = set()

        # 第一轮：分类型召回
        for memory_type, handler in memory_handlers.items():
            if not handler:
                continue

            memories = await handler.recall(
                processed_query, limit=per_type_limit, include_consolidated=True
            )
            # 从分类型召回的结果中移除已在实体记忆中的ID
            filtered_memories = [mem for mem in memories if mem.id not in seen_ids]
            if filtered_memories:
                recalled_by_type[memory_type] = filtered_memories
                all_recalled_ids.update([m.id for m in filtered_memories])

        # 第二轮：从关联中补充记忆
        for memory_type, memories in list(recalled_by_type.items()):
            for memory in memories:
                if not memory.associations:
                    continue

                for assoc_id, assoc_strength in memory.associations.items():
                    # 确保关联记忆不在已见ID中（包括实体记忆和第一轮召回的记忆）
                    if assoc_id in all_recalled_ids or assoc_id in seen_ids:
                        continue

                    try:
                        assoc_results = self.collection.get(ids=[assoc_id])
                        if not assoc_results["metadatas"]:
                            continue

                        metadata = assoc_results["metadatas"][0]
                        assoc_type = metadata.get("memory_type", "知识记忆")

                        assoc_memory = self._build_memory_from_metadata(
                            assoc_id,
                            metadata,
                            assoc_results["documents"][0]
                            if assoc_results["documents"]
                            else "",
                        )

                        if assoc_memory:
                            if assoc_type not in recalled_by_type:
                                recalled_by_type[assoc_type] = []

                            recalled_by_type[assoc_type].append(assoc_memory)
                            all_recalled_ids.add(assoc_id)

                    except Exception as e:
                        self.logger.warning(f"获取关联记忆 {assoc_id} 失败: {str(e)}")
                        continue

        # 统计权重：出现次数 × 强度
        memory_count = defaultdict(int)
        memory_obj_map = {}

        # 统计出现次数
        for memory_type, memories in recalled_by_type.items():
            for memory in memories:
                memory_count[memory.id] += 1
                memory_obj_map[memory.id] = memory

        # 计算权重
        weights = {}
        for memory_id, count in memory_count.items():
            memory = memory_obj_map[memory_id]
            weights[memory_id] = count * memory.strength

        # 加权随机抽取常规记忆
        regular_memories = []
        remaining_ids = list(weights.keys())

        for _ in range(min(final_limit, len(remaining_ids))):
            if not remaining_ids:
                break
            total_weight = sum(weights[mid] for mid in remaining_ids)
            if total_weight == 0:  # 避免除以零
                break

            rand = random.uniform(0, total_weight)
            cumulative = 0
            selected_id = None

            for mid in remaining_ids:
                cumulative += weights[mid]
                if cumulative >= rand:
                    selected_id = mid
                    break

            if selected_id:
                regular_memories.append(memory_obj_map[selected_id])
                remaining_ids.remove(selected_id)
                del weights[selected_id]  # 移除已选择的记忆权重

        return {
            "entity_memories": entity_memories,
            "regular_memories": regular_memories,
        }

    async def _recall_entities_by_strength(
        self, entities: List[str], per_entity_limit: int
    ) -> List[BaseMemory]:
        """
        为每个实体检索出强度最高的N条记忆。
        使用 get() + 内存过滤的方式，并按强度排序。
        """
        all_results = []
        seen_ids = set()

        # 1. 获取一个足够大的候选集（例如最新或全部的记忆，根据实际情况调整）
        # 这里为了避免过度消耗内存和时间，我们获取最新的一部分记忆作为候选。
        # 实际生产环境中，这个 limit 可能需要根据数据库大小和性能进行优化。
        # 如果需要检索所有记忆，可以分批获取或者调整 ChromaDB 配置。
        candidate_results = self.collection.get(
            limit=1000,  # 获取最新的1000条记忆作为筛选池，确保覆盖面
            include=["metadatas", "documents"],
        )

        if not candidate_results or not candidate_results["ids"]:
            self.logger.debug("实体直召：未从长期记忆获取到任何候选记忆。")
            return []

        # 将原始数据转换为 BaseMemory 对象
        candidate_memories = []
        for i, mem_id in enumerate(candidate_results["ids"]):
            metadata = candidate_results["metadatas"][i]
            document = (
                candidate_results["documents"][i]
                if candidate_results["documents"]
                else ""
            )
            memory_obj = self._build_memory_from_metadata(mem_id, metadata, document)
            if memory_obj:
                candidate_memories.append(memory_obj)

        if not candidate_memories:
            self.logger.debug("实体直召：从候选数据中未构建出有效记忆。")
            return []

        # 2. 为每个实体进行筛选和排序
        for entity_name in entities:
            # a. 筛选出所有包含该实体tag的记忆
            # 注意：这里我们假设 memory.tags 是一个列表，直接使用 'in' 操作
            memories_for_entity = [
                mem
                for mem in candidate_memories
                if hasattr(mem, "tags")
                and isinstance(mem.tags, list)
                and entity_name in mem.tags
                and mem.id not in seen_ids
            ]

            # b. 按强度降序排序
            memories_for_entity.sort(key=lambda x: x.strength, reverse=True)

            # c. 取强度最高的N条
            top_memories = memories_for_entity[:per_entity_limit]

            all_results.extend(top_memories)
            seen_ids.update(mem.id for mem in top_memories)  # 更新已见ID，避免重复添加

        return all_results

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
                is_consolidated=metadata.get("is_consolidated", False),
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
            if len(group) >= 2:
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
