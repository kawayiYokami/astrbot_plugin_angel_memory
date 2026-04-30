"""
记忆管理器 - 处理记忆的高级功能。

这个模块包含了记忆系统的高级功能，如记忆巩固、强化、
链式回忆、记忆合并等复杂操作。

优化版变更（2026-04-29）：
- Fix 1: _build_memory_from_metadata 补全 created_at / useful_count / useful_score / last_recalled_at
- Fix 2: 新增 _apply_time_decay，在 comprehensive_recall 三个返回路径统一应用
- Fix 3: merge_memories 不再设置 is_consolidated（该字段已迁移废弃）
- Fix 4: _build_memory_from_metadata 移除未使用的 document 参数
- Fix 5: _consolidate_previous_task_memory 查询条件从 is_consolidated=False 改为 strength>0
"""

from typing import List, Optional, Dict, Any
import logging
import time
from collections import defaultdict

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)
from ..models.data_models import BaseMemory, MemoryType, ValidationError
from ..config.system_config import system_config
from .memory_decay_policy import MemoryDecayConfig, MemoryDecayPolicy

# 导入查询处理器（用于统一检索词预处理）
from ...core.utils.query_processor import get_query_processor


class MemoryManager:
    """
    记忆管理器类 - 处理记忆的高级功能。

    负责记忆的巩固、强化、链式回忆、记忆合并等复杂操作。
    这些功能不直接与特定记忆类型绑定，而是处理记忆的通用行为。
    """

    def __init__(
        self,
        main_collection,
        vector_store,
        memory_sql_manager=None,
        memory_index_collection=None,
        decay_config: MemoryDecayConfig | None = None,
    ):
        """
        初始化记忆管理器。

        Args:
            main_collection: 用于操作的主集合实例
            vector_store: 向量存储实例 (用于调用 recall, update_memory 等高级方法)
        """
        self.collection = main_collection
        self.store = vector_store
        self.memory_sql_manager = memory_sql_manager
        self.memory_index_collection = memory_index_collection
        self.logger = logger
        self.decay_policy = MemoryDecayPolicy(decay_config)

        # 初始化查询处理器（用于统一检索词预处理）
        self.query_processor = get_query_processor()

    # ===== 记忆巩固和强化 =====

    async def consolidate_memories(self):
        """
        执行记忆清理过程（睡眠模式）。

        清理规则：
        - 删除被动记忆中强度 ≤ 0 的记忆
        - 主动记忆永不删除（不受强度影响）
        """
        self.logger.debug("开始记忆清理过程...")

        if self.memory_sql_manager is not None:
            await self.memory_sql_manager.consolidate_memories()
            return

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
        if self.memory_sql_manager is not None:
            await self.memory_sql_manager.reinforce_memories(memory_ids, delta=1)
            return

        for memory_id in memory_ids:
            try:
                # 根据 metadata 中的 id 查找记忆
                result = self.collection.get(where={"id": memory_id}, limit=1)
                if result and result["metadatas"] and result["ids"]:
                    current_meta = result["metadatas"][0]
                    current_strength = current_meta.get("strength", 1)
                    current_useful_count = int(current_meta.get("useful_count", 0) or 0)
                    current_useful_score = float(current_meta.get("useful_score", 0.0) or 0.0)
                    decay_config = getattr(self.decay_policy, "config", None)
                    consolidate_speed = float(
                        getattr(decay_config, "consolidate_speed", 2.5) or 2.5
                    )
                    now_ts = time.time()
                    # 获取 ChromaDB 的文档 ID
                    chroma_doc_id = result["ids"][0]
                    # 使用 ChromaDB 文档 ID 进行更新
                    await self.store.update_memory(
                        self.collection,
                        chroma_doc_id,
                        {
                            "strength": current_strength + 1,
                            "useful_count": current_useful_count + 1,
                            "useful_score": current_useful_score + consolidate_speed,
                            "last_recalled_at": now_ts,
                        },
                    )
            except Exception as e:
                self.logger.error(f"强化记忆 {memory_id} 失败: {str(e)}")

    # ===== 高级回忆功能 =====

    @staticmethod
    def _build_scope_where_filter(memory_scope: str) -> Dict[str, Any]:
        """构建严格的 memory_scope 过滤条件。"""
        scope = str(memory_scope or "").strip()
        if not scope:
            raise ValueError("memory_scope 为空，拒绝执行检索")

        if scope == "public":
            return {"memory_scope": "public"}

        return {"$or": [{"memory_scope": scope}, {"memory_scope": "public"}]}

    @staticmethod
    def _is_scope_allowed(memory_scope: str, target_scope: str) -> bool:
        """检查给定的 memory_scope 是否在 target_scope 允许范围内（含 public 通配）"""
        scope = str(memory_scope or "").strip() or "public"
        target = str(target_scope or "").strip()
        if target == "public":
            return scope == "public"
        return scope in {target, "public"}

    @staticmethod
    def _safe_parse_timestamp(value, default: float) -> float:
        """安全解析时间戳，兼容脏数据（空字符串、非数字字符串等）"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return float(stripped)
            except ValueError:
                pass
            try:
                from datetime import datetime

                return datetime.fromisoformat(stripped).timestamp()
            except (ValueError, AttributeError):
                pass
        return default

    def _apply_time_decay(self, memories: List[BaseMemory], decay_rate: float = 0.05) -> List[BaseMemory]:
        """
        对被动记忆应用时间衰减，并重新排序。

        主动记忆（is_active=True）不受时间衰减影响。
        被动记忆的相似度乘以衰减因子：1 / (1 + decay_rate * age_days)
        约 20 天后衰减至原来的 50%。

        Args:
            memories: 记忆列表
            decay_rate: 衰减率，默认 0.05

        Returns:
            衰减并重新排序后的记忆列表
        """
        if not memories:
            return memories

        now = time.time()
        for mem in memories:
            if getattr(mem, 'is_active', False):
                continue
            created_at = self._safe_parse_timestamp(getattr(mem, 'created_at', None), now)
            age_days = max(0.0, (now - created_at) / 86400.0)
            decay_factor = 1.0 / (1.0 + decay_rate * age_days)
            mem.similarity = getattr(mem, 'similarity', 0.0) * decay_factor

        memories.sort(key=lambda m: getattr(m, 'similarity', 0.0), reverse=True)
        return memories

    async def comprehensive_recall(
        self,
        query: str,
        limit: int = None,
        event=None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        """
        统一记忆检索 - 使用混合检索获取所有相关记忆。

        Args:
            query: 搜索查询字符串
            limit: 最大返回数量
            event: 消息事件（用于查询词预处理）
            vector: 可选的预计算向量，如果提供则直接使用

        Returns:
            记忆列表（相似度 >= 0.5），已按时间衰减重排
        """
        # 预处理查询词
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(
                query, event
            )

        if limit is None:
            limit = system_config.fresh_recall_limit
        where_filter = self._build_scope_where_filter(memory_scope)

        if self.memory_sql_manager is not None and self.memory_index_collection is not None:
            id_scores = await self.store.recall_memory_ids(
                collection=self.memory_index_collection,
                query=processed_query,
                limit=limit * 3,
                vector=vector,
                similarity_threshold=0.5,
            )
            score_map = {mid: score for mid, score in id_scores}
            hybrid_memories = await self.memory_sql_manager.recall_by_tags(
                query=processed_query,
                limit=limit * 3,
                memory_scope=memory_scope,
                vector_scores=score_map if score_map else None,
            )
            if hybrid_memories:
                # [Fix 2] Hybrid 主路径：应用时间衰减
                return self._apply_time_decay(hybrid_memories)[:limit]

            if not id_scores:
                return []
            ordered_ids = [mid for mid, _ in id_scores]
            sql_memories = await self.memory_sql_manager.get_memories_by_ids(ordered_ids)
            filtered: List[BaseMemory] = []
            for mem in sql_memories:
                if not self._is_scope_allowed(getattr(mem, "memory_scope", "public"), memory_scope):
                    continue
                mem.similarity = float(score_map.get(mem.id, 0.0))
                filtered.append(mem)
            # [Fix 2] Hybrid 回退路径：应用时间衰减
            return self._apply_time_decay(filtered)[:limit]

        # 直接使用混合检索，相似度阈值0.5
        if vector is not None:
            memories = await self.store.recall_with_vector(
                collection=self.collection,
                vector=vector,
                query=processed_query,
                limit=limit,
                where_filter=where_filter,
                similarity_threshold=0.5,
            )
        else:
            memories = await self.store.recall(
                collection=self.collection,
                query=processed_query,
                limit=limit,
                where_filter=where_filter,
                similarity_threshold=0.5,
            )

        # [Fix 2] 纯向量路径：应用时间衰减
        memories = self._apply_time_decay(memories)

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
        memory_scope: str = "public",
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
            所有相关记忆列表（已由 comprehensive_recall 完成时间衰减）
        """
        # 预处理查询词
        processed_query = query
        if self.query_processor and event:
            processed_query = self.query_processor.process_query_for_memory(query, event)

        # 步骤1: 混合检索获取候选池（相似度≥0.5的所有记忆，已含时间衰减）
        candidate_pool = await self.comprehensive_recall(
            query=processed_query,
            limit=100,  # 足够大的限制，让相似度阈值0.5来过滤
            event=event,
            vector=vector,
            memory_scope=memory_scope,
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

        self.logger.debug(f"链式回忆: 实体记忆={len(entity_memories)}, 类型记忆={sum(len(v) for v in type_memories.values())}, 总计={len(all_memories)}")

        return all_memories


    def _build_memory_from_metadata(
        self, memory_id: str, metadata: dict
    ) -> Optional[BaseMemory]:
        """
        从 ChromaDB 元数据构建 BaseMemory 对象。

        Fix 1: 补全 created_at，避免合并路径时间戳丢失。
        Fix 3: 补全 useful_count / useful_score / last_recalled_at，避免合并后衰减状态丢失。
        """
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
                created_at=metadata.get("created_at", time.time()),
                memory_scope=metadata.get("memory_scope", "public"),
                useful_count=metadata.get("useful_count", 0),
                useful_score=metadata.get("useful_score", 0.0),
                last_recalled_at=metadata.get("last_recalled_at", 0.0),
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

                # 使用统一的构建方法
                memory_obj = self._build_memory_from_metadata(
                    memory_id, metadata
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

        non_public_scopes = {
            str(getattr(memory_obj, "memory_scope", "public") or "public").strip()
            for memory_obj in memories_to_merge
            if str(getattr(memory_obj, "memory_scope", "public") or "public").strip() != "public"
        }
        if len(non_public_scopes) > 1:
            raise ValidationError(
                "禁止合并来自不同私有分类域的记忆：检测到多个非 public memory_scope。"
            )
        merged_scope = next(iter(non_public_scopes), "public")

        # 步骤 2: 创建新记忆（默认为知识记忆）
        # 使用去重后的强度，避免相同ID重复累加
        unique_strength = sum(
            memory_obj.strength for memory_obj in set(memories_to_merge)
        )

        # Fix 4: 不再设置 is_consolidated — 该字段已通过 comprehensive_recall 迁移删除
        new_memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment=new_judgment,
            reasoning=new_reasoning,
            tags=new_tags,
            strength=unique_strength,  # 强度等于去重后记忆之和
            memory_scope=merged_scope,
            is_consolidated=False,  # 读取侧全部迁移前先保留兼容字段
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
        recalled_memory_ids: List[str] = None,
        new_memories: List[dict] = None,
        merge_groups: List[List[str]] = None,
        memory_handlers: Dict[str, object] = None,
        memory_scope: str = "public",
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
        1. 标记有用回忆并强化
        2. 批量创建新记忆
        3. 合并重复记忆
        """
        if self.memory_sql_manager is not None:
            result = await self.memory_sql_manager.process_feedback(
                useful_memory_ids=useful_memory_ids,
                recalled_memory_ids=recalled_memory_ids,
                new_memories=new_memories,
                merge_groups=merge_groups,
                memory_scope=memory_scope,
            )
            # 修复：中央库模式合并记忆时，同步删除向量索引并为合并记忆写入新向量
            if merge_groups and self.memory_index_collection is not None:
                ids_to_delete = []
                for group in merge_groups:
                    if isinstance(group, list):
                        ids_to_delete.extend([str(mid) for mid in group if mid])

                if ids_to_delete:
                    try:
                        import asyncio
                        await asyncio.to_thread(
                            self.memory_index_collection.delete,
                            ids=ids_to_delete,
                        )
                        self.logger.debug(
                            f"已从 memory_index 删除 {len(ids_to_delete)} 条合并前的记忆向量"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"从 memory_index 删除合并记忆向量失败: {e}"
                        )

                # 为新合并记忆写入向量索引（不嵌套在 ids_to_delete 内，upsert 是幂等的）
                if result:
                    rows_to_index = []
                    for merged_memory in result:
                        if merged_memory and hasattr(merged_memory, 'id'):
                            vector_text = self.memory_sql_manager.build_vector_text(
                                judgment=getattr(merged_memory, 'judgment', ''),
                                tags=getattr(merged_memory, 'tags', []),
                            )
                            if vector_text:
                                rows_to_index.append({
                                    'id': merged_memory.id,
                                    'vector_text': vector_text,
                                })
                    if rows_to_index:
                        try:
                            await self.store.upsert_memory_index_rows(
                                collection=self.memory_index_collection,
                                rows=rows_to_index,
                            )
                            self.logger.debug(
                                f"已为 {len(rows_to_index)} 个合并记忆添加向量索引"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"为合并记忆添加向量索引失败（不影响主流程）: {e}"
                            )
            return result

        useful_memory_ids = useful_memory_ids or []
        recalled_memory_ids = recalled_memory_ids or []
        new_memories = new_memories or []
        merge_groups = merge_groups or []
        memory_handlers = memory_handlers or {}
        resolved_scope = str(memory_scope or "").strip()
        if not resolved_scope:
            raise ValidationError("memory_scope 为空，拒绝写入反馈记忆")

        created_memories = []  # 新增：收集创建的记忆对象

        if useful_memory_ids:
            # 强化记忆强度
            await self.reinforce_memories(useful_memory_ids)

        # 1.5 被召回但无用：仅衰减 T1（兼容向量直连链路）
        if recalled_memory_ids:
            useful_set = set(str(x) for x in useful_memory_ids)
            useless_ids = [str(x) for x in recalled_memory_ids if str(x) not in useful_set]
            if useless_ids:
                for memory_id in useless_ids:
                    try:
                        result = self.collection.get(where={"id": memory_id}, limit=1)
                        if not result or not result.get("metadatas") or not result.get("ids"):
                            continue
                        meta = result["metadatas"][0] or {}
                        if bool(meta.get("is_active", False)):
                            continue
                        useful_score = float(meta.get("useful_score", 0.0) or 0.0)
                        tier = self.decay_policy.tier_of(useful_score)
                        if tier != 1:
                            continue
                        current_strength = int(meta.get("strength", 0) or 0)
                        new_strength = max(0, current_strength - 1)
                        if new_strength == current_strength:
                            continue
                        chroma_doc_id = result["ids"][0]
                        await self.store.update_memory(
                            self.collection,
                            chroma_doc_id,
                            {"strength": new_strength},
                        )
                    except Exception as e:
                        self.logger.warning(f"召回无用衰减失败 id={memory_id}: {e}")
        # 2. 批量创建新记忆
        for mem_data in new_memories:
            mem_type = mem_data.get("type", "knowledge")

            # 检查必需字段
            judgment = mem_data.get("judgment")
            reasoning = mem_data.get("reasoning")

            if not judgment:
                self.logger.warning(
                    f"跳过缺少必需字段的记忆: judgment={bool(judgment)}"
                )
                continue

            # tags 是可选的，默认为空列表
            tags = mem_data.get("tags", [])

            # 根据类型创建记忆，获取返回的对象
            new_memory_object = None
            handler = memory_handlers.get(mem_type)
            if handler:
                new_memory_object = await handler.remember(
                    judgment, reasoning, tags, memory_scope=resolved_scope
                )
            else:
                self.logger.warning(f"未找到记忆类型 {mem_type} 的处理器，跳过创建")

            if new_memory_object:
                created_memories.append(new_memory_object)  # 收集创建的对象

                # 任务记忆特殊状态管理：新任务记忆创建时，将之前最新的任务记忆转为已巩固状态
                if mem_type == "task":
                    await self._consolidate_previous_task_memory(new_memory_object.id)

        # 3. 合并重复记忆
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

        Fix 5: 查询条件从 is_consolidated=False 改为 strength>0，
        因为 is_consolidated 字段已通过 comprehensive_recall 迁移删除。
        改用 strength>0 和 memory_type 过滤，确保查到所有活跃任务记忆。
        """
        try:
            # 先获取当前任务的 memory_scope，防止跨范围误更新
            current_scope = "public"
            current_task_data = self.collection.get(
                where={"id": current_task_id},
                limit=1,
                include=["metadatas"],
            )
            if current_task_data and current_task_data.get("metadatas"):
                current_scope = (
                    str(
                        current_task_data["metadatas"][0].get(
                            "memory_scope", "public"
                        )
                    ).strip()
                    or "public"
                )

            # 查找除了当前记忆之外的所有任务记忆（strength>0 表示未过期）
            where_filter = {
                "$and": [
                    {"memory_type": "任务记忆"},
                    {"strength": {"$gt": 0}},
                    {"memory_scope": current_scope},
                ]
            }

            fresh_task_results = self.collection.get(where=where_filter)
            if not fresh_task_results or not fresh_task_results["ids"]:
                return  # 没有其他任务记忆

            # 过滤掉当前刚创建的任务记忆
            previous_task_ids = [
                mem_id
                for mem_id in fresh_task_results["ids"]
                if mem_id != current_task_id
            ]

            if not previous_task_ids:
                return  # 没有之前的任务记忆需要巩固

            # 将之前的任务记忆强度降至 1（标记为已巩固，但保留最低强度不删除）
            for task_id in previous_task_ids:
                await self.store.update_memory(
                    self.collection, task_id, {"strength": 1}
                )

            self.logger.debug(
                f"已将 {len(previous_task_ids)} 个之前的任务记忆转为已巩固状态"
            )

        except Exception as e:
            self.logger.error(f"巩固之前任务记忆失败: {str(e)}")
