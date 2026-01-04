"""
记忆关联管理器。

负责管理记忆单元之间的动态关联关系，实现赫布理论（"一起激活，一起连接"）。
关联关系现在存储在每个记忆对象的内部关联字段中，而非独立的记忆类型。
"""

from typing import Dict, List, Optional
from itertools import combinations
from dataclasses import dataclass
import logging

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)
from .vector_store import VectorStore
from ..models.data_models import BaseMemory


@dataclass
class MemorySnapshot:
    memory: BaseMemory
    metadata: dict
    document: str
    embedding: Optional[List[float]]


class AssociationManager:
    """
    记忆关联管理器。

    管理记忆之间的动态关联网络，支持强化、查询和清理关联关系。
    现在通过操作记忆内部的关联字段来实现关联功能，而非创建独立的关联对象。
    """

    def __init__(self, main_collection, vector_store: VectorStore):
        """
        初始化关联管理器。

        Args:
            main_collection: 用于操作的主集合实例
            vector_store: VectorStore 实例 (用于重新生成嵌入等)
        """
        self.collection = main_collection
        self.vector_store = vector_store

        # 设置日志记录器
        self.logger = logger

    def preload_memories(
        self, memory_ids: List[str], cache: Optional[Dict[str, MemorySnapshot]] = None
    ) -> Dict[str, MemorySnapshot]:
        """批量加载记忆快照，减少重复的数据库访问。"""
        cache = cache or {}
        if not memory_ids:
            return cache

        missing_ids = [memory_id for memory_id in memory_ids if memory_id not in cache]
        if not missing_ids:
            return cache

        try:
            results = self.collection.get(
                ids=missing_ids, include=["metadatas", "documents", "embeddings"]
            )
        except Exception as e:
            self.logger.error(f"批量加载记忆失败: {str(e)}")
            return cache

        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []
        embeddings = results.get("embeddings")
        if embeddings is None:
            embeddings = []
        ids = results.get("ids") or []

        for idx, memory_id in enumerate(ids):
            if not metadatas or idx >= len(metadatas) or not metadatas[idx]:
                continue

            metadata = metadatas[idx]
            document = documents[idx] if idx < len(documents) else ""
            embedding = embeddings[idx] if idx < len(embeddings) else None
            try:
                memory_obj = BaseMemory.from_dict(metadata)
            except Exception as e:
                self.logger.error(f"构建记忆对象失败 {memory_id}: {str(e)}")
                continue

            cache[memory_id] = MemorySnapshot(
                memory=memory_obj,
                metadata=metadata,
                document=document,
                embedding=embedding,
            )

        return cache

    async def _persist_snapshot(self, snapshot: MemorySnapshot) -> bool:
        """将内存中的记忆快照写回存储，返回是否重新生成嵌入。"""
        embedding_regenerated = False
        try:
            metadata = snapshot.metadata

            # 更新关联字段
            import json

            metadata["associations"] = (
                json.dumps(snapshot.memory.associations)
                if snapshot.memory.associations
                else "{}"
            )

            embedding = snapshot.embedding
            if embedding is None:
                semantic_core = snapshot.memory.get_semantic_core()
                # 使用异步方法生成embedding
                embedding = (await self.vector_store.embedding_provider.embed_documents(
                    [semantic_core]
                ))[0]
                snapshot.embedding = embedding
                embedding_regenerated = True
                self.logger.debug(
                    f"[assoc] regenerated embedding for {snapshot.memory.id}"
                )

            self.collection.upsert(
                ids=[snapshot.memory.id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[snapshot.document],
            )
            return embedding_regenerated

        except Exception as e:
            self.logger.error(f"更新记忆 {snapshot.memory.id} 的关联失败: {str(e)}")
            return False

    async def reinforce_association(self, memory_ids: List[str]):
        """
        强化记忆之间的关联强度。

        当多个记忆同时被认为有用时，在它们之间建立或强化关联。
        遵循赫布理论："一起激活，一起连接"。

        Args:
            memory_ids: 同时被激活的记忆ID列表
        """
        if len(memory_ids) < 2:
            return  # 需要至少2个记忆才能建立关联

        # 获取所有记忆对象
        snapshot_cache = self.preload_memories(memory_ids)

        # 对所有ID对（两两组合）进行关联强化
        for id1, id2 in combinations(sorted(memory_ids), 2):
            # 找到对应的记忆对象
            snapshot1 = snapshot_cache.get(id1)
            snapshot2 = snapshot_cache.get(id2)
            mem1 = snapshot1.memory if snapshot1 else None
            mem2 = snapshot2.memory if snapshot2 else None

            if mem1 and mem2:
                # 在记忆1的关联字段中强化对记忆2的关联
                if id2 in mem1.associations:
                    mem1.associations[id2] += 1
                else:
                    mem1.associations[id2] = 1

                # 在记忆2的关联字段中强化对记忆1的关联（双向关联）
                if id1 in mem2.associations:
                    mem2.associations[id1] += 1
                else:
                    mem2.associations[id1] = 1

                # 更新存储中的关联信息
                if snapshot1:
                    await self._persist_snapshot(snapshot1)
                if snapshot2:
                    await self._persist_snapshot(snapshot2)

    async def _add_or_update_association(
        self,
        id1: str,
        id2: str,
        strength_increase: int = 1,
        cache: Optional[Dict[str, MemorySnapshot]] = None,
    ):
        """
        【内部方法】在两个记忆之间添加或更新关联（单向）。

        Args:
            id1: 源记忆ID
            id2: 目标记忆ID
            strength_increase: 关联强度增量（默认1）

        功能：
        - 如果关联已存在，则累加强度
        - 如果关联不存在，则创建新关联（初始强度 = strength_increase）
        - 注意：此方法只处理 id1 → id2 的单向关联
        - 如需双向关联，需要调用两次（id1→id2 和 id2→id1）
        """
        try:
            # 基本格式验证：快速过滤无效ID
            if (
                not id1
                or not id2
                or id1 == id2
                or not isinstance(id1, str)
                or not isinstance(id2, str)
            ):
                return

            # 获取源记忆
            cache = cache or {}
            snapshot = cache.get(id1)
            if not snapshot:
                cache = self.preload_memories([id1], cache)
                snapshot = cache.get(id1)

            # 如果源记忆不存在，静默跳过（不再记录警告）
            if not snapshot:
                return

            # 验证目标记忆是否存在
            target_snapshot = cache.get(id2)
            if not target_snapshot:
                # 尝试单独加载目标记忆
                target_cache = self.preload_memories([id2])
                target_snapshot = target_cache.get(id2)
                if not target_snapshot:
                    # 目标记忆不存在，静默跳过
                    return

            memory_obj = snapshot.memory

            # 更新关联强度（去重、累加）
            if id2 in memory_obj.associations:
                memory_obj.associations[id2] += strength_increase
            else:
                memory_obj.associations[id2] = strength_increase

            # 持久化到存储
            await self._persist_snapshot(snapshot)
            cache[id1] = snapshot

        except Exception as e:
            # 记录警告日志而不是静默忽略异常
            self.logger.warning(f"Failed to update association strength: {e}")

    def get_associations_for_memory(
        self, memory_id: str, min_strength: int = 1
    ) -> List[BaseMemory]:
        """
        获取与指定记忆相关的所有关联记忆。

        Args:
            memory_id: 记忆ID
            min_strength: 最小强度阈值

        Returns:
            相关的记忆列表
        """
        try:
            # 获取指定记忆的关联信息
            memory_results = self.collection.get(ids=[memory_id])
            if not memory_results or not memory_results["metadatas"]:
                return []

            memory_data = memory_results["metadatas"][0]
            # 解析associations（可能是JSON字符串或字典）
            from ..models.data_models import BaseMemory

            associations = BaseMemory._parse_associations(
                memory_data.get("associations", {})
            )

            # 过滤出强度大于阈值的关联
            strong_associations = {
                assoc_id: strength
                for assoc_id, strength in associations.items()
                if strength >= min_strength
            }

            # 获取关联的记忆对象
            associated_memories = []
            for assoc_id in strong_associations.keys():
                try:
                    assoc_results = self.collection.get(ids=[assoc_id])
                    if assoc_results and assoc_results["metadatas"]:
                        assoc_memory = BaseMemory.from_dict(
                            assoc_results["metadatas"][0]
                        )
                        associated_memories.append(assoc_memory)
                except Exception as e:
                    self.logger.error(f"获取关联记忆 {assoc_id} 失败: {str(e)}")
                    continue

            return associated_memories

        except Exception as e:
            self.logger.error(f"获取记忆 {memory_id} 的关联失败: {str(e)}")
            return []

    def get_association_strength(self, id1: str, id2: str) -> Optional[int]:
        """
        获取两个记忆之间的关联强度。

        Args:
            id1: 第一个记忆ID
            id2: 第二个记忆ID

        Returns:
            关联强度，如果不存在则返回None
        """
        try:
            # 获取第一个记忆的关联信息
            memory_results = self.collection.get(ids=[id1])
            if not memory_results or not memory_results["metadatas"]:
                return None

            memory_data = memory_results["metadatas"][0]
            # 解析associations（可能是JSON字符串或字典）
            from ..models.data_models import BaseMemory

            associations = BaseMemory._parse_associations(
                memory_data.get("associations", {})
            )

            return associations.get(id2)

        except Exception as e:
            self.logger.error(f"获取记忆 {id1} 和 {id2} 之间的关联强度失败: {str(e)}")
            return None

    async def cleanup_weak_associations(self, strength_threshold: int = 1):
        """
        清理强度过低的关联（在记忆巩固过程中调用）。

        Args:
            strength_threshold: 强度阈值，低于此值的关联将被删除
        """
        try:
            # 获取所有记忆
            all_results = self.collection.get()
            if not all_results or not all_results["metadatas"]:
                return

            cleaned_count = 0
            for i, metadata in enumerate(all_results["metadatas"]):
                memory_id = all_results["ids"][i]
                # 解析associations（可能是JSON字符串或字典）
                from ..models.data_models import BaseMemory

                associations = BaseMemory._parse_associations(
                    metadata.get("associations", {})
                )

                # 过滤出强度大于等于阈值的关联
                filtered_associations = {
                    assoc_id: strength
                    for assoc_id, strength in associations.items()
                    if strength >= strength_threshold
                }

                # 如果有关联被清理，更新存储
                if len(filtered_associations) < len(associations):
                    cleaned_count += len(associations) - len(filtered_associations)
                    # 序列化为JSON字符串
                    import json

                    metadata["associations"] = (
                        json.dumps(filtered_associations)
                        if filtered_associations
                        else "{}"
                    )

                    # 获取当前嵌入和文档
                    current_embedding = (
                        all_results["embeddings"][i]
                        if all_results["embeddings"]
                        and i < len(all_results["embeddings"])
                        else None
                    )
                    current_document = (
                        all_results["documents"][i]
                        if all_results["documents"]
                        and i < len(all_results["documents"])
                        else ""
                    )

                    # 如果没有嵌入，需要重新生成
                    if current_embedding is None:
                        memory_obj = BaseMemory.from_dict(metadata)
                        semantic_core = memory_obj.get_semantic_core()
                        # 使用异步方法生成embedding
                        current_embedding = (
                            await self.vector_store.embedding_provider.embed_documents(
                                [semantic_core]
                            )
                        )[0]

                    # 更新存储
                    self.collection.upsert(
                        ids=[memory_id],
                        embeddings=[current_embedding],
                        metadatas=[metadata],
                        documents=[current_document],
                    )

            self.logger.info(
                f"已清理 {cleaned_count} 个强度低于 {strength_threshold} 的关联"
            )

        except Exception as e:
            self.logger.error(f"清理弱关联失败: {str(e)}")
