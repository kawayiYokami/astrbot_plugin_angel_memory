"""
记忆关联管理器。

负责管理记忆单元之间的动态关联关系，实现赫布理论（"一起激活，一起连接"）。
关联关系现在存储在每个记忆对象的内部关联字段中，而非独立的记忆类型。
"""

from typing import List, Optional, Tuple, Dict
from itertools import combinations

from .vector_store import VectorStore
from ..models.data_models import BaseMemory
from ...core.logger import get_logger


class AssociationManager:
    """
    记忆关联管理器。

    管理记忆之间的动态关联网络，支持强化、查询和清理关联关系。
    现在通过操作记忆内部的关联字段来实现关联功能，而非创建独立的关联对象。
    """

    def __init__(self, vector_store: VectorStore):
        """
        初始化关联管理器。

        Args:
            vector_store: VectorStore 实例
        """
        self.vector_store = vector_store

        # 设置日志记录器
        self.logger = get_logger()

    def reinforce_association(self, memory_ids: List[str]):
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
        memories = []
        for memory_id in memory_ids:
            try:
                memory_results = self.vector_store.collection.get(ids=[memory_id])
                if memory_results and memory_results['metadatas']:
                    memory_data = memory_results['metadatas'][0]
                    memory_obj = BaseMemory.from_dict(memory_data)
                    memories.append(memory_obj)
            except Exception as e:
                self.logger.error(f"获取记忆 {memory_id} 失败: {str(e)}")
                continue

        # 对所有ID对（两两组合）进行关联强化
        for id1, id2 in combinations(sorted(memory_ids), 2):
            # 找到对应的记忆对象
            mem1 = next((m for m in memories if m.id == id1), None)
            mem2 = next((m for m in memories if m.id == id2), None)

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
                self._update_memory_associations(mem1)
                self._update_memory_associations(mem2)

    def _add_or_update_association(self, id1: str, id2: str, strength_increase: int = 1):
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
            # 获取源记忆
            memory_results = self.vector_store.collection.get(ids=[id1])
            if not memory_results or not memory_results['metadatas']:
                self.logger.warning(f"记忆 {id1} 不存在，无法建立关联")
                return

            memory_data = memory_results['metadatas'][0]
            memory_obj = BaseMemory.from_dict(memory_data)

            # 更新关联强度（去重、累加）
            if id2 in memory_obj.associations:
                memory_obj.associations[id2] += strength_increase
            else:
                memory_obj.associations[id2] = strength_increase

            # 持久化到存储
            self._update_memory_associations(memory_obj)

        except Exception as e:
            self.logger.error(f"添加/更新关联 {id1} → {id2} 失败: {str(e)}")

    def _update_memory_associations(self, memory: BaseMemory):
        """
        更新记忆的关联字段到存储中。

        Args:
            memory: 要更新的记忆对象
        """
        try:
            # 获取记忆的完整信息
            current_data = self.vector_store.collection.get(ids=[memory.id])
            if not current_data or not current_data['metadatas']:
                self.logger.warning(f"记忆 {memory.id} 不存在，无法更新关联")
                return

            current_meta = current_data['metadatas'][0]
            current_document = current_data['documents'][0] if current_data['documents'] else ""

            # 获取当前的嵌入（如果可用）
            if current_data['embeddings'] and len(current_data['embeddings']) > 0:
                current_embedding = current_data['embeddings'][0]
            else:
                # 如果没有嵌入，我们需要重新生成一个
                semantic_core = memory.get_semantic_core()
                current_embedding = self.vector_store.embedding_model.encode(semantic_core).tolist()

            # 更新关联字段（序列化为JSON字符串，因为ChromaDB不支持嵌套字典）
            import json
            current_meta['associations'] = json.dumps(memory.associations) if memory.associations else "{}"

            # 重新存储
            self.vector_store.collection.upsert(
                ids=[memory.id],
                embeddings=[current_embedding],
                metadatas=[current_meta],
                documents=[current_document]
            )

        except Exception as e:
            self.logger.error(f"更新记忆 {memory.id} 的关联失败: {str(e)}")

    def get_associations_for_memory(self, memory_id: str, min_strength: int = 1) -> List[BaseMemory]:
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
            memory_results = self.vector_store.collection.get(ids=[memory_id])
            if not memory_results or not memory_results['metadatas']:
                return []

            memory_data = memory_results['metadatas'][0]
            # 解析associations（可能是JSON字符串或字典）
            from ..models.data_models import BaseMemory
            associations = BaseMemory._parse_associations(memory_data.get('associations', {}))

            # 过滤出强度大于阈值的关联
            strong_associations = {assoc_id: strength for assoc_id, strength in associations.items() if strength >= min_strength}

            # 获取关联的记忆对象
            associated_memories = []
            for assoc_id in strong_associations.keys():
                try:
                    assoc_results = self.vector_store.collection.get(ids=[assoc_id])
                    if assoc_results and assoc_results['metadatas']:
                        assoc_memory = BaseMemory.from_dict(assoc_results['metadatas'][0])
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
            memory_results = self.vector_store.collection.get(ids=[id1])
            if not memory_results or not memory_results['metadatas']:
                return None

            memory_data = memory_results['metadatas'][0]
            # 解析associations（可能是JSON字符串或字典）
            from ..models.data_models import BaseMemory
            associations = BaseMemory._parse_associations(memory_data.get('associations', {}))

            return associations.get(id2)

        except Exception as e:
            self.logger.error(f"获取记忆 {id1} 和 {id2} 之间的关联强度失败: {str(e)}")
            return None

    def cleanup_weak_associations(self, strength_threshold: int = 1):
        """
        清理强度过低的关联（在记忆巩固过程中调用）。

        Args:
            strength_threshold: 强度阈值，低于此值的关联将被删除
        """
        try:
            # 获取所有记忆
            all_results = self.vector_store.collection.get()
            if not all_results or not all_results['metadatas']:
                return

            cleaned_count = 0
            for i, metadata in enumerate(all_results['metadatas']):
                memory_id = all_results['ids'][i]
                # 解析associations（可能是JSON字符串或字典）
                from ..models.data_models import BaseMemory
                associations = BaseMemory._parse_associations(metadata.get('associations', {}))

                # 过滤出强度大于等于阈值的关联
                filtered_associations = {assoc_id: strength for assoc_id, strength in associations.items() if strength >= strength_threshold}

                # 如果有关联被清理，更新存储
                if len(filtered_associations) < len(associations):
                    cleaned_count += len(associations) - len(filtered_associations)
                    # 序列化为JSON字符串
                    import json
                    metadata['associations'] = json.dumps(filtered_associations) if filtered_associations else "{}"

                    # 获取当前嵌入和文档
                    current_embedding = all_results['embeddings'][i] if all_results['embeddings'] and i < len(all_results['embeddings']) else None
                    current_document = all_results['documents'][i] if all_results['documents'] and i < len(all_results['documents']) else ""

                    # 如果没有嵌入，需要重新生成
                    if current_embedding is None:
                        memory_obj = BaseMemory.from_dict(metadata)
                        semantic_core = memory_obj.get_semantic_core()
                        current_embedding = self.vector_store.embedding_model.encode(semantic_core).tolist()

                    # 更新存储
                    self.vector_store.collection.upsert(
                        ids=[memory_id],
                        embeddings=[current_embedding],
                        metadatas=[metadata],
                        documents=[current_document]
                    )

            self.logger.info(f"已清理 {cleaned_count} 个强度低于 {strength_threshold} 的关联")

        except Exception as e:
            self.logger.error(f"清理弱关联失败: {str(e)}")
