"""
测试记忆管理器模块
"""

import pytest

from ..models.data_models import BaseMemory, MemoryType, ValidationError
from ..service.memory_manager import MemoryManager
from ..components.association_manager import AssociationManager


class TestMemoryManager:
    """测试MemoryManager类"""

    @pytest.fixture
    def memory_manager(self, vector_store):
        """创建记忆管理器实例"""
        association_manager = AssociationManager(vector_store)
        return MemoryManager(vector_store, association_manager)

    def test_initialization(self, memory_manager, vector_store):
        """测试初始化"""
        assert memory_manager.store == vector_store
        assert memory_manager.association_manager is not None

    def test_reinforce_memories(self, memory_manager, vector_store):
        """测试强化记忆"""
        # 创建一条记忆
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试",
            reasoning="测试",
            tags=["测试"]
        )
        vector_store.remember(memory)

        # 强化
        memory_manager.reinforce_memories([memory.id])

        # 验证强度增加
        results = vector_store.collection.get(ids=[memory.id])
        assert results['metadatas'][0]['strength'] == 2

    def test_reinforce_multiple_times(self, memory_manager, vector_store):
        """测试多次强化"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试",
            reasoning="测试",
            tags=["测试"]
        )
        vector_store.remember(memory)

        # 强化3次
        for _ in range(3):
            memory_manager.reinforce_memories([memory.id])

        # 验证强度为4
        results = vector_store.collection.get(ids=[memory.id])
        assert results['metadatas'][0]['strength'] == 4

    def test_consolidate_memories(self, memory_manager, vector_store):
        """测试记忆巩固"""
        # 创建新鲜记忆
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试",
            reasoning="测试",
            tags=["测试"],
            is_consolidated=False
        )
        vector_store.remember(memory)

        # 巩固
        memory_manager.consolidate_memories()

        # 验证已巩固
        results = vector_store.collection.get(ids=[memory.id])
        assert results['metadatas'][0]['is_consolidated'] == True

    def test_comprehensive_recall(self, memory_manager, vector_store):
        """测试综合回忆"""
        # 创建新鲜记忆和已巩固记忆
        fresh_memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="新鲜记忆",
            reasoning="测试",
            tags=["测试"],
            is_consolidated=False
        )
        vector_store.remember(fresh_memory)

        consolidated_memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="已巩固记忆",
            reasoning="测试",
            tags=["测试"],
            is_consolidated=True
        )
        vector_store.remember(consolidated_memory)

        # 综合回忆
        results = memory_manager.comprehensive_recall("测试", fresh_limit=5, consolidated_limit=5)

        assert len(results) >= 2

    def test_comprehensive_recall_deduplication(self, memory_manager, vector_store):
        """测试综合回忆去重"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试记忆",
            reasoning="测试",
            tags=["测试"]
        )
        vector_store.remember(memory)

        # 综合回忆（可能返回重复）
        results = memory_manager.comprehensive_recall("测试", fresh_limit=10, consolidated_limit=10)

        # 验证没有重复ID
        ids = [m.id for m in results]
        assert len(ids) == len(set(ids))

    def test_merge_memories(self, memory_manager, vector_store):
        """测试记忆合并"""
        # 创建两条记忆
        memory1 = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="记忆1",
            reasoning="理由1",
            tags=["测试"],
            strength=2
        )
        vector_store.remember(memory1)

        memory2 = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="记忆2",
            reasoning="理由2",
            tags=["测试"],
            strength=3
        )
        vector_store.remember(memory2)

        # 合并
        new_memory = memory_manager.merge_memories(
            memories_to_merge_ids=[memory1.id, memory2.id],
            new_judgment="合并后的记忆",
            new_reasoning="合并理由",
            new_tags=["测试", "合并"]
        )

        # 验证新记忆强度等于旧记忆之和
        assert new_memory.strength == 5

        # 验证旧记忆已删除
        results = vector_store.collection.get(ids=[memory1.id, memory2.id])
        assert len(results['ids']) == 0

    def test_merge_memories_insufficient(self, memory_manager, vector_store):
        """测试合并不足2条记忆"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="记忆",
            reasoning="理由",
            tags=["测试"]
        )
        vector_store.remember(memory)

        with pytest.raises(ValidationError):
            memory_manager.merge_memories(
                memories_to_merge_ids=[memory.id],
                new_judgment="合并",
                new_reasoning="理由",
                new_tags=["测试"]
            )

    def test_build_memory_from_metadata(self, memory_manager):
        """测试从元数据构建记忆"""
        metadata = {
            "memory_type": "知识记忆",
            "judgment": "测试判断",
            "tags": "标签1, 标签2",
            "strength": 3,
            "is_consolidated": False,
            "associations": "{}"
        }
        document = "测试理由"

        memory = memory_manager._build_memory_from_metadata("test_id", metadata, document)

        assert memory is not None
        assert memory.id == "test_id"
        assert memory.memory_type == MemoryType.KNOWLEDGE
        assert memory.judgment == "测试判断"
        assert memory.reasoning == "测试理由"
        assert memory.strength == 3

    def test_build_memory_exclude_task(self, memory_manager):
        """测试构建记忆时排除任务记忆"""
        metadata = {
            "memory_type": "任务记忆",
            "judgment": "任务",
            "tags": "",
            "strength": 1,
            "is_consolidated": False
        }

        memory = memory_manager._build_memory_from_metadata("test_id", metadata, "")
        assert memory is None

    def test_process_feedback_reinforce(self, memory_manager, vector_store):
        """测试反馈处理 - 强化记忆"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试",
            reasoning="测试",
            tags=["测试"]
        )
        vector_store.remember(memory)

        # 提供反馈
        memory_manager.process_feedback(
            query="测试",
            useful_memory_ids=[memory.id],
            memory_handlers={}
        )

        # 验证强化
        results = vector_store.collection.get(ids=[memory.id])
        assert results['metadatas'][0]['strength'] > 1

    def test_process_feedback_create_new(self, memory_manager, vector_store):
        """测试反馈处理 - 创建新记忆"""
        from ..service.memory_handlers import MemoryHandlerFactory

        factory = MemoryHandlerFactory(vector_store)

        # 提供反馈创建新记忆
        memory_manager.process_feedback(
            query="测试",
            new_memories=[
                {
                    "type": "knowledge",
                    "judgment": "新判断",
                    "reasoning": "新理由",
                    "tags": ["新标签"]
                }
            ],
            memory_handlers=factory.handlers
        )

        # 验证新记忆已创建
        results = vector_store.recall("新判断", limit=5)
        assert len(results) > 0
