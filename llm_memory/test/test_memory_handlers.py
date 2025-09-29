"""
测试记忆处理器模块
"""

import pytest

from ..models.data_models import MemoryType
from ..service.memory_handlers import MemoryHandler, MemoryHandlerFactory


class TestMemoryHandler:
    """测试MemoryHandler类"""

    def test_create_handler(self, vector_store):
        """测试创建处理器"""
        handler = MemoryHandler(MemoryType.KNOWLEDGE, vector_store)
        assert handler.memory_type == MemoryType.KNOWLEDGE
        assert handler.store == vector_store

    def test_remember(self, vector_store):
        """测试记住记忆"""
        handler = MemoryHandler(MemoryType.KNOWLEDGE, vector_store)

        memory_id = handler.remember(
            judgment="Python是编程语言",
            reasoning="因为它有解释器和语法",
            tags=["编程", "Python"]
        )

        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    def test_recall(self, vector_store):
        """测试回忆记忆"""
        handler = MemoryHandler(MemoryType.KNOWLEDGE, vector_store)

        # 先记住一条记忆
        handler.remember(
            judgment="Python是编程语言",
            reasoning="因为它有解释器",
            tags=["编程", "Python"]
        )

        # 回忆
        results = handler.recall("Python", limit=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0].memory_type == MemoryType.KNOWLEDGE

    def test_recall_filter_consolidated(self, vector_store):
        """测试回忆时过滤已巩固记忆"""
        handler = MemoryHandler(MemoryType.KNOWLEDGE, vector_store)

        # 记住一条记忆
        memory_id = handler.remember(
            judgment="测试记忆",
            reasoning="测试",
            tags=["测试"]
        )

        # 标记为已巩固
        vector_store.update_memory(memory_id, {"is_consolidated": True})

        # 不包含已巩固记忆
        results = handler.recall("测试", limit=5, include_consolidated=False)
        assert len(results) == 0

        # 包含已巩固记忆
        results = handler.recall("测试", limit=5, include_consolidated=True)
        assert len(results) > 0


class TestMemoryHandlerFactory:
    """测试MemoryHandlerFactory类"""

    def test_create_factory(self, vector_store):
        """测试创建工厂"""
        factory = MemoryHandlerFactory(vector_store)
        assert factory.store == vector_store
        assert len(factory.handlers) == 5  # 5种记忆类型

    def test_get_handler(self, vector_store):
        """测试获取处理器"""
        factory = MemoryHandlerFactory(vector_store)

        handler = factory.get_handler("knowledge")
        assert isinstance(handler, MemoryHandler)
        assert handler.memory_type == MemoryType.KNOWLEDGE

    def test_get_all_handler_types(self, vector_store):
        """测试获取所有类型的处理器"""
        factory = MemoryHandlerFactory(vector_store)

        types = ["event", "knowledge", "skill", "emotional", "task"]
        expected_types = [
            MemoryType.EVENT,
            MemoryType.KNOWLEDGE,
            MemoryType.SKILL,
            MemoryType.EMOTIONAL,
            MemoryType.TASK
        ]

        for type_str, expected_type in zip(types, expected_types):
            handler = factory.get_handler(type_str)
            assert handler.memory_type == expected_type

    def test_get_invalid_handler(self, vector_store):
        """测试获取无效类型的处理器"""
        factory = MemoryHandlerFactory(vector_store)

        with pytest.raises(ValueError):
            factory.get_handler("invalid_type")

    def test_handlers_share_store(self, vector_store):
        """测试所有处理器共享同一个存储"""
        factory = MemoryHandlerFactory(vector_store)

        handler1 = factory.get_handler("knowledge")
        handler2 = factory.get_handler("event")

        assert handler1.store is handler2.store
        assert handler1.store is vector_store
