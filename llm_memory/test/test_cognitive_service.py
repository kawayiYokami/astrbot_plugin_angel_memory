"""
测试认知服务模块
"""

import pytest


class TestCognitiveService:
    """测试CognitiveService类"""

    def test_initialization(self, cognitive_service):
        """测试初始化"""
        assert cognitive_service.main_store is not None
        assert cognitive_service.association_manager is not None
        assert cognitive_service.memory_handler_factory is not None
        assert cognitive_service.memory_manager is not None

    def test_remember_generic(self, cognitive_service):
        """测试通用记忆接口"""
        memory_id = cognitive_service.remember(
            memory_type="knowledge",
            judgment="Python是编程语言",
            reasoning="因为它有解释器",
            tags=["编程", "Python"]
        )

        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    def test_recall_generic(self, cognitive_service):
        """测试通用回忆接口"""
        # 先记住
        cognitive_service.remember(
            memory_type="knowledge",
            judgment="Python是编程语言",
            reasoning="因为它有解释器",
            tags=["编程", "Python"]
        )

        # 回忆
        results = cognitive_service.recall(
            memory_type="knowledge",
            query="Python",
            limit=5
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_remember_knowledge(self, cognitive_service):
        """测试记住知识记忆"""
        memory_id = cognitive_service.remember_knowledge(
            judgment="测试判断",
            reasoning="测试理由",
            tags=["测试"]
        )

        assert isinstance(memory_id, str)

    def test_recall_knowledge(self, cognitive_service):
        """测试回忆知识记忆"""
        cognitive_service.remember_knowledge(
            judgment="测试判断",
            reasoning="测试理由",
            tags=["测试"]
        )

        results = cognitive_service.recall_knowledge("测试", limit=5)
        assert len(results) > 0

    def test_remember_event(self, cognitive_service):
        """测试记住事件记忆"""
        memory_id = cognitive_service.remember_event(
            judgment="用户询问问题",
            reasoning="在对话中",
            tags=["对话"]
        )

        assert isinstance(memory_id, str)

    def test_recall_event(self, cognitive_service):
        """测试回忆事件记忆"""
        cognitive_service.remember_event(
            judgment="用户询问问题",
            reasoning="在对话中",
            tags=["对话"]
        )

        results = cognitive_service.recall_event("对话", limit=5)
        assert len(results) > 0

    def test_remember_skill(self, cognitive_service):
        """测试记住技能记忆"""
        memory_id = cognitive_service.remember_skill(
            judgment="使用pip安装包",
            reasoning="执行pip install命令",
            tags=["Python", "pip"]
        )

        assert isinstance(memory_id, str)

    def test_comprehensive_recall(self, cognitive_service):
        """测试综合回忆"""
        # 记住几条记忆
        cognitive_service.remember_knowledge(
            judgment="Python是编程语言",
            reasoning="理由",
            tags=["Python"]
        )

        cognitive_service.remember_event(
            judgment="用户问Python问题",
            reasoning="对话",
            tags=["Python"]
        )

        # 综合回忆
        results = cognitive_service.comprehensive_recall(
            query="Python",
            fresh_limit=5,
            consolidated_limit=5
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_consolidate_memories(self, cognitive_service):
        """测试记忆巩固"""
        # 记住一条记忆
        memory_id = cognitive_service.remember_knowledge(
            judgment="测试记忆",
            reasoning="测试",
            tags=["测试"]
        )

        # 巩固记忆
        cognitive_service.consolidate_memories()

        # 验证记忆已巩固
        results = cognitive_service.main_store.collection.get(ids=[memory_id])
        assert results['metadatas'][0]['is_consolidated'] == True

    def test_feedback(self, cognitive_service):
        """测试反馈处理"""
        # 记住一条记忆
        memory_id = cognitive_service.remember_knowledge(
            judgment="测试记忆",
            reasoning="测试",
            tags=["测试"]
        )

        # 提供反馈
        cognitive_service.feedback(
            query="测试",
            useful_memory_ids=[memory_id],
            new_memories=[
                {
                    "type": "knowledge",
                    "judgment": "新记忆",
                    "reasoning": "新理由",
                    "tags": ["新标签"]
                }
            ]
        )

        # 验证记忆被强化
        results = cognitive_service.main_store.collection.get(ids=[memory_id])
        assert results['metadatas'][0]['strength'] > 1

    def test_clear_all_memories(self, cognitive_service):
        """测试清空所有记忆"""
        # 记住几条记忆
        cognitive_service.remember_knowledge(
            judgment="记忆1",
            reasoning="测试",
            tags=["测试"]
        )

        cognitive_service.remember_event(
            judgment="记忆2",
            reasoning="测试",
            tags=["测试"]
        )

        # 清空
        cognitive_service.clear_all_memories()

        # 验证已清空
        count = cognitive_service.main_store.collection.count()
        assert count == 0

    def test_get_prompt(self):
        """测试获取提示词"""
        from ..service.cognitive_service import CognitiveService

        prompt = CognitiveService.get_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_set_storage_path(self, cognitive_service, temp_dir):
        """测试设置存储路径"""
        new_path = temp_dir / "new_storage"
        cognitive_service.set_storage_path(str(new_path))

        # 验证路径已更新
        assert cognitive_service.main_store.persist_directory == str(new_path)

    def test_multiple_memory_types(self, cognitive_service):
        """测试多种记忆类型共存"""
        # 记住不同类型的记忆
        id1 = cognitive_service.remember_knowledge(
            judgment="知识",
            reasoning="理由",
            tags=["测试"]
        )

        id2 = cognitive_service.remember_event(
            judgment="事件",
            reasoning="理由",
            tags=["测试"]
        )

        id3 = cognitive_service.remember_skill(
            judgment="技能",
            reasoning="理由",
            tags=["测试"]
        )

        # 验证都能回忆
        all_results = cognitive_service.comprehensive_recall("测试")
        assert len(all_results) >= 3
