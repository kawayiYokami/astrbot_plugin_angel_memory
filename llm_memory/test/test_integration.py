"""
集成测试 - 测试完整的工作流程
"""

import pytest


class TestIntegration:
    """测试完整的记忆系统工作流"""

    def test_basic_workflow(self, cognitive_service):
        """测试基本工作流：记忆 -> 回忆 -> 反馈"""
        # 步骤1: 记住几条记忆
        id1 = cognitive_service.remember_knowledge(
            judgment="Python是一门编程语言",
            reasoning="它有解释器、语法和标准库",
            tags=["编程", "Python"]
        )

        id2 = cognitive_service.remember_event(
            judgment="用户询问Python的问题",
            reasoning="在今天的对话中",
            tags=["对话", "Python"]
        )

        id3 = cognitive_service.remember_skill(
            judgment="使用pip安装包",
            reasoning="执行pip install package_name",
            tags=["Python", "pip", "工具"]
        )

        # 步骤2: 回忆相关记忆
        results = cognitive_service.comprehensive_recall("Python", fresh_limit=10, consolidated_limit=10)

        assert len(results) >= 3
        result_ids = [m.id for m in results]
        assert id1 in result_ids
        assert id2 in result_ids
        assert id3 in result_ids

        # 步骤3: 提供反馈
        cognitive_service.feedback(
            query="Python",
            useful_memory_ids=[id1, id2],
            new_memories=[
                {
                    "type": "knowledge",
                    "judgment": "Python广泛应用于数据科学",
                    "reasoning": "因为有numpy、pandas等库",
                    "tags": ["Python", "数据科学"]
                }
            ]
        )

        # 验证反馈效果
        updated_results = cognitive_service.main_store.collection.get(ids=[id1, id2])
        assert updated_results['metadatas'][0]['strength'] > 1
        assert updated_results['metadatas'][1]['strength'] > 1

    def test_consolidation_workflow(self, cognitive_service):
        """测试记忆巩固工作流"""
        # 步骤1: 记住新鲜记忆
        memory_id = cognitive_service.remember_knowledge(
            judgment="测试记忆",
            reasoning="测试理由",
            tags=["测试"]
        )

        # 验证是新鲜记忆
        results = cognitive_service.recall_knowledge("测试", include_consolidated=False)
        assert len(results) > 0

        # 步骤2: 巩固记忆
        cognitive_service.consolidate_memories()

        # 步骤3: 验证已巩固
        fresh_results = cognitive_service.recall_knowledge("测试", include_consolidated=False)
        assert len(fresh_results) == 0

        consolidated_results = cognitive_service.recall_knowledge("测试", include_consolidated=True)
        assert len(consolidated_results) > 0

    def test_memory_merge_workflow(self, cognitive_service):
        """测试记忆合并工作流"""
        # 步骤1: 记住相似的记忆
        id1 = cognitive_service.remember_knowledge(
            judgment="Python是一门编程语言",
            reasoning="理由1",
            tags=["Python"]
        )

        id2 = cognitive_service.remember_knowledge(
            judgment="Python用于编程",
            reasoning="理由2",
            tags=["Python"]
        )

        # 步骤2: 通过反馈合并记忆
        cognitive_service.feedback(
            query="Python",
            merge_groups=[[id1, id2]]
        )

        # 验证旧记忆已删除
        results = cognitive_service.main_store.collection.get(ids=[id1, id2])
        assert len(results['ids']) == 0

        # 新记忆应该存在
        all_memories = cognitive_service.recall_knowledge("Python")
        assert len(all_memories) > 0

    def test_multi_type_recall(self, cognitive_service):
        """测试多类型记忆检索"""
        # 记住不同类型的记忆，都关于同一个主题
        cognitive_service.remember_knowledge(
            judgment="机器学习是AI的一个分支",
            reasoning="它使用数据训练模型",
            tags=["AI", "机器学习"]
        )

        cognitive_service.remember_event(
            judgment="用户问了关于机器学习的问题",
            reasoning="在讨论AI时",
            tags=["对话", "机器学习"]
        )

        cognitive_service.remember_skill(
            judgment="训练机器学习模型的步骤",
            reasoning="准备数据、选择模型、训练、评估",
            tags=["机器学习", "流程"]
        )

        # 综合回忆应该返回所有类型
        results = cognitive_service.comprehensive_recall("机器学习")

        memory_types = set(m.memory_type.value for m in results)
        assert len(memory_types) >= 2

    def test_association_workflow(self, cognitive_service):
        """测试关联建立工作流"""
        # 记住两条记忆
        id1 = cognitive_service.remember_knowledge(
            judgment="记忆1",
            reasoning="理由",
            tags=["测试"]
        )

        id2 = cognitive_service.remember_knowledge(
            judgment="记忆2",
            reasoning="理由",
            tags=["测试"]
        )

        # 通过反馈建立关联
        cognitive_service.feedback(
            query="测试",
            useful_memory_ids=[id1, id2]
        )

        # 验证关联已建立
        mem1 = cognitive_service.main_store.collection.get(ids=[id1])
        associations = cognitive_service.main_store._parse_associations(
            mem1['metadatas'][0].get('associations', '{}')
        )

        assert id2 in associations

    def test_strength_accumulation(self, cognitive_service):
        """测试强度累积"""
        # 记住一条记忆
        memory_id = cognitive_service.remember_knowledge(
            judgment="重要记忆",
            reasoning="很重要",
            tags=["重要"]
        )

        # 多次反馈强化
        for _ in range(3):
            cognitive_service.feedback(
                query="重要",
                useful_memory_ids=[memory_id]
            )

        # 验证强度累积
        results = cognitive_service.main_store.collection.get(ids=[memory_id])
        assert results['metadatas'][0]['strength'] >= 4

    def test_empty_recall(self, cognitive_service):
        """测试空回忆"""
        results = cognitive_service.comprehensive_recall("不存在的内容xyz123")
        assert isinstance(results, list)
        # 可能为空或返回不相关的结果

    def test_clear_and_rebuild(self, cognitive_service):
        """测试清空和重建"""
        # 记住一些记忆
        cognitive_service.remember_knowledge(
            judgment="记忆1",
            reasoning="理由",
            tags=["测试"]
        )

        cognitive_service.remember_event(
            judgment="记忆2",
            reasoning="理由",
            tags=["测试"]
        )

        # 清空
        cognitive_service.clear_all_memories()

        # 验证已清空
        results = cognitive_service.comprehensive_recall("测试")
        count = cognitive_service.main_store.collection.count()
        assert count == 0

        # 重建
        new_id = cognitive_service.remember_knowledge(
            judgment="新记忆",
            reasoning="新理由",
            tags=["新测试"]
        )

        # 验证可以正常使用
        new_results = cognitive_service.recall_knowledge("新测试")
        assert len(new_results) > 0

    def test_large_batch_operations(self, cognitive_service):
        """测试批量操作"""
        # 批量记住
        ids = []
        for i in range(10):
            memory_id = cognitive_service.remember_knowledge(
                judgment=f"记忆{i}",
                reasoning=f"理由{i}",
                tags=["批量", f"编号{i}"]
            )
            ids.append(memory_id)

        # 批量回忆
        results = cognitive_service.comprehensive_recall("批量", fresh_limit=20)
        assert len(results) >= 10

        # 批量强化
        cognitive_service.feedback(
            query="批量",
            useful_memory_ids=ids[:5]
        )

        # 验证部分被强化
        reinforced = cognitive_service.main_store.collection.get(ids=ids[:5])
        for metadata in reinforced['metadatas']:
            assert metadata['strength'] > 1

    def test_different_tags_recall(self, cognitive_service):
        """测试不同标签的回忆"""
        # 记住带有不同标签的记忆
        cognitive_service.remember_knowledge(
            judgment="Python教程",
            reasoning="入门教程",
            tags=["Python", "教程", "入门"]
        )

        cognitive_service.remember_knowledge(
            judgment="Python进阶",
            reasoning="高级特性",
            tags=["Python", "进阶", "高级"]
        )

        # 按不同关键词回忆
        basic_results = cognitive_service.recall_knowledge("入门")
        advanced_results = cognitive_service.recall_knowledge("进阶")
        general_results = cognitive_service.recall_knowledge("Python")

        assert len(general_results) >= len(basic_results)
        assert len(general_results) >= len(advanced_results)
