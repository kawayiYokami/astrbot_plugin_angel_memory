"""
SessionMemoryManager 测试
"""

import pytest
import threading
import time
import sys
import os
import importlib.util

# 直接加载模块文件，避免触发 __init__.py
spec = importlib.util.spec_from_file_location(
    "session_memory",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "session_memory.py")
)
session_memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_memory_module)

# 从加载的模块获取类
SessionMemoryManager = session_memory_module.SessionMemoryManager
SessionMemory = session_memory_module.SessionMemory
MemoryItem = session_memory_module.MemoryItem


class TestSessionMemory:
    """测试 SessionMemory 类"""

    def test_session_memory_initialization(self):
        """测试会话记忆初始化"""
        session = SessionMemory("test_session", capacity_multiplier=2)

        # 验证容量设置
        assert session.capacity_multiplier == 2
        assert len(session.memories) == 5  # 五种记忆类型
        assert session.memories['knowledge'].maxlen == 28  # 14 * 2
        assert session.memories['emotional'].maxlen == 2   # 1 * 2
        assert session.memories['skill'].maxlen == 14      # 7 * 2
        assert session.memories['task'].maxlen == 2        # 1 * 2
        assert session.memories['event'].maxlen == 6       # 3 * 2

    def test_add_memories(self):
        """测试添加记忆"""
        session = SessionMemory("test_session")

        # 创建知识记忆
        from ..llm_memory.models.data_models import KnowledgeMemory
        knowledge_memory = KnowledgeMemory(
            judgment="测试知识",
            reasoning="测试理由",
            tags=["测试"],
            id="test_id",
            strength=0.8
        )

        session.add_memories([knowledge_memory])

        # 验证记忆已添加
        assert len(session.memories['knowledge']) == 1
        assert session.memories['knowledge'][0].id == knowledge_memory.id

    def test_get_memories_by_type(self):
        """测试按类型获取记忆"""
        session = SessionMemory("test_session")

        # 添加不同类型的记忆
        from ..llm_memory.models.data_models import KnowledgeMemory, SkillMemory
        knowledge_memory = KnowledgeMemory("知识1", "", [], id="id1", strength=0.8)
        skill_memory = SkillMemory("技能1", "", [], id="id2", strength=0.7)

        session.add_memories([knowledge_memory])
        session.add_memories([skill_memory])

        # 获取知识记忆
        knowledge_memories = session.get_memories_by_type("knowledge")
        assert len(knowledge_memories) == 1
        assert knowledge_memories[0].id == knowledge_memory.id

        # 获取技能记忆
        skill_memories = session.get_memories_by_type("skill")
        assert len(skill_memories) == 1
        assert skill_memories[0].id == skill_memory.id

        # 获取空的记忆类型
        emotional_memories = session.get_memories_by_type("emotional")
        assert len(emotional_memories) == 0

    def test_get_all_memories(self):
        """测试获取所有记忆"""
        session = SessionMemory("test_session")

        # 添加不同类型的记忆
        from ..llm_memory.models.data_models import KnowledgeMemory, SkillMemory
        memory1 = KnowledgeMemory("知识1", "", [], id="id1", strength=0.8)
        memory2 = SkillMemory("技能1", "", [], id="id2", strength=0.7)

        session.add_memories([memory1])
        session.add_memories([memory2])

        # 获取所有记忆
        all_memories = session.get_memories()
        assert len(all_memories) == 2
        assert all_memories[0].id == memory1.id
        assert all_memories[1].id == memory2.id

    def test_capacity_limit(self):
        """测试容量限制"""
        session = SessionMemory("test_session", capacity_multiplier=1)

        # 添加超过容量的知识记忆（容量为14）
        from ..llm_memory.models.data_models import KnowledgeMemory
        for i in range(20):
            memory = KnowledgeMemory(f"知识{i}", "", [], id=f"id_{i}", strength=0.8)
            session.add_memories([memory])

        # 验证只保留最新的14个
        knowledge_memories = session.get_memories_by_type("knowledge")
        assert len(knowledge_memories) == 14

        # 验证保留的是最新的记忆（ID从6到19）
        memory_ids = [m.id for m in knowledge_memories]
        assert "id_0" not in memory_ids  # 最早的被移除
        assert "id_19" in memory_ids     # 最新的保留

    def test_thread_safety(self):
        """测试线程安全性"""
        session = SessionMemory("test_session")
        results = []
        errors = []

        def add_memories(start_id):
            try:
                from ..llm_memory.models.data_models import KnowledgeMemory
                for i in range(10):
                    memory = KnowledgeMemory(
                        f"知识{start_id}_{i}",
                        "",
                        [],
                        id=f"id_{start_id}_{i}",
                        strength=0.8
                    )
                    session.add_memories([memory])
                    time.sleep(0.001)  # 短暂延迟增加竞争
                results.append(len(session.get_memories_by_type("knowledge")))
            except Exception as e:
                errors.append(e)

        # 创建多个线程同时添加记忆
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_memories, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有记忆都被添加（注意：由于容量限制，可能不会是50个）
        final_memories = session.get_memories_by_type("knowledge")
        # 验证添加了一些记忆
        assert len(final_memories) > 0


class TestSessionMemoryManager:
    """测试 SessionMemoryManager 类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建管理器实例
        self.manager = SessionMemoryManager(capacity_multiplier=1)

    def test_manager_initialization(self):
        """测试管理器初始化"""
        assert self.manager.capacity_multiplier == 1
        assert len(self.manager.sessions) == 0

    def test_get_or_create_session(self):
        """测试获取或创建会话"""
        # 获取新会话
        session1 = self.manager.get_or_create_session("test_session")
        assert isinstance(session1, SessionMemory)
        assert session1.session_id == "test_session"
        assert len(self.manager.sessions) == 1

        # 再次获取同一会话（应该返回同一个实例）
        session2 = self.manager.get_or_create_session("test_session")
        assert session1 is session2
        assert len(self.manager.sessions) == 1

        # 获取不同会话
        session3 = self.manager.get_or_create_session("another_session")
        assert session3.session_id == "another_session"
        assert len(self.manager.sessions) == 2

    def test_add_memories_to_session(self):
        """测试向会话添加记忆"""
        from ..llm_memory.models.data_models import KnowledgeMemory, SkillMemory
        memories = [
            KnowledgeMemory("知识1", "", [], id="id1", strength=0.8),
            SkillMemory("技能1", "", [], id="id2", strength=0.7)
        ]

        # 添加到新会话
        self.manager.add_memories_to_session("test_session", memories)

        # 验证记忆已添加
        session = self.manager.get_or_create_session("test_session")
        all_memories = session.get_memories()
        assert len(all_memories) == 2

    def test_get_session_memories(self):
        """测试获取会话记忆"""
        from ..llm_memory.models.data_models import KnowledgeMemory, SkillMemory
        memories = [
            KnowledgeMemory("知识1", "", [], id="id1", strength=0.8),
            SkillMemory("技能1", "", [], id="id2", strength=0.7)
        ]

        # 添加记忆到会话
        self.manager.add_memories_to_session("test_session", memories)

        # 获取会话记忆
        session_memories = self.manager.get_session_memories("test_session")
        assert len(session_memories) == 2

        # 获取不存在的会话记忆
        empty_memories = self.manager.get_session_memories("nonexistent")
        assert len(empty_memories) == 0

    def test_clear_session(self):
        """测试清空会话记忆"""
        from ..llm_memory.models.data_models import KnowledgeMemory
        memories = [KnowledgeMemory("知识1", "", [], id="id1", strength=0.8)]

        # 添加记忆到会话
        self.manager.add_memories_to_session("test_session", memories)
        assert len(self.manager.get_session_memories("test_session")) == 1

        # 清空会话
        self.manager.clear_session("test_session")
        assert len(self.manager.get_session_memories("test_session")) == 0
        assert "test_session" not in self.manager.sessions

    def test_get_all_session_stats(self):
        """测试获取所有会话统计信息"""
        from ..llm_memory.models.data_models import KnowledgeMemory, SkillMemory
        memories1 = [KnowledgeMemory("知识1", "", [], id="id1", strength=0.8)]
        memories2 = [SkillMemory("技能1", "", [], id="id2", strength=0.7)]

        # 添加到不同会话
        self.manager.add_memories_to_session("session1", memories1)
        self.manager.add_memories_to_session("session2", memories2)

        # 获取统计信息
        all_stats = self.manager.get_all_session_stats()

        assert "session1" in all_stats
        assert "session2" in all_stats
        assert all_stats["session1"]["total_memories"] == 1
        assert all_stats["session2"]["total_memories"] == 1
        assert all_stats["session1"]["by_type"]["knowledge"]["current"] == 1
        assert all_stats["session2"]["by_type"]["skill"]["current"] == 1