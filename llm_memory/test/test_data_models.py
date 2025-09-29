"""
测试数据模型模块
"""

import pytest
import uuid

from ..models.data_models import (
    BaseMemory, MemoryType, ValidationError
)


class TestBaseMemory:
    """测试BaseMemory类"""

    def test_create_knowledge_memory(self):
        """测试创建知识记忆"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试判断",
            reasoning="测试理由",
            tags=["标签1", "标签2"]
        )

        assert memory.memory_type == MemoryType.KNOWLEDGE
        assert memory.judgment == "测试判断"
        assert memory.reasoning == "测试理由"
        assert memory.tags == ["标签1", "标签2"]
        assert memory.strength == 1
        assert memory.is_consolidated == False
        assert isinstance(memory.id, str)
        assert len(memory.id) > 0

    def test_create_with_custom_id(self):
        """测试使用自定义ID创建记忆"""
        custom_id = str(uuid.uuid4())
        memory = BaseMemory(
            memory_type=MemoryType.EVENT,
            judgment="测试",
            reasoning="测试",
            tags=[],
            id=custom_id
        )

        assert memory.id == custom_id

    def test_create_all_memory_types(self):
        """测试创建所有记忆类型"""
        types = [
            MemoryType.KNOWLEDGE,
            MemoryType.EVENT,
            MemoryType.SKILL,
            MemoryType.TASK,
            MemoryType.EMOTIONAL
        ]

        for mem_type in types:
            memory = BaseMemory(
                memory_type=mem_type,
                judgment="测试",
                reasoning="测试",
                tags=[]
            )
            assert memory.memory_type == mem_type

    def test_get_semantic_core(self):
        """测试语义核心生成"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="Python是编程语言",
            reasoning="理由",
            tags=["编程", "Python"]
        )

        core = memory.get_semantic_core()
        assert "Python是编程语言" in core
        assert "编程" in core
        assert "Python" in core

    def test_to_dict(self):
        """测试转换为字典"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="判断",
            reasoning="理由",
            tags=["标签1", "标签2"],
            strength=5,
            is_consolidated=True
        )

        data = memory.to_dict()

        assert data["memory_type"] == "知识记忆"
        assert data["judgment"] == "判断"
        assert data["reasoning"] == "理由"
        assert data["tags"] == "标签1, 标签2"  # 转换为字符串
        assert data["strength"] == 5
        assert data["is_consolidated"] == True
        assert "id" in data

    def test_from_dict(self):
        """测试从字典创建记忆"""
        data = {
            "id": str(uuid.uuid4()),
            "memory_type": "知识记忆",
            "judgment": "判断",
            "reasoning": "理由",
            "tags": "标签1, 标签2",
            "strength": 3,
            "is_consolidated": False,
            "associations": "{}"
        }

        memory = BaseMemory.from_dict(data)

        assert memory.memory_type == MemoryType.KNOWLEDGE
        assert memory.judgment == "判断"
        assert memory.reasoning == "理由"
        assert memory.tags == ["标签1", "标签2"]
        assert memory.strength == 3
        assert memory.is_consolidated == False

    def test_from_dict_with_list_tags(self):
        """测试从字典创建记忆（tags为列表）"""
        data = {
            "id": str(uuid.uuid4()),
            "memory_type": "事件记忆",
            "judgment": "判断",
            "reasoning": "理由",
            "tags": ["标签1", "标签2"],
            "strength": 1,
            "is_consolidated": False
        }

        memory = BaseMemory.from_dict(data)
        assert memory.tags == ["标签1", "标签2"]

    def test_from_dict_invalid_type(self):
        """测试从无效类型创建记忆"""
        with pytest.raises(ValidationError):
            BaseMemory.from_dict("not a dict")

    def test_from_dict_missing_id(self):
        """测试从缺少ID的字典创建记忆"""
        data = {
            "memory_type": "知识记忆",
            "judgment": "判断",
            "reasoning": "理由",
            "tags": []
        }

        with pytest.raises(ValidationError):
            BaseMemory.from_dict(data)

    def test_from_dict_default_memory_type(self):
        """测试默认记忆类型"""
        data = {
            "id": str(uuid.uuid4()),
            "judgment": "判断",
            "reasoning": "理由",
            "tags": []
        }

        memory = BaseMemory.from_dict(data)
        assert memory.memory_type == MemoryType.EVENT  # 默认为事件记忆

    def test_associations(self):
        """测试关联字段"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="测试",
            reasoning="测试",
            tags=[],
            associations={"id1": 5, "id2": 3}
        )

        assert memory.associations == {"id1": 5, "id2": 3}

    def test_parse_tags_empty_string(self):
        """测试解析空字符串标签"""
        tags = BaseMemory._parse_tags("")
        assert tags == []

    def test_parse_tags_with_spaces(self):
        """测试解析带空格的标签"""
        tags = BaseMemory._parse_tags("标签1 , 标签2 , 标签3")
        assert tags == ["标签1", "标签2", "标签3"]

    def test_str_representation(self):
        """测试字符串表示"""
        memory = BaseMemory(
            memory_type=MemoryType.KNOWLEDGE,
            judgment="这是一个很长的判断内容用来测试截断功能",
            reasoning="理由",
            tags=["标签1"]
        )

        str_repr = str(memory)
        assert "Memory" in str_repr
        assert "知识记忆" in str_repr
        assert memory.id[:8] in str_repr
