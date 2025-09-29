"""
pytest配置文件，提供测试fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from ..service.cognitive_service import CognitiveService
from ..components.vector_store import VectorStore
from ..models.data_models import BaseMemory, MemoryType


@pytest.fixture(scope="function")
def temp_dir():
    """创建临时目录用于测试"""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture(scope="function")
def vector_store(temp_dir):
    """创建临时向量存储实例"""
    store = VectorStore(collection_name="test_collection")
    store.set_storage_path(str(temp_dir / "test_db"))
    yield store
    # 清理
    try:
        store.clear_all()
    except:
        pass


@pytest.fixture(scope="function")
def cognitive_service(temp_dir):
    """创建临时认知服务实例"""
    service = CognitiveService()
    service.set_storage_path(str(temp_dir / "test_db"))
    yield service
    # 清理
    try:
        service.clear_all_memories()
    except:
        pass


@pytest.fixture
def sample_knowledge_memory():
    """示例知识记忆"""
    return BaseMemory(
        memory_type=MemoryType.KNOWLEDGE,
        judgment="Python是一门编程语言",
        reasoning="因为它有解释器、语法规则和标准库",
        tags=["编程", "语言", "Python"]
    )


@pytest.fixture
def sample_event_memory():
    """示例事件记忆"""
    return BaseMemory(
        memory_type=MemoryType.EVENT,
        judgment="用户询问了关于Python的问题",
        reasoning="在2024年的对话中",
        tags=["对话", "Python", "问答"]
    )


@pytest.fixture
def sample_skill_memory():
    """示例技能记忆"""
    return BaseMemory(
        memory_type=MemoryType.SKILL,
        judgment="如何使用pip安装包",
        reasoning="执行命令: pip install package_name",
        tags=["Python", "pip", "安装"]
    )
