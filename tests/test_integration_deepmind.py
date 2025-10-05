# tests/test_integration_deepmind.py

import sys
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock
from core.deepmind import DeepMind
from core.config import MemoryConfig
from llm_memory.service.note_service import NoteService
from llm_memory.components.vector_store import VectorStore
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

# 确保测试可以找到 llm_memory 和 astrbot 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# 假设 astrbot 在项目根目录的上一级的上一级的上一级
# e.g., from e:/github/ai-qq/astrbot/data/plugins/astrbot_plugin_angel_memory
# to e:/github/ai-qq/
sys.path.insert(0, str(project_root.parent.parent.parent))

# 从单元测试中导入测试数据fixture，实现复用

# --- Fixtures for Integration Test ---

@pytest.fixture(scope="function")
def note_service_with_data(tmp_path: Path, linked_notes_fixture):
    """
    创建一个预填充了链式笔记数据的、真实的NoteService实例。
    """
    db_path = tmp_path / "test_deepmind_db"
    db_path.mkdir()

    vector_store = VectorStore(db_path=str(db_path))
    note_service = NoteService(vector_store=vector_store)

    # 将测试数据添加到数据库
    for note_id, note_data in linked_notes_fixture.items():
        note_service.add_note(
            content=note_data['content'],
            tags=["test_data"], # 提供一个非空标签以满足验证
            metadata=note_data['metadata'],
            note_id=note_id
        )

    yield note_service

    import shutil
    shutil.rmtree(db_path, ignore_errors=True)

@pytest.fixture
def mock_llm_provider():
    """
    创建一个模拟的LLM Provider，其text_chat方法是可配置的。
    """
    mock_provider = MagicMock()
    mock_provider.text_chat = AsyncMock()
    return mock_provider

@pytest.fixture
def deepmind_instance(note_service_with_data, mock_llm_provider):
    """
    创建一个用于测试的DeepMind实例。
    """
    config = MemoryConfig(config={
        "provider_id": "mock_provider",
        "sleep_interval": 0  # 禁用睡眠定时器
    })

    mock_context = MagicMock()
    mock_context.get_provider_by_id.return_value = mock_llm_provider

    deepmind = DeepMind(config, mock_context, note_service_with_data.vector_store, provider_id="mock_provider")

    # 确保 memory_system 被正确初始化
    assert deepmind.memory_system is not None
    deepmind.memory_system.note_service = note_service_with_data

    return deepmind

# --- Integration Test for DeepMind ---

@pytest.mark.asyncio
async def test_organize_and_inject_memories_full_flow(
    deepmind_instance: DeepMind,
    mock_llm_provider: MagicMock,
    linked_notes_fixture
):
    """
    **集成测试: 验证完整的笔记增强流程**

    - **场景**: 模拟一个发往主模型的请求被 DeepMind 拦截。
    - **行为**: DeepMind 调用小模型，小模型选择了一个相关的笔记片段。
    - **验证**: DeepMind 能够基于小模型的选择，将完整的笔记内容扩展并正确注入到原始请求的 `system_prompt` 中。
    """
    # 1. Arrange

    # 模拟小模型返回的JSON，它选择了我们链式笔记的中间部分 ("block_3")
    # 我们需要一个真实的短ID，所以从 MemoryIDResolver 生成它
    from core.utils.memory_id_resolver import MemoryIDResolver
    short_id_for_block_3 = MemoryIDResolver.generate_short_id("block_3")

    mock_llm_response_json = {
        "useful_notes": [short_id_for_block_3],
        "feedback_data": {
            "thoughts": "用户在询问相关概念，笔记block_3看起来最相关。",
            "useful_memory_ids": [],
            "merge_groups": [],
            "new_memories": {}
        }
    }
    mock_llm_provider.text_chat.return_value.completion_text = json.dumps(mock_llm_response_json)

    # 模拟一个合法的 AstrMessageEvent
    mock_platform_meta = MagicMock()
    mock_platform_meta.id = "test_platform"
    mock_message_obj = MagicMock()
    class MockMessageType:
        value = "private"
    mock_message_obj.type = MockMessageType()
    event = AstrMessageEvent(
        message_str="test message",
        message_obj=mock_message_obj,
        platform_meta=mock_platform_meta,
        session_id="test_session"
    )
    event.angelmemory_context = json.dumps({
        "session_id": "test_session",
        "recall_query": "关于中间部分的一些问题"
    })

    # 模拟一个即将发往主模型的 ProviderRequest
    # 它的 system_prompt 是我们关心的目标
    request = ProviderRequest(model="any_main_model", prompt="tell me more")
    request.system_prompt = "Original system prompt."

    # 2. Act
    # 调用被测函数，它应该会修改 request 对象
    await deepmind_instance.organize_and_inject_memories(event, request)

    # 3. Assert

    # 验证小模型确实被调用了
    mock_llm_provider.text_chat.assert_called_once()

    # **核心断言**: 验证 request.system_prompt 是否被正确增强
    final_prompt = request.system_prompt

    # 检查原始提示词是否保留
    assert "Original system prompt." in final_prompt
    # 检查笔记上下文的标题是否被添加
    assert "相关笔记上下文：" in final_prompt

    # 验证所有5个块的内容都按顺序出现在最终的提示中
    content_parts = [
        "这是第一部分。", "这是第二部分。", "这是中间部分。", "这是第四部分。", "这是最后一部分。"
    ]
    for part in content_parts:
        assert part in final_prompt

    # 验证它们的相对顺序是否正确
    positions = [final_prompt.find(part) for part in content_parts]
    assert all(positions[i] < positions[i+1] for i in range(len(positions)-1))
