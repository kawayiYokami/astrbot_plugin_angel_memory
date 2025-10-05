# tests/test_unit_note_context_builder.py

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from core.utils.note_context_builder import NoteContextBuilder
from llm_memory.utils.token_utils import count_tokens

# 确保测试可以找到 llm_memory 和 astrbot 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# 假设 astrbot 在项目根目录的上一级的上一级的上一级
# e.g., from e:/github/ai-qq/astrbot/data/plugins/astrbot_plugin_angel_memory
# to e:/github/ai-qq/
sys.path.insert(0, str(project_root.parent.parent.parent))

# --- Test Data Fixtures ---

@pytest.fixture(scope="module")
def linked_notes_fixture():
    """
    创建一个由5个块组成的、互相链接的虚拟文档，用于测试。
    返回一个以块ID为键，块字典为值的字典。
    """
    source_path = "dummy/doc.md"
    notes = {}

    # 定义内容
    contents = {
        "block_1": "这是第一部分。",
        "block_2": "这是第二部分。",
        "block_3": "这是中间部分。",
        "block_4": "这是第四部分。",
        "block_5": "这是最后一部分。"
    }

    # 定义链接关系
    links = {
        "block_1": [None, "block_2"],
        "block_2": ["block_1", "block_3"],
        "block_3": ["block_2", "block_4"],
        "block_4": ["block_3", "block_5"],
        "block_5": ["block_4", None]
    }

    for i in range(1, 6):
        block_id = f"block_{i}"
        prev_id, next_id = links[block_id]

        # 将链接格式化为 'prev,next' 字符串，以匹配数据库中的存储格式
        related_ids_str = f"{prev_id or ''},{next_id or ''}"

        notes[block_id] = {
            "id": block_id,
            "content": contents[block_id],
            "metadata": {
                "source_file_path": source_path,
                "related_block_ids": related_ids_str
            }
        }
    return notes

@pytest.fixture
def mock_note_service(linked_notes_fixture):
    """
    创建一个模拟的 NoteService，其 get_note 方法会从测试数据中返回笔记。
    """
    mock_service = MagicMock()

    # 配置 get_note 的行为
    def get_note_side_effect(note_id):
        return linked_notes_fixture.get(note_id)

    mock_service.get_note.side_effect = get_note_side_effect
    return mock_service

# --- Unit Tests for NoteContextBuilder ---

class TestNoteContextBuilder:
    """NoteContextBuilder 的单元测试"""

    def test_expand_from_middle_full_context(self, mock_note_service, linked_notes_fixture):
        """测试: 从中间块开始，无token限制，应返回完整上下文"""
        # Arrange
        start_id = "block_3"
        max_tokens = 1000 # 足够大的token

        # Act
        expanded_blocks = NoteContextBuilder._expand_bidirectional(
            start_id, mock_note_service, max_tokens
        )

        # Assert
        assert len(expanded_blocks) == 5
        expected_order = [
            linked_notes_fixture["block_1"]["content"],
            linked_notes_fixture["block_2"]["content"],
            linked_notes_fixture["block_3"]["content"],
            linked_notes_fixture["block_4"]["content"],
            linked_notes_fixture["block_5"]["content"],
        ]
        assert expanded_blocks == expected_order

    def test_expand_from_start_of_chain(self, mock_note_service, linked_notes_fixture):
        """测试: 从链头开始，应只向后扩展"""
        # Arrange
        start_id = "block_1"
        max_tokens = 1000

        # Act
        expanded_blocks = NoteContextBuilder._expand_bidirectional(
            start_id, mock_note_service, max_tokens
        )

        # Assert
        assert len(expanded_blocks) == 5 # 依然会扩展完整
        assert expanded_blocks[0] == linked_notes_fixture["block_1"]["content"]

    def test_expand_from_end_of_chain(self, mock_note_service, linked_notes_fixture):
        """测试: 从链尾开始，应只向前扩展"""
        # Arrange
        start_id = "block_5"
        max_tokens = 1000

        # Act
        expanded_blocks = NoteContextBuilder._expand_bidirectional(
            start_id, mock_note_service, max_tokens
        )

        # Assert
        assert len(expanded_blocks) == 5 # 依然会扩展完整
        assert expanded_blocks[4] == linked_notes_fixture["block_5"]["content"]

    def test_expand_with_token_limit(self, mock_note_service, linked_notes_fixture):
        """测试: 存在token限制时，应在达到限制时优雅停止"""
        # Arrange
        start_id = "block_3"
        # 计算一个只能容纳3个块的token预算
        # (内容 + 分隔符) * 3
        token_for_3_blocks = count_tokens(
            "\n\n".join([
                linked_notes_fixture["block_2"]["content"],
                linked_notes_fixture["block_3"]["content"],
                linked_notes_fixture["block_4"]["content"],
            ])
        )
        max_tokens = token_for_3_blocks + 1 # 给予少量余量

        # Act
        expanded_blocks = NoteContextBuilder._expand_bidirectional(
            start_id, mock_note_service, max_tokens
        )

        # Assert
        # 预期结果是中心块、前一个块、后一个块
        assert len(expanded_blocks) == 3
        expected_order = [
            linked_notes_fixture["block_2"]["content"],
            linked_notes_fixture["block_3"]["content"],
            linked_notes_fixture["block_4"]["content"],
        ]
        assert expanded_blocks == expected_order

    def test_public_expand_from_ids_with_mapping(self, mock_note_service, linked_notes_fixture):
        """测试: 公共接口能否正确使用ID映射并拼接上下文"""
        # Arrange
        short_ids = ["short_3", "short_5"]
        id_mapping = {
            "short_3": "block_3",
            "short_5": "block_5"
        }
        total_token_budget = 2000

        # Act
        full_context = NoteContextBuilder.expand_context_from_note_ids(
            short_ids, mock_note_service, total_token_budget, id_mapping
        )

        # Assert
        # 验证两个扩展后的上下文都被包含，并由分隔符隔开
        full_doc_content = "\n\n".join([
            linked_notes_fixture[f"block_{i}"]["content"] for i in range(1, 6)
        ])
        assert full_doc_content in full_context
        assert "\n\n---\n\n" in full_context

    def test_build_candidate_list_for_prompt(self):
        """测试: 候选列表的格式化是否正确"""
        # Arrange
        notes = [
            {"id": "uuid-1", "content": "这是第一个笔记"},
            {"id": "uuid-2", "content": "这是第二个笔记，内容非常长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长-f.g"}
        ]

        # Act
        result = NoteContextBuilder.build_candidate_list_for_prompt(notes)

        # Assert
        assert "[ID: uuid-1]" in result
        assert "这是第一个笔记" in result
        assert "[ID: uuid-2]" in result
        assert "..." in result # 验证内容被截断
        assert len(result.split('\n')) == 4 # 标题行 + 2个笔记行 + 1个空行