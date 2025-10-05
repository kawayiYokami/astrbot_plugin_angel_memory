"""
NoteService 集成测试 (重构后)

测试 NoteService 与一个独立的、通过依赖注入传入的 VectorStore 实例的协同工作能力。
"""

import sys
from pathlib import Path
import pytest
import shutil
from llm_memory.components.vector_store import VectorStore
from llm_memory.service.note_service import NoteService, NoteNotFoundError

# 确保测试可以找到 llm_memory 和 astrbot 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# 假设 astrbot 在项目根目录的上一级的上一级的上一级
# e.g., from e:/github/ai-qq/astrbot/data/plugins/astrbot_plugin_angel_memory
# to e:/github/ai-qq/
sys.path.insert(0, str(project_root.parent.parent.parent))

class TestNoteServiceIntegration:
    """集成测试，验证服务层到数据库的完整流程。"""

    @pytest.fixture(scope="function")
    def temp_db_path(self, tmp_path: Path) -> Path:
        """创建一个临时的数据库目录，并在测试结束后清理。"""
        db_path = tmp_path / "test_chroma_db"
        db_path.mkdir()
        yield db_path
        # 使用 ignore_errors=True 增加清理的稳健性
        shutil.rmtree(db_path, ignore_errors=True)

    @pytest.fixture(scope="function")
    def note_service_instance(self, temp_db_path: Path):
        """
        创建一个使用独立 VectorStore 的 NoteService 实例。
        """
        # 1. 创建一个全新的、隔离的 VectorStore 实例
        vector_store = VectorStore(db_path=str(temp_db_path))

        # 2. 将 VectorStore 实例注入到 NoteService
        note_service = NoteService(vector_store=vector_store)

        yield note_service

    def test_add_and_get_note(self, note_service_instance):
        """测试添加一个笔记，然后通过 get_note 成功检索。"""
        # 1. Arrange
        note_content = "这是一个关于AI的集成测试笔记。"
        note_tags = ["AI", "Testing"]
        note_metadata = {"source": "integration_test"}

        # 2. Act
        note_id = note_service_instance.add_note(
            content=note_content,
            tags=note_tags,
            metadata=note_metadata
        )
        retrieved_note = note_service_instance.get_note(note_id)

        # 3. Assert
        assert retrieved_note is not None
        assert retrieved_note['id'] == note_id
        # 验证 get_note 返回的是存储在数据库中的“融合内容”
        expected_stored_content = note_content + " \n\nTags: " + " ".join(note_tags)
        assert retrieved_note['content'] == expected_stored_content
        # 比较集合以忽略顺序问题
        assert set(retrieved_note['tags']) == set(note_tags)
        assert retrieved_note['metadata']['source'] == "integration_test"

    def test_update_note_content(self, note_service_instance):
        """测试更新笔记的内容和标签。"""
        # 1. Arrange: 添加一个初始笔记
        note_id = note_service_instance.add_note(
            content="这是关于Python的初始笔记。",
            tags=["Python"],
            metadata={"filepath": "dummy/python.md"}
        )

        # 2. Act: 更新笔记
        updated_content = "这是更新后的Python笔记，增加了关于测试的内容。"
        updated_tags = ["Python", "Pytest"]
        note_service_instance.update_note(note_id, content=updated_content, tags=updated_tags)

        # 3. Assert: 验证更新是否成功
        retrieved_note = note_service_instance.get_note(note_id)
        expected_stored_content = updated_content + " \n\nTags: " + " ".join(updated_tags)
        assert retrieved_note['content'] == expected_stored_content
        assert set(retrieved_note['tags']) == set(updated_tags)
        assert retrieved_note['metadata']['updated'] is True

    def test_delete_note(self, note_service_instance):
        """测试删除一个笔记后，数据库中不再存在该记录。"""
        # 1. Arrange: 添加一个笔记
        note_id = note_service_instance.add_note(
            content="这是一个即将被删除的笔记。",
            tags=["ephemeral"],
            metadata={"filepath": "dummy/to_delete.md"}
        )
        assert note_service_instance.main_collection.count() == 1

        # 2. Act: 删除笔记
        delete_result = note_service_instance.delete_note(note_id)

        # 3. Assert: 验证删除结果和数据库状态
        assert delete_result is True
        assert note_service_instance.main_collection.count() == 0
        with pytest.raises(NoteNotFoundError):
            note_service_instance.get_note(note_id)

    def test_search_notes_hybrid_retrieval(self, note_service_instance):
        """测试混合搜索能否准确召回并排序。"""
        # 1. Arrange: 添加多个纯中文笔记
        note_id_poem = note_service_instance.add_note(
            content="床前明月光，疑是地上霜。",
            tags=["唐诗", "李白"],
            metadata={"filepath": "dummy/tang_poem.md"}
        )
        note_service_instance.add_note(
            content="红烧肉是一道经典的中国菜。",
            tags=["菜谱", "猪肉"],
            metadata={"filepath": "dummy/recipe.md"}
        )
        note_service_instance.add_note(
            content="机器学习是人工智能的一个分支。",
            tags=["AI", "计算机科学"],
            metadata={"filepath": "dummy/ai.md"}
        )
        # 等待数据写入
        import time
        time.sleep(2)
        assert note_service_instance.main_collection.count() == 3

        # 2. Act: 搜索与 "古诗" 相关的笔记
        search_results = note_service_instance.search_notes("关于李白的古诗")

        # 3. Assert: 验证搜索结果
        assert len(search_results) == 1
        assert search_results[0]['id'] == note_id_poem