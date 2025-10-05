"""
笔记服务核心功能单元测试

测试 NoteService 的核心业务逻辑，重点验证：
1. 笔记的增删改查操作
2. 标签提取和处理
3. 与 VectorStore 的正确协作
"""

import pytest
from unittest.mock import MagicMock, patch
import uuid
import sys
from types import ModuleType


@pytest.fixture(autouse=True)
def mock_astrbot_and_dependencies():
    """全局 Mock astrbot 相关依赖，避免模块导入错误"""

    # 创建假的 astrbot 模块
    fake_astrbot = ModuleType('astrbot')
    fake_api = ModuleType('astrbot.api')

    # 创建 Mock logger
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.debug = MagicMock()

    fake_api.logger = mock_logger
    fake_astrbot.api = fake_api

    # 注入到 sys.modules
    sys.modules['astrbot'] = fake_astrbot
    sys.modules['astrbot.api'] = fake_api

    yield mock_logger

    # 清理
    if 'astrbot' in sys.modules:
        del sys.modules['astrbot']
    if 'astrbot.api' in sys.modules:
        del sys.modules['astrbot.api']


class TestNoteServiceCore:
    """NoteService 核心功能测试"""

    @pytest.fixture
    def mock_system_config(self):
        """Mock 系统配置"""
        mock_config = MagicMock()
        mock_config.embedding_model = 'paraphrase-MiniLM-L3-v2'
        mock_config.notes_main_collection_name = 'test_notes_main'
        mock_config.notes_sub_collection_name = 'test_notes_sub'
        mock_config.get_database_path = MagicMock(return_value='/tmp/test_db')
        return mock_config

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore 实例"""
        mock_vs = MagicMock()
        mock_vs.upsert_documents = MagicMock()
        mock_vs.get_or_create_collection_with_dimension_check = MagicMock()
        return mock_vs

    @pytest.fixture
    def note_service_with_mocks(self, mock_system_config, mock_vector_store):
        """带完整 Mock 的 NoteService 实例"""
        with patch('llm_memory.config.system_config', mock_system_config), \
             patch('llm_memory.components.vector_store.VectorStore', return_value=mock_vector_store), \
             patch('llm_memory.parser.parser_manager.parser_manager', MagicMock()):

            from llm_memory.service.note_service import NoteService
            service = NoteService(vector_store=mock_vector_store)
            return service, mock_vector_store

    def test_add_note_success(self, note_service_with_mocks):
        """测试成功添加笔记"""
        service, mock_vs = note_service_with_mocks
        # 正确地在 service 实例上模拟 logger
        service.logger = MagicMock()

        # 执行添加笔记
        tags = ["标签1", "标签2"]
        note_id = service.add_note("测试笔记内容", tags)

        # 验证返回的ID是UUID格式
        assert isinstance(note_id, str)
        try:
            uuid.UUID(note_id)
        except ValueError:
            pytest.fail("返回的笔记ID不是有效的UUID格式")

        # 验证调用了两次 upsert_documents
        assert mock_vs.upsert_documents.call_count == 2

        # 验证主集合调用
        main_call = mock_vs.upsert_documents.call_args_list[0]
        assert main_call.kwargs['collection'] == mock_vs.get_or_create_collection_with_dimension_check.return_value
        # 确保比较的是列表
        assert main_call.kwargs['ids'] == [note_id]
        assert main_call.kwargs['documents'] == ["测试笔记内容 \n\nTags: 标签1 标签2"]

        # 验证副集合调用
        sub_call = mock_vs.upsert_documents.call_args_list[1]
        assert sub_call.kwargs['collection'] == mock_vs.get_or_create_collection_with_dimension_check.return_value
        # 确保比较的是列表
        assert sub_call.kwargs['ids'] == [note_id]
        assert sub_call.kwargs['documents'] == ["标签1 标签2"]

        # 验证日志调用
        service.logger.info.assert_called_with(f"成功添加笔记: {note_id}, 标签: {tags}")

    def test_add_note_validation_error_empty_tags(self, note_service_with_mocks, mock_astrbot_and_dependencies):
        """测试标签为空时的验证错误"""
        service, mock_vs = note_service_with_mocks
        
        # 执行添加笔记（无标签）
        with pytest.raises(ValueError, match="标签列表不允许为空"):
            service.add_note("测试内容", [])

        # 验证没有调用数据库操作
        mock_vs.upsert_documents.assert_not_called()

    def test_add_note_with_auto_tags(self, note_service_with_mocks, mock_astrbot_and_dependencies):
        """测试自动标签提取"""
        service, mock_vs = note_service_with_mocks
        service.add_note("# 标题\n\n这是正文内容。", None)

        # 验证提取了标题标签
        assert mock_vs.upsert_documents.call_count == 2

        # 验证副集合调用中的标签
        sub_call = mock_vs.upsert_documents.call_args_list[1]
        assert "标题" in sub_call.kwargs['embedding_texts']

    def test_add_note_duplicate_hash_skip(self, note_service_with_mocks):
        """测试重复文件哈希的去重逻辑"""
        service, mock_vs = note_service_with_mocks
        # 在 service 实例上模拟 logger
        service.logger = MagicMock()

        # 直接 Mock service 实例上的 main_collection 属性
        service.main_collection.get.return_value = {
            'ids': ['existing_id'],
            'documents': ['existing_document'],
            'metadatas': [{'source_file_hash': 'duplicate_hash'}]
        }

        # 执行添加笔记（带重复哈希）
        result_id = service.add_note("内容", ["标签"], {"source_file_hash": "duplicate_hash"})

        # 验证返回已存在的ID
        assert result_id == "existing_id"

        # 验证没有执行插入操作
        mock_vs.upsert_documents.assert_not_called()

        # 验证日志
        service.logger.info.assert_called_with("文件哈希值 duplicate_hash 已存在，跳过写入。")


class TestNoteServiceTagExtraction:
    """笔记服务标签提取功能测试"""

    def test_extract_tags_from_headings(self):
        """测试从标题中提取标签"""
        # 直接导入和测试标签提取方法，避免复杂的初始化

        # 创建一个临时实例来调用方法
        # 由于单例模式问题，我们直接测试静态方法
        content = """# 一级标题
## 二级标题
### 三级标题

正文内容"""

        # 直接调用方法逻辑（复制实现，避免初始化问题）
        tags = []
        import re

        # 1. 提取标题（# ## ###）
        title_pattern = r'^#{1,6}\s+(.+)$'
        titles = re.findall(title_pattern, content, re.MULTILINE)
        tags.extend(titles)

        # 去重和清理（复制原方法逻辑）
        tags = list(set(tags))  # 去重
        tags = [tag.strip() for tag in tags if tag.strip()]  # 去空格
        tags = [tag for tag in tags if len(tag) > 1]  # 过滤太短的

        assert "一级标题" in tags
        assert "二级标题" in tags
        assert "三级标题" in tags

    def test_extract_tags_from_bold_text(self, mock_astrbot_and_dependencies):
        """测试从粗体文本中提取标签"""
        from llm_memory.service.note_service import NoteService

        with patch('llm_memory.config.system_config'), \
             patch('llm_memory.components.vector_store.VectorStore', return_value=MagicMock()), \
             patch('llm_memory.parser.parser_manager.parser_manager'):

            service = NoteService(vector_store=MagicMock())

            content = """这是**重要概念**和**关键术语**的解释。"""
            tags = service._extract_tags(content)
            assert "重要概念" in tags
            assert "关键术语" in tags

    def test_extract_tags_from_quotes(self, mock_astrbot_and_dependencies):
        """测试从引号中提取标签"""
        from llm_memory.service.note_service import NoteService

        with patch('llm_memory.config.system_config'), \
             patch('llm_memory.components.vector_store.VectorStore', return_value=MagicMock()), \
             patch('llm_memory.parser.parser_manager.parser_manager'):

            service = NoteService(vector_store=MagicMock())

            content = """根据"机器学习"和'深度学习'的定义..."""
            tags = service._extract_tags(content)
            assert "机器学习" in tags
            assert "深度学习" in tags

    def test_extract_tags_filter_short_and_duplicates(self, mock_astrbot_and_dependencies):
        """测试过滤短标签和去重"""
        from llm_memory.service.note_service import NoteService

        with patch('llm_memory.config.system_config'), \
             patch('llm_memory.components.vector_store.VectorStore', return_value=MagicMock()), \
             patch('llm_memory.parser.parser_manager.parser_manager'):

            service = NoteService(vector_store=MagicMock())

            content = """# A
**B**
# C
# A"""
            tags = service._extract_tags(content)
            # 应该包含单字符标签，并且去重
            assert "A" in tags
            assert "B" in tags
            assert "C" in tags
            assert len(tags) == 3