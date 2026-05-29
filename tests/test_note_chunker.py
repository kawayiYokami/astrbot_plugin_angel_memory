"""笔记切片器单元测试"""

import pytest

from llm_memory.parser.note_chunker import (
    chunk_file,
    _identify_special_ranges,
    _split_by_paragraphs,
    MAX_CHUNK_CHARS,
)


class TestChunkFileBasic:
    """基础切片功能"""

    def test_empty_content_returns_path_only(self):
        """空文件只产出路径行"""
        chunks = chunk_file("", "notes/empty.md")
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 1
        assert chunks[0]["line_start"] == 0
        assert chunks[0]["line_end"] == 0
        assert chunks[0]["content"] == "notes/empty.md"

    def test_whitespace_only_returns_path_only(self):
        """纯空白文件只产出路径行"""
        chunks = chunk_file("   \n\n  \n", "notes/blank.md")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "notes/blank.md"

    def test_single_line_content(self):
        """单行内容：路径行 + 内容"""
        chunks = chunk_file("Hello world", "test.md")
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 1
        assert chunks[0]["line_start"] == 0  # 包含路径行
        assert chunks[0]["line_end"] == 1
        assert "test.md" in chunks[0]["content"]
        assert "Hello world" in chunks[0]["content"]

    def test_path_is_line_zero(self):
        """第一个切片的 line_start 为 0（路径虚拟行）"""
        chunks = chunk_file("Line 1\nLine 2", "docs/readme.md")
        assert chunks[0]["line_start"] == 0
        # 内容第一行是路径
        lines = chunks[0]["content"].split("\n")
        assert lines[0] == "docs/readme.md"

    def test_multiple_paragraphs_split(self):
        """多段落按空行切分"""
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_file(content, "test.md")
        # 应该产出 3 个切片（每段一个）
        assert len(chunks) == 3
        # 第一个切片包含路径行
        assert "test.md" in chunks[0]["content"]
        assert "Paragraph one." in chunks[0]["content"]
        # 后续切片不含路径行
        assert chunks[1]["content"] == "Paragraph two."
        assert chunks[2]["content"] == "Paragraph three."

    def test_chunk_index_sequential(self):
        """切片序号从 1 开始递增"""
        content = "A\n\nB\n\nC"
        chunks = chunk_file(content, "test.md")
        for i, chunk in enumerate(chunks, start=1):
            assert chunk["chunk_index"] == i


class TestChunkFileSpecialBlocks:
    """代码块和表格保持完整"""

    def test_code_block_not_split(self):
        """代码块不拆分"""
        content = "Before code.\n\n```python\ndef foo():\n    pass\n```\n\nAfter code."
        chunks = chunk_file(content, "test.md")
        # 应该有 3 个切片：前文、代码块、后文
        assert len(chunks) == 3
        # 代码块完整保留
        code_chunk = chunks[1]
        assert "```python" in code_chunk["content"]
        assert "def foo():" in code_chunk["content"]
        assert "```" in code_chunk["content"]

    def test_table_not_split(self):
        """表格不拆分"""
        content = "Before table.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nAfter table."
        chunks = chunk_file(content, "test.md")
        assert len(chunks) == 3
        table_chunk = chunks[1]
        assert "| A | B |" in table_chunk["content"]
        assert "| 1 | 2 |" in table_chunk["content"]


class TestChunkFileLongContent:
    """超长段落强制截断"""

    def test_long_paragraph_multiline_split_by_char_limit(self):
        """多行超过 max_chunk_chars 的段落被按行截断"""
        # 创建多行内容，每行约 20 字符，总计远超 100
        lines = [f"这是第{i}行的内容文字。" for i in range(30)]
        content = "\n".join(lines)  # 无空行，算一个段落
        chunks = chunk_file(content, "test.md", max_chunk_chars=100)
        # 应该被拆成多个切片
        assert len(chunks) > 1

    def test_single_long_line_stays_intact(self):
        """单行超长文本不强制拆分（与 Story 项目行为一致）"""
        long_line = "这是一段很长的文字。" * 50
        content = long_line
        chunks = chunk_file(content, "test.md", max_chunk_chars=100)
        # 单行不拆分，保持完整
        assert len(chunks) == 1
        assert long_line in chunks[0]["content"]


class TestChunkFileLineNumbers:
    """行号定位正确性"""

    def test_line_numbers_correct(self):
        """行号与实际文件行对应"""
        content = "Line 1\nLine 2\nLine 3\n\nLine 5\nLine 6"
        chunks = chunk_file(content, "test.md")
        # 第一个切片：路径行(0) + Line1-3
        assert chunks[0]["line_start"] == 0
        assert chunks[0]["line_end"] == 3
        # 第二个切片：Line5-6
        assert chunks[1]["line_start"] == 5
        assert chunks[1]["line_end"] == 6

    def test_code_block_line_numbers(self):
        """代码块行号正确"""
        content = "Intro\n\n```\ncode line 1\ncode line 2\n```\n\nOutro"
        chunks = chunk_file(content, "test.md")
        # Intro: line 1, code block: lines 3-6, Outro: line 8
        assert chunks[0]["line_end"] == 1  # Intro
        code_chunk = chunks[1]
        assert code_chunk["line_start"] == 3
        assert code_chunk["line_end"] == 6


class TestChunkStore:
    """切片存储集成测试"""

    def test_upsert_and_query(self, tmp_path):
        """写入切片后可以查询"""
        from llm_memory.components.note_chunk_store import NoteChunkStore

        db_path = tmp_path / "test_chunks.db"
        store = NoteChunkStore(str(db_path))

        chunks = [
            {"chunk_index": 1, "line_start": 0, "line_end": 3, "content": "path\nline1\nline2\nline3"},
            {"chunk_index": 2, "line_start": 4, "line_end": 6, "content": "line4\nline5\nline6"},
        ]
        count = store.upsert_chunks("file_001", "notes/test.md", chunks)
        assert count == 2

        # 按路径查询
        result = store.get_chunks_by_path("notes/test.md")
        assert len(result) == 2
        assert result[0]["chunk_index"] == 1
        assert result[1]["chunk_index"] == 2
        assert result[0]["content"] == "path\nline1\nline2\nline3"

        # 按 file_id 查询
        result2 = store.get_chunks_by_file_id("file_001")
        assert len(result2) == 2

        store.close()

    def test_upsert_replaces_old_data(self, tmp_path):
        """重复写入同一 file_id 会替换旧数据"""
        from llm_memory.components.note_chunk_store import NoteChunkStore

        db_path = tmp_path / "test_chunks.db"
        store = NoteChunkStore(str(db_path))

        chunks_v1 = [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "old content"},
        ]
        store.upsert_chunks("file_001", "test.md", chunks_v1)

        chunks_v2 = [
            {"chunk_index": 1, "line_start": 0, "line_end": 2, "content": "new content"},
            {"chunk_index": 2, "line_start": 3, "line_end": 4, "content": "extra"},
        ]
        store.upsert_chunks("file_001", "test.md", chunks_v2)

        result = store.get_chunks_by_file_id("file_001")
        assert len(result) == 2
        assert result[0]["content"] == "new content"

        store.close()

    def test_delete_by_file_id(self, tmp_path):
        """按 file_id 删除"""
        from llm_memory.components.note_chunk_store import NoteChunkStore

        db_path = tmp_path / "test_chunks.db"
        store = NoteChunkStore(str(db_path))

        store.upsert_chunks("file_001", "a.md", [{"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "a"}])
        store.upsert_chunks("file_002", "b.md", [{"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "b"}])

        deleted = store.delete_by_file_id("file_001")
        assert deleted == 1

        assert store.get_chunks_by_file_id("file_001") == []
        assert len(store.get_chunks_by_file_id("file_002")) == 1

        store.close()

    def test_stats(self, tmp_path):
        """统计信息正确"""
        from llm_memory.components.note_chunk_store import NoteChunkStore

        db_path = tmp_path / "test_chunks.db"
        store = NoteChunkStore(str(db_path))

        store.upsert_chunks("f1", "a.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "a1"},
            {"chunk_index": 2, "line_start": 2, "line_end": 3, "content": "a2"},
        ])
        store.upsert_chunks("f2", "b.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "b1"},
        ])

        stats = store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["total_files"] == 2

        store.close()
