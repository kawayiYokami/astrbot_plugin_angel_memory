"""笔记切片搜索引擎测试"""

import pytest

from llm_memory.components.note_chunk_search import (
    NoteChunkSearchEngine,
    normalize_search_text,
    iter_cjk_fragments,
    select_required_cjk_ngrams,
    tokenize_for_index,
    _build_all_terms_query,
    _build_any_terms_query,
    _resolve_hit_limit,
)


class TestTokenizer:
    """CJK bigram 分词测试"""

    def test_normalize_fullwidth(self):
        assert normalize_search_text("Ｈｅｌｌｏ") == "hello"

    def test_normalize_chinese_quotes(self):
        # 引号被统一为半角后，首尾引号被去除
        assert normalize_search_text("\u201c测试\u201d") == "测试"

    def test_cjk_bigram(self):
        fragments = iter_cjk_fragments("红烧肉")
        # 应包含 unigram: 红, 烧, 肉 和 bigram: 红烧, 烧肉
        assert "红" in fragments
        assert "烧" in fragments
        assert "肉" in fragments
        assert "红烧" in fragments
        assert "烧肉" in fragments

    def test_cjk_bigram_with_english(self):
        fragments = iter_cjk_fragments("Python编程")
        assert "python" in fragments
        assert "编程" in fragments
        assert "编" in fragments
        assert "程" in fragments

    def test_select_required_short_word(self):
        """短词（≤2字）保留原样"""
        terms = select_required_cjk_ngrams("猫")
        assert "猫" in terms

    def test_select_required_long_word(self):
        """长词拆成 bigram"""
        terms = select_required_cjk_ngrams("红烧肉")
        assert "红烧" in terms
        assert "烧肉" in terms

    def test_tokenize_for_index(self):
        result = tokenize_for_index("Hello 世界")
        assert "hello" in result
        assert "世界" in result
        assert "世" in result
        assert "界" in result


class TestQueryBuilding:
    """查询构建测试"""

    def test_all_terms_query(self):
        q = _build_all_terms_query(["红烧", "烧肉"])
        assert "+红烧" in q
        assert "+烧肉" in q

    def test_any_terms_query(self):
        q = _build_any_terms_query(["红烧", "烧肉"])
        assert "红烧" in q
        assert "烧肉" in q
        assert "+" not in q

    def test_resolve_hit_limit_single_char(self):
        """单字查询放大 hit_limit"""
        limit = _resolve_hit_limit("猫", 50)
        assert limit >= 800

    def test_resolve_hit_limit_short(self):
        """短查询放大"""
        limit = _resolve_hit_limit("红烧", 50)
        assert limit >= 240

    def test_resolve_hit_limit_normal(self):
        """正常查询不放大"""
        limit = _resolve_hit_limit("这是一个正常的查询", 50)
        assert limit == 50


try:
    import tantivy
    HAS_TANTIVY = True
except Exception:
    HAS_TANTIVY = False


@pytest.mark.skipif(not HAS_TANTIVY, reason="tantivy not installed")
class TestSearchEngine:
    """搜索引擎集成测试"""

    def test_index_and_search(self, tmp_path):
        """索引切片后可以搜索到"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        chunks = [
            {"chunk_index": 1, "line_start": 0, "line_end": 3, "content": "notes/cooking/红烧肉做法.md\n# 红烧肉做法\n材料：五花肉、酱油、冰糖"},
            {"chunk_index": 2, "line_start": 4, "line_end": 6, "content": "步骤：先焯水，再炒糖色，最后炖煮"},
        ]
        engine.index_chunks("file_001", "notes/cooking/红烧肉做法.md", chunks)

        results = engine.search("红烧肉", limit=10)
        assert len(results) > 0
        assert any("红烧肉" in r.get("content", "") for r in results)

    def test_search_by_path(self, tmp_path):
        """可以通过路径关键词搜索"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        chunks = [
            {"chunk_index": 1, "line_start": 0, "line_end": 2, "content": "notes/travel/日本旅行.md\n# 东京攻略\n新宿是购物天堂"},
        ]
        engine.index_chunks("file_002", "notes/travel/日本旅行.md", chunks)

        results = engine.search("旅行", limit=10)
        assert len(results) > 0

    def test_multi_file_search(self, tmp_path):
        """多文件搜索，每文件限制切片数"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        # 文件1：3个切片
        for i in range(1, 4):
            engine.index_chunks(
                "file_a", "a.md",
                [{"chunk_index": i, "line_start": i, "line_end": i, "content": f"猫咪的故事第{i}章"}],
            )
        # 文件2：3个切片
        for i in range(1, 4):
            engine.index_chunks(
                "file_b", "b.md",
                [{"chunk_index": i, "line_start": i, "line_end": i, "content": f"猫咪的冒险第{i}章"}],
            )

        results = engine.search("猫咪", limit=10, max_chunks_per_file=2)
        # 每个文件最多2个切片
        file_a_count = sum(1 for r in results if r["file_id"] == "file_a")
        file_b_count = sum(1 for r in results if r["file_id"] == "file_b")
        assert file_a_count <= 2
        assert file_b_count <= 2

    def test_multi_term_search_caps_chunks_per_file_after_merge(self, tmp_path):
        """多词合并后仍按文件限制切片数"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        engine.index_chunks("file_a", "a.md", [
            {"chunk_index": 1, "line_start": 1, "line_end": 1, "content": "alpha only"},
            {"chunk_index": 2, "line_start": 2, "line_end": 2, "content": "beta only"},
        ])
        engine.index_chunks("file_b", "b.md", [
            {"chunk_index": 1, "line_start": 1, "line_end": 1, "content": "alpha beta together"},
        ])

        results = engine.search("alpha beta", limit=10, max_chunks_per_file=1)

        per_file = {}
        for item in results:
            per_file[item["file_id"]] = per_file.get(item["file_id"], 0) + 1
        assert per_file["file_a"] == 1
        assert per_file["file_b"] == 1

    def test_delete_removes_from_index(self, tmp_path):
        """删除后搜索不到"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        chunks = [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "独特内容：量子纠缠"},
        ]
        engine.index_chunks("file_x", "x.md", chunks)

        results_before = engine.search("量子纠缠", limit=10)
        assert len(results_before) > 0

        engine.delete_by_file_id("file_x")
        results_after = engine.search("量子纠缠", limit=10)
        assert len(results_after) == 0

    def test_rebuild_all(self, tmp_path):
        """全量重建"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        # 先写入一些数据
        engine.index_chunks("old", "old.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "旧数据"},
        ])

        # 全量重建
        all_chunks = [
            {"file_id": "new1", "source_file_path": "new1.md", "chunk_index": 1, "line_start": 0, "line_end": 1, "content": "新数据一"},
            {"file_id": "new2", "source_file_path": "new2.md", "chunk_index": 1, "line_start": 0, "line_end": 1, "content": "新数据二"},
        ]
        count = engine.rebuild_all(all_chunks)
        assert count == 2

        # 旧数据搜不到
        assert engine.search("旧数据", limit=10) == []
        # 新数据能搜到
        assert len(engine.search("新数据", limit=10)) > 0

    def test_empty_query_returns_empty(self, tmp_path):
        """空查询返回空"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))
        assert engine.search("", limit=10) == []
        assert engine.search("   ", limit=10) == []

    def test_staged_search_strict_first(self, tmp_path):
        """严格匹配优先（stage_rank=0）"""
        engine = NoteChunkSearchEngine(index_dir=str(tmp_path))

        # 文件1：同时包含"红烧"和"肉"
        engine.index_chunks("exact", "exact.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "红烧肉的做法很简单"},
        ])
        # 文件2：只包含"红烧"
        engine.index_chunks("partial", "partial.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "红烧茄子也很好吃"},
        ])

        results = engine.search("红烧肉", limit=10)
        assert len(results) >= 1
        # 严格匹配的应该排在前面
        if len(results) >= 2:
            assert results[0]["file_id"] == "exact"

    @pytest.mark.asyncio
    async def test_search_async_awaits_async_rerank(self, tmp_path):
        """异步搜索会等待异步 rerank_provider。"""

        class AsyncReranker:
            async def rerank(self, query, documents):
                return [
                    {"index": 1, "relevance_score": 0.99},
                    {"index": 0, "relevance_score": 0.10},
                ]

        engine = NoteChunkSearchEngine(index_dir=str(tmp_path), rerank_provider=AsyncReranker())
        engine.index_chunks("a", "a.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "猫咪普通记录"},
        ])
        engine.index_chunks("b", "b.md", [
            {"chunk_index": 1, "line_start": 0, "line_end": 1, "content": "猫咪重要记录"},
        ])

        baseline = engine.search("猫咪", limit=2)
        assert len(baseline) == 2

        results = await engine.search_async("猫咪", limit=2)
        assert len(results) == 2
        assert results[0]["file_id"] == baseline[1]["file_id"]
        assert results[0]["score"] == pytest.approx(0.99)
