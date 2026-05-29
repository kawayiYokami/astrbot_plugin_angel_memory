from llm_memory.components.bm25_retriever import TantivyBM25Retriever


def test_rrf_fuse_uses_relative_threshold_for_rrf_scores():
    rows = [
        {"item_id": "a", "bm25_score": 9.0, "bm25_normalized_score": 1.0},
        {"item_id": "b", "bm25_score": 8.0, "bm25_normalized_score": 0.8},
    ]
    vector_scores = {"a": 0.95, "b": 0.9}

    hits = TantivyBM25Retriever._rrf_fuse(
        bm25_rows=rows,
        vector_scores=vector_scores,
        threshold=0.5,
        limit=10,
    )

    assert [hit["id"] for hit in hits] == ["a", "b"]
    assert all(hit["score_kind"] == "rrf" for hit in hits)
    assert all(0.0 < hit["final_score"] < 0.5 for hit in hits)
