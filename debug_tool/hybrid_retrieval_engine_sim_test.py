from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_memory.components.hybrid_retrieval_engine import HybridRetrievalEngine  # noqa: E402


class FakeRetriever:
    """模拟 BM25 检索器。"""

    def __init__(self):
        self.bm25_calls: int = 0
        self.fusion_calls: int = 0

    def bm25_only(self, query: str, limit: int) -> List[Dict[str, Any]]:
        self.bm25_calls += 1
        _ = query
        base = [
            {"id": "m1", "final_score": 0.9},
            {"id": "m2", "final_score": 0.6},
            {"id": "m3", "final_score": 0.4},
        ]
        return base[: max(0, int(limit))]

    def fusion(self, query: str, limit: int, bm25_limit: int, vector_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        self.fusion_calls += 1
        _ = query, bm25_limit
        rows = []
        for item_id, score in sorted(vector_scores.items(), key=lambda x: float(x[1]), reverse=True):
            rows.append({"id": str(item_id), "final_score": float(score)})
        return rows[: max(0, int(limit))]


class FakeRerankProvider:
    """模拟上游 rerank_provider。"""

    def __init__(self):
        self.called = False

    def rerank(self, query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        self.called = True
        _ = query, top_n
        # 用文本中携带的 "rank:N" 人工控制重排顺序。
        items = []
        for idx, doc in enumerate(documents):
            marker = "rank:"
            score = 0.0
            if marker in doc:
                try:
                    score = float(doc.split(marker, 1)[1].strip().split()[0])
                except Exception:
                    score = 0.0
            items.append({"index": idx, "score": score})
        items.sort(key=lambda x: float(x["score"]), reverse=True)
        return {"code": 0, "results": items}


class FailingRerankProvider:
    """模拟重排异常。"""

    def rerank(self, query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        _ = query, documents, top_n
        raise RuntimeError("mock rerank failure")


def _build_doc_text_map(ids: List[str]) -> Dict[str, str]:
    # 手工指定 rank 值，让重排顺序可预测。
    rank_map = {
        "m1": 0.2,
        "m2": 0.95,
        "m3": 0.5,
        "m4": 0.8,
    }
    return {item_id: f"doc {item_id} rank:{rank_map.get(item_id, 0.0)}" for item_id in ids}


async def _run_all() -> None:
    fake = FakeRetriever()

    # 档1：无嵌入、无重排 -> BM25 only
    engine_no_rerank = HybridRetrievalEngine(retriever=fake, rerank_provider=None)
    tier1 = await engine_no_rerank.search_with_strategy(
        query="猫毛过敏",
        limit=2,
        candidate_limit=6,
        bm25_limit=6,
        vector_scores=None,
        bm25_only_search=fake.bm25_only,
        fusion_search=fake.fusion,
        build_doc_text_map=_build_doc_text_map,
    )
    assert [x["id"] for x in tier1] == ["m1", "m2"], f"档1失败: {tier1}"
    assert fake.bm25_calls >= 1 and fake.fusion_calls == 0, "档1调用路径异常"

    # 档2：有嵌入、无重排 -> fusion
    tier2 = await engine_no_rerank.search_with_strategy(
        query="猫毛过敏",
        limit=2,
        candidate_limit=6,
        bm25_limit=6,
        vector_scores={"m3": 0.99, "m1": 0.88, "m2": 0.77},
        bm25_only_search=fake.bm25_only,
        fusion_search=fake.fusion,
        build_doc_text_map=_build_doc_text_map,
    )
    assert [x["id"] for x in tier2] == ["m3", "m1"], f"档2失败: {tier2}"
    assert fake.fusion_calls >= 1, "档2未走融合路径"

    # 档3：有嵌入、有重排 -> 合并去重后 rerank
    rerank_provider = FakeRerankProvider()
    engine_with_rerank = HybridRetrievalEngine(retriever=fake, rerank_provider=rerank_provider)
    tier3 = await engine_with_rerank.search_with_strategy(
        query="猫毛过敏",
        limit=3,
        candidate_limit=6,
        bm25_limit=6,
        vector_scores={"m1": 0.9, "m4": 0.85},
        bm25_only_search=fake.bm25_only,
        fusion_search=fake.fusion,
        build_doc_text_map=_build_doc_text_map,
    )
    # 预期按 rank 值排序: m2(0.95), m4(0.8), m3(0.5)...
    assert [x["id"] for x in tier3[:3]] == ["m2", "m4", "m3"], f"档3失败: {tier3}"
    assert rerank_provider.called, "档3未触发重排"

    # 档3降级A：有向量 + 重排失败 -> 降级到融合
    engine_fail_rerank_vector = HybridRetrievalEngine(
        retriever=fake,
        rerank_provider=FailingRerankProvider(),
    )
    tier3_fallback_vector = await engine_fail_rerank_vector.search_with_strategy(
        query="猫毛过敏",
        limit=2,
        candidate_limit=6,
        bm25_limit=6,
        vector_scores={"m3": 0.99, "m1": 0.88, "m2": 0.77},
        bm25_only_search=fake.bm25_only,
        fusion_search=fake.fusion,
        build_doc_text_map=_build_doc_text_map,
    )
    assert [x["id"] for x in tier3_fallback_vector] == ["m3", "m1"], (
        f"重排失败降级(有向量)失败: {tier3_fallback_vector}"
    )

    # 档3降级B：无向量 + 重排失败 -> 降级到 BM25
    engine_fail_rerank_bm25 = HybridRetrievalEngine(
        retriever=fake,
        rerank_provider=FailingRerankProvider(),
    )
    tier3_fallback_bm25 = await engine_fail_rerank_bm25.search_with_strategy(
        query="猫毛过敏",
        limit=2,
        candidate_limit=6,
        bm25_limit=6,
        vector_scores=None,
        bm25_only_search=fake.bm25_only,
        fusion_search=fake.fusion,
        build_doc_text_map=_build_doc_text_map,
    )
    assert [x["id"] for x in tier3_fallback_bm25] == ["m1", "m2"], (
        f"重排失败降级(无向量)失败: {tier3_fallback_bm25}"
    )

    print("HybridRetrievalEngine 模拟测试通过：档1/档2/档3 全部通过")


if __name__ == "__main__":
    asyncio.run(_run_all())
