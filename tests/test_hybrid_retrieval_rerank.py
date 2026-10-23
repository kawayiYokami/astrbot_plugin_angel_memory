"""记忆混合检索的重排分数契约。

上游 rerank provider 返回的是 AstrBot 的 RerankResult 对象（字段
relevance_score），重排必须按该分数排序；缺失分数时才降级为按返回顺序。
"""

from dataclasses import dataclass

import pytest

from llm_memory.components.hybrid_retrieval_engine import HybridRetrievalEngine


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class _Provider:
    def __init__(self, results):
        self._results = results

    async def rerank(self, query, documents):
        return self._results


@pytest.mark.asyncio
async def test_rerank_orders_candidates_by_relevance_score():
    engine = HybridRetrievalEngine(
        retriever=None,
        rerank_provider=_Provider([RerankResult(1, 0.8), RerankResult(0, 0.2)]),
    )

    scored = await engine.rerank_candidates(
        "查询",
        ["a", "b"],
        {"a": "文档甲", "b": "文档乙"},
    )

    assert [row["id"] for row in scored] == ["b", "a"]
    assert scored[0]["final_score"] == pytest.approx(1.0)
    assert scored[1]["final_score"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_rerank_falls_back_to_return_order_when_scores_missing():
    class _NoScoreResult:
        def __init__(self, index):
            self.index = index

    engine = HybridRetrievalEngine(
        retriever=None,
        rerank_provider=_Provider([_NoScoreResult(1), _NoScoreResult(0)]),
    )

    scored = await engine.rerank_candidates(
        "查询",
        ["a", "b"],
        {"a": "文档甲", "b": "文档乙"},
    )

    assert [row["id"] for row in scored] == ["b", "a"]
    assert scored[0]["final_score"] == pytest.approx(1.0)
    assert scored[1]["final_score"] == pytest.approx(0.0)
