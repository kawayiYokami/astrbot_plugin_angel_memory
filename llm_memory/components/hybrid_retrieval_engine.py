from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional


class HybridRetrievalEngine:
    """统一混合检索策略引擎（按能力分档）。"""

    def __init__(self, retriever: Any, rerank_provider: Optional[Any] = None):
        self.retriever = retriever
        self.rerank_provider = rerank_provider

    def has_rerank(self) -> bool:
        return self.rerank_provider is not None and hasattr(self.rerank_provider, "rerank")

    @staticmethod
    def _rank_normalize(size: int) -> List[float]:
        if size <= 0:
            return []
        if size == 1:
            return [1.0]
        denom = float(size - 1)
        return [1.0 - (idx / denom) for idx in range(size)]

    @staticmethod
    def _extract_rerank_results(rerank_resp: Any) -> List[Dict[str, Any]]:
        if rerank_resp is None:
            return []
        if isinstance(rerank_resp, dict):
            if rerank_resp.get("code") not in (None, 0, 200, "0", "200"):
                return []
            items = rerank_resp.get("results")
            if items is None:
                data = rerank_resp.get("data")
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("results") or data.get("items")
        else:
            items = rerank_resp
        if not isinstance(items, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append(
                    {
                        "id": getattr(item, "id", None),
                        "index": getattr(item, "index", None),
                        "score": getattr(item, "score", None),
                    }
                )
        return normalized

    async def rerank_candidates(
        self,
        query: str,
        ordered_ids: List[str],
        doc_text_map: Dict[str, str],
        top_n: int,
    ) -> List[Dict[str, float]]:
        if not ordered_ids or not self.has_rerank():
            return []

        try:
            passages: List[Dict[str, str]] = []
            documents: List[str] = []
            for item_id in ordered_ids:
                text = str(doc_text_map.get(item_id, "") or "").strip()
                if not text:
                    text = item_id
                passages.append({"id": item_id, "text": text})
                documents.append(text)

            rerank_method = self.rerank_provider.rerank
            rerank_resp = rerank_method(query=str(query or ""), documents=documents, top_n=int(top_n))
            if inspect.isawaitable(rerank_resp):
                rerank_resp = await rerank_resp
            items = self._extract_rerank_results(rerank_resp)
            if not items:
                return []

            scored: List[Dict[str, float]] = []
            for idx, item in enumerate(items):
                item_id = str(item.get("id") or "").strip()
                if not item_id and isinstance(item.get("index"), int):
                    i = int(item["index"])
                    if 0 <= i < len(passages):
                        item_id = passages[i]["id"]
                if not item_id:
                    text = str(item.get("text") or item.get("document") or "").strip()
                    if text:
                        for p in passages:
                            if p["text"] == text:
                                item_id = p["id"]
                                break
                if not item_id:
                    continue
                scored.append(
                    {
                        "id": item_id,
                        "rerank_score": float(item.get("score", 0.0) or 0.0),
                        "rank_index": idx,
                    }
                )
            if not scored:
                return []

            max_score = max(float(x["rerank_score"]) for x in scored)
            fallback_norm = self._rank_normalize(len(scored))
            for idx, row in enumerate(scored):
                if max_score > 0:
                    row["final_score"] = float(row["rerank_score"]) / max_score
                else:
                    row["final_score"] = float(fallback_norm[idx])
            scored.sort(
                key=lambda x: (float(x.get("final_score", 0.0)), -int(x.get("rank_index", 0))),
                reverse=True,
            )
            return scored
        except Exception:
            return []

    @staticmethod
    def _merge_ids(vector_scores: Dict[str, float], bm25_hits: List[Dict[str, Any]], candidate_limit: int) -> List[str]:
        vector_sorted = sorted(
            vector_scores.items(),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:candidate_limit]
        vector_ids = [str(item_id).strip() for item_id, _ in vector_sorted if str(item_id).strip()]
        bm25_ids = [str(item.get("id") or "").strip() for item in bm25_hits if str(item.get("id") or "").strip()]

        merged_ids: List[str] = []
        seen = set()
        for item_id in vector_ids + bm25_ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            merged_ids.append(item_id)
        return merged_ids

    async def search_with_strategy(
        self,
        query: str,
        limit: int,
        candidate_limit: int,
        bm25_limit: int,
        vector_scores: Optional[Dict[str, float]],
        bm25_only_search: Callable[[str, int], List[Dict[str, Any]]],
        fusion_search: Callable[[str, int, int, Dict[str, float]], List[Dict[str, Any]]],
        build_doc_text_map: Callable[[List[str]], Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        has_vector = bool(vector_scores)
        has_rerank = self.has_rerank()
        safe_query = str(query or "").strip()
        if not safe_query:
            return []

        # 先拿 BM25 候选（永远可用）
        bm25_hits = bm25_only_search(safe_query, max(1, int(candidate_limit)))

        # 无重排：有向量走融合；无向量直接 BM25。
        if not has_rerank:
            if not has_vector:
                return bm25_hits[: max(1, int(limit))]
            return fusion_search(
                safe_query,
                max(1, int(limit)),
                max(1, int(bm25_limit)),
                vector_scores or {},
            )

        # 有重排：重排能力独立于向量能力。
        if has_vector:
            merged_ids = self._merge_ids(vector_scores or {}, bm25_hits, max(1, int(candidate_limit)))
        else:
            merged_ids = [str(item.get("id") or "").strip() for item in bm25_hits if str(item.get("id") or "").strip()]

        if not merged_ids:
            return []
        doc_text_map = build_doc_text_map(merged_ids)
        reranked = await self.rerank_candidates(
            query=safe_query,
            ordered_ids=merged_ids,
            doc_text_map=doc_text_map,
            top_n=max(1, int(limit)),
        )
        if reranked:
            return [{"id": x["id"], "final_score": float(x.get("final_score", 0.0))} for x in reranked]
        # 重排失败/无结果：降级到无重排策略。
        if has_vector:
            return fusion_search(
                safe_query,
                max(1, int(limit)),
                max(1, int(bm25_limit)),
                vector_scores or {},
            )
        return bm25_hits[: max(1, int(limit))]
