"""
Tantivy BM25 检索组件（模块化，可单独测试）。

设计目标：
1. 使用 Tantivy 做 BM25 稀疏检索。
2. 中文分词由 CJK bigram 完成（无词典依赖）。
3. 统一记忆融合算法，支持三档检索中的前两档：
   - BM25 only
   - 向量 + BM25 融合
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from .note_chunk_search import iter_cjk_fragments, normalize_search_text, tokenize_for_index

import tantivy


class TantivyBM25Retriever:
    """Tantivy BM25 + 向量融合检索器。"""

    def __init__(
        self,
        db_path: str,
        memory_threshold: float = 0.5,
        memory_default_vector_score: float = 0.5,
    ):
        self.db_path = str(db_path)
        self.memory_threshold = float(memory_threshold)
        self.memory_default_vector_score = float(memory_default_vector_score)
        self._init_indexes()

    def _init_indexes(self) -> None:
        base_dir = Path(self.db_path).parent / "tantivy_indexes"
        self._memory_dir = base_dir / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # 使用 TEXT 字段存储预分词文本，检索时直接按 token 查询。
        memory_builder = tantivy.SchemaBuilder()
        memory_builder.add_text_field("item_id", stored=True)
        memory_builder.add_text_field("tags_text", stored=False)
        memory_builder.add_text_field("judgment_text", stored=False)
        self._memory_schema = memory_builder.build()

        self._memory_index = tantivy.Index(self._memory_schema, path=str(self._memory_dir))

    @staticmethod
    def _normalize_token(token: str) -> str:
        return str(token or "").strip().lower()

    def _tokenize(self, text: str) -> List[str]:
        """使用 CJK bigram 分词（无词典依赖）"""
        return iter_cjk_fragments(text, include_unigram=True, max_ngram=2)

    def _pretokenized_text(self, text: str) -> str:
        return tokenize_for_index(text)

    def _pretokenized_tags(self, tags: Iterable[str]) -> str:
        if tags is None:
            return ""
        source = " ".join([str(t) for t in tags if str(t).strip()])
        return self._pretokenized_text(source)

    @staticmethod
    def _score_normalize(rows: List[Dict]) -> List[Dict]:
        if not rows:
            return []
        max_score = 0.0
        for row in rows:
            score = float(row.get("bm25_score", 0.0))
            if score > max_score:
                max_score = score
        normalized_rows: List[Dict] = []
        for row in rows:
            score = float(row.get("bm25_score", 0.0))
            normalized = (score / max_score) if max_score > 0.0 else 0.0
            item = dict(row)
            item["bm25_normalized_score"] = normalized
            normalized_rows.append(item)
        return normalized_rows

    @staticmethod
    def _extract_doc_values(doc: object, field_name: str) -> List[str]:
        """
        兼容不同 tantivy.Document Python 绑定形态：
        - dict-like（.get）
        - .to_dict()
        - __getitem__
        - 属性访问
        """
        if doc is None:
            return []

        values: object = None

        try:
            if hasattr(doc, "to_dict"):
                maybe_dict = doc.to_dict()
                if isinstance(maybe_dict, dict):
                    values = maybe_dict.get(field_name)
        except Exception:
            values = None

        if values is None:
            try:
                if isinstance(doc, dict):
                    values = doc.get(field_name)
                elif hasattr(doc, "get"):
                    values = doc.get(field_name)
            except Exception:
                values = None

        if values is None:
            try:
                values = doc[field_name]  # type: ignore[index]
            except Exception:
                values = None

        if values is None:
            try:
                values = getattr(doc, field_name, None)
            except Exception:
                values = None

        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return [str(v) for v in values if str(v).strip()]
        text = str(values).strip()
        return [text] if text else []

    def clear_index(self, target: str) -> None:
        normalized_target = str(target or "").strip().lower()
        if normalized_target == "memory":
            index = self._memory_index
        else:
            raise ValueError("clear_index target 非法，允许值为 'memory'")
        writer = index.writer()
        writer.delete_all_documents()
        writer.commit()

    def upsert_memory(self, item_id: str, tags: Iterable[str], judgment: str) -> None:
        if not item_id:
            return
        writer = self._memory_index.writer()
        writer.delete_documents("item_id", str(item_id))
        writer.add_document(
            tantivy.Document(
                item_id=[str(item_id)],
                tags_text=[self._pretokenized_tags(tags)],
                judgment_text=[self._pretokenized_text(judgment)],
            )
        )
        writer.commit()

    def delete_memory(self, item_id: str) -> None:
        writer = self._memory_index.writer()
        writer.delete_documents("item_id", str(item_id))
        writer.commit()

    def rebuild_memory(self, rows: List[Dict]) -> None:
        writer = self._memory_index.writer()
        writer.delete_all_documents()
        for row in rows:
            item_id = str(row.get("id") or "").strip()
            if not item_id:
                continue
            writer.add_document(
                tantivy.Document(
                    item_id=[item_id],
                    tags_text=[self._pretokenized_tags(row.get("tags") or [])],
                    judgment_text=[self._pretokenized_text(str(row.get("judgment") or ""))],
                )
            )
        writer.commit()

    def _build_query_text(self, query: str) -> str:
        tokens = self._tokenize(query)
        if not tokens:
            return ""
        return " OR ".join(tokens)

    def _staged_search(self, index, fields: List[str], query: str, limit: int) -> List[Dict]:
        """多级搜索：先严格 AND，不够再 OR 补充"""
        from .note_chunk_search import select_required_cjk_ngrams, _build_all_terms_query, _build_any_terms_query

        required_terms = select_required_cjk_ngrams(query)
        all_terms = iter_cjk_fragments(query, include_unigram=True, max_ngram=2)

        if not required_terms and not all_terms:
            return []

        # Stage 0: AND
        strict_query = _build_all_terms_query(required_terms) if required_terms else ""
        candidates: List[Dict] = []
        if strict_query:
            try:
                searcher = index.searcher()
                parsed = index.parse_query(strict_query, fields)
                result = searcher.search(parsed, int(limit * 3))
                for score, addr in result.hits:
                    doc = searcher.doc(addr)
                    item_ids = self._extract_doc_values(doc, "item_id")
                    item_id = str(item_ids[0]) if item_ids else ""
                    if item_id:
                        candidates.append({"item_id": item_id, "bm25_score": float(score)})
            except Exception:
                candidates = []

        # Stage 1: OR (if not enough)
        if len(candidates) < limit and all_terms:
            loose_query = _build_any_terms_query(all_terms)
            if loose_query:
                try:
                    searcher = index.searcher()
                    parsed = index.parse_query(loose_query, fields)
                    result = searcher.search(parsed, int(limit * 3))
                    seen_ids = {r["item_id"] for r in candidates}
                    for score, addr in result.hits:
                        doc = searcher.doc(addr)
                        item_ids = self._extract_doc_values(doc, "item_id")
                        item_id = str(item_ids[0]) if item_ids else ""
                        if item_id and item_id not in seen_ids:
                            candidates.append({"item_id": item_id, "bm25_score": float(score)})
                            seen_ids.add(item_id)
                except Exception:
                    pass

        return self._score_normalize(candidates[:limit * 3])

    def _search_memory_bm25(self, query: str, limit: int = 50) -> List[Dict]:
        return self._staged_search(
            self._memory_index, ["tags_text", "judgment_text"], query, limit
        )

    @staticmethod
    def _rrf_fuse(
        bm25_rows: List[Dict],
        vector_scores: Optional[Dict[str, float]],
        threshold: float,
        limit: int,
        k: int = 60,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) 融合 BM25 和向量分数"""
        if not bm25_rows:
            return []

        # BM25 排名
        bm25_sorted = sorted(bm25_rows, key=lambda x: float(x.get("bm25_score", 0)), reverse=True)
        bm25_rank: Dict[str, int] = {}
        for rank, row in enumerate(bm25_sorted):
            item_id = str(row.get("item_id") or "").strip()
            if item_id:
                bm25_rank[item_id] = rank + 1

        # 向量排名
        vector_rank: Dict[str, int] = {}
        if vector_scores:
            vec_sorted = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (item_id, _) in enumerate(vec_sorted):
                vector_rank[item_id] = rank + 1

        # RRF 融合
        all_ids = set(bm25_rank.keys())
        if vector_scores:
            all_ids.update(vector_scores.keys())

        fused: List[Dict] = []
        for item_id in all_ids:
            bm25_r = bm25_rank.get(item_id, len(bm25_sorted) + 1)
            vec_r = vector_rank.get(item_id, len(vector_rank) + 100) if vector_scores else k + 1
            rrf_score = 1.0 / (k + bm25_r) + 1.0 / (k + vec_r)

            bm25_row = next((r for r in bm25_rows if str(r.get("item_id", "")).strip() == item_id), None)
            bm25_score = float(bm25_row.get("bm25_score", 0.0)) if bm25_row else 0.0
            bm25_norm = float(bm25_row.get("bm25_normalized_score", 0.0)) if bm25_row else 0.0
            vector_score = float(vector_scores.get(item_id, 0.0)) if vector_scores else 0.0

            fused.append({
                "id": item_id,
                "bm25_score": bm25_score,
                "bm25_normalized_score": bm25_norm,
                "vector_score": vector_score,
                "final_score": rrf_score,
                "score_kind": "rrf",
            })

        fused.sort(key=lambda x: x["final_score"], reverse=True)
        if min_final_score is None and threshold > 0.0 and fused:
            top_score = float(fused[0].get("final_score", 0.0) or 0.0)
            if top_score > 0.0:
                keep_ratio = max(0.0, min(1.0, float(threshold)))
                keep_threshold = top_score * keep_ratio
                fused = [
                    item for item in fused
                    if float(item.get("final_score", 0.0) or 0.0) >= keep_threshold
                ]
        elif min_final_score is not None:
            absolute_threshold = max(0.0, float(min_final_score))
            fused = [
                item for item in fused
                if float(item.get("final_score", 0.0) or 0.0) >= absolute_threshold
            ]
        return fused[:max(0, int(limit))]

    @staticmethod
    def _fuse_scores(
        bm25_rows: List[Dict],
        vector_scores: Optional[Dict[str, float]],
        bm25_weight: float,
        vector_weight: float,
        threshold: float,
        default_vector_score: float,
        limit: int,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        """兼容旧接口，内部委托 RRF"""
        return TantivyBM25Retriever._rrf_fuse(
            bm25_rows=bm25_rows,
            vector_scores=vector_scores,
            threshold=threshold,
            limit=limit,
            min_final_score=min_final_score,
        )

    def search_memory(
        self,
        query: str,
        limit: int = 20,
        fts_limit: int = 80,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        vector_scores: Optional[Dict[str, float]] = None,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        rows = self._search_memory_bm25(query=query, limit=fts_limit)
        return self._fuse_scores(
            bm25_rows=rows,
            vector_scores=vector_scores,
            bm25_weight=fts_weight,
            vector_weight=vector_weight,
            threshold=self.memory_threshold,
            default_vector_score=self.memory_default_vector_score,
            limit=limit,
            min_final_score=min_final_score,
        )

    def search_memory_bm25_only(self, query: str, limit: int = 20) -> List[Dict]:
        rows = self._search_memory_bm25(query=query, limit=limit)
        return [
            {
                "id": str(r.get("item_id") or ""),
                "bm25_score": float(r.get("bm25_score", 0.0)),
                "bm25_normalized_score": float(r.get("bm25_normalized_score", 0.0)),
                "final_score": float(r.get("bm25_normalized_score", 0.0)),
                "score_kind": "normalized",
            }
            for r in rows[:limit]
            if str(r.get("item_id") or "").strip()
        ]

# 兼容旧导入路径（避免一次性大面积修改）。
FTS5HybridRetriever = TantivyBM25Retriever
