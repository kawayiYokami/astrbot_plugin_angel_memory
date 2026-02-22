"""
Tantivy BM25 检索组件（模块化，可单独测试）。

设计目标：
1. 使用 Tantivy 做 BM25 稀疏检索。
2. 中文分词由 Python 侧 jieba 预分词完成（索引/查询同流程）。
3. 统一记忆/笔记融合算法，支持三档检索中的前两档：
   - BM25 only
   - 向量 + BM25 融合
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re
import threading

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

import jieba

try:
    import tantivy
except Exception:  # pragma: no cover - 依赖缺失时由运行时显式报错
    tantivy = None


class TantivyBM25Retriever:
    """Tantivy BM25 + 向量融合检索器。"""

    _jieba_lock = threading.Lock()
    _jieba_loaded_words: set[str] = set()

    def __init__(
        self,
        db_path: str,
        tokenizer_mode: str = "jieba_cut",
        memory_threshold: float = 0.5,
        note_threshold: float = 0.6,
        memory_default_vector_score: float = 0.5,
        note_default_vector_score: float = 0.6,
    ):
        if tantivy is None:
            raise RuntimeError(
                "Tantivy 依赖不可用，请先安装依赖后再启动（例如: pip install tantivy）"
            )

        self.db_path = str(db_path)
        self.tokenizer_mode = tokenizer_mode
        self.memory_threshold = float(memory_threshold)
        self.note_threshold = float(note_threshold)
        self.memory_default_vector_score = float(memory_default_vector_score)
        self.note_default_vector_score = float(note_default_vector_score)
        self._init_indexes()

    def _init_indexes(self) -> None:
        base_dir = Path(self.db_path).parent / "tantivy_indexes"
        self._memory_dir = base_dir / "memory"
        self._note_dir = base_dir / "note"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._note_dir.mkdir(parents=True, exist_ok=True)

        # 使用 TEXT 字段存储预分词文本，检索时直接按 token 查询。
        memory_builder = tantivy.SchemaBuilder()
        memory_builder.add_text_field("item_id", stored=True)
        memory_builder.add_text_field("tags_text", stored=False)
        memory_builder.add_text_field("judgment_text", stored=False)
        self._memory_schema = memory_builder.build()

        note_builder = tantivy.SchemaBuilder()
        note_builder.add_text_field("item_id", stored=True)
        note_builder.add_text_field("tags_text", stored=False)
        self._note_schema = note_builder.build()

        self._memory_index = tantivy.Index(self._memory_schema, path=str(self._memory_dir))
        self._note_index = tantivy.Index(self._note_schema, path=str(self._note_dir))

    @staticmethod
    def _normalize_token(token: str) -> str:
        return str(token or "").strip().lower()

    def _tokenize(self, text: str) -> List[str]:
        raw_text = str(text or "")
        if not raw_text.strip():
            return []

        if self.tokenizer_mode == "jieba_cut_for_search":
            words = jieba.cut_for_search(raw_text)
        else:
            words = jieba.cut(raw_text)

        tokens: List[str] = []
        for word in words:
            tok = self._normalize_token(word)
            if not tok:
                continue
            if re.fullmatch(r"[\W_]+", tok):
                continue
            tokens.append(tok)
        return tokens

    def _pretokenized_text(self, text: str) -> str:
        return " ".join(self._tokenize(text))

    def _pretokenized_tags(self, tags: Iterable[str]) -> str:
        if tags is None:
            return ""
        source = " ".join([str(t) for t in tags if str(t).strip()])
        return self._pretokenized_text(source)

    @classmethod
    def add_jieba_words(cls, words: Iterable[str]) -> Dict[str, int]:
        """批量注入自定义词到 jieba 词典。"""
        total = 0
        added = 0
        skipped = 0
        if words is None:
            return {"total": 0, "added": 0, "skipped": 0}

        with cls._jieba_lock:
            for word in words:
                text = str(word or "").strip()
                if not text:
                    continue
                total += 1
                if text in cls._jieba_loaded_words:
                    skipped += 1
                    continue
                jieba.add_word(text)
                cls._jieba_loaded_words.add(text)
                added += 1
        return {"total": total, "added": added, "skipped": skipped}

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
        elif normalized_target == "note":
            index = self._note_index
        else:
            raise ValueError("clear_index target 非法，允许值为 'memory' 或 'note'")
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

    def upsert_note(self, item_id: str, tags: Iterable[str]) -> None:
        if not item_id:
            return
        writer = self._note_index.writer()
        writer.delete_documents("item_id", str(item_id))
        writer.add_document(
            tantivy.Document(
                item_id=[str(item_id)],
                tags_text=[self._pretokenized_tags(tags)],
            )
        )
        writer.commit()

    def delete_memory(self, item_id: str) -> None:
        writer = self._memory_index.writer()
        writer.delete_documents("item_id", str(item_id))
        writer.commit()

    def delete_note(self, item_id: str) -> None:
        writer = self._note_index.writer()
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

    def rebuild_note(self, rows: List[Dict]) -> None:
        writer = self._note_index.writer()
        writer.delete_all_documents()
        for row in rows:
            item_id = str(row.get("id") or "").strip()
            if not item_id:
                continue
            writer.add_document(
                tantivy.Document(
                    item_id=[item_id],
                    tags_text=[self._pretokenized_tags(row.get("tags") or [])],
                )
            )
        writer.commit()

    def _build_query_text(self, query: str) -> str:
        tokens = self._tokenize(query)
        if not tokens:
            return ""
        return " OR ".join(tokens)

    def _search_memory_bm25(self, query: str, limit: int = 50) -> List[Dict]:
        query_text = self._build_query_text(query)
        if not query_text:
            return []

        searcher = self._memory_index.searcher()
        parsed_query = self._memory_index.parse_query(query_text, ["tags_text", "judgment_text"])
        search_result = searcher.search(parsed_query, int(limit))
        rows: List[Dict] = []
        for score, addr in search_result.hits:
            doc = searcher.doc(addr)
            item_ids = self._extract_doc_values(doc, "item_id")
            item_id = str(item_ids[0]) if item_ids else ""
            if not item_id:
                continue
            rows.append({"item_id": item_id, "bm25_score": float(score)})
        return self._score_normalize(rows)

    def _search_note_bm25(self, query: str, limit: int = 50) -> List[Dict]:
        query_text = self._build_query_text(query)
        if not query_text:
            return []

        searcher = self._note_index.searcher()
        parsed_query = self._note_index.parse_query(query_text, ["tags_text"])
        search_result = searcher.search(parsed_query, int(limit))
        rows: List[Dict] = []
        for score, addr in search_result.hits:
            doc = searcher.doc(addr)
            item_ids = self._extract_doc_values(doc, "item_id")
            item_id = str(item_ids[0]) if item_ids else ""
            if not item_id:
                continue
            rows.append({"item_id": item_id, "bm25_score": float(score)})
        return self._score_normalize(rows)

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
        if not bm25_rows:
            return []
        fused: List[Dict] = []
        effective_threshold = (
            float(min_final_score) if min_final_score is not None else float(threshold)
        )
        for row in bm25_rows:
            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                continue
            bm25_norm = float(row.get("bm25_normalized_score", 0.0))
            if vector_scores is None:
                vector_score = float(default_vector_score)
            else:
                vector_score = float(vector_scores.get(item_id, default_vector_score))
            final_score = (float(vector_weight) * vector_score) + (
                float(bm25_weight) * bm25_norm
            )
            if final_score < effective_threshold:
                continue
            fused.append(
                {
                    "id": item_id,
                    "bm25_score": float(row.get("bm25_score", 0.0)),
                    "bm25_normalized_score": bm25_norm,
                    "vector_score": vector_score,
                    "final_score": final_score,
                }
            )
        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return fused[: max(0, int(limit))]

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

    def search_note(
        self,
        query: str,
        limit: int = 20,
        fts_limit: int = 80,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        vector_scores: Optional[Dict[str, float]] = None,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        rows = self._search_note_bm25(query=query, limit=fts_limit)
        return self._fuse_scores(
            bm25_rows=rows,
            vector_scores=vector_scores,
            bm25_weight=fts_weight,
            vector_weight=vector_weight,
            threshold=self.note_threshold,
            default_vector_score=self.note_default_vector_score,
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
            }
            for r in rows
            if str(r.get("item_id") or "").strip()
        ]

    def search_note_bm25_only(self, query: str, limit: int = 20) -> List[Dict]:
        rows = self._search_note_bm25(query=query, limit=limit)
        return [
            {
                "id": str(r.get("item_id") or ""),
                "bm25_score": float(r.get("bm25_score", 0.0)),
                "bm25_normalized_score": float(r.get("bm25_normalized_score", 0.0)),
                "final_score": float(r.get("bm25_normalized_score", 0.0)),
            }
            for r in rows
            if str(r.get("item_id") or "").strip()
        ]


# 兼容旧导入路径（避免一次性大面积修改）。
FTS5HybridRetriever = TantivyBM25Retriever
