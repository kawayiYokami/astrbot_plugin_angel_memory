"""
FTS5 混合检索组件（模块化，可单独测试）。

设计目标：
1. 使用 SQLite FTS5 做稀疏检索（预分词 + 空格拼接）。
2. 统一记忆/笔记的融合算法，索引物理分离。
3. 支持离线单测（不依赖主流程组件）。
"""

from __future__ import annotations

import sqlite3
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


class FTS5HybridRetriever:
    """SQLite FTS5 + 向量分数融合检索器。"""

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
        self.db_path = str(db_path)
        self.tokenizer_mode = tokenizer_mode
        self.memory_threshold = float(memory_threshold)
        self.note_threshold = float(note_threshold)
        self.memory_default_vector_score = float(memory_default_vector_score)
        self.note_default_vector_score = float(note_default_vector_score)
        self._ensure_db_parent()
        self.init_schema()

    def _ensure_db_parent(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_schema(self) -> None:
        """初始化 FTS5 索引表。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    item_id UNINDEXED,
                    tags,
                    judgment,
                    tokenize='unicode61'
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
                    item_id UNINDEXED,
                    tags,
                    tokenize='unicode61'
                )
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = str(token or "").strip().lower()
        return token

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
        """
        批量注入自定义词到 jieba 词典（默认参数，不设 freq/tag）。
        使用全局锁保护，避免并发 add_word 的竞态问题。
        """
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
    def _build_match_query(tokens: List[str]) -> str:
        # FTS5 MATCH 查询中将 token 视为字面量：
        # 1) 转义反斜杠和双引号
        # 2) 去除控制字符
        # 3) 使用双引号包裹，避免操作符/特殊字符造成语法注入
        escaped_tokens: List[str] = []
        for token in tokens:
            raw = str(token or "").strip()
            if not raw:
                continue
            safe = re.sub(r"[\x00-\x1f]+", " ", raw).strip()
            if not safe:
                continue
            safe = safe.replace('"', '""')
            escaped_tokens.append(f'"{safe}"')
        return " OR ".join(escaped_tokens)

    @staticmethod
    def _rank_normalize(size: int) -> List[float]:
        if size <= 0:
            return []
        if size == 1:
            return [1.0]
        denom = float(size - 1)
        return [1.0 - (idx / denom) for idx in range(size)]

    def clear_index(self, target: str) -> None:
        normalized_target = str(target or "").strip().lower()
        table_map = {
            "memory": "memory_fts",
            "note": "note_fts",
        }
        if normalized_target not in table_map:
            raise ValueError(
                "clear_index target 非法，允许值为 'memory' 或 'note'"
            )
        table = table_map[normalized_target]
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {table}")
            conn.commit()

    def upsert_memory(self, item_id: str, tags: Iterable[str], judgment: str) -> None:
        if not item_id:
            return
        tags_text = self._pretokenized_tags(tags)
        judgment_text = self._pretokenized_text(judgment)
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_fts WHERE item_id = ?", (str(item_id),))
            conn.execute(
                "INSERT INTO memory_fts(item_id, tags, judgment) VALUES(?, ?, ?)",
                (str(item_id), tags_text, judgment_text),
            )
            conn.commit()

    def upsert_note(self, item_id: str, tags: Iterable[str]) -> None:
        if not item_id:
            return
        tags_text = self._pretokenized_tags(tags)
        with self._connect() as conn:
            conn.execute("DELETE FROM note_fts WHERE item_id = ?", (str(item_id),))
            conn.execute(
                "INSERT INTO note_fts(item_id, tags) VALUES(?, ?)",
                (str(item_id), tags_text),
            )
            conn.commit()

    def delete_memory(self, item_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_fts WHERE item_id = ?", (str(item_id),))
            conn.commit()

    def delete_note(self, item_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM note_fts WHERE item_id = ?", (str(item_id),))
            conn.commit()

    def rebuild_memory(self, rows: List[Dict]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_fts")
            for row in rows:
                item_id = str(row.get("id") or "").strip()
                if not item_id:
                    continue
                tags_text = self._pretokenized_tags(row.get("tags") or [])
                judgment_text = self._pretokenized_text(str(row.get("judgment") or ""))
                conn.execute(
                    "INSERT INTO memory_fts(item_id, tags, judgment) VALUES(?, ?, ?)",
                    (item_id, tags_text, judgment_text),
                )
            conn.commit()

    def rebuild_note(self, rows: List[Dict]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM note_fts")
            for row in rows:
                item_id = str(row.get("id") or "").strip()
                if not item_id:
                    continue
                tags_text = self._pretokenized_tags(row.get("tags") or [])
                conn.execute(
                    "INSERT INTO note_fts(item_id, tags) VALUES(?, ?)",
                    (item_id, tags_text),
                )
            conn.commit()

    def _search_memory_fts(self, query: str, limit: int = 50) -> List[Dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        match_query = self._build_match_query(tokens)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    item_id,
                    bm25(memory_fts, 2.0, 1.0) AS bm25_score
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def _search_note_fts(self, query: str, limit: int = 50) -> List[Dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        match_query = self._build_match_query(tokens)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    item_id,
                    bm25(note_fts, 1.0) AS bm25_score
                FROM note_fts
                WHERE note_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _fuse_scores(
        fts_rows: List[Dict],
        vector_scores: Optional[Dict[str, float]],
        fts_weight: float,
        vector_weight: float,
        threshold: float,
        default_vector_score: float,
        limit: int,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        if not fts_rows:
            return []
        normalized = FTS5HybridRetriever._rank_normalize(len(fts_rows))
        fused: List[Dict] = []
        effective_threshold = (
            float(min_final_score) if min_final_score is not None else float(threshold)
        )
        for idx, row in enumerate(fts_rows):
            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                continue
            fts_score = float(normalized[idx])
            if vector_scores is None:
                vector_score = float(default_vector_score)
            else:
                vector_score = float(vector_scores.get(item_id, default_vector_score))
            final_score = (float(fts_weight) * fts_score) + (
                float(vector_weight) * vector_score
            )
            if final_score < effective_threshold:
                continue
            fused.append(
                {
                    "id": item_id,
                    "bm25_score": float(row.get("bm25_score", 0.0)),
                    "fts_score": fts_score,
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
        fts_weight: float = 0.7,
        vector_weight: float = 0.3,
        vector_scores: Optional[Dict[str, float]] = None,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        rows = self._search_memory_fts(query=query, limit=fts_limit)
        return self._fuse_scores(
            fts_rows=rows,
            vector_scores=vector_scores,
            fts_weight=fts_weight,
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
        fts_weight: float = 0.85,
        vector_weight: float = 0.15,
        vector_scores: Optional[Dict[str, float]] = None,
        min_final_score: Optional[float] = None,
    ) -> List[Dict]:
        rows = self._search_note_fts(query=query, limit=fts_limit)
        return self._fuse_scores(
            fts_rows=rows,
            vector_scores=vector_scores,
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            threshold=self.note_threshold,
            default_vector_score=self.note_default_vector_score,
            limit=limit,
            min_final_score=min_final_score,
        )


def _demo_self_test() -> None:
    """模块级最小自测，可单独运行验证 FTS5 索引和检索。"""
    demo_db = Path("data") / "fts5_hybrid_demo.db"
    retriever = FTS5HybridRetriever(db_path=str(demo_db))
    retriever.rebuild_memory(
        [
            {"id": "m1", "tags": ["猫", "过敏", "药物"], "judgment": "用户对猫毛过敏，春季会加重"},
            {"id": "m2", "tags": ["宠物", "饮食"], "judgment": "用户给猫买了低敏猫粮"},
            {"id": "m3", "tags": ["过敏", "旅行"], "judgment": "用户去海边旅行时过敏减轻"},
        ]
    )
    retriever.rebuild_note(
        [
            {"id": "n1", "tags": ["notes", "family", "travel", "plan.md"]},
            {"id": "n2", "tags": ["notes", "health", "allergy", "daily.md"]},
        ]
    )

    logger.info("[检索] 开始 FTS5 模块自测")
    mem_results = retriever.search_memory(query="我最近猫毛过敏又犯了", limit=5)
    note_results = retriever.search_note(query="旅行 计划", limit=5)
    logger.info(f"[检索] 完成 记忆结果={len(mem_results)} 笔记结果={len(note_results)}")
    print("memory:", mem_results)
    print("note:", note_results)


if __name__ == "__main__":
    _demo_self_test()
