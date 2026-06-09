from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class SqliteTextIndex:
    """Provider-local flat cosine vector index backed by SQLite."""

    INDEX_VERSION = "sqlite_flat_cosine_v1"

    def __init__(
        self,
        *,
        name: str,
        db_path: Path,
        embedding_provider: Any,
        provider_id: str,
        model_key: str,
        log=None,
    ):
        self.name = str(name or "").strip()
        if not self.name:
            raise ValueError("SQLite向量索引名称不能为空")
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self.provider_id = str(provider_id or "unknown").strip() or "unknown"
        self.model_key = str(model_key or "unknown").strip() or "unknown"
        self.logger = log or logger
        self._lock = threading.RLock()
        self._matrix_cache: Optional[Tuple[List[str], np.ndarray]] = None

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS vector_rows (
                    item_id TEXT PRIMARY KEY,
                    vector_text TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_vector_rows_updated_at
                    ON vector_rows(updated_at);
                """
            )
            conn.commit()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    @staticmethod
    def _encode_vector(vector: np.ndarray) -> bytes:
        arr = np.asarray(vector, dtype="float32").reshape(-1)
        return arr.tobytes(order="C")

    @staticmethod
    def _decode_vector(blob: bytes, dimension: int) -> Optional[np.ndarray]:
        if not blob or int(dimension or 0) <= 0:
            return None
        arr = np.frombuffer(blob, dtype="float32")
        if int(arr.size) != int(dimension):
            return None
        return arr

    def _invalidate_cache_locked(self) -> None:
        self._matrix_cache = None

    def _get_meta_locked(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM index_meta").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _write_meta_locked(self, dimension: int) -> None:
        values = {
            "provider_id": self.provider_id,
            "embedding_model": self.model_key,
            "dimension": str(int(dimension)),
            "index_path": str(self.db_path),
            "index_version": self.INDEX_VERSION,
            "updated_at": str(time.time()),
        }
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
                list(values.items()),
            )
            conn.commit()

    def _is_meta_compatible_locked(self, dimension: Optional[int] = None) -> bool:
        meta = self._get_meta_locked()
        if not meta:
            return True
        if meta.get("provider_id") != self.provider_id:
            return False
        if meta.get("embedding_model") != self.model_key:
            return False
        if meta.get("index_version") not in ("", self.INDEX_VERSION):
            return False
        if dimension is not None:
            try:
                if int(meta.get("dimension", "0")) != int(dimension):
                    return False
            except ValueError:
                return False
        return True

    def _clear_locked(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vector_rows")
            conn.execute("DELETE FROM index_meta")
            conn.commit()
        self._invalidate_cache_locked()

    def clear(self) -> None:
        with self._lock:
            self._clear_locked()

    async def _embed_documents(
        self,
        texts: List[str],
        *,
        is_query: bool = False,
        timeout: int = 3,
    ) -> Optional[List[List[float]]]:
        if not texts:
            return []
        if is_query:
            try:
                return await asyncio.wait_for(
                    self.embedding_provider.embed_documents(texts),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"[SQLite向量索引] 查询向量化超时 index={self.name} timeout={timeout}s"
                )
                return None
            except Exception as e:
                self.logger.warning(f"[SQLite向量索引] 查询向量化失败 index={self.name} 异常={e}")
                return None
        return await self.embedding_provider.embed_documents(texts)

    async def upsert_rows(self, rows: List[Dict[str, str]]) -> None:
        clean_rows: List[Dict[str, str]] = []
        seen = set()
        for row in rows or []:
            item_id = str(row.get("id") or "").strip()
            vector_text = str(row.get("vector_text") or "").strip()
            if not item_id or not vector_text or item_id in seen:
                continue
            seen.add(item_id)
            clean_rows.append({"id": item_id, "vector_text": vector_text})
        if not clean_rows:
            return

        embed_start = time.time()
        self.logger.info(
            "[SQLite向量索引] 开始 "
            f"任务名=批量向量化 index={self.name} provider_id={self.provider_id} "
            f"embedding_model={self.model_key} 条数={len(clean_rows)}"
        )
        embeddings = await self._embed_documents([row["vector_text"] for row in clean_rows])
        if not embeddings:
            self.logger.warning(
                "[SQLite向量索引] 失败 "
                f"任务名=批量向量化 index={self.name} provider_id={self.provider_id} "
                f"原因=嵌入结果为空 条数={len(clean_rows)}"
            )
            return

        vectors = self._normalize(np.asarray(embeddings, dtype="float32"))
        dimension = int(vectors.shape[1])
        embed_ms = int((time.time() - embed_start) * 1000)
        self.logger.info(
            "[SQLite向量索引] 完成 "
            f"任务名=批量向量化 index={self.name} provider_id={self.provider_id} "
            f"embedding_model={self.model_key} 条数={len(clean_rows)} "
            f"dimension={dimension} 耗时毫秒={embed_ms}"
        )

        now = time.time()
        with self._lock:
            if not self._is_meta_compatible_locked(dimension):
                self.logger.info(
                    "[SQLite向量索引] 检测到模型或维度变化，清空当前索引 "
                    f"index={self.name} provider_id={self.provider_id} "
                    f"embedding_model={self.model_key} dimension={dimension}"
                )
                self._clear_locked()

            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO vector_rows(
                        item_id, vector_text, embedding_blob, dimension, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row["id"],
                            row["vector_text"],
                            sqlite3.Binary(self._encode_vector(vectors[idx])),
                            dimension,
                            now,
                        )
                        for idx, row in enumerate(clean_rows)
                    ],
                )
                conn.commit()
            self._write_meta_locked(dimension)
            self._invalidate_cache_locked()

    async def rebuild(self, rows: List[Dict[str, str]]) -> Dict[str, int]:
        start = time.time()
        clean_rows = [
            {
                "id": str(row.get("id") or "").strip(),
                "vector_text": str(row.get("vector_text") or "").strip(),
            }
            for row in (rows or [])
        ]
        clean_rows = [row for row in clean_rows if row["id"] and row["vector_text"]]
        with self._lock:
            self._clear_locked()
        if clean_rows:
            await self.upsert_rows(clean_rows)
        return {"total": len(clean_rows), "cost_ms": int((time.time() - start) * 1000)}

    async def sync_rows(self, rows: List[Dict[str, str]], *, force_rebuild: bool = False) -> Dict[str, int]:
        clean_rows = [
            {
                "id": str(row.get("id") or "").strip(),
                "vector_text": str(row.get("vector_text") or "").strip(),
            }
            for row in (rows or [])
        ]
        clean_rows = [row for row in clean_rows if row["id"] and row["vector_text"]]
        sql_ids = {row["id"] for row in clean_rows}

        with self._lock:
            meta_compatible = self._is_meta_compatible_locked()
            vector_total_before = self._count_rows_locked()
        vector_ids = set(self.list_ids())
        needs_rebuild = bool(force_rebuild or not meta_compatible)

        if needs_rebuild:
            result = await self.rebuild(clean_rows)
            return {
                "sql_total": len(clean_rows),
                "vector_total_before": vector_total_before,
                "vector_total": result["total"],
                "missing": result["total"],
                "orphan": 0,
                "changed": 0,
                "migrated": result["total"],
                "deleted": 0,
                "rebuilt": 1,
                "failed": 0,
            }

        missing_ids = sql_ids - vector_ids
        orphan_ids = vector_ids - sql_ids
        changed_rows = []
        text_map = {row["id"]: row["vector_text"] for row in clean_rows}
        with self._connect() as conn:
            existing_rows = conn.execute("SELECT item_id, vector_text FROM vector_rows").fetchall()
        for row in existing_rows:
            item_id = str(row["item_id"])
            if item_id in sql_ids and str(row["vector_text"] or "") != text_map.get(item_id, ""):
                changed_rows.append(item_id)

        upsert_ids = missing_ids | set(changed_rows)
        add_rows = [row for row in clean_rows if row["id"] in upsert_ids]
        if add_rows:
            await self.upsert_rows(add_rows)
        deleted = self.delete(list(orphan_ids)) if orphan_ids else 0

        return {
            "sql_total": len(clean_rows),
            "vector_total_before": vector_total_before,
            "vector_total": self._count_rows_locked(),
            "missing": len(missing_ids),
            "orphan": len(orphan_ids),
            "changed": len(changed_rows),
            "migrated": len(add_rows),
            "deleted": int(deleted),
            "rebuilt": 0,
            "failed": 0,
        }

    def _count_rows_locked(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) FROM vector_rows").fetchone()
        return int(row[0] if row else 0)

    def delete(self, ids: List[str], **_: Any) -> int:
        clean_ids = [str(item_id).strip() for item_id in (ids or []) if str(item_id).strip()]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    f"DELETE FROM vector_rows WHERE item_id IN ({placeholders})",
                    tuple(clean_ids),
                )
                conn.commit()
            self._invalidate_cache_locked()
        return int(cur.rowcount or 0)

    def list_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_id FROM vector_rows ORDER BY updated_at ASC, item_id ASC"
            ).fetchall()
        return [str(row["item_id"]) for row in rows]

    def get(self, limit: Optional[int] = None, offset: int = 0, include: Optional[List[str]] = None) -> Dict[str, Any]:
        del include
        sql = "SELECT item_id FROM vector_rows ORDER BY updated_at ASC, item_id ASC"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (max(0, int(limit)), max(0, int(offset)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {"ids": [str(row["item_id"]) for row in rows]}

    def _load_matrix_locked(self) -> Tuple[List[str], np.ndarray]:
        if self._matrix_cache is not None:
            return self._matrix_cache

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT item_id, embedding_blob, dimension
                FROM vector_rows
                ORDER BY updated_at ASC, item_id ASC
                """
            ).fetchall()

        ids: List[str] = []
        vectors: List[np.ndarray] = []
        expected_dimension: Optional[int] = None
        for row in rows:
            dimension = int(row["dimension"] or 0)
            vector = self._decode_vector(row["embedding_blob"], dimension)
            if vector is None:
                continue
            if expected_dimension is None:
                expected_dimension = int(vector.size)
            if int(vector.size) != expected_dimension:
                continue
            ids.append(str(row["item_id"]))
            vectors.append(vector)

        if not vectors:
            matrix = np.empty((0, 0), dtype="float32")
        else:
            matrix = np.vstack(vectors).astype("float32", copy=False)
        self._matrix_cache = (ids, matrix)
        return self._matrix_cache

    async def search(
        self,
        *,
        query: str,
        limit: int = 10,
        vector: Optional[List[float]] = None,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        query_vector = vector
        if query_vector is None:
            embeddings = await self._embed_documents([str(query or "").strip()], is_query=True)
            if not embeddings:
                return []
            query_vector = embeddings[0]
        if query_vector is None:
            return []

        vector_arr = self._normalize(np.asarray(query_vector, dtype="float32")).reshape(-1)
        with self._lock:
            if not self._is_meta_compatible_locked(int(vector_arr.size)):
                self.logger.warning(
                    "[SQLite向量索引] 查询跳过：索引模型或维度与当前配置不一致 "
                    f"index={self.name} provider_id={self.provider_id} embedding_model={self.model_key}"
                )
                return []
            ids, matrix = self._load_matrix_locked()

        if not ids or matrix.size == 0:
            return []
        if int(matrix.shape[1]) != int(vector_arr.size):
            self.logger.warning(
                f"[SQLite向量索引] 查询跳过：向量维度不匹配 index={self.name} "
                f"query_dim={vector_arr.size} index_dim={matrix.shape[1]}"
            )
            return []

        scores = matrix @ vector_arr
        if scores.size == 0:
            return []
        top_k = min(max(1, int(limit)), int(scores.size))
        order = np.argsort(scores)[::-1][:top_k]

        recalled: List[Tuple[str, float]] = []
        for idx in order.tolist():
            score = float(scores[idx])
            if score < float(similarity_threshold):
                continue
            recalled.append((ids[idx], score))
        return recalled


class SqliteVectorStore:
    """SQLite-backed exact cosine vector store for memory_index."""

    backend_name = "sqlite"

    def __init__(
        self,
        *,
        embedding_provider: Any,
        index_dir: Path,
        provider_id: str,
        rerank_provider: Optional[Any] = None,
    ):
        if embedding_provider is None:
            raise ValueError("SqliteVectorStore需要有效的embedding_provider")
        self.embedding_provider = embedding_provider
        self.index_dir = Path(index_dir)
        self.provider_id = str(provider_id or "unknown").strip() or "unknown"
        self.rerank_provider = rerank_provider
        self.logger = logger
        self.client = f"sqlite-vector:{self.index_dir}"
        self._collections: Dict[str, SqliteTextIndex] = {}
        self._lock = threading.RLock()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_key = self._build_model_key()
        self.logger.info(
            "[SQLite向量索引] 完成 "
            f"provider_id={self.provider_id} embedding_model={self.model_key} index_dir={self.index_dir}"
        )

    def _build_model_key(self) -> str:
        try:
            info = self.embedding_provider.get_model_info() or {}
        except Exception:
            info = {}
        parts = [
            str(info.get("provider_id") or "").strip(),
            str(info.get("model_name") or info.get("model") or "").strip(),
            str(info.get("dimension") or "").strip(),
        ]
        key = "|".join([part for part in parts if part])
        return key or self.provider_id

    def get_or_create_collection_with_dimension_check(self, name: str) -> SqliteTextIndex:
        safe_name = str(name or "").strip()
        if not safe_name:
            raise ValueError("SQLite向量集合名称不能为空")
        with self._lock:
            existing = self._collections.get(safe_name)
            if existing is not None:
                return existing
            collection = SqliteTextIndex(
                name=safe_name,
                db_path=self.index_dir / f"{safe_name}.sqlite",
                embedding_provider=self.embedding_provider,
                provider_id=self.provider_id,
                model_key=self.model_key,
                log=self.logger,
            )
            self._collections[safe_name] = collection
            return collection

    def clear_collection(self, collection: SqliteTextIndex) -> None:
        collection.clear()

    async def embed_documents(
        self,
        documents: List[str],
        is_query: bool = False,
        timeout: int = 3,
    ) -> Optional[List[List[float]]]:
        texts = [str(doc or "").strip() for doc in (documents or [])]
        if not texts:
            return []
        if is_query:
            if any(not text for text in texts):
                return None
            try:
                return await asyncio.wait_for(
                    self.embedding_provider.embed_documents(texts),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"[SQLite向量索引] 查询向量化超时 timeout={timeout}s")
                return None
            except Exception as e:
                self.logger.warning(f"[SQLite向量索引] 查询向量化失败 异常={e}")
                return None
        return await self.embedding_provider.embed_documents(texts)

    async def embed_single_document(
        self,
        document: str,
        is_query: bool = False,
        timeout: int = 3,
    ) -> Optional[List[float]]:
        text = str(document or "").strip()
        if not text:
            return None
        embeddings = await self.embed_documents([text], is_query=is_query, timeout=timeout)
        if embeddings:
            return embeddings[0]
        return None

    async def upsert_memory_index_rows(
        self,
        collection: SqliteTextIndex,
        rows: List[Dict[str, str]],
    ) -> None:
        await collection.upsert_rows(rows)

    async def recall_memory_ids(
        self,
        collection: SqliteTextIndex,
        query: str,
        limit: int = 10,
        vector: Optional[List[float]] = None,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        return await collection.search(
            query=query,
            limit=limit,
            vector=vector,
            similarity_threshold=similarity_threshold,
        )
