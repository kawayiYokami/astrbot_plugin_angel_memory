from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from ..config.system_config import system_config
from ..models.data_models import BaseMemory, MemoryType, ValidationError


class MemorySqlManager:
    """SimpleMemory 的 SQL 存储管理器。"""

    def __init__(self, db_path: Path):
        self.logger = logger
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tag_names: set[str] = set()

        self._init_db()
        self._load_tag_cache()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    judgment TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    strength INTEGER NOT NULL,
                    is_active INTEGER NOT NULL,
                    memory_scope TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS global_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS memory_tag_rel (
                    memory_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    UNIQUE(memory_id, tag_id)
                );

                CREATE TABLE IF NOT EXISTS note_tag_rel (
                    source_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    UNIQUE(source_id, tag_id)
                );
                
                CREATE TABLE IF NOT EXISTS note_index_records (
                    source_id TEXT PRIMARY KEY,
                    note_short_id INTEGER UNIQUE,
                    file_id TEXT NOT NULL,
                    source_file_path TEXT NOT NULL,
                    heading_h1 TEXT,
                    heading_h2 TEXT,
                    heading_h3 TEXT,
                    heading_h4 TEXT,
                    heading_h5 TEXT,
                    heading_h6 TEXT,
                    total_lines INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_scope_created_at
                    ON memory_records(memory_scope, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_active_strength
                    ON memory_records(is_active, strength);
                CREATE INDEX IF NOT EXISTS idx_memory_judgment
                    ON memory_records(judgment);
                CREATE INDEX IF NOT EXISTS idx_memory_tag_rel_tag_memory
                    ON memory_tag_rel(tag_id, memory_id);
                CREATE INDEX IF NOT EXISTS idx_note_tag_rel_tag_source
                    ON note_tag_rel(tag_id, source_id);
                CREATE INDEX IF NOT EXISTS idx_note_index_file_id
                    ON note_index_records(file_id);
                CREATE INDEX IF NOT EXISTS idx_note_index_path
                    ON note_index_records(source_file_path);
                CREATE INDEX IF NOT EXISTS idx_note_index_short_id
                    ON note_index_records(note_short_id);

                CREATE TABLE IF NOT EXISTS note_short_id_seq (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    next_id INTEGER NOT NULL
                );
                """
            )
            # 兼容迁移：若旧表 memory_tags 存在，则迁移到 global_tags
            has_legacy = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_tags'"
            ).fetchone()
            if has_legacy:
                cur.execute(
                    "INSERT OR IGNORE INTO global_tags(name) SELECT name FROM memory_tags"
                )
            # 兼容迁移：note_index_records 补充 total_lines 列
            note_columns = cur.execute("PRAGMA table_info(note_index_records)").fetchall()
            column_names = {str(row[1]) for row in note_columns}
            if "note_short_id" not in column_names:
                cur.execute("ALTER TABLE note_index_records ADD COLUMN note_short_id INTEGER")
            if "total_lines" not in column_names:
                cur.execute(
                    "ALTER TABLE note_index_records ADD COLUMN total_lines INTEGER NOT NULL DEFAULT 0"
                )
            # 初始化短ID序列
            cur.execute("INSERT OR IGNORE INTO note_short_id_seq(id, next_id) VALUES (1, 0)")
            # 兼容迁移：给历史 note_index_records 补 note_short_id（从0开始）
            rows_without_short_id = cur.execute(
                """
                SELECT source_id
                FROM note_index_records
                WHERE note_short_id IS NULL
                ORDER BY source_id ASC
                """
            ).fetchall()
            if rows_without_short_id:
                seq_row = cur.execute(
                    "SELECT next_id FROM note_short_id_seq WHERE id = 1"
                ).fetchone()
                next_id = int(seq_row[0]) if seq_row else 0
                for row in rows_without_short_id:
                    cur.execute(
                        "UPDATE note_index_records SET note_short_id = ? WHERE source_id = ?",
                        (next_id, str(row[0])),
                    )
                    next_id += 1
                cur.execute(
                    "UPDATE note_short_id_seq SET next_id = ? WHERE id = 1",
                    (next_id,),
                )
            conn.commit()

    def _load_tag_cache(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM global_tags").fetchall()
            with self._lock:
                self._tag_names = {str(row["name"]) for row in rows if row["name"]}
        self.logger.info(f"SimpleMemory 标签缓存加载完成: {len(self._tag_names)} 个")

    @staticmethod
    def _normalize_scope(memory_scope: str) -> str:
        scope = str(memory_scope or "").strip()
        if not scope:
            raise ValidationError("memory_scope 为空，拒绝执行")
        return scope

    @staticmethod
    def _normalize_tags(tags: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for tag in tags or []:
            t = str(tag).strip()
            if not t or t in seen:
                continue
            seen.add(t)
            normalized.append(t)
        return normalized

    @staticmethod
    def _to_memory_type(memory_type: str) -> MemoryType:
        raw = str(memory_type or "").strip()
        mapping = {
            "knowledge": MemoryType.KNOWLEDGE,
            "event": MemoryType.EVENT,
            "skill": MemoryType.SKILL,
            "task": MemoryType.TASK,
            "emotional": MemoryType.EMOTIONAL,
        }
        if raw in mapping:
            return mapping[raw]
        try:
            return MemoryType(raw)
        except ValueError:
            return MemoryType.KNOWLEDGE

    @staticmethod
    def _scope_sql(memory_scope: str, alias: str = "mr") -> Tuple[str, List[str]]:
        scope = MemorySqlManager._normalize_scope(memory_scope)
        if scope == "public":
            return f"{alias}.memory_scope = ?", ["public"]
        return f"({alias}.memory_scope = ? OR {alias}.memory_scope = 'public')", [scope]

    @staticmethod
    def _row_to_memory(row: sqlite3.Row, tags: List[str]) -> BaseMemory:
        return BaseMemory(
            memory_type=MemorySqlManager._to_memory_type(row["memory_type"]),
            judgment=row["judgment"],
            reasoning=row["reasoning"],
            tags=tags,
            id=row["id"],
            strength=int(row["strength"]),
            is_active=bool(row["is_active"]),
            created_at=float(row["created_at"]),
            memory_scope=row["memory_scope"],
        )

    def _fetch_tags_for_memory_ids(
        self, conn: sqlite3.Connection, memory_ids: Sequence[str]
    ) -> Dict[str, List[str]]:
        if not memory_ids:
            return {}
        placeholders = ",".join(["?" for _ in memory_ids])
        sql = f"""
            SELECT mtr.memory_id AS memory_id, mt.name AS tag_name
            FROM memory_tag_rel mtr
            JOIN global_tags mt ON mt.id = mtr.tag_id
            WHERE mtr.memory_id IN ({placeholders})
        """
        rows = conn.execute(sql, tuple(memory_ids)).fetchall()
        result: Dict[str, List[str]] = {mid: [] for mid in memory_ids}
        for row in rows:
            memory_id = str(row["memory_id"])
            tag_name = str(row["tag_name"])
            result.setdefault(memory_id, []).append(tag_name)
        return result

    def _upsert_tags_and_bind(
        self, conn: sqlite3.Connection, memory_id: str, tags: List[str]
    ) -> None:
        if tags:
            conn.executemany(
                "INSERT OR IGNORE INTO global_tags(name) VALUES (?)",
                [(tag,) for tag in tags],
            )
            placeholders = ",".join(["?" for _ in tags])
            rows = conn.execute(
                f"SELECT id, name FROM global_tags WHERE name IN ({placeholders})",
                tuple(tags),
            ).fetchall()
            tag_id_by_name = {str(row["name"]): int(row["id"]) for row in rows}

            rel_rows = [
                (memory_id, tag_id_by_name[tag])
                for tag in tags
                if tag in tag_id_by_name
            ]
            if rel_rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_tag_rel(memory_id, tag_id) VALUES (?, ?)",
                    rel_rows,
                )
            with self._lock:
                self._tag_names.update(tags)

    def _replace_memory_tags(
        self, conn: sqlite3.Connection, memory_id: str, tags: List[str]
    ) -> None:
        conn.execute("DELETE FROM memory_tag_rel WHERE memory_id = ?", (memory_id,))
        self._upsert_tags_and_bind(conn, memory_id, tags)

    def _get_or_create_tag_ids_sync(self, tag_names: List[str]) -> List[int]:
        normalized = self._normalize_tags(tag_names)
        if not normalized:
            return []
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO global_tags(name) VALUES (?)",
                [(tag,) for tag in normalized],
            )
            placeholders = ",".join(["?" for _ in normalized])
            rows = conn.execute(
                f"SELECT id, name FROM global_tags WHERE name IN ({placeholders})",
                tuple(normalized),
            ).fetchall()
        with self._lock:
            self._tag_names.update(normalized)
        by_name = {str(row["name"]): int(row["id"]) for row in rows}
        return [by_name[tag] for tag in normalized if tag in by_name]

    def _get_or_create_tag_ids_with_conn(
        self, conn: sqlite3.Connection, tag_names: List[str]
    ) -> List[int]:
        normalized = self._normalize_tags(tag_names)
        if not normalized:
            return []
        conn.executemany(
            "INSERT OR IGNORE INTO global_tags(name) VALUES (?)",
            [(tag,) for tag in normalized],
        )
        placeholders = ",".join(["?" for _ in normalized])
        rows = conn.execute(
            f"SELECT id, name FROM global_tags WHERE name IN ({placeholders})",
            tuple(normalized),
        ).fetchall()
        with self._lock:
            self._tag_names.update(normalized)
        by_name = {str(row["name"]): int(row["id"]) for row in rows}
        return [by_name[tag] for tag in normalized if tag in by_name]

    async def get_or_create_tag_ids(self, tag_names: List[str]) -> List[int]:
        return await asyncio.to_thread(self._get_or_create_tag_ids_sync, tag_names)

    async def bind_note_tags(self, source_id: str, tag_names: List[str]) -> None:
        await asyncio.to_thread(self._bind_note_tags_sync, source_id, tag_names)

    def bind_note_tags_sync(self, source_id: str, tag_names: List[str]) -> None:
        self._bind_note_tags_sync(source_id, tag_names)

    def _bind_note_tags_sync(self, source_id: str, tag_names: List[str]) -> None:
        sid = str(source_id or "").strip()
        if not sid:
            return
        tag_ids = self._get_or_create_tag_ids_sync(tag_names)
        with self._connect() as conn:
            conn.execute("DELETE FROM note_tag_rel WHERE source_id = ?", (sid,))
            if tag_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO note_tag_rel(source_id, tag_id) VALUES (?, ?)",
                    [(sid, tid) for tid in tag_ids],
                )
            conn.commit()

    def _bind_note_tags_with_conn(
        self, conn: sqlite3.Connection, source_id: str, tag_names: List[str]
    ) -> None:
        sid = str(source_id or "").strip()
        if not sid:
            return
        tag_ids = self._get_or_create_tag_ids_with_conn(conn, tag_names)
        conn.execute("DELETE FROM note_tag_rel WHERE source_id = ?", (sid,))
        if tag_ids:
            conn.executemany(
                "INSERT OR IGNORE INTO note_tag_rel(source_id, tag_id) VALUES (?, ?)",
                [(sid, tid) for tid in tag_ids],
            )

    @staticmethod
    def _safe_heading(value: Any) -> str:
        text = str(value or "").strip()
        return text[:500] if text else ""

    @staticmethod
    def _build_note_vector_text(tags: List[str]) -> str:
        # notes_index 仅使用 tags 生成向量文本，避免正文/路径语义噪声
        deduped = list(dict.fromkeys([str(t).strip() for t in tags if str(t).strip()]))
        return " ".join(deduped).strip()

    @staticmethod
    def _alloc_next_note_short_id(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT next_id FROM note_short_id_seq WHERE id = 1"
        ).fetchone()
        next_id = int(row["next_id"]) if row else 0
        conn.execute(
            "UPDATE note_short_id_seq SET next_id = ? WHERE id = 1",
            (next_id + 1,),
        )
        return next_id

    async def upsert_note_index_entries(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        return await asyncio.to_thread(self._upsert_note_index_entries_sync, entries)

    def _upsert_note_index_entries_sync(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        scanned = len(entries or [])
        upserted = 0
        failed = 0
        with self._connect() as conn:
            for item in entries or []:
                try:
                    source_id = str(item.get("source_id") or "").strip()
                    file_id = str(item.get("file_id") or "").strip()
                    source_file_path = str(item.get("source_file_path") or "").strip()
                    if not source_id or not file_id or not source_file_path:
                        continue
                    existing = conn.execute(
                        "SELECT note_short_id FROM note_index_records WHERE source_id = ?",
                        (source_id,),
                    ).fetchone()
                    if existing and existing["note_short_id"] is not None:
                        note_short_id = int(existing["note_short_id"])
                    else:
                        note_short_id = self._alloc_next_note_short_id(conn)

                    headings = [self._safe_heading(item.get(f"h{i}", "")) for i in range(1, 7)]
                    tags = self._normalize_tags(item.get("tags") or [])
                    total_lines = int(item.get("total_lines") or 0)
                    updated_at = float(item.get("updated_at") or time.time())

                    conn.execute(
                        """
                        INSERT INTO note_index_records(
                            source_id, note_short_id, file_id, source_file_path,
                            heading_h1, heading_h2, heading_h3,
                            heading_h4, heading_h5, heading_h6,
                            total_lines, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_id) DO UPDATE SET
                            note_short_id=excluded.note_short_id,
                            file_id=excluded.file_id,
                            source_file_path=excluded.source_file_path,
                            heading_h1=excluded.heading_h1,
                            heading_h2=excluded.heading_h2,
                            heading_h3=excluded.heading_h3,
                            heading_h4=excluded.heading_h4,
                            heading_h5=excluded.heading_h5,
                            heading_h6=excluded.heading_h6,
                            total_lines=excluded.total_lines,
                            updated_at=excluded.updated_at
                        """,
                        (
                            source_id,
                            note_short_id,
                            file_id,
                            source_file_path,
                            headings[0],
                            headings[1],
                            headings[2],
                            headings[3],
                            headings[4],
                            headings[5],
                            total_lines,
                            updated_at,
                        ),
                    )
                    # 使用同一连接写入 tags，避免事务内嵌套连接导致 SQLite 锁冲突
                    self._bind_note_tags_with_conn(conn, source_id, tags)
                    upserted += 1
                except Exception:
                    failed += 1
                    self.logger.exception("笔记索引写入失败")
            conn.commit()
        return {"scanned": scanned, "upserted": upserted, "failed": failed}

    async def delete_note_index_by_file_id(self, file_id: str) -> List[str]:
        return await asyncio.to_thread(self._delete_note_index_by_file_id_sync, file_id)

    def _delete_note_index_by_file_id_sync(self, file_id: str) -> List[str]:
        fid = str(file_id or "").strip()
        if not fid:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_id FROM note_index_records WHERE file_id = ?",
                (fid,),
            ).fetchall()
            source_ids = [str(row["source_id"]) for row in rows]
            if source_ids:
                placeholders = ",".join(["?" for _ in source_ids])
                conn.execute(
                    f"DELETE FROM note_tag_rel WHERE source_id IN ({placeholders})",
                    tuple(source_ids),
                )
            conn.execute("DELETE FROM note_index_records WHERE file_id = ?", (fid,))
            conn.commit()
        return source_ids

    async def get_note_index_by_source_ids(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_note_index_by_source_ids_sync, source_ids)

    def _get_note_index_by_source_ids_sync(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        ids = [str(sid).strip() for sid in (source_ids or []) if str(sid).strip()]
        if not ids:
            return []
        placeholders = ",".join(["?" for _ in ids])
        sql = f"""
            SELECT
                nir.source_id,
                nir.note_short_id,
                nir.file_id,
                nir.source_file_path,
                nir.heading_h1,
                nir.heading_h2,
                nir.heading_h3,
                nir.heading_h4,
                nir.heading_h5,
                nir.heading_h6,
                nir.total_lines,
                nir.updated_at,
                IFNULL(tags.tags_text, '') AS tags_text
            FROM note_index_records nir
            LEFT JOIN (
                SELECT
                    ntr.source_id AS source_id,
                    GROUP_CONCAT(gt.name, ', ') AS tags_text
                FROM note_tag_rel ntr
                JOIN global_tags gt ON gt.id = ntr.tag_id
                GROUP BY ntr.source_id
            ) tags ON tags.source_id = nir.source_id
            WHERE nir.source_id IN ({placeholders})
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(ids)).fetchall()
        return [dict(row) for row in rows]

    async def search_note_index_by_tags(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._search_note_index_by_tags_sync, query, limit)

    def _search_note_index_by_tags_sync(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        text = str(query or "")
        if not text.strip():
            return []
        with self._lock:
            matched_tags = [tag for tag in self._tag_names if tag and tag in text]
        if not matched_tags:
            return []

        placeholders = ",".join(["?" for _ in matched_tags])
        sql = f"""
            SELECT
                nir.source_id,
                nir.note_short_id,
                nir.file_id,
                nir.source_file_path,
                nir.heading_h1,
                nir.heading_h2,
                nir.heading_h3,
                nir.heading_h4,
                nir.heading_h5,
                nir.heading_h6,
                nir.total_lines,
                nir.updated_at,
                COUNT(DISTINCT gt.id) AS hit_count,
                CAST(nir.updated_at / 86400 AS INTEGER) AS day_bucket,
                IFNULL(tags.tags_text, '') AS tags_text
            FROM note_index_records nir
            JOIN note_tag_rel ntr ON ntr.source_id = nir.source_id
            JOIN global_tags gt ON gt.id = ntr.tag_id
            LEFT JOIN (
                SELECT
                    ntr2.source_id AS source_id,
                    GROUP_CONCAT(gt2.name, ', ') AS tags_text
                FROM note_tag_rel ntr2
                JOIN global_tags gt2 ON gt2.id = ntr2.tag_id
                GROUP BY ntr2.source_id
            ) tags ON tags.source_id = nir.source_id
            WHERE gt.name IN ({placeholders})
            GROUP BY nir.source_id
            ORDER BY hit_count DESC, day_bucket DESC, nir.source_file_path ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple([*matched_tags, int(limit)])).fetchall()
        return [dict(row) for row in rows]

    async def list_note_index_vector_rows(self) -> List[Dict[str, str]]:
        return await asyncio.to_thread(self._list_note_index_vector_rows_sync)

    def _list_note_index_vector_rows_sync(self) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    nir.source_id,
                    nir.note_short_id,
                    nir.source_file_path,
                    nir.heading_h1,
                    nir.heading_h2,
                    nir.heading_h3,
                    nir.heading_h4,
                    nir.heading_h5,
                    nir.heading_h6,
                    nir.total_lines,
                    IFNULL(tags.tags_text, '') AS tags_text
                FROM note_index_records nir
                LEFT JOIN (
                    SELECT
                        ntr.source_id AS source_id,
                        GROUP_CONCAT(gt.name, ' ') AS tags_text
                    FROM note_tag_rel ntr
                    JOIN global_tags gt ON gt.id = ntr.tag_id
                    GROUP BY ntr.source_id
                ) tags ON tags.source_id = nir.source_id
                """
            ).fetchall()
        result: List[Dict[str, str]] = []
        for row in rows:
            tags = [t for t in str(row["tags_text"] or "").split(" ") if t]
            vector_text = self._build_note_vector_text(tags=tags)
            if not vector_text:
                continue
            result.append({"id": str(row["source_id"] or ""), "vector_text": vector_text})
        return [r for r in result if r["id"]]

    async def get_note_index_by_short_id(self, note_short_id: int) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_note_index_by_short_id_sync, note_short_id)

    def _get_note_index_by_short_id_sync(self, note_short_id: int) -> Optional[Dict[str, Any]]:
        sid = int(note_short_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    nir.source_id,
                    nir.note_short_id,
                    nir.file_id,
                    nir.source_file_path,
                    nir.heading_h1,
                    nir.heading_h2,
                    nir.heading_h3,
                    nir.heading_h4,
                    nir.heading_h5,
                    nir.heading_h6,
                    nir.total_lines,
                    nir.updated_at
                FROM note_index_records nir
                WHERE nir.note_short_id = ?
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
        return dict(row) if row else None

    async def find_matched_tag_ids(self, query_text: str) -> List[int]:
        return await asyncio.to_thread(self._find_matched_tag_ids_sync, query_text)

    def _find_matched_tag_ids_sync(self, query_text: str) -> List[int]:
        text = str(query_text or "")
        if not text.strip():
            return []
        with self._lock:
            matched_names = [name for name in self._tag_names if name and name in text]
        if not matched_names:
            return []
        placeholders = ",".join(["?" for _ in matched_names])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id FROM global_tags WHERE name IN ({placeholders})",
                tuple(matched_names),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def search_memory_ids_by_tag_ids(self, tag_ids: List[int]) -> List[str]:
        return await asyncio.to_thread(self._search_memory_ids_by_tag_ids_sync, tag_ids)

    def _search_memory_ids_by_tag_ids_sync(self, tag_ids: List[int]) -> List[str]:
        ids = [int(tid) for tid in (tag_ids or [])]
        if not ids:
            return []
        placeholders = ",".join(["?" for _ in ids])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id
                FROM memory_tag_rel
                WHERE tag_id IN ({placeholders})
                GROUP BY memory_id
                ORDER BY COUNT(DISTINCT tag_id) DESC
                """,
                tuple(ids),
            ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    async def search_note_source_ids_by_tag_ids(self, tag_ids: List[int]) -> List[str]:
        return await asyncio.to_thread(self._search_note_source_ids_by_tag_ids_sync, tag_ids)

    def _search_note_source_ids_by_tag_ids_sync(self, tag_ids: List[int]) -> List[str]:
        ids = [int(tid) for tid in (tag_ids or [])]
        if not ids:
            return []
        placeholders = ",".join(["?" for _ in ids])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT source_id
                FROM note_tag_rel
                WHERE tag_id IN ({placeholders})
                GROUP BY source_id
                ORDER BY COUNT(DISTINCT tag_id) DESC
                """,
                tuple(ids),
            ).fetchall()
        return [str(row["source_id"]) for row in rows]

    async def unified_tag_hit_search(self, query_text: str) -> Dict[str, Any]:
        """
        统一 tags 命中入口：一次解析，返回命中 tag_ids + 记忆ID + 笔记source_id。
        """
        return await asyncio.to_thread(self._unified_tag_hit_search_sync, query_text)

    def _unified_tag_hit_search_sync(self, query_text: str) -> Dict[str, Any]:
        tag_ids = self._find_matched_tag_ids_sync(query_text)
        memory_ids = self._search_memory_ids_by_tag_ids_sync(tag_ids)
        note_source_ids = self._search_note_source_ids_by_tag_ids_sync(tag_ids)
        return {
            "tag_ids": tag_ids,
            "memory_ids": memory_ids,
            "note_source_ids": note_source_ids,
        }

    async def remember(
        self,
        memory_type: str,
        judgment: str,
        reasoning: str,
        tags: List[str],
        is_active: bool = False,
        strength: Optional[int] = None,
        memory_scope: str = "public",
    ) -> BaseMemory:
        return await asyncio.to_thread(
            self._remember_sync,
            memory_type,
            judgment,
            reasoning,
            tags,
            is_active,
            strength,
            memory_scope,
        )

    async def upsert_memory(self, memory: BaseMemory) -> BaseMemory:
        """
        用指定ID将记忆写入中央库（用于向量模式镜像写入）。
        """
        return await asyncio.to_thread(self._upsert_memory_sync, memory)

    def _remember_sync(
        self,
        memory_type: str,
        judgment: str,
        reasoning: str,
        tags: List[str],
        is_active: bool = False,
        strength: Optional[int] = None,
        memory_scope: str = "public",
    ) -> BaseMemory:
        scope = self._normalize_scope(memory_scope)
        normalized_tags = self._normalize_tags(tags)
        now = time.time()
        memory = BaseMemory(
            memory_type=self._to_memory_type(memory_type),
            judgment=str(judgment or "").strip(),
            reasoning=str(reasoning or "").strip(),
            tags=normalized_tags,
            id=str(uuid.uuid4()),
            strength=int(
                strength
                if strength is not None
                else system_config.default_passive_strength
            ),
            is_active=bool(is_active),
            created_at=now,
            memory_scope=scope,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_records(
                    id, memory_type, judgment, reasoning, strength, is_active,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.memory_type.value,
                    memory.judgment,
                    memory.reasoning,
                    memory.strength,
                    1 if memory.is_active else 0,
                    scope,
                    memory.created_at,
                    now,
                ),
            )
            self._upsert_tags_and_bind(conn, memory.id, normalized_tags)
            conn.commit()

        return memory

    def _upsert_memory_sync(self, memory: BaseMemory) -> BaseMemory:
        scope = self._normalize_scope(getattr(memory, "memory_scope", "public"))
        normalized_tags = self._normalize_tags(getattr(memory, "tags", []))
        now = time.time()
        memory_id = str(getattr(memory, "id", "") or uuid.uuid4())
        created_at = float(getattr(memory, "created_at", now) or now)
        memory_type = getattr(getattr(memory, "memory_type", None), "value", None)
        if not memory_type:
            memory_type = "知识记忆"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_records(
                    id, memory_type, judgment, reasoning, strength, is_active,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    memory_type=excluded.memory_type,
                    judgment=excluded.judgment,
                    reasoning=excluded.reasoning,
                    strength=excluded.strength,
                    is_active=excluded.is_active,
                    memory_scope=excluded.memory_scope,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    str(memory_type),
                    str(getattr(memory, "judgment", "") or "").strip(),
                    str(getattr(memory, "reasoning", "") or "").strip(),
                    int(getattr(memory, "strength", 1) or 1),
                    1 if bool(getattr(memory, "is_active", False)) else 0,
                    scope,
                    created_at,
                    now,
                ),
            )
            self._replace_memory_tags(conn, memory_id, normalized_tags)
            conn.commit()
        return memory

    async def upsert_memories_by_judgment(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        return await asyncio.to_thread(self._upsert_memories_by_judgment_sync, memories)

    def _upsert_memories_by_judgment_sync(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        按 judgment 幂等写入（同 judgment 仅保留最新 created_at）。

        Returns:
            {"scanned": int, "deduped": int, "upserted": int, "failed": int}
        """
        scanned = len(memories or [])
        deduped_map: Dict[str, Dict[str, Any]] = {}
        failed = 0

        for item in memories or []:
            judgment = str(item.get("judgment") or "").strip()
            if not judgment:
                continue
            try:
                created_at = float(item.get("created_at") or 0)
            except (TypeError, ValueError):
                created_at = 0.0

            current = deduped_map.get(judgment)
            if current is None:
                deduped_map[judgment] = dict(item)
                deduped_map[judgment]["created_at"] = created_at
                continue

            current_created = float(current.get("created_at") or 0)
            if created_at >= current_created:
                deduped_map[judgment] = dict(item)
                deduped_map[judgment]["created_at"] = created_at

        upserted = 0
        now = time.time()

        with self._connect() as conn:
            for judgment, raw in deduped_map.items():
                try:
                    created_at = float(raw.get("created_at") or now)
                    normalized_tags = self._normalize_tags(BaseMemory._parse_tags(raw.get("tags", [])))
                    reasoning = str(raw.get("reasoning") or "").strip()
                    memory_type = str(raw.get("memory_type") or "知识记忆").strip() or "知识记忆"
                    strength = int(raw.get("strength", 1) or 1)
                    is_active = 1 if bool(raw.get("is_active", False)) else 0
                    memory_scope = self._normalize_scope(raw.get("memory_scope", "public"))

                    existing_rows = conn.execute(
                        """
                        SELECT id, created_at
                        FROM memory_records
                        WHERE judgment = ?
                        ORDER BY created_at DESC, updated_at DESC
                        """,
                        (judgment,),
                    ).fetchall()

                    if existing_rows:
                        keep_id = str(existing_rows[0]["id"])
                        existing_created_at = float(existing_rows[0]["created_at"] or 0)
                        if created_at >= existing_created_at:
                            conn.execute(
                                """
                                UPDATE memory_records
                                SET memory_type = ?,
                                    reasoning = ?,
                                    strength = ?,
                                    is_active = ?,
                                    memory_scope = ?,
                                    created_at = ?,
                                    updated_at = ?
                                WHERE id = ?
                                """,
                                (
                                    memory_type,
                                    reasoning,
                                    strength,
                                    is_active,
                                    memory_scope,
                                    created_at,
                                    now,
                                    keep_id,
                                ),
                            )
                            self._replace_memory_tags(conn, keep_id, normalized_tags)
                            upserted += 1

                        duplicate_ids = [str(row["id"]) for row in existing_rows[1:]]
                        if duplicate_ids:
                            placeholders = ",".join(["?" for _ in duplicate_ids])
                            conn.execute(
                                f"DELETE FROM memory_records WHERE id IN ({placeholders})",
                                tuple(duplicate_ids),
                            )
                            conn.execute(
                                f"DELETE FROM memory_tag_rel WHERE memory_id IN ({placeholders})",
                                tuple(duplicate_ids),
                            )
                    else:
                        # 中央记忆库ID必须由本项目生成，禁止复用外部传入ID。
                        memory_id = str(uuid.uuid4())
                        conn.execute(
                            """
                            INSERT INTO memory_records(
                                id, memory_type, judgment, reasoning, strength, is_active,
                                memory_scope, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                memory_id,
                                memory_type,
                                judgment,
                                reasoning,
                                strength,
                                is_active,
                                memory_scope,
                                created_at,
                                now,
                            ),
                        )
                        self._replace_memory_tags(conn, memory_id, normalized_tags)
                        upserted += 1

                except Exception:
                    failed += 1
                    self.logger.exception("SimpleMemory 备份写入失败 (judgment=%s)", judgment)

            conn.execute(
                """
                DELETE FROM memory_tag_rel
                WHERE memory_id NOT IN (SELECT id FROM memory_records)
                """
            )
            conn.commit()

        return {
            "scanned": scanned,
            "deduped": len(deduped_map),
            "upserted": upserted,
            "failed": failed,
        }

    async def list_all_memories_for_vector_sync(self) -> List[Dict[str, Any]]:
        """导出 simple 库全部记忆（含 tags）用于向量回灌。"""
        return await asyncio.to_thread(self._list_all_memories_for_vector_sync_sync)

    def _list_all_memories_for_vector_sync_sync(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            sql = """
                SELECT
                    mr.id,
                    mr.memory_type,
                    mr.judgment,
                    mr.reasoning,
                    mr.strength,
                    mr.is_active,
                    mr.memory_scope,
                    mr.created_at,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                LEFT JOIN (
                    SELECT
                        mtr.memory_id AS memory_id,
                        GROUP_CONCAT(mt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr
                    JOIN global_tags mt ON mt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                ORDER BY mr.created_at DESC
            """
            rows = conn.execute(sql).fetchall()

        records: List[Dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "id": row["id"],
                    "memory_type": row["memory_type"],
                    "judgment": row["judgment"],
                    "reasoning": row["reasoning"],
                    "strength": int(row["strength"] or 1),
                    "is_active": bool(row["is_active"]),
                    "memory_scope": row["memory_scope"] or "public",
                    "created_at": float(row["created_at"] or 0),
                    "tags": BaseMemory._parse_tags(row["tags"] or ""),
                }
            )
        return records

    @staticmethod
    def _build_vector_text(judgment: str, tags: List[str]) -> str:
        """构建轻量向量索引文本。"""
        safe_judgment = str(judgment or "").strip()
        safe_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        if safe_tags:
            return f"{safe_judgment} {' '.join(safe_tags)}".strip()
        return safe_judgment

    async def list_memory_index_rows(self) -> List[Dict[str, str]]:
        """
        导出记忆轻量向量索引数据（仅 id + vector_text）。
        """
        return await asyncio.to_thread(self._list_memory_index_rows_sync)

    def _list_memory_index_rows_sync(self) -> List[Dict[str, str]]:
        with self._connect() as conn:
            sql = """
                SELECT
                    mr.id,
                    mr.judgment,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                LEFT JOIN (
                    SELECT
                        mtr.memory_id AS memory_id,
                        GROUP_CONCAT(gt.name, ' ') AS tags_text
                    FROM memory_tag_rel mtr
                    JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
            """
            rows = conn.execute(sql).fetchall()

        result: List[Dict[str, str]] = []
        for row in rows:
            memory_id = str(row["id"])
            judgment = str(row["judgment"] or "")
            tag_tokens = [t for t in str(row["tags"] or "").split(" ") if t]
            vector_text = self._build_vector_text(judgment, tag_tokens)
            if not vector_text:
                continue
            result.append({"id": memory_id, "vector_text": vector_text})
        return result

    async def export_backup_snapshot(self) -> Dict[str, Any]:
        """导出中央记忆库快照（JSON冷备份使用）。"""
        return await asyncio.to_thread(self._export_backup_snapshot_sync)

    def _export_backup_snapshot_sync(self) -> Dict[str, Any]:
        with self._connect() as conn:
            records_rows = conn.execute(
                """
                SELECT id, memory_type, judgment, reasoning, strength, is_active,
                       memory_scope, created_at, updated_at
                FROM memory_records
                ORDER BY created_at DESC
                """
            ).fetchall()
            tags_rows = conn.execute(
                "SELECT id, name FROM global_tags ORDER BY id ASC"
            ).fetchall()
            rel_rows = conn.execute(
                "SELECT memory_id, tag_id FROM memory_tag_rel"
            ).fetchall()

        records = [dict(row) for row in records_rows]
        tags = [dict(row) for row in tags_rows]
        relations = [dict(row) for row in rel_rows]
        return {
            "schema_version": 1,
            "exported_at": int(time.time()),
            "records": records,
            "global_tags": tags,
            "memory_tag_rel": relations,
        }

    async def recall_by_tags(
        self,
        query: str,
        limit: int,
        memory_scope: str,
    ) -> List[BaseMemory]:
        scope_sql, scope_params = self._scope_sql(memory_scope)
        text = str(query or "")
        with self._lock:
            matched_tags = [tag for tag in self._tag_names if tag and tag in text]

        if not text.strip():
            return []

        if not matched_tags:
            return []

        placeholders = ",".join(["?" for _ in matched_tags])
        sql = f"""
            SELECT
                mr.id,
                mr.memory_type,
                mr.judgment,
                mr.reasoning,
                mr.strength,
                mr.is_active,
                mr.memory_scope,
                mr.created_at,
                COUNT(DISTINCT mt.id) AS hit_count,
                CAST(mr.created_at / 86400 AS INTEGER) AS day_bucket
            FROM memory_records mr
            JOIN memory_tag_rel mtr ON mtr.memory_id = mr.id
            JOIN global_tags mt ON mt.id = mtr.tag_id
            WHERE {scope_sql}
              AND mt.name IN ({placeholders})
            GROUP BY mr.id
            ORDER BY hit_count DESC, day_bucket DESC, mr.strength DESC, mr.created_at DESC
            LIMIT ?
        """

        params: List[Any] = [*scope_params, *matched_tags, int(limit)]
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            memory_ids = [str(row["id"]) for row in rows]
            tags_map = self._fetch_tags_for_memory_ids(conn, memory_ids)
            memories = [
                self._row_to_memory(row, tags_map.get(str(row["id"]), []))
                for row in rows
            ]
        return memories

    async def reinforce_memories(self, memory_ids: List[str], delta: int = 3) -> None:
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return
        placeholders = ",".join(["?" for _ in ids])
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_records
                SET strength = strength + ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                tuple([int(delta), now, *ids]),
            )
            conn.commit()

    async def decay_memories(self, memory_ids: List[str], delta: int = 1) -> None:
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return
        placeholders = ",".join(["?" for _ in ids])
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_records
                SET strength = CASE WHEN strength - ? < 0 THEN 0 ELSE strength - ? END,
                    updated_at = ?
                WHERE id IN ({placeholders}) AND is_active = 0
                """,
                tuple([int(delta), int(delta), now, *ids]),
            )
            conn.commit()

    async def consolidate_memories(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_records WHERE is_active = 0 AND strength <= 0")
            conn.execute(
                """
                DELETE FROM memory_tag_rel
                WHERE memory_id NOT IN (SELECT id FROM memory_records)
                """
            )
            conn.commit()

    def _get_memories_by_ids_sync(self, memory_ids: List[str]) -> List[BaseMemory]:
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return []
        placeholders = ",".join(["?" for _ in ids])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, memory_type, judgment, reasoning, strength,
                       is_active, memory_scope, created_at
                FROM memory_records
                WHERE id IN ({placeholders})
                """,
                tuple(ids),
            ).fetchall()
            tags_map = self._fetch_tags_for_memory_ids(conn, [str(row["id"]) for row in rows])
        return [self._row_to_memory(row, tags_map.get(str(row["id"]), [])) for row in rows]

    async def get_memories_by_ids(self, memory_ids: List[str]) -> List[BaseMemory]:
        return await asyncio.to_thread(self._get_memories_by_ids_sync, memory_ids)

    async def merge_group(self, memory_ids: List[str]) -> Optional[BaseMemory]:
        return await asyncio.to_thread(self._merge_group_sync, memory_ids)

    def _merge_group_sync(self, memory_ids: List[str]) -> Optional[BaseMemory]:
        memories = self._get_memories_by_ids_sync(memory_ids)
        if len(memories) < 2:
            return None

        non_public_scopes = {
            str(mem.memory_scope or "public").strip()
            for mem in memories
            if str(mem.memory_scope or "public").strip() != "public"
        }
        if len(non_public_scopes) > 1:
            raise ValidationError("禁止合并不同私有分类域的记忆")

        merged_scope = next(iter(non_public_scopes), "public")
        first = memories[0]
        merged_strength = sum(mem.strength for mem in memories)
        now = time.time()
        new_memory = BaseMemory(
            memory_type=self._to_memory_type("knowledge"),
            judgment=first.judgment or "合并记忆",
            reasoning=first.reasoning or "合并多个相似记忆",
            tags=self._normalize_tags(first.tags),
            id=str(uuid.uuid4()),
            strength=merged_strength,
            is_active=False,
            created_at=now,
            memory_scope=merged_scope,
        )

        ids = [mem.id for mem in memories]
        placeholders = ",".join(["?" for _ in ids])
        with self._lock:
            cache_snapshot = set(self._tag_names)
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO memory_records(
                        id, memory_type, judgment, reasoning, strength, is_active,
                        memory_scope, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_memory.id,
                        new_memory.memory_type.value,
                        new_memory.judgment,
                        new_memory.reasoning,
                        new_memory.strength,
                        1 if new_memory.is_active else 0,
                        new_memory.memory_scope,
                        new_memory.created_at,
                        now,
                    ),
                )
                self._upsert_tags_and_bind(conn, new_memory.id, new_memory.tags)
                conn.execute(
                    f"DELETE FROM memory_tag_rel WHERE memory_id IN ({placeholders})",
                    tuple(ids),
                )
                conn.execute(
                    f"DELETE FROM memory_records WHERE id IN ({placeholders})",
                    tuple(ids),
                )
            except Exception:
                with self._lock:
                    self._tag_names = cache_snapshot
                raise

        return new_memory

    async def process_feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        new_memories: Optional[List[Dict[str, Any]]] = None,
        merge_groups: Optional[List[List[str]]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        resolved_scope = self._normalize_scope(memory_scope)
        created_memories: List[BaseMemory] = []

        if useful_memory_ids:
            await self.reinforce_memories(useful_memory_ids, delta=3)

        for mem_data in (new_memories or []):
            judgment = str(mem_data.get("judgment") or "").strip()
            reasoning = str(mem_data.get("reasoning") or "").strip()
            if not judgment or not reasoning:
                continue
            memory = await self.remember(
                memory_type=str(mem_data.get("type") or "knowledge"),
                judgment=judgment,
                reasoning=reasoning,
                tags=mem_data.get("tags") or [],
                is_active=bool(mem_data.get("is_active", False)),
                strength=mem_data.get("strength"),
                memory_scope=resolved_scope,
            )
            created_memories.append(memory)

        for group in (merge_groups or []):
            if not isinstance(group, list) or len(group) < 2:
                continue
            merged = await self.merge_group(group)
            if merged:
                created_memories.append(merged)

        return created_memories

    def close(self) -> None:
        with self._lock:
            self._tag_names.clear()

    def shutdown(self) -> None:
        self.close()
