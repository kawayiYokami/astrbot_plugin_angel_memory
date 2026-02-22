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
from ..service.memory_decay_policy import MemoryDecayConfig, MemoryDecayPolicy
from .bm25_retriever import TantivyBM25Retriever
from .hybrid_retrieval_engine import HybridRetrievalEngine


class MemorySqlManager:
    """SimpleMemory 的 SQL 存储管理器。"""

    def __init__(
        self,
        db_path: Path,
        decay_config: Optional[MemoryDecayConfig] = None,
        rerank_provider: Optional[Any] = None,
    ):
        self.logger = logger
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fts_lock = threading.Lock()
        self._tag_names: set[str] = set()
        self._jieba_dict_ready = False
        self._jieba_pending_tags: set[str] = set()
        self._jieba_new_tags_since_rebuild = 0
        self._fts_rebuild_required = False
        self._jieba_rebuild_added_threshold = 50
        self._jieba_rebuild_ratio_threshold = 0.10
        self.decay_policy = MemoryDecayPolicy(decay_config)
        self._rerank_provider = rerank_provider
        self._fts_retriever = TantivyBM25Retriever(
            db_path=str(self.db_path),
            tokenizer_mode="jieba_cut",
            memory_threshold=0.5,
            note_threshold=0.6,
            memory_default_vector_score=0.5,
            note_default_vector_score=0.6,
        )
        self._hybrid_engine = HybridRetrievalEngine(
            retriever=self._fts_retriever,
            rerank_provider=self._rerank_provider,
        )
        self._fts_ready = False

        self._init_db()
        self._load_tag_cache()
        # 启动阶段主动完成一次 FTS5 索引构建，避免首次查询触发冷启动延迟。
        start_ts = time.time()
        self._ensure_fts_ready_sync(force_rebuild=True)
        elapsed_ms = int((time.time() - start_ts) * 1000)
        self.logger.info(f"[FTS5重建] 启动预构建完成 耗时={elapsed_ms}ms")

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
                    useful_count INTEGER NOT NULL DEFAULT 0,
                    useful_score REAL NOT NULL DEFAULT 0,
                    last_recalled_at REAL NOT NULL DEFAULT 0,
                    last_decay_at REAL NOT NULL DEFAULT 0,
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
            # 兼容迁移：memory_records 补充三档记忆字段
            memory_columns = cur.execute("PRAGMA table_info(memory_records)").fetchall()
            memory_column_names = {str(row[1]) for row in memory_columns}
            if "useful_count" not in memory_column_names:
                cur.execute(
                    "ALTER TABLE memory_records ADD COLUMN useful_count INTEGER NOT NULL DEFAULT 0"
                )
            if "useful_score" not in memory_column_names:
                cur.execute(
                    "ALTER TABLE memory_records ADD COLUMN useful_score REAL NOT NULL DEFAULT 0"
                )
            if "last_recalled_at" not in memory_column_names:
                cur.execute(
                    "ALTER TABLE memory_records ADD COLUMN last_recalled_at REAL NOT NULL DEFAULT 0"
                )
            if "last_decay_at" not in memory_column_names:
                cur.execute(
                    "ALTER TABLE memory_records ADD COLUMN last_decay_at REAL NOT NULL DEFAULT 0"
                )
            # 索引创建放在补列之后，避免旧库因缺列导致初始化失败
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_tier_fields
                    ON memory_records(is_active, useful_score, last_recalled_at)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_note_index_short_id
                    ON note_index_records(note_short_id)
                """
            )
            conn.commit()

    def _load_tag_cache(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM global_tags").fetchall()
            with self._lock:
                self._tag_names = {str(row["name"]) for row in rows if row["name"]}
        self.logger.info(f"SimpleMemory 标签缓存加载完成: {len(self._tag_names)} 个")

    def _list_global_tag_names_sync(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM global_tags").fetchall()
        names: List[str] = []
        seen = set()
        for row in rows:
            tag = str(row["name"] or "").strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            names.append(tag)
        return names

    def _mark_new_tags_sync(self, tags: List[str]) -> None:
        normalized = self._normalize_tags(tags)
        if not normalized:
            return
        with self._lock:
            new_tags = [tag for tag in normalized if tag not in self._tag_names]
            if not new_tags:
                return
            self._tag_names.update(new_tags)
            self._jieba_pending_tags.update(new_tags)
            self._jieba_new_tags_since_rebuild += len(new_tags)
            # 新词加入后标记词典待刷新，查询路径会懒加载注词。
            self._jieba_dict_ready = False

    def _load_jieba_tags_dict_locked_sync(self, trigger: str) -> Dict[str, int]:
        """
        加载并注入 tags 词表到 jieba（需在 _fts_lock 保护下调用）。
        - 数据源：global_tags（统一不区分 memory/note）
        - 注词：jieba.add_word(word)，不设 freq/tag
        """
        start_ts = time.time()
        with self._lock:
            pending = list(self._jieba_pending_tags)
        if pending:
            target_words = pending
        else:
            target_words = self._list_global_tag_names_sync()

        stats = self._fts_retriever.add_jieba_words(target_words)
        tags_total = len(target_words)
        elapsed_ms = int((time.time() - start_ts) * 1000)

        with self._lock:
            if pending:
                self._jieba_pending_tags.difference_update(set(target_words))
            self._jieba_dict_ready = True

        self.logger.info(
            "[FTS5词表] 完成 "
            f"trigger={trigger} tags_total={tags_total} "
            f"added={int(stats.get('added', 0))} skipped={int(stats.get('skipped', 0))} "
            f"cost_ms={elapsed_ms}"
        )
        return {
            "tags_total": tags_total,
            "added": int(stats.get("added", 0)),
            "skipped": int(stats.get("skipped", 0)),
            "cost_ms": elapsed_ms,
        }

    def _ensure_jieba_tags_ready_sync(self, trigger: str) -> None:
        with self._fts_lock:
            with self._lock:
                pending_exists = bool(self._jieba_pending_tags)
                dict_ready = bool(self._jieba_dict_ready)
            if dict_ready and not pending_exists:
                return
            try:
                self._load_jieba_tags_dict_locked_sync(trigger=trigger)
            except Exception as e:
                self._jieba_dict_ready = False
                self.logger.error(f"[FTS5词表] 失败 trigger={trigger} 异常={e}", exc_info=True)
                raise

    def _list_memory_rows_for_fts_sync(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    mr.id,
                    mr.judgment,
                    IFNULL(tags.tags_text, '') AS tags_text
                FROM memory_records mr
                LEFT JOIN (
                    SELECT
                        mtr.memory_id AS memory_id,
                        GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr
                    JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                """
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            tags = [t.strip() for t in str(row["tags_text"] or "").split(",") if t.strip()]
            result.append(
                {
                    "id": str(row["id"] or ""),
                    "judgment": str(row["judgment"] or ""),
                    "tags": tags,
                }
            )
        return [r for r in result if r["id"]]

    def _list_note_rows_for_fts_sync(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    nir.source_id,
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
                """
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            tags = [t.strip() for t in str(row["tags_text"] or "").split(",") if t.strip()]
            result.append({"id": str(row["source_id"] or ""), "tags": tags})
        return [r for r in result if r["id"]]

    def _ensure_fts_ready_sync(self, force_rebuild: bool = False) -> None:
        rebuilt = False
        with self._fts_lock:
            with self._lock:
                pending_exists = bool(self._jieba_pending_tags)
                dict_ready = bool(self._jieba_dict_ready)
            if force_rebuild or pending_exists or not dict_ready:
                try:
                    self._load_jieba_tags_dict_locked_sync(
                        trigger="rebuild" if force_rebuild else "query-lazy"
                    )
                except Exception as e:
                    self._jieba_dict_ready = False
                    self.logger.error(
                        f"[FTS5词表] 失败 trigger={'rebuild' if force_rebuild else 'query-lazy'} 异常={e}",
                        exc_info=True,
                    )
                    raise

            with self._lock:
                total_tags = max(1, len(self._tag_names))
                added_since_rebuild = int(self._jieba_new_tags_since_rebuild)
            threshold_reached = (
                added_since_rebuild >= int(self._jieba_rebuild_added_threshold)
                or (added_since_rebuild / float(total_tags)) >= float(self._jieba_rebuild_ratio_threshold)
            )
            if threshold_reached and self._fts_ready and not force_rebuild:
                self._fts_rebuild_required = True
                self.logger.info(
                    "[FTS5重建] 触发阈值达到，准备执行全量重建 "
                    f"added={added_since_rebuild} total_tags={total_tags}"
                )

            effective_force_rebuild = bool(force_rebuild or self._fts_rebuild_required)
            if self._fts_ready and not effective_force_rebuild:
                return
            previous_ready = self._fts_ready
            try:
                self._fts_retriever.rebuild_memory(self._list_memory_rows_for_fts_sync())
                self._fts_retriever.rebuild_note(self._list_note_rows_for_fts_sync())
                self._fts_ready = True
                self._fts_rebuild_required = False
                self._jieba_new_tags_since_rebuild = 0
                rebuilt = True
            except Exception as e:
                self._fts_ready = previous_ready
                self.logger.error(f"[FTS5重建] 失败，异常={e}", exc_info=True)
                raise
        if rebuilt:
            self._log_fts_index_size_sync()

    def _log_fts_index_size_sync(self) -> None:
        """输出检索索引构建后的体积信息。"""
        db_size_bytes = 0
        try:
            db_size_bytes = int(self.db_path.stat().st_size)
        except Exception:
            db_size_bytes = 0

        db_size_mb = db_size_bytes / (1024 * 1024)
        self.logger.info(
            f"[检索索引] 完成 db_size={db_size_bytes}B ({db_size_mb:.2f}MB)"
        )

    def _sync_memory_fts_by_id_sync(self, memory_id: str) -> None:
        if not self._fts_ready:
            return
        self._ensure_jieba_tags_ready_sync(trigger="incremental")
        mid = str(memory_id or "").strip()
        if not mid:
            return
        memories = self._get_memories_by_ids_sync([mid])
        if not memories:
            self._fts_retriever.delete_memory(mid)
            return
        mem = memories[0]
        self._fts_retriever.upsert_memory(
            item_id=mem.id,
            tags=getattr(mem, "tags", []) or [],
            judgment=str(getattr(mem, "judgment", "") or ""),
        )

    def _sync_memory_fts_batch_sync(
        self,
        upsert_ids: Optional[List[str]] = None,
        delete_ids: Optional[List[str]] = None,
    ) -> None:
        if not self._fts_ready:
            return
        upsert_list = [str(x).strip() for x in (upsert_ids or []) if str(x).strip()]
        delete_list = [str(x).strip() for x in (delete_ids or []) if str(x).strip()]
        if delete_list:
            for memory_id in delete_list:
                self._fts_retriever.delete_memory(memory_id)
        for memory_id in upsert_list:
            self._sync_memory_fts_by_id_sync(memory_id)

    def _sync_note_fts_by_source_id_sync(self, source_id: str) -> None:
        if not self._fts_ready:
            return
        self._ensure_jieba_tags_ready_sync(trigger="incremental")
        sid = str(source_id or "").strip()
        if not sid:
            return
        rows = self._get_note_index_by_source_ids_sync([sid])
        if not rows:
            self._fts_retriever.delete_note(sid)
            return
        row = rows[0]
        tags_text = str(row.get("tags_text") or "")
        tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        self._fts_retriever.upsert_note(item_id=sid, tags=tags)

    async def audit_and_repair_fts_indexes(self, sample_size: int = 50) -> Dict[str, Any]:
        return await asyncio.to_thread(self._audit_and_repair_fts_indexes_sync, sample_size)

    def _audit_and_repair_fts_indexes_sync(self, sample_size: int = 50) -> Dict[str, Any]:
        del sample_size
        self._ensure_fts_ready_sync(force_rebuild=True)
        with self._connect() as conn:
            sql_memory_count = int(conn.execute("SELECT COUNT(1) FROM memory_records").fetchone()[0] or 0)
            sql_note_count = int(conn.execute("SELECT COUNT(1) FROM note_index_records").fetchone()[0] or 0)
        # Tantivy 模式下无法直接通过 SQL 查询索引行数，这里仅保留 SQL 侧计数。
        self.logger.info(
            "[检索巡检] 完成 "
            f"memory_sql={sql_memory_count} note_sql={sql_note_count} auto_repaired=是"
        )
        return {
            "memory_sql_count": sql_memory_count,
            "memory_fts_count": -1,
            "note_sql_count": sql_note_count,
            "note_fts_count": -1,
            "auto_repaired": True,
        }

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
    def _is_scope_allowed(memory_scope: str, target_scope: str) -> bool:
        scope = str(memory_scope or "").strip() or "public"
        target = str(target_scope or "").strip()
        if target == "public":
            return scope == "public"
        return scope in {target, "public"}

    @staticmethod
    def _row_to_memory(row: sqlite3.Row, tags: List[str]) -> BaseMemory:
        keys = set(row.keys()) if hasattr(row, "keys") else set()

        def _v(name: str, default: Any) -> Any:
            if keys and name in keys:
                return row[name]
            return default

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
            useful_count=int(_v("useful_count", 0) or 0),
            useful_score=float(_v("useful_score", 0.0) or 0.0),
            last_recalled_at=float(_v("last_recalled_at", 0.0) or 0.0),
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
            self._mark_new_tags_sync(tags)

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
        self._mark_new_tags_sync(normalized)
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
        self._mark_new_tags_sync(normalized)
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
        synced_source_ids: List[str] = []
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
                    synced_source_ids.append(source_id)
                except Exception:
                    failed += 1
                    self.logger.exception("笔记索引写入失败")
            conn.commit()
        if self._fts_ready and synced_source_ids:
            for source_id in synced_source_ids:
                self._sync_note_fts_by_source_id_sync(source_id)
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
        if self._fts_ready and source_ids:
            for source_id in source_ids:
                self._fts_retriever.delete_note(source_id)
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

    def _build_note_doc_text_map_by_ids(self, source_ids: List[str]) -> Dict[str, str]:
        rows = self._get_note_index_by_source_ids_sync(source_ids)
        row_map = {str(row.get("source_id") or ""): row for row in rows}
        doc_text_map: Dict[str, str] = {}
        for sid in source_ids:
            row = row_map.get(str(sid))
            if row is None:
                continue
            tags_text = str(row.get("tags_text") or "")
            h_values = [str(row.get(f"heading_h{i}") or "").strip() for i in range(1, 7)]
            heading_text = " / ".join([h for h in h_values if h])
            source_file_path = str(row.get("source_file_path") or "")
            doc_text_map[str(sid)] = f"{source_file_path}\n{heading_text}\n{tags_text}".strip()
        return doc_text_map

    def _build_memory_doc_text_map_by_ids(self, memory_ids: List[str]) -> Dict[str, str]:
        memories = self._get_memories_by_ids_sync(memory_ids)
        memory_map = {mem.id: mem for mem in memories}
        doc_text_map: Dict[str, str] = {}
        for memory_id in memory_ids:
            mem = memory_map.get(str(memory_id))
            if mem is None:
                continue
            tags = " ".join(
                [str(t).strip() for t in (getattr(mem, "tags", []) or []) if str(t).strip()]
            )
            doc_text_map[str(memory_id)] = (
                f"{getattr(mem, 'judgment', '')}\n{getattr(mem, 'reasoning', '')}\n{tags}"
            ).strip()
        return doc_text_map

    async def search_note_index_by_tags(
        self,
        query: str,
        limit: int = 20,
        vector_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            return []

        await asyncio.to_thread(self._ensure_fts_ready_sync)
        candidate_limit = max(10, int(limit) * 6)
        # rerank 依赖当前事件循环任务上下文，不能在线程里 asyncio.run。
        if self._hybrid_engine.has_rerank():
            hits = await self._hybrid_engine.search_with_strategy(
                query=text,
                limit=max(1, int(limit)),
                candidate_limit=candidate_limit,
                bm25_limit=candidate_limit,
                vector_scores=vector_scores,
                bm25_only_search=lambda q, k: self._fts_retriever.search_note_bm25_only(query=q, limit=k),
                fusion_search=lambda q, k, bk, scores: self._fts_retriever.search_note(
                    query=q,
                    limit=k,
                    fts_limit=bk,
                    fts_weight=0.3,
                    vector_weight=0.7,
                    vector_scores=scores,
                    min_final_score=0.0,
                ),
                build_doc_text_map=self._build_note_doc_text_map_by_ids,
            )
        else:
            hits = await asyncio.to_thread(
                self._search_note_index_by_tags_without_rerank_sync,
                text,
                limit,
                vector_scores,
            )

        if not hits:
            return []

        source_ids = [
            str(item.get("id") or "").strip()
            for item in hits
            if str(item.get("id") or "").strip()
        ]
        rows = await asyncio.to_thread(self._get_note_index_by_source_ids_sync, source_ids)
        row_map = {str(row.get("source_id") or ""): row for row in rows}

        result: List[Dict[str, Any]] = []
        for item in hits:
            source_id = str(item.get("id") or "").strip()
            row = row_map.get(source_id)
            if row is None:
                continue
            row = dict(row)
            row["similarity"] = float(item.get("final_score", 0.0))
            result.append(row)
        return result[: max(0, int(limit))]

    def _search_note_index_by_tags_without_rerank_sync(
        self,
        query: str,
        limit: int,
        vector_scores: Optional[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            return []
        self._ensure_fts_ready_sync()
        candidate_limit = max(10, int(limit) * 6)
        if vector_scores:
            return self._fts_retriever.search_note(
                query=text,
                limit=max(1, int(limit)),
                fts_limit=candidate_limit,
                fts_weight=0.3,
                vector_weight=0.7,
                vector_scores=vector_scores,
                min_final_score=0.0,
            )
        return self._fts_retriever.search_note_bm25_only(
            query=text,
            limit=max(1, int(limit)),
        )

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
            useful_count=0,
            useful_score=0.0,
            last_recalled_at=0.0,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_records(
                    id, memory_type, judgment, reasoning, strength, is_active,
                    useful_count, useful_score, last_recalled_at,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.memory_type.value,
                    memory.judgment,
                    memory.reasoning,
                    memory.strength,
                    1 if memory.is_active else 0,
                    memory.useful_count,
                    memory.useful_score,
                    memory.last_recalled_at,
                    scope,
                    memory.created_at,
                    now,
                ),
            )
            self._upsert_tags_and_bind(conn, memory.id, normalized_tags)
            conn.commit()
        self._sync_memory_fts_by_id_sync(memory.id)

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
                    useful_count, useful_score, last_recalled_at,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    memory_type=excluded.memory_type,
                    judgment=excluded.judgment,
                    reasoning=excluded.reasoning,
                    strength=excluded.strength,
                    is_active=excluded.is_active,
                    useful_count=excluded.useful_count,
                    useful_score=excluded.useful_score,
                    last_recalled_at=excluded.last_recalled_at,
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
                    int(getattr(memory, "useful_count", 0) or 0),
                    float(getattr(memory, "useful_score", 0.0) or 0.0),
                    float(getattr(memory, "last_recalled_at", 0.0) or 0.0),
                    scope,
                    created_at,
                    now,
                ),
            )
            self._replace_memory_tags(conn, memory_id, normalized_tags)
            conn.commit()
        self._sync_memory_fts_by_id_sync(memory_id)
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
        fts_upsert_ids: List[str] = []
        fts_delete_ids: List[str] = []

        with self._connect() as conn:
            for judgment, raw in deduped_map.items():
                try:
                    created_at = float(raw.get("created_at") or now)
                    normalized_tags = self._normalize_tags(BaseMemory._parse_tags(raw.get("tags", [])))
                    reasoning = str(raw.get("reasoning") or "").strip()
                    memory_type = str(raw.get("memory_type") or "知识记忆").strip() or "知识记忆"
                    strength = int(raw.get("strength", 1) or 1)
                    is_active = 1 if bool(raw.get("is_active", False)) else 0
                    useful_count = int(raw.get("useful_count", 0) or 0)
                    useful_score = float(raw.get("useful_score", 0.0) or 0.0)
                    last_recalled_at = float(raw.get("last_recalled_at", 0.0) or 0.0)
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
                                    useful_count = ?,
                                    useful_score = ?,
                                    last_recalled_at = ?,
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
                                    useful_count,
                                    useful_score,
                                    last_recalled_at,
                                    memory_scope,
                                    created_at,
                                    now,
                                    keep_id,
                                ),
                            )
                            self._replace_memory_tags(conn, keep_id, normalized_tags)
                            upserted += 1
                            fts_upsert_ids.append(keep_id)

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
                            fts_delete_ids.extend(duplicate_ids)
                    else:
                        # 中央记忆库ID必须由本项目生成，禁止复用外部传入ID。
                        memory_id = str(uuid.uuid4())
                        conn.execute(
                            """
                            INSERT INTO memory_records(
                                id, memory_type, judgment, reasoning, strength, is_active,
                                useful_count, useful_score, last_recalled_at,
                                memory_scope, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                memory_id,
                                memory_type,
                                judgment,
                                reasoning,
                                strength,
                                is_active,
                                useful_count,
                                useful_score,
                                last_recalled_at,
                                memory_scope,
                                created_at,
                                now,
                            ),
                        )
                        self._replace_memory_tags(conn, memory_id, normalized_tags)
                        upserted += 1
                        fts_upsert_ids.append(memory_id)

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
        self._sync_memory_fts_batch_sync(
            upsert_ids=fts_upsert_ids,
            delete_ids=fts_delete_ids,
        )

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
                    mr.useful_count,
                    mr.useful_score,
                    mr.last_recalled_at,
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
                    "useful_count": int(row["useful_count"] or 0),
                    "useful_score": float(row["useful_score"] or 0.0),
                    "last_recalled_at": float(row["last_recalled_at"] or 0.0),
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
                       useful_count, useful_score, last_recalled_at,
                       last_decay_at, memory_scope, created_at, updated_at
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
        vector_scores: Optional[Dict[str, float]] = None,
    ) -> List[BaseMemory]:
        text = str(query or "").strip()
        if not text:
            return []

        self._ensure_fts_ready_sync()
        candidate_limit = max(20, int(limit) * 20)
        bm25_limit = max(50, int(limit) * 30)
        hits = await self._hybrid_engine.search_with_strategy(
            query=text,
            limit=candidate_limit,
            candidate_limit=candidate_limit,
            bm25_limit=bm25_limit,
            vector_scores=vector_scores,
            bm25_only_search=lambda q, k: self._fts_retriever.search_memory_bm25_only(query=q, limit=k),
            fusion_search=lambda q, k, bk, scores: self._fts_retriever.search_memory(
                query=q,
                limit=k,
                fts_limit=bk,
                fts_weight=0.3,
                vector_weight=0.7,
                vector_scores=scores,
                min_final_score=0.0,
            ),
            build_doc_text_map=self._build_memory_doc_text_map_by_ids,
        )
        if not hits:
            return []

        ordered_ids = [
            str(item.get("id") or "").strip()
            for item in hits
            if str(item.get("id") or "").strip()
        ]
        score_map = {
            str(item.get("id") or "").strip(): float(item.get("final_score", 0.0))
            for item in hits
            if str(item.get("id") or "").strip()
        }
        memories = self._get_memories_by_ids_sync(ordered_ids)
        memory_map = {mem.id: mem for mem in memories}

        ordered: List[BaseMemory] = []
        for memory_id in ordered_ids:
            mem = memory_map.get(memory_id)
            if mem is None:
                continue
            if not self._is_scope_allowed(getattr(mem, "memory_scope", "public"), memory_scope):
                continue
            final_score = float(score_map.get(memory_id, 0.0))
            if final_score < 0.5:
                continue
            mem.similarity = final_score
            ordered.append(mem)
            if len(ordered) >= int(limit):
                break
        return ordered

    async def reinforce_memories(self, memory_ids: List[str], delta: int = 1) -> None:
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return
        placeholders = ",".join(["?" for _ in ids])
        now = time.time()
        score_delta = float(self.decay_policy.config.consolidate_speed)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_records
                SET strength = strength + ?,
                    useful_count = useful_count + 1,
                    useful_score = useful_score + ?,
                    last_recalled_at = ?,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                tuple([int(delta), score_delta, now, now, *ids]),
            )
            conn.commit()

    async def decay_memories(self, memory_ids: List[str], delta: int = 1) -> None:
        """
        兼容旧接口：仅对 T1（待证档）执行“召回无用衰减”。
        """
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return
        placeholders = ",".join(["?" for _ in ids])
        now = time.time()
        tier1_min_score = float(self.decay_policy.config.tier0_threshold)
        tier1_max_score = float(self.decay_policy.config.tier1_threshold)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_records
                SET strength = CASE WHEN strength - ? < 0 THEN 0 ELSE strength - ? END,
                    updated_at = ?
                WHERE id IN ({placeholders})
                  AND is_active = 0
                  AND useful_score >= ?
                  AND useful_score < ?
                """,
                tuple([int(delta), int(delta), now, tier1_min_score, tier1_max_score, *ids]),
            )
            conn.commit()

    async def decay_recalled_but_useless(self, memory_ids: List[str], delta: int = 1) -> None:
        await self.decay_memories(memory_ids, delta=delta)

    async def natural_decay_tier0(self, now_ts: Optional[float] = None) -> int:
        """
        仅对 T0（易逝档）做时间驱动遗忘。
        """
        now = float(now_ts or time.time())
        cycle_days = int(self.decay_policy.tier0_decay_cycle_days())
        cycle_seconds = float(cycle_days * 86400)
        tier0_max_score = float(self.decay_policy.config.tier0_threshold)
        with self._connect() as conn:
            # 批量计算并更新：避免逐条 Python 循环与 IO 放大。
            # ref_time 语义：
            # - 已有 last_decay_at: 续算
            # - 无 last_decay_at: 取 max(created_at, last_recalled_at)
            cur = conn.execute(
                """
                UPDATE memory_records
                SET
                    strength = CASE
                        WHEN strength - CAST((? - CASE
                            WHEN last_decay_at > 0 THEN last_decay_at
                            WHEN last_recalled_at > created_at THEN last_recalled_at
                            ELSE created_at
                        END) / ? AS INTEGER) < 0
                        THEN 0
                        ELSE strength - CAST((? - CASE
                            WHEN last_decay_at > 0 THEN last_decay_at
                            WHEN last_recalled_at > created_at THEN last_recalled_at
                            ELSE created_at
                        END) / ? AS INTEGER)
                    END,
                    last_decay_at = (
                        CASE
                            WHEN last_decay_at > 0 THEN last_decay_at
                            WHEN last_recalled_at > created_at THEN last_recalled_at
                            ELSE created_at
                        END
                    ) + (
                        CAST((? - CASE
                            WHEN last_decay_at > 0 THEN last_decay_at
                            WHEN last_recalled_at > created_at THEN last_recalled_at
                            ELSE created_at
                        END) / ? AS INTEGER) * ?
                    ),
                    updated_at = ?
                WHERE is_active = 0
                  AND useful_score < ?
                  AND strength > 0
                  AND CAST((? - CASE
                        WHEN last_decay_at > 0 THEN last_decay_at
                        WHEN last_recalled_at > created_at THEN last_recalled_at
                        ELSE created_at
                    END) / ? AS INTEGER) > 0
                """,
                (
                    now,
                    cycle_seconds,
                    now,
                    cycle_seconds,
                    now,
                    cycle_seconds,
                    cycle_seconds,
                    now,
                    tier0_max_score,
                    now,
                    cycle_seconds,
                ),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    async def consolidate_memories(self) -> None:
        # T0 时间衰减（T1/T2 不参与自然遗忘）
        await self.natural_decay_tier0()
        deleted_ids: List[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_records WHERE is_active = 0 AND strength <= 0"
            ).fetchall()
            deleted_ids = [str(row["id"]) for row in rows]
            conn.execute("DELETE FROM memory_records WHERE is_active = 0 AND strength <= 0")
            conn.execute(
                """
                DELETE FROM memory_tag_rel
                WHERE memory_id NOT IN (SELECT id FROM memory_records)
                """
            )
            conn.commit()
        self._sync_memory_fts_batch_sync(delete_ids=deleted_ids)

    def _get_memories_by_ids_sync(self, memory_ids: List[str]) -> List[BaseMemory]:
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return []
        placeholders = ",".join(["?" for _ in ids])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, memory_type, judgment, reasoning, strength,
                       is_active, useful_count, useful_score, last_recalled_at,
                       memory_scope, created_at
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
        merged_useful_count = sum(int(getattr(mem, "useful_count", 0) or 0) for mem in memories)
        merged_useful_score = max(float(getattr(mem, "useful_score", 0.0) or 0.0) for mem in memories)
        merged_last_recalled_at = max(float(getattr(mem, "last_recalled_at", 0.0) or 0.0) for mem in memories)
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
            useful_count=merged_useful_count,
            useful_score=merged_useful_score,
            last_recalled_at=merged_last_recalled_at,
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
                        useful_count, useful_score, last_recalled_at,
                        memory_scope, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_memory.id,
                        new_memory.memory_type.value,
                        new_memory.judgment,
                        new_memory.reasoning,
                        new_memory.strength,
                        1 if new_memory.is_active else 0,
                        new_memory.useful_count,
                        new_memory.useful_score,
                        new_memory.last_recalled_at,
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
        self._sync_memory_fts_batch_sync(
            upsert_ids=[new_memory.id],
            delete_ids=ids,
        )

        return new_memory

    async def process_feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        recalled_memory_ids: Optional[List[str]] = None,
        new_memories: Optional[List[Dict[str, Any]]] = None,
        merge_groups: Optional[List[List[str]]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        resolved_scope = self._normalize_scope(memory_scope)
        created_memories: List[BaseMemory] = []

        useful_ids = [str(x).strip() for x in (useful_memory_ids or []) if str(x).strip()]
        recalled_ids = [str(x).strip() for x in (recalled_memory_ids or []) if str(x).strip()]
        if useful_ids:
            await self.reinforce_memories(useful_ids, delta=1)
        if recalled_ids:
            useful_set = set(useful_ids)
            useless_recalled = [mid for mid in recalled_ids if mid not in useful_set]
            if useless_recalled:
                await self.decay_recalled_but_useless(useless_recalled, delta=1)

        for mem_data in (new_memories or []):
            judgment = str(mem_data.get("judgment") or "").strip()
            reasoning = str(mem_data.get("reasoning") or "").strip()
            if not judgment:
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
