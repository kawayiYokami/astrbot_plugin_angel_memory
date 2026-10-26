"""旧 SQL 库播种工具（仅测试用）。

生产侧已不再写旧库，迁移测试需要一个可读的旧库样本，
这里直接用 sqlite3 建同构表并插入记录。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LEGACY_SCHEMA = """
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
CREATE TABLE IF NOT EXISTS user_ledger (
    platform TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_names TEXT NOT NULL DEFAULT '[]',
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    PRIMARY KEY (platform, user_id)
);
CREATE TABLE IF NOT EXISTS group_ledger (
    group_id TEXT PRIMARY KEY,
    group_name TEXT NOT NULL DEFAULT '',
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
"""


class LegacyDb:
    """一个最小可读的旧 SQL 库，形状与历史版本一致。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_LEGACY_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def add_memory(
        self,
        judgment: str,
        *,
        memory_id: Optional[str] = None,
        reasoning: str = "旧库背景",
        tags: Optional[List[str]] = None,
        memory_type: str = "知识记忆",
        strength: int = 50,
        is_active: bool = False,
        useful_count: int = 0,
        useful_score: float = 0.0,
        last_recalled_at: float = 0.0,
        last_decay_at: float = 0.0,
        memory_scope: str = "public",
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
    ) -> str:
        import uuid

        mid = str(memory_id or uuid.uuid4())
        now = time.time()
        created = float(created_at if created_at is not None else now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_records(
                    id, memory_type, judgment, reasoning, strength, is_active,
                    useful_count, useful_score, last_recalled_at, last_decay_at,
                    memory_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid, str(memory_type), str(judgment), str(reasoning),
                    int(strength), 1 if is_active else 0,
                    int(useful_count), float(useful_score),
                    float(last_recalled_at), float(last_decay_at),
                    str(memory_scope), created,
                    float(updated_at if updated_at is not None else now),
                ),
            )
            for name in tags or []:
                conn.execute("INSERT OR IGNORE INTO global_tags(name) VALUES (?)", (name,))
                row = conn.execute("SELECT id FROM global_tags WHERE name = ?", (name,)).fetchone()
                if row is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_tag_rel(memory_id, tag_id) VALUES (?, ?)",
                        (mid, int(row["id"])),
                    )
        return mid

    def add_user(self, platform: str, user_id: str, names: List[str]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_ledger(platform, user_id, user_names, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (platform, user_id, json.dumps(names, ensure_ascii=False), now, now),
            )

    def add_group(self, group_id: str, group_name: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO group_ledger(group_id, group_name, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?)",
                (group_id, group_name, now, now),
            )

    def dump(self) -> Dict[str, Any]:
        with self._connect() as conn:
            return {
                "records": [dict(r) for r in conn.execute("SELECT * FROM memory_records").fetchall()],
                "tags": [dict(r) for r in conn.execute("SELECT * FROM global_tags").fetchall()],
            }
