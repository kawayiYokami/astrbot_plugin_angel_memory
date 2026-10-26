"""账本管理器：只承载用户/群账本、短 ID 注册表与旧库迁移源读取。

记忆与笔记的读写都在 p-memory，这里不再做检索、衰减、合并或笔记索引。
旧表（memory_records / global_tags / memory_tag_rel）只读，供自动迁移导出使用。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemorySqlManager:
    """账本管理器：仅承载用户/群账本、标签与迁移源数据，不做检索。"""

    def __init__(self, db_path: Path):
        self.logger = logger
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._init_db()
        self._init_short_id_registry()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
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
            )
            # 兼容迁移：若旧表 memory_tags 存在，则迁移到 global_tags
            has_legacy = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_tags'"
            ).fetchone()
            if has_legacy:
                cur.execute(
                    "INSERT OR IGNORE INTO global_tags(name) SELECT name FROM memory_tags"
                )
            conn.commit()

    # ===== 全局短 ID 注册表（内存态） =====

    def _init_short_id_registry(self) -> None:
        """启动时加载全部记忆 ID，分配纯数字短 ID（从 10000 开始递增）。"""
        self._short_to_full: Dict[str, str] = {}
        self._full_to_short: Dict[str, str] = {}
        self._next_short_id: int = 10000

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_records ORDER BY created_at ASC"
            ).fetchall()

        for row in rows:
            full_id = str(row["id"]).strip()
            if full_id and full_id not in self._full_to_short:
                short_id = str(self._next_short_id)
                self._short_to_full[short_id] = full_id
                self._full_to_short[full_id] = short_id
                self._next_short_id += 1

        self.logger.debug(
            f"[短ID注册表] 初始化完成 记忆数={len(self._full_to_short)} "
            f"下一个短ID={self._next_short_id}"
        )

    def register_short_id(self, full_id: str) -> str:
        """为新记忆分配短 ID 并注册。如果已存在则直接返回。"""
        full_id = str(full_id or "").strip()
        if not full_id:
            return ""
        with self._lock:
            if full_id in self._full_to_short:
                return self._full_to_short[full_id]
            short_id = str(self._next_short_id)
            self._short_to_full[short_id] = full_id
            self._full_to_short[full_id] = short_id
            self._next_short_id += 1
        return short_id

    def get_short_id(self, full_id: str) -> str:
        """根据完整 ID 获取短 ID，未注册则返回空字符串。"""
        return self._full_to_short.get(str(full_id or "").strip(), "")

    def get_full_id(self, short_id: str) -> str:
        """根据短 ID 获取完整 ID，未注册则返回空字符串。"""
        return self._short_to_full.get(str(short_id or "").strip(), "")

    def build_id_mapping(self, full_ids: List[str]) -> Dict[str, str]:
        """为一组完整 ID 构建 {短ID: 完整ID} 映射（用于反思等场景）。"""
        mapping: Dict[str, str] = {}
        for full_id in full_ids:
            short_id = self.get_short_id(full_id)
            if short_id:
                mapping[short_id] = full_id
        return mapping

    # ===== 用户/群账本 =====

    def upsert_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """登记用户到账本：平台 + 用户 ID 复合主键，同记录合并昵称列表（去重）。

        统一拒绝占位身份（空、0、none、null、unknown/user/assistant），
        与 _is_valid_user_id 及 deepmind 登记处共用同一集合。
        """
        plat = str(platform or "").strip()
        uid = str(user_id or "").strip()
        invalid_ids = {"", "0", "unknown", "user", "assistant", "none", "null"}
        if not plat or uid.lower() in invalid_ids:
            return
        name = str(user_name or "").strip()
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT user_names FROM user_ledger WHERE platform = ? AND user_id = ?",
                    (plat, uid),
                ).fetchone()
                if row is None:
                    names = [name] if name else []
                    conn.execute(
                        "INSERT INTO user_ledger (platform, user_id, user_names, first_seen_at, last_seen_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (plat, uid, json.dumps(names, ensure_ascii=False), now, now),
                    )
                    return
                try:
                    names = json.loads(str(row["user_names"] or "[]"))
                except (json.JSONDecodeError, ValueError):
                    names = []
                if not isinstance(names, list):
                    names = []
                if name and name not in names:
                    names.append(name)
                conn.execute(
                    "UPDATE user_ledger SET user_names = ?, last_seen_at = ? "
                    "WHERE platform = ? AND user_id = ?",
                    (json.dumps(names, ensure_ascii=False), now, plat, uid),
                )

    def get_known_user_ids(self, platform: str = "") -> List[str]:
        """返回账本中全部已知用户 ID；platform 非空时只返回该平台下的 ID。

        不带平台时返回跨平台去重后的 ID 集合（仅用于画像判定锚点），
        带平台时返回该平台下的完整列表（可能含不同用户同号，但同平台内唯一）。
        """
        plat = str(platform or "").strip()
        with self._connect() as conn:
            if plat:
                rows = conn.execute(
                    "SELECT user_id FROM user_ledger WHERE platform = ?",
                    (plat,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT user_id FROM user_ledger").fetchall()
        ids = [str(row["user_id"] or "").strip() for row in rows if str(row["user_id"] or "").strip()]
        if plat:
            return ids
        seen = set()
        unique = []
        for uid in ids:
            if uid not in seen:
                seen.add(uid)
                unique.append(uid)
        return unique

    def get_user_names(self, platform: str, user_id: str) -> List[str]:
        """返回某平台下某用户账本中登记过的全部昵称。"""
        plat = str(platform or "").strip()
        uid = str(user_id or "").strip()
        if not plat or not uid:
            return []
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_names FROM user_ledger WHERE platform = ? AND user_id = ?",
                (plat, uid),
            ).fetchone()
        if row is None:
            return []
        try:
            names = json.loads(str(row["user_names"] or "[]"))
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(names, list):
            return []
        return [str(name).strip() for name in names if isinstance(name, str) and str(name).strip()]

    def upsert_group(self, group_id: str, group_name: str = "") -> None:
        """登记群聊到群账本：同群 ID 更新群名（用最近拿到的值）与最近活跃时间。"""
        gid = str(group_id or "").strip()
        if not gid or gid in {"Unknown", "unknown", "0"}:
            return
        gname = str(group_name or "").strip()
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT group_name FROM group_ledger WHERE group_id = ?",
                    (gid,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO group_ledger (group_id, group_name, first_seen_at, last_seen_at) "
                        "VALUES (?, ?, ?, ?)",
                        (gid, gname, now, now),
                    )
                    return
                if gname:
                    conn.execute(
                        "UPDATE group_ledger SET group_name = ?, last_seen_at = ? WHERE group_id = ?",
                        (gname, now, gid),
                    )
                else:
                    conn.execute(
                        "UPDATE group_ledger SET last_seen_at = ? WHERE group_id = ?",
                        (now, gid),
                    )

    def get_known_group_ids(self) -> List[str]:
        """返回群账本中全部已知群 ID。"""
        with self._connect() as conn:
            rows = conn.execute("SELECT group_id FROM group_ledger").fetchall()
        return [str(row["group_id"] or "").strip() for row in rows if str(row["group_id"] or "").strip()]

    def get_group_name(self, group_id: str) -> str:
        """返回某群的群名，未登记或无名返回空串。"""
        gid = str(group_id or "").strip()
        if not gid:
            return ""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT group_name FROM group_ledger WHERE group_id = ?",
                (gid,),
            ).fetchone()
        if row is None:
            return ""
        return str(row["group_name"] or "").strip()

    def list_ledger_users(self) -> List[Dict[str, Any]]:
        """返回账本全部用户，最近活跃在前（供 WebUI 画像列表使用）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT platform, user_id, user_names, first_seen_at, last_seen_at "
                "FROM user_ledger ORDER BY last_seen_at DESC"
            ).fetchall()
        users: List[Dict[str, Any]] = []
        for row in rows:
            platform = str(row["platform"] or "").strip()
            user_id = str(row["user_id"] or "").strip()
            if not platform or not user_id:
                continue
            try:
                names = json.loads(str(row["user_names"] or "[]"))
            except (json.JSONDecodeError, ValueError):
                names = []
            if not isinstance(names, list):
                names = []
            users.append({
                "platform": platform,
                "user_id": user_id,
                "nickname": names[-1] if names else "",
                "nicknames": names,
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
            })
        return users

    def list_ledger_groups(self) -> List[Dict[str, Any]]:
        """返回账本全部群，最近活跃在前。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT group_id, group_name, first_seen_at, last_seen_at "
                "FROM group_ledger ORDER BY last_seen_at DESC"
            ).fetchall()
        groups: List[Dict[str, Any]] = []
        for row in rows:
            group_id = str(row["group_id"] or "").strip()
            if not group_id:
                continue
            groups.append({
                "group_id": group_id,
                "group_name": str(row["group_name"] or "").strip(),
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
            })
        return groups

    # ===== 旧库迁移源（只读） =====

    async def export_backup_snapshot(self) -> Dict[str, Any]:
        """导出旧 SQL 记忆快照（只被自动迁移读取，不再作为运行期数据源）。"""
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
