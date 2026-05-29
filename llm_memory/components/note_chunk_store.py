"""
笔记切片存储 - 独立 SQLite 数据库

与记忆中央库分离，切片库可随时重建（删库重新扫描即可）。
仅负责切片数据的持久化，不涉及检索逻辑。
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class NoteChunkStore:
    """笔记切片独立存储（SQLite）"""

    def __init__(self, db_path: str):
        """
        初始化切片存储。

        Args:
            db_path: 切片数据库文件路径
        """
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()
        logger.info(f"笔记切片存储初始化完成: {self._db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构"""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    source_file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_id
                    ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_path
                    ON chunks(source_file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_path_index
                    ON chunks(source_file_path, chunk_index);
                """
            )
            conn.commit()

    def upsert_chunks(self, file_id: str, source_file_path: str, chunks: List[Dict]) -> int:
        """
        写入切片（先删旧数据再插入新数据）。

        Args:
            file_id: 文件 ID
            source_file_path: 相对路径
            chunks: 切片列表，每项包含:
                - chunk_index: int 切片序号
                - line_start: int 起始行
                - line_end: int 结束行
                - content: str 切片文本

        Returns:
            写入的切片数量
        """
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            if not chunks:
                conn.commit()
                return 0
            rows = [
                (
                    file_id,
                    source_file_path,
                    chunk["chunk_index"],
                    chunk["line_start"],
                    chunk["line_end"],
                    chunk["content"],
                    now,
                )
                for chunk in chunks
            ]
            conn.executemany(
                """
                INSERT INTO chunks (file_id, source_file_path, chunk_index, line_start, line_end, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return len(rows)

    def delete_by_file_id(self, file_id: str) -> int:
        """
        按 file_id 删除切片。

        Returns:
            删除的行数
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.commit()
            return cursor.rowcount

    def get_chunks_by_path(self, source_file_path: str) -> List[Dict]:
        """
        按路径获取所有切片（按 chunk_index 排序）。

        Returns:
            切片列表
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT chunk_id, file_id, source_file_path, chunk_index, line_start, line_end, content, created_at
            FROM chunks
            WHERE source_file_path = ?
            ORDER BY chunk_index ASC
            """,
            (source_file_path,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_chunks_by_file_id(self, file_id: str) -> List[Dict]:
        """
        按 file_id 获取所有切片（按 chunk_index 排序）。

        Returns:
            切片列表
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT chunk_id, file_id, source_file_path, chunk_index, line_start, line_end, content, created_at
            FROM chunks
            WHERE file_id = ?
            ORDER BY chunk_index ASC
            """,
            (file_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_all_chunks(self) -> List[Dict]:
        """
        获取全部切片，用于全量重建搜索索引。

        Returns:
            切片列表，按路径和切片序号稳定排序。
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT chunk_id, file_id, source_file_path, chunk_index, line_start, line_end, content, created_at
            FROM chunks
            ORDER BY source_file_path ASC, chunk_index ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """获取切片库统计信息"""
        conn = self._get_conn()
        total_chunks = int(
            conn.execute("SELECT COUNT(1) FROM chunks").fetchone()[0] or 0
        )
        total_files = int(
            conn.execute("SELECT COUNT(DISTINCT file_id) FROM chunks").fetchone()[0] or 0
        )
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "db_path": self._db_path,
        }

    def clear_all(self) -> int:
        """清空所有切片数据（重建时使用）。"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM chunks")
            conn.commit()
            count = cursor.rowcount
            logger.info(f"切片库已清空，删除 {count} 条记录")
            return count

    def close(self) -> None:
        """关闭数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None
