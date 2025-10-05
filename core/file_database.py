"""
文件数据库管理器

使用SQLite存储文件时间戳信息，提升频繁读写场景下的性能。
"""

import sqlite3
import threading
import os
from pathlib import Path
from typing import Dict
from astrbot.api import logger


class FileDatabase:
    """文件时间戳数据库管理器"""

    def __init__(self, data_directory: str):
        """
        初始化文件数据库

        Args:
            data_directory: 插件数据目录
        """
        self.logger = logger
        self.data_directory = Path(data_directory)
        self.db_path = self.data_directory / "file_timestamps.db"
        self._lock = threading.Lock()

        # 确保数据目录存在
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库表"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 创建文件时间戳表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS file_timestamps (
                        file_path TEXT PRIMARY KEY,
                        modified_time REAL NOT NULL
                    )
                ''')

                # 创建索引以提升查询性能
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_modified_time
                    ON file_timestamps(modified_time)
                ''')

                conn.commit()
                conn.close()

                self.logger.info("文件时间戳数据库初始化完成")
        except Exception as e:
            self.logger.error(f"初始化文件数据库失败: {e}")
            raise

    def is_file_changed(self, file_path: str) -> bool:
        """
        检查文件是否已更改

        Args:
            file_path: 文件路径

        Returns:
            文件是否已更改
        """
        try:
            # 获取文件当前修改时间
            current_modified_time = os.path.getmtime(file_path)

            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 查询存储的修改时间
                cursor.execute(
                    'SELECT modified_time FROM file_timestamps WHERE file_path = ?',
                    (file_path,)
                )
                result = cursor.fetchone()
                conn.close()

                # 如果没有记录或修改时间不同，则文件已更改
                if result is None:
                    return True

                stored_modified_time = result[0]
                return current_modified_time != stored_modified_time

        except Exception as e:
            self.logger.error(f"检查文件变更状态失败: {file_path}, 错误: {e}")
            # 出错时默认认为文件已更改
            return True

    def update_file_timestamp(self, file_path: str):
        """
        更新文件时间戳

        Args:
            file_path: 文件路径
        """
        try:
            # 获取文件当前修改时间
            modified_time = os.path.getmtime(file_path)

            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 使用UPSERT操作更新或插入记录
                cursor.execute('''
                    INSERT OR REPLACE INTO file_timestamps (file_path, modified_time)
                    VALUES (?, ?)
                ''', (file_path, modified_time))

                conn.commit()
                conn.close()

        except Exception as e:
            self.logger.error(f"更新文件时间戳失败: {file_path}, 错误: {e}")

    def remove_file_timestamp(self, file_path: str):
        """
        移除文件时间戳记录

        Args:
            file_path: 文件路径
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    'DELETE FROM file_timestamps WHERE file_path = ?',
                    (file_path,)
                )

                conn.commit()
                conn.close()

        except Exception as e:
            self.logger.error(f"移除文件时间戳记录失败: {file_path}, 错误: {e}")

    def get_all_file_timestamps(self) -> Dict[str, float]:
        """
        获取所有文件时间戳

        Returns:
            文件路径到修改时间的映射
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('SELECT file_path, modified_time FROM file_timestamps')
                results = cursor.fetchall()
                conn.close()

                return {file_path: modified_time for file_path, modified_time in results}

        except Exception as e:
            self.logger.error(f"获取所有文件时间戳失败: {e}")
            return {}

    def close(self):
        """关闭数据库连接"""
        # SQLite会自动管理连接，这里主要是为了API完整性
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()