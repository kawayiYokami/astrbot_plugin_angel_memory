"""
时间戳管理器

使用SQLite数据库管理文件时间戳信息，提升频繁读写场景下的性能。
"""

from pathlib import Path
from typing import Dict
from astrbot.api import logger
from .file_database import FileDatabase


class TimestampManager:
    """时间戳管理器"""

    def __init__(self, data_directory: str):
        """
        初始化时间戳管理器

        Args:
            data_directory: 数据目录
        """
        self.logger = logger
        self.data_directory = Path(data_directory)

        # 使用SQLite数据库替代JSON文件
        self.file_db = FileDatabase(data_directory)

    def is_file_changed(self, file_path: str) -> bool:
        """
        检查文件是否已更改

        Args:
            file_path: 文件路径

        Returns:
            文件是否已更改
        """
        return self.file_db.is_file_changed(file_path)

    def update_file_timestamp(self, file_path: str):
        """
        更新文件的时间戳

        Args:
            file_path: 文件路径
        """
        self.file_db.update_file_timestamp(file_path)

    def remove_file_timestamp(self, file_path: str):
        """
        移除文件的时间戳

        Args:
            file_path: 文件路径
        """
        self.file_db.remove_file_timestamp(file_path)

    def get_all_file_timestamps(self) -> Dict[str, float]:
        """
        获取所有文件时间戳

        Returns:
            文件路径到修改时间的映射
        """
        return self.file_db.get_all_file_timestamps()

    def close(self):
        """关闭时间戳管理器"""
        if self.file_db:
            self.file_db.close()
