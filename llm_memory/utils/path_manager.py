"""
路径管理工具

提供统一的路径管理功能，简化项目中的路径操作。
"""

import sys
from pathlib import Path
from typing import Optional


class PathManager:
    """路径管理工具类"""

    _project_root: Optional[Path] = None

    @classmethod
    def get_project_root(cls) -> Path:
        """获取项目根目录"""
        if cls._project_root is None:
            # 从当前文件位置向上查找项目根目录
            current_dir = Path(__file__).parent
            # 假设项目根目录是llm_memory的上级目录的上级目录
            cls._project_root = current_dir.parent.parent
        return cls._project_root

    @classmethod
    def get_prompt_path(cls) -> Path:
        """获取提示词文件路径"""
        return cls.get_project_root() / "llm_memory" / "prompts" / "memory_system_guide.md"

    @classmethod
    def get_storage_dir(cls) -> Path:
        """获取存储目录路径"""
        return cls.get_project_root() / "storage"

    @classmethod
    def get_index_dir(cls) -> Path:
        """获取索引目录路径"""
        return cls.get_storage_dir() / "index"

    @classmethod
    def ensure_project_in_path(cls):
        """确保项目根目录在Python路径中"""
        project_root = str(cls.get_project_root())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    @classmethod
    def ensure_directories_exist(cls):
        """确保所有必要的目录都存在"""
        storage_dir = cls.get_storage_dir()
        index_dir = cls.get_index_dir()

        storage_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)