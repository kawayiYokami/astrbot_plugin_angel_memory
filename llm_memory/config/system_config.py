"""
记忆系统配置管理
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


# 记忆系统常量
KNOWLEDGE_CORE_SEPARATOR = ", "  # 用于连接 judgment 和 tags 的分隔符


@dataclass
class MemorySystemConfig:
    """记忆系统配置类"""

    # 嵌入模型
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    ))

    # 嵌入式模型提供商ID
    astrbot_embedding_provider_id: str = field(default_factory=lambda: os.getenv(
        "MEMORY_ASTRBOT_EMBEDDING_PROVIDER_ID", ""
    ))

    # 集合名称配置
    collection_name: str = field(default_factory=lambda: os.getenv(
        "MEMORY_COLLECTION_NAME", "personal_memory_v1"
    ))

    # 笔记服务专用集合名称
    notes_main_collection_name: str = field(default_factory=lambda: os.getenv(
        "NOTES_MAIN_COLLECTION_NAME", "notes_main"
    ))

    notes_sub_collection_name: str = field(default_factory=lambda: os.getenv(
        "NOTES_SUB_COLLECTION_NAME", "notes_sub"
    ))

    # 注意：storage_dir 和 index_dir 应该从 PathManager 实例获取（在设置供应商后）
    # 这里不再设置默认值，由外部传入或通过 PathManager 动态获取
    storage_dir: Path = field(default=None)
    index_dir: Path = field(default=None)

    # 阈值配置
    strength_threshold: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_STRENGTH_THRESHOLD", "2"
    )))

    association_threshold: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_ASSOCIATION_THRESHOLD", "1"
    )))

    # 检索限制
    default_recall_limit: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_DEFAULT_RECALL_LIMIT", "10"
    )))

    fresh_recall_limit: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_FRESH_RECALL_LIMIT", "5"
    )))

    consolidated_recall_limit: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_CONSOLIDATED_RECALL_LIMIT", "5"
    )))

    # 巩固间隔（小时）
    consolidation_interval_hours: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_CONSOLIDATION_INTERVAL_HOURS", "24"
    )))

    def __post_init__(self):
        """初始化后处理"""
        # 确保路径是Path对象
        if isinstance(self.storage_dir, str):
            self.storage_dir = Path(self.storage_dir)
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir)

        # 验证配置
        if self.strength_threshold < 1:
            raise ValueError("strength_threshold 必须 >= 1")
        if self.association_threshold < 0:
            raise ValueError("association_threshold 必须 >= 0")
        if self.consolidation_interval_hours < 1:
            raise ValueError("consolidation_interval_hours 必须 >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "embedding_model": self.embedding_model,
            "astrbot_embedding_provider_id": self.astrbot_embedding_provider_id,
            "collection_name": self.collection_name,
            "storage_dir": str(self.storage_dir),
            "index_dir": str(self.index_dir),
            "strength_threshold": self.strength_threshold,
            "association_threshold": self.association_threshold,
            "default_recall_limit": self.default_recall_limit,
            "fresh_recall_limit": self.fresh_recall_limit,
            "consolidated_recall_limit": self.consolidated_recall_limit,
            "consolidation_interval_hours": self.consolidation_interval_hours,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemorySystemConfig':
        """从字典创建配置"""
        if 'storage_dir' in config_dict:
            config_dict['storage_dir'] = Path(config_dict['storage_dir'])
        if 'index_dir' in config_dict:
            config_dict['index_dir'] = Path(config_dict['index_dir'])

        return cls(**config_dict)

    def ensure_directories_exist(self):
        """确保目录存在"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def get_database_path(self, provider_id: Optional[str] = None) -> Path:
        """
        获取数据库路径

        Args:
            provider_id: 提供商ID，如果提供则使用提供商专用路径

        Returns:
            数据库路径
        """
        if provider_id:
            return self.index_dir / f"chromadb_{provider_id}"
        else:
            return self.index_dir / "chromadb"


# 全局配置实例
system_config = MemorySystemConfig()