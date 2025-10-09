"""
路径管理工具 - 中央路径管理器

提供统一的路径管理功能，支持供应商数据库分离，
每个供应商拥有独立的数据目录结构。
"""

import re
from pathlib import Path
from typing import Optional, Dict

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PathManager:
    """中央路径管理器 - 统一管理所有路径和供应商数据库分离"""

    _instance = None
    _project_root: Optional[Path] = None
    _current_provider: Optional[str] = None
    _base_dir: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'PathManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _get_plugin_root(cls) -> Path:
        """获取插件根目录（仅内部使用，用于定位提示词文件）"""
        if cls._project_root is None:
            # 从当前文件位置向上查找插件根目录
            current_dir = Path(__file__).parent
            # llm_memory/utils -> llm_memory -> astrbot_plugin_angel_memory
            cls._project_root = current_dir.parent.parent
        return cls._project_root

    # === 供应商管理 ===

    def set_provider(self, provider_id: str, base_data_dir: str):
        """
        设置供应商并配置所有路径

        Args:
            provider_id: 供应商ID（用于数据库分表）
            base_data_dir: 基础数据目录（必需，由外部传入）
        """
        # 只检查空值，不验证格式（上游会验证）
        if not provider_id or not provider_id.strip():
            raise ValueError("供应商ID不能为空")

        # 强制要求传入数据目录
        if not base_data_dir:
            raise ValueError("必须提供 base_data_dir 参数！数据目录应由外部传入，不应自动推测。")

        # 使用过滤后的安全ID（仅用于文件名）
        safe_provider_id = re.sub(r'[<>:"/\\|?*]', '_', provider_id.strip())

        self._current_provider = safe_provider_id

        # 设置基础目录
        self._base_dir = Path(base_data_dir) / f"memory_{safe_provider_id}"

        # 创建目录结构
        self._ensure_provider_directories_exist()

        logger.info(f"PathManager供应商设置完成: {provider_id} -> {safe_provider_id}, 基础目录: {self._base_dir}")

    
    def get_current_provider(self) -> str:
        """获取当前供应商ID"""
        if self._current_provider is None:
            raise ValueError("供应商未设置，请先调用set_provider()")
        return self._current_provider

    def is_provider_set(self) -> bool:
        """检查是否已设置供应商"""
        return self._current_provider is not None

    def _ensure_provider_directories_exist(self):
        """确保供应商目录结构存在"""
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            (self._base_dir / "index").mkdir(exist_ok=True)
            (self._base_dir / "chromadb").mkdir(exist_ok=True)
            (self._base_dir / "logs").mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"创建供应商目录失败: {e}")
            raise

    # === 路径获取方法 ===

    def get_index_dir(self) -> Path:
        """获取索引目录"""
        if not self.is_provider_set():
            raise ValueError("供应商未设置，无法获取索引目录")
        return self._base_dir / "index"

    def get_tag_db_path(self) -> Path:
        """获取标签数据库路径"""
        if not self.is_provider_set():
            raise ValueError("供应商未设置，无法获取标签数据库路径")
        return self.get_index_dir() / f"tag_{self._current_provider}.db"

    def get_file_db_path(self) -> Path:
        """获取文件数据库路径"""
        if not self.is_provider_set():
            raise ValueError("供应商未设置，无法获取文件数据库路径")
        return self.get_index_dir() / f"file_{self._current_provider}.db"

    def get_chroma_db_path(self) -> Path:
        """获取Chroma数据库路径"""
        if not self.is_provider_set():
            raise ValueError("供应商未设置，无法获取Chroma数据库路径")
        return self._base_dir / "chromadb"

    def get_logs_dir(self) -> Path:
        """获取日志目录"""
        if not self.is_provider_set():
            raise ValueError("供应商未设置，无法获取日志目录")
        return self._base_dir / "logs"

    # === 对外路径方法（仅提示词路径） ===

    @classmethod
    def get_prompt_path(cls) -> Path:
        """获取提示词文件路径（插件根目录的唯一对外用途）"""
        return cls._get_plugin_root() / "llm_memory" / "prompts" / "memory_system_guide.md"

    # === 便捷方法 ===

    def list_databases(self) -> Dict[str, Path]:
        """列出当前供应商的所有数据库文件"""
        if not self.is_provider_set():
            return {}

        return {
            "tag_db": self.get_tag_db_path(),
            "file_db": self.get_file_db_path(),
            "chroma_db": self.get_chroma_db_path()
        }

    def get_database_info(self) -> Dict[str, any]:
        """获取当前供应商的数据库信息"""
        if not self.is_provider_set():
            return {}

        return {
            "provider_id": self.get_current_provider(),
            "base_dir": str(self._base_dir),
            "databases": self.list_databases(),
            "provider_name": self._get_provider_display_name()
        }

    def _get_provider_display_name(self) -> str:
        """获取供应商显示名称"""
        provider_id = self.get_current_provider()
        provider_names = {
            "local": "本地模型",
            "openai": "OpenAI",
            "claude": "Claude",
            "gemini": "Gemini",
            "azure": "Azure OpenAI"
        }
        return provider_names.get(provider_id, provider_id.upper())

    def cleanup_provider_data(self) -> bool:
        """清理当前供应商的所有数据（谨慎使用）"""
        if not self.is_provider_set():
            return False

        try:
            import shutil
            provider_id = self.get_current_provider()

            logger.warning(f"正在清理供应商 {provider_id} 的所有数据: {self._base_dir}")
            shutil.rmtree(self._base_dir, ignore_errors=True)

            # 重置状态
            self._current_provider = None
            self._base_dir = None

            logger.info(f"供应商 {provider_id} 数据清理完成")
            return True
        except Exception as e:
            logger.error(f"清理供应商数据失败: {e}")
            return False

    def __str__(self) -> str:
        """字符串表示"""
        if self.is_provider_set():
            return f"PathManager(provider={self.get_current_provider()})"
        return "PathManager(未设置供应商)"

    def __repr__(self) -> str:
        """详细表示"""
        if self.is_provider_set():
            return (f"PathManager("
                    f"provider='{self.get_current_provider()}', "
                    f"base_dir='{self._base_dir}', "
                    f"databases={list(self.list_databases().keys())}"
                    f")")
        return "PathManager(未设置供应商)"