"""
ID转换服务 - 精简版ID转换接口

专注提供核心的ID转换功能：批量标签ID转换、文件ID转换和资源管理。
封装TagManager和FileIndexManager，为上层服务提供高效的API。
"""

from typing import List, Optional

from ..components.tag_manager import TagManager
from ..components.file_index_manager import FileIndexManager

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class IDServiceError(Exception):
    """ID服务异常基类"""
    pass


class IDConversionError(IDServiceError):
    """ID转换失败异常"""
    pass


class IDService:
    """
    精简版ID转换服务

    专注核心功能，提供批量标签ID转换、单个文件ID转换和资源管理。
    封装TagManager和FileIndexManager，为上层服务提供高效的API。
    """

    def __init__(self, plugin_context=None):
        """
        初始化ID服务

        Args:
            plugin_context: PluginContext插件上下文（可选，如果不提供则使用默认值）
        """
        self.logger = logger
        self.plugin_context = plugin_context

        # 从PluginContext获取资源，或使用默认值
        if plugin_context:
            self.data_directory = str(plugin_context.get_index_dir())
            self.provider_id = plugin_context.get_current_provider()
        else:
            # 回退到默认值
            self.data_directory = "storage/index"
            self.provider_id = "default"
            self.logger.warning("PluginContext未提供，使用默认配置")

        # 初始化底层管理器
        self.tag_manager = TagManager(self.data_directory, self.provider_id)
        self.file_manager = FileIndexManager(self.data_directory, self.provider_id)

        self.logger.info(f"ID服务初始化完成 (提供商: {self.provider_id}, 目录: {self.data_directory})")

    def ids_to_tags(self, tag_ids: List[int]) -> List[str]:
        """
        批量ID转标签名

        Args:
            tag_ids: 标签ID列表

        Returns:
            标签名称列表，顺序与输入ID对应。为保持向后兼容，返回空列表而非抛出异常
        """
        if not tag_ids:
            return []

        try:
            return self.tag_manager.get_tag_names(tag_ids)
        except Exception as e:
            self.logger.error(f"批量ID转标签失败: {tag_ids}, 错误: {e}")
            return []

    def file_to_id(self, file_path: str, timestamp: int = 0) -> int:
        """
        单个文件路径转ID

        Args:
            file_path: 文件路径
            timestamp: 文件时间戳（可选）

        Returns:
            文件ID

        Raises:
            IDConversionError: 当转换失败时抛出异常
        """
        try:
            return self.file_manager.get_or_create_file_id(file_path, timestamp)
        except Exception as e:
            self.logger.error(f"文件转ID失败: {file_path}, 错误: {e}")
            raise IDConversionError(f"文件转ID失败: {file_path}") from e

    def id_to_file(self, file_id: int) -> Optional[str]:
        """
        单个ID转文件路径

        Args:
            file_id: 文件ID

        Returns:
            文件路径，如果不存在返回None
        """
        try:
            return self.file_manager.get_file_path(file_id)
        except Exception as e:
            self.logger.error(f"ID转文件失败: {file_id}, 错误: {e}")
            return None

    def close(self):
        """关闭服务，释放资源"""
        try:
            self.tag_manager.close()
            self.file_manager.close()
        except Exception as e:
            self.logger.error(f"关闭ID服务失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        """字符串表示"""
        return f"IDService(provider={self.provider_id}, dir={self.data_directory})"

    @classmethod
    def from_plugin_context(cls, plugin_context):
        """
        从PluginContext创建IDService实例

        Args:
            plugin_context: PluginContext插件上下文

        Returns:
            IDService实例
        """
        return cls(plugin_context)