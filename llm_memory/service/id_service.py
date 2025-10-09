"""
ID转换服务 - 统一的ID转换接口

提供标签和文件路径与整数ID之间的双向转换功能，
封装TagManager和FileIndexManager，为上层服务提供统一的API。
"""

from typing import Dict, List, Optional, Any

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
    统一的ID转换服务

    封装TagManager和FileIndexManager，提供语义清晰的ID转换接口。
    支持标签和文件路径与整数ID之间的双向转换。
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

    # ===== 标签ID转换 =====

    def tag_to_id(self, tag_name: str) -> int:
        """
        单个标签名转ID

        Args:
            tag_name: 标签名称

        Returns:
            标签ID
        """
        try:
            return self.tag_manager.get_or_create_tag_id(tag_name)
        except Exception as e:
            self.logger.error(f"标签转ID失败: {tag_name}, 错误: {e}")
            raise IDConversionError(f"标签转ID失败: {tag_name}") from e

    def id_to_tag(self, tag_id: int) -> Optional[str]:
        """
        单个ID转标签名

        Args:
            tag_id: 标签ID

        Returns:
            标签名称，如果不存在返回None
        """
        try:
            return self.tag_manager.get_tag_name(tag_id)
        except Exception as e:
            self.logger.error(f"ID转标签失败: {tag_id}, 错误: {e}")
            return None

    def tags_to_ids(self, tag_names: List[str]) -> List[int]:
        """
        批量标签名转ID

        Args:
            tag_names: 标签名称列表

        Returns:
            标签ID列表，顺序与输入标签名称对应
        """
        if not tag_names:
            return []

        try:
            return self.tag_manager.get_or_create_tag_ids(tag_names)
        except Exception as e:
            self.logger.error(f"批量标签转ID失败: {tag_names}, 错误: {e}")
            raise IDConversionError(f"批量标签转ID失败: {tag_names}") from e

    def ids_to_tags(self, tag_ids: List[int]) -> List[str]:
        """
        批量ID转标签名

        Args:
            tag_ids: 标签ID列表

        Returns:
            标签名称列表，顺序与输入ID对应
        """
        if not tag_ids:
            return []

        try:
            return self.tag_manager.get_tag_names(tag_ids)
        except Exception as e:
            self.logger.error(f"批量ID转标签失败: {tag_ids}, 错误: {e}")
            return []

    # ===== 文件ID转换 =====

    def file_to_id(self, file_path: str, timestamp: int = 0) -> int:
        """
        单个文件路径转ID

        Args:
            file_path: 文件路径
            timestamp: 文件时间戳（可选）

        Returns:
            文件ID
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

    def files_to_ids(self, file_paths: List[str], timestamps: List[int] = None) -> List[int]:
        """
        批量文件路径转ID

        Args:
            file_paths: 文件路径列表
            timestamps: 对应的时间戳列表（可选）

        Returns:
            文件ID列表，顺序与输入文件路径对应
        """
        if not file_paths:
            return []

        try:
            return self.file_manager.batch_get_or_create_file_ids(file_paths, timestamps)
        except Exception as e:
            self.logger.error(f"批量文件转ID失败: {file_paths}, 错误: {e}")
            raise IDConversionError(f"批量文件转ID失败: {file_paths}") from e

    def ids_to_files(self, file_ids: List[int]) -> List[str]:
        """
        批量ID转文件路径

        Args:
            file_ids: 文件ID列表

        Returns:
            文件路径列表，顺序与输入ID对应
        """
        if not file_ids:
            return []

        try:
            return self.file_manager.batch_get_file_paths(file_ids)
        except Exception as e:
            self.logger.error(f"批量ID转文件失败: {file_ids}, 错误: {e}")
            return []

    # ===== 高级转换接口 =====

    def convert_note_data(self, tags: List[str] = None, file_path: str = None,
                         timestamp: int = 0) -> Dict[str, Any]:
        """
        统一的笔记数据转换接口

        Args:
            tags: 标签列表（可选）
            file_path: 文件路径（可选）
            timestamp: 文件时间戳（可选）

        Returns:
            包含转换结果的字典:
            - tag_ids: 标签ID列表（如果提供了tags）
            - file_id: 文件ID（如果提供了file_path）
            - tag_names: 原始标签名称列表（如果提供了tags）
            - file_path: 原始文件路径（如果提供了file_path）
        """
        result = {}

        try:
            if tags:
                result['tag_ids'] = self.tags_to_ids(tags)
                result['tag_names'] = tags  # 保存原始标签名

            if file_path:
                result['file_id'] = self.file_to_id(file_path, timestamp)
                result['file_path'] = file_path  # 保存原始文件路径

            return result

        except Exception as e:
            self.logger.error(f"笔记数据转换失败: tags={tags}, file_path={file_path}, 错误: {e}")
            raise IDConversionError("笔记数据转换失败") from e

    def batch_convert(self, data: Dict[str, List]) -> Dict[str, List[int]]:
        """
        批量转换接口

        Args:
            data: 包含待转换数据的字典，支持的键:
                  - 'tags': 标签名称列表
                  - 'files': 文件路径列表

        Returns:
            包含转换结果的字典:
            - 'tag_ids': 标签ID列表（如果输入包含'tags'）
            - 'file_ids': 文件ID列表（如果输入包含'files'）
        """
        result = {}

        try:
            if 'tags' in data:
                result['tag_ids'] = self.tags_to_ids(data['tags'])

            if 'files' in data:
                result['file_ids'] = self.files_to_ids(data['files'])

            return result

        except Exception as e:
            self.logger.error(f"批量转换失败: {data}, 错误: {e}")
            raise IDConversionError("批量转换失败") from e

    # ===== 管理功能 =====

    def get_all_tags(self) -> List[Dict]:
        """
        获取所有标签

        Returns:
            标签列表，每个标签包含id和name
        """
        try:
            return self.tag_manager.get_all_tags()
        except Exception as e:
            self.logger.error(f"获取所有标签失败: {e}")
            return []

    def get_all_files(self) -> List[Dict]:
        """
        获取所有文件索引

        Returns:
            文件索引列表，每个文件包含id, relative_path, file_timestamp
        """
        try:
            return self.file_manager.get_all_files()
        except Exception as e:
            self.logger.error(f"获取所有文件失败: {e}")
            return []

    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            包含统计信息的字典:
            - total_tags: 标签总数
            - total_files: 文件总数
        """
        try:
            tag_stats = self.tag_manager.get_tag_statistics()
            file_stats = self.file_manager.get_file_statistics() if hasattr(self.file_manager, 'get_file_statistics') else {'total_files': 0}

            return {
                'total_tags': tag_stats.get('total_tags', 0),
                'total_files': file_stats.get('total_files', 0)
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {'total_tags': 0, 'total_files': 0}

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

    def __repr__(self) -> str:
        """详细表示"""
        return (f"IDService(provider='{self.provider_id}', "
                f"data_directory='{self.data_directory}', "
                f"tag_manager={self.tag_manager}, "
                f"file_manager={self.file_manager})")

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

    def update_plugin_context(self, plugin_context):
        """
        更新PluginContext（用于重新配置）

        Args:
            plugin_context: 新的PluginContext
        """
        self.plugin_context = plugin_context
        self.data_directory = str(plugin_context.get_index_dir())
        self.provider_id = plugin_context.get_current_provider()

        # 重新初始化底层管理器
        self.tag_manager.close()
        self.file_manager.close()

        self.tag_manager = TagManager(self.data_directory, self.provider_id)
        self.file_manager = FileIndexManager(self.data_directory, self.provider_id)

        self.logger.info(f"ID服务已更新 (提供商: {self.provider_id}, 目录: {self.data_directory})")