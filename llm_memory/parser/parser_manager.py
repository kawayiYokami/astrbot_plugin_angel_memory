"""
解析器管理器

用于动态注册和管理不同的文件解析器。
"""

from typing import Dict, Type, Set
from pathlib import Path
from .markdown_parser import MarkdownParser
from .universal_parser import UniversalParser

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ParserManager:
    """解析器管理器"""

    def __init__(self):
        """初始化解析器管理器"""
        self.logger = logger

        # 解析器注册表：扩展名 -> 解析器类
        self.parsers: Dict[str, Type] = {}

        # 自动注册默认解析器
        self._register_default_parsers()

    def _register_default_parsers(self):
        """注册默认解析器"""
        # 注册Markdown解析器支持的所有扩展名
        for extension in MarkdownParser.SUPPORTED_EXTENSIONS:
            self.register_parser(extension, MarkdownParser)

        # 默认启用MarkItDown功能，注册通用解析器支持的扩展名（除了MarkdownParser已支持的）
        universal_extensions = UniversalParser.SUPPORTED_EXTENSIONS - MarkdownParser.SUPPORTED_EXTENSIONS
        for extension in universal_extensions:
            self.register_parser(extension, UniversalParser)

    def register_parser(self, extension: str, parser_class: Type):
        """
        注册解析器

        Args:
            extension: 文件扩展名（如 '.md'）
            parser_class: 解析器类
        """
        extension = extension.lower()
        self.parsers[extension] = parser_class
        logger.info(f"注册解析器: {extension} -> {parser_class.__name__}")

    def get_parser_for_extension(self, extension: str, tag_manager=None):
        """
        获取指定扩展名的解析器实例

        Args:
            extension: 文件扩展名（如 '.md'）
            tag_manager: 标签管理器实例（可选）

        Returns:
            解析器实例，如果未找到则返回None
        """
        extension = extension.lower()
        parser_class = self.parsers.get(extension)
        if parser_class:
            return parser_class(tag_manager)
        return None

    def get_supported_extensions(self) -> Set[str]:
        """
        获取所有支持的文件扩展名

        Returns:
            支持的文件扩展名集合
        """
        return set(self.parsers.keys())

    def is_supported_extension(self, extension: str) -> bool:
        """
        检查是否支持指定的文件扩展名

        Args:
            extension: 文件扩展名（如 '.md'）

        Returns:
            是否支持
        """
        return extension.lower() in self.parsers

    def get_parser_for_file(self, file_path: str, tag_manager=None):
        """
        根据文件路径获取相应的解析器实例

        Args:
            file_path: 文件路径
            tag_manager: 标签管理器实例（可选）

        Returns:
            解析器实例，如果未找到则返回None
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        return self.get_parser_for_extension(extension, tag_manager)


# 全局解析器管理器实例
parser_manager = ParserManager()