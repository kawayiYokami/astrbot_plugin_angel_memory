"""
通用解析器

封装MarkItDown + MarkdownParser，支持多种文件格式转换为文档块。
"""

import asyncio
from typing import List, Set
from pathlib import Path

from ..models.document_models import DocumentBlock
from .markdown_parser import MarkdownParser
from astrbot.api import logger

# 尝试导入MarkItDown，如果失败则记录错误
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError as e:
    MARKITDOWN_AVAILABLE = False
    logger.warning(f"MarkItDown导入失败: {e}")


class UniversalParser:
    """通用文件解析器，支持多种文件格式"""

    # 支持的文件扩展名（包括MarkItDown支持的格式）
    SUPPORTED_EXTENSIONS: Set[str] = {
        # MarkdownParser支持的格式
        '.md', '.txt',
        # MarkItDown支持的文档格式
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.html', '.epub',
        # MarkItDown支持的数据格式
        '.csv', '.json', '.xml',
        # MarkItDown支持的媒体格式
        '.jpg', '.jpeg', '.png', '.mp3', '.wav',
        # MarkItDown支持的容器格式
        '.zip'
    }

    def __init__(self):
        """初始化通用解析器"""
        self.logger = logger
        self.markdown_parser = MarkdownParser()

        # 初始化MarkItDown转换器
        if MARKITDOWN_AVAILABLE:
            try:
                self.markitdown = MarkItDown()
                self.logger.info("MarkItDown初始化成功")
            except Exception as e:
                self.markitdown = None
                self.logger.error(f"MarkItDown初始化失败: {e}")
        else:
            self.markitdown = None
            self.logger.warning("MarkItDown不可用，仅支持Markdown和TXT格式")

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """
        检查是否支持指定的文件扩展名

        Args:
            extension: 文件扩展名（如 '.pdf'）

        Returns:
            是否支持
        """
        return extension.lower() in cls.SUPPORTED_EXTENSIONS

    def parse(self, content: str, file_path: str = "") -> List[DocumentBlock]:
        """
        解析文件内容

        Args:
            content: 文件内容（对于非文本文件，这应该是文件路径）
            file_path: 文件路径

        Returns:
            文档块列表
        """
        try:
            # 获取文件扩展名
            path = Path(file_path)
            extension = path.suffix.lower()

            # 如果是Markdown或TXT文件，直接使用MarkdownParser
            if extension in ['.md', '.txt']:
                return self.markdown_parser.parse(content, file_path)

            # 对于其他支持的格式，使用MarkItDown转换
            if self.markitdown and extension in self.SUPPORTED_EXTENSIONS:
                return self._parse_with_markitdown(file_path)

            # 如果不支持该格式，返回空列表
            self.logger.warning(f"不支持的文件格式: {extension}")
            return []

        except Exception as e:
            self.logger.error(f"解析文件失败: {file_path}, 错误: {e}")
            # 出错时返回空列表而不是抛出异常
            return []

    def _parse_with_markitdown(self, file_path: str) -> List[DocumentBlock]:
        """
        使用MarkItDown解析文件

        Args:
            file_path: 文件路径

        Returns:
            文档块列表
        """
        try:
            # 使用MarkItDown转换文件为Markdown
            with open(file_path, 'rb') as f:
                conversion_result = self.markitdown.convert_stream(f)

            markdown_content = conversion_result.text_content

            # 使用MarkdownParser解析转换后的Markdown内容
            return self.markdown_parser.parse(markdown_content, file_path)

        except Exception as e:
            self.logger.error(f"使用MarkItDown解析文件失败: {file_path}, 错误: {e}")
            # 出错时返回空列表而不是抛出异常
            return []

    async def async_parse(self, file_path: str) -> List[DocumentBlock]:
        """
        异步解析文件（不阻塞主流程）

        Args:
            file_path: 文件路径

        Returns:
            文档块列表
        """
        # 在线程池中运行CPU密集型任务
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse_file, file_path)

    def parse_file(self, file_path: str) -> List[DocumentBlock]:
        """
        解析文件（读取文件内容并解析）

        Args:
            file_path: 文件路径

        Returns:
            文档块列表
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()

            # 对于二进制文件，直接传递文件路径给MarkItDown
            if extension in ['.md', '.txt']:
                # 文本文件读取内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.parse(content, file_path)
            else:
                # 二进制文件直接使用文件路径
                return self.parse("", file_path)

        except Exception as e:
            self.logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            return []