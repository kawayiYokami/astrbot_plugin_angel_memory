"""
通用解析器

仅支持 Markdown 和 TXT 格式。MarkItDown 依赖 onnxruntime（CUDA 绑定问题），
已在 requirements.txt 中移除依赖，如有需要可自行安装。
"""

import asyncio
from typing import List, Set
from pathlib import Path

from ..models.note_models import NoteData
from .markdown_parser import MarkdownParser

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# MarkItDown 已禁用，依赖 onnxruntime 导致安装问题
# 如需启用文档转换功能，请自行安装 markitdown[all] 并取消下方注释
# try:
#     from markitdown import MarkItDown
#     MARKITDOWN_AVAILABLE = True
# except ImportError as e:
#     logger.warning(f"MarkItDown导入失败: {e}")
MARKITDOWN_AVAILABLE = False


class UniversalParser:
    """通用文件解析器，仅保留 Markdown/TXT 路径。"""

    SUPPORTED_EXTENSIONS: Set[str] = {
        ".md",
        ".txt",
    }

    def __init__(self, tag_manager=None):
        """初始化通用解析器"""
        self.logger = logger
        self.markdown_parser = MarkdownParser(tag_manager)
        self.tag_manager = tag_manager

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

    def parse(self, content: str, file_id: int, file_path: str = "") -> List[NoteData]:
        """
        解析文件内容

        Args:
            content: 文件内容（对于非文本文件，这应该是文件路径）
            file_id: 文件索引ID
            file_path: 文件路径

        Returns:
            NoteData列表
        """
        try:
            # 获取文件扩展名
            path = Path(file_path)
            extension = path.suffix.lower()

            # 如果是Markdown或TXT文件，直接使用MarkdownParser
            if extension in [".md", ".txt"]:
                return self.markdown_parser.parse(content, file_id, file_path)

            # 如果不支持该格式，返回空列表
            self.logger.warning(f"不支持的文件格式: {extension}")
            return []

        except Exception as e:
            self.logger.error(f"解析文件失败: {file_path}, 错误: {e}")
            # 出错时返回空列表而不是抛出异常
            return []


    async def async_parse(
        self, file_path: str, tag_manager=None, file_id: int = 0
    ) -> List[NoteData]:
        """
        异步解析文件（不阻塞主流程）

        Args:
            file_path: 文件路径
            tag_manager: 标签管理器
            file_id: 文件索引ID

        Returns:
            NoteData列表
        """
        # 在线程池中运行CPU密集型任务
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.parse_file, file_path, tag_manager, file_id
        )

    def parse_file(
        self, file_path: str, tag_manager=None, file_id: int = 0
    ) -> List[NoteData]:
        """
        解析文件（读取文件内容并解析）

        Args:
            file_path: 文件路径
            tag_manager: 标签管理器
            file_id: 文件索引ID

        Returns:
            NoteData列表
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()

            # 如果提供了tag_manager，更新markdown_parser
            if tag_manager and hasattr(self.markdown_parser, "tag_manager"):
                self.markdown_parser.tag_manager = tag_manager

            # 获取file_id（如果没有提供则生成）
            if file_id == 0:
                file_timestamp = int(path.stat().st_mtime)
                # 临时使用文件名作为相对路径
                relative_path = path.name
                # 如果没有file_index_manager，使用简单hash生成file_id
                file_id = hash(relative_path + str(file_timestamp)) & 0x7FFFFFFF

            # 读取文本文件
            if extension in [".md", ".txt"]:
                # 文本文件读取内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return self.parse(content, file_id, file_path)

        except Exception as e:
            self.logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            return []
