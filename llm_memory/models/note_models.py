"""
笔记数据模型 - 精简后的笔记数据结构

本模块定义了笔记系统的专用数据模型，经过优化删除了冗余字段。
"""

import json
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NoteData:
    """
    笔记数据类 - 精简的笔记数据模型

    设计原则：
    1. 只保留必要字段
    2. 使用ID引用替代字符串存储
    3. 优化检索性能
    4. 简化数据结构
    """

    # 核心字段
    id: str  # 唯一标识符
    content: str  # 笔记内容
    file_id: int  # 文件索引ID（引用file_index表）
    tag_ids: List[int]  # 标签ID列表（引用tag表）
    related_block_ids: Optional[str] = None  # 逗号分隔的关联块ID，用于上下文重建

    @classmethod
    def create_file_block(
        cls,
        block_id: str,
        content: str,
        file_id: int,
        tag_ids: List[int],
        related_ids: List[str] = None,
    ) -> "NoteData":
        """
        创建文件块笔记

        Args:
            block_id: 块ID
            content: 内容
            file_id: 文件索引ID
            tag_ids: 标签ID列表
            related_ids: 关联块ID列表

        Returns:
            NoteData实例
        """
        return cls(
            id=block_id,
            content=content,
            file_id=file_id,
            tag_ids=tag_ids,
            related_block_ids=",".join(related_ids) if related_ids else "",
        )

    @classmethod
    def create_user_note(cls, content: str, tag_ids: List[int] = None) -> "NoteData":
        """
        创建用户笔记（不关联文件）

        Args:
            content: 笔记内容
            tag_ids: 标签ID列表

        Returns:
            NoteData实例
        """
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            file_id=0,  # 用户笔记不关联文件，使用0表示
            tag_ids=tag_ids or [],
            related_block_ids=None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（用于存储）

        Returns:
            字典格式的笔记数据
        """
        return {
            "id": self.id,
            "content": self.content,
            "file_id": self.file_id,
            "tag_ids": json.dumps(self.tag_ids),  # 转为JSON字符串存储
            "related_block_ids": self.related_block_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoteData":
        """
        从字典创建NoteData实例

        Args:
            data: 字典数据

        Returns:
            NoteData实例
        """
        # 解析tag_ids JSON字符串为列表
        tag_ids_str = data.get("tag_ids", "[]")
        try:
            tag_ids = (
                json.loads(tag_ids_str) if isinstance(tag_ids_str, str) else tag_ids_str
            )
        except json.JSONDecodeError:
            tag_ids = []

        return cls(
            id=data["id"],
            content=data["content"],
            file_id=data.get("file_id", 0),
            tag_ids=tag_ids,
            related_block_ids=data.get("related_block_ids"),
        )

    def get_embedding_text(self, tag_names: List[str] = None) -> str:
        """
        获取用于向量化的文本

        Args:
            tag_names: 标签名称列表（需要从外部传入，因为内部只有tag_ids）

        Returns:
            用于向量化的文本
        """
        if tag_names:
            tags_text = " ".join(tag_names)
            return f"{self.content}\n\nTags: {tags_text}"
        return self.content

    def get_tags_text(self, tag_names: List[str]) -> str:
        """
        获取纯标签文本（用于副集合）

        Args:
            tag_names: 标签名称列表

        Returns:
            标签文本
        """
        return " ".join(tag_names)

    def __str__(self) -> str:
        """字符串表示"""
        content_preview = (
            self.content[:30] + "..." if len(self.content) > 30 else self.content
        )
        return f"Note(id={self.id[:8]}..., file_id={self.file_id}, content='{content_preview}', tags={len(self.tag_ids)})"
