"""
文档数据模型

用于文档分析的结构化数据模型。
"""

from dataclasses import dataclass
from typing import List
import uuid
import time


@dataclass
class DocumentBlock:
    """文档块模型"""

    id: str                # 唯一标识符
    content: str           # 正文内容
    tags: List[str]        # 标签列表
    tag_vector: List[float] # 标签向量值
    source_file_hash: str  # 源文件哈希值
    created_at: float      # 创建时间
    related_block_ids: List[str]  # 关联块ID列表（按顺序的前后块）
    source_file_path: str = ""  # 源文件路径

    def __post_init__(self):
        """初始化后处理"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not hasattr(self, 'created_at') or not self.created_at:
            self.created_at = time.time()
        if not hasattr(self, 'related_block_ids'):
            self.related_block_ids = []
        if not hasattr(self, 'source_file_path'):
            self.source_file_path = ""