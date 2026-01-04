"""
LLM记忆系统的数据模型 - 统一的三元组记忆体系。

本模块定义了统一的 BaseMemory 类，使用 (judgment, reasoning, tags) 三元组结构。
所有记忆类型共享相同的数据结构，通过 memory_type 字段区分：
- 知识记忆：存储"AI认为是什么"
- 事件记忆：记录"AI经历了什么"
- 技能记忆：存储"AI能做什么及如何做"
- 任务记忆：存储"AI正在处理什么"
- 情感记忆：存储"AI感觉怎么样"
"""

from enum import Enum
from typing import List, Optional
import uuid
import time

from ..config.system_config import KNOWLEDGE_CORE_SEPARATOR


# ===== 记忆类型枚举 =====


class MemoryType(Enum):
    """记忆类型的中文枚举"""

    KNOWLEDGE = "知识记忆"
    EVENT = "事件记忆"
    SKILL = "技能记忆"
    TASK = "任务记忆"
    EMOTIONAL = "情感记忆"


# ===== 异常处理层 =====


class MemoryError(Exception):
    """记忆系统基础异常类"""

    pass


class VectorizationError(MemoryError):
    """向量化和嵌入相关异常"""

    pass


class StorageError(MemoryError):
    """存储和检索相关异常"""

    pass


class ValidationError(MemoryError):
    """数据验证相关异常"""

    pass


class BaseMemory:
    """
    记忆的基类 - 统一的三元组结构（judgment, reasoning, tags）

    所有记忆都使用相同的三元组结构，通过 memory_type 区分类型。
    """

    def __init__(
        self,
        memory_type: MemoryType,
        judgment: str,
        reasoning: str,
        tags: List[str],
        id: str = None,
        strength: int = 1,
        is_active: bool = False,
        associations: dict = None,
        created_at: float = None,
        is_consolidated: bool = None,  # 废弃字段，仅保留兼容性
        state_snapshot: dict = None,   # 灵魂状态快照
    ):
        self.id = id or str(uuid.uuid4())
        self.memory_type = memory_type
        self.judgment = judgment
        self.reasoning = reasoning
        self.tags = tags if isinstance(tags, list) else []
        self.strength = strength
        self.is_active = is_active  # True=主动记忆(不衰减), False=被动记忆(会衰减)
        self.associations = associations or {}  # 记忆关联字段：{memory_id: strength}
        self.created_at = created_at or time.time()  # 自动记录创建时间
        self.state_snapshot = state_snapshot or {} # 记录生成该记忆时的灵魂状态
        self.similarity = 0.0  # 相似度分数（仅用于检索时传递，不存储到数据库）

    def get_semantic_core(self) -> str:
        """返回用于向量化的核心语义文本：judgment + tags"""
        tags_text = KNOWLEDGE_CORE_SEPARATOR.join(self.tags)
        return f"{self.judgment}{KNOWLEDGE_CORE_SEPARATOR} {tags_text}"

    def to_dict(self) -> dict:
        """转换为字典以进行JSON序列化。"""
        import json

        # ChromaDB不支持嵌套字典，需要将associations序列化为JSON字符串
        associations_str = json.dumps(self.associations) if self.associations else "{}"

        # 将tags列表转换为字符串以兼容ChromaDB
        tags_str = ", ".join(self.tags) if isinstance(self.tags, list) else ""

        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "judgment": self.judgment,
            "reasoning": self.reasoning,
            "tags": tags_str,
            "strength": self.strength,
            "is_active": self.is_active,
            "associations": associations_str,
            "created_at": self.created_at,
            "state_snapshot": json.dumps(self.state_snapshot) if self.state_snapshot else "{}",
        }

    @staticmethod
    def _parse_associations(associations_data) -> dict:
        """解析associations字段（可能是JSON字符串或字典）"""
        import json

        if isinstance(associations_data, str):
            try:
                return json.loads(associations_data)
            except (json.JSONDecodeError, ValueError):
                return {}
        elif isinstance(associations_data, dict):
            return associations_data
        else:
            return {}

    @staticmethod
    def _parse_tags(tags_data) -> List[str]:
        """解析tags字段（可能是字符串或列表）"""
        if isinstance(tags_data, str):
            return [tag.strip() for tag in tags_data.split(",") if tag.strip()]
        elif isinstance(tags_data, list):
            return tags_data
        else:
            return []

    @classmethod
    def from_dict(cls, data: dict) -> Optional["BaseMemory"]:
        """从字典创建实例（用于JSON反序列化）。"""
        if not isinstance(data, dict):
            raise ValidationError(f"输入必须是字典类型，实际为: {type(data)}")

        # 验证必需字段
        if "id" not in data:
            raise ValidationError("缺少必需字段: id")

        # 解析memory_type
        memory_type_str = data.get("memory_type")
        if not memory_type_str:
            return None  # 未指定类型时不创建记忆

        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            return None  # 无效类型时也不创建记忆

        # 解析tags
        tags = cls._parse_tags(data.get("tags", []))

        # 兼容旧字段名
        judgment = data.get(
            "judgment", data.get("content", data.get("emotion_type", ""))
        )
        reasoning = data.get(
            "reasoning", data.get("context", data.get("trigger_event", ""))
        )

        try:
            return cls(
                memory_type=memory_type,
                judgment=judgment,
                reasoning=reasoning,
                tags=tags,
                id=data.get("id", str(uuid.uuid4())),
                strength=data.get("strength", 1),
                is_active=data.get("is_active", False),
                associations=cls._parse_associations(data.get("associations", {})),
                created_at=data.get("created_at", time.time()),
                state_snapshot=cls._parse_associations(data.get("state_snapshot", {})), # 复用json解析逻辑
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValidationError(f"从字典创建记忆失败: {str(e)}")

    def __str__(self) -> str:
        """字符串表示形式，用于调试。"""
        return f"Memory(type={self.memory_type.value}, id={self.id[:8]}..., judgment='{self.judgment[:30]}...', tags={self.tags})"
