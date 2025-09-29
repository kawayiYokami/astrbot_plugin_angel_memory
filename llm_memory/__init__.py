"""
LLM Memory System - 统一的三元组记忆体系

基于统一的 (judgment, reasoning, tags) 三元组结构。
通过 memory_type 区分五种记忆类型：知识、事件、技能、任务、情感。
"""

from .service.cognitive_service import CognitiveService
from .models.data_models import (
    BaseMemory, MemoryType,
    MemoryError, VectorizationError, StorageError, ValidationError
)
from .config.system_config import system_config, MemorySystemConfig, KNOWLEDGE_CORE_SEPARATOR

__version__ = "6.0.0"
__all__ = [
    # 核心接口
    "CognitiveService", "BaseMemory", "MemoryType",
    # 异常
    "MemoryError", "VectorizationError", "StorageError", "ValidationError",
    # 配置
    "system_config", "MemorySystemConfig", "KNOWLEDGE_CORE_SEPARATOR"
]