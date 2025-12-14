"""
LLM Memory System - 统一的三元组记忆体系

基于统一的 (judgment, reasoning, tags) 三元组结构。
通过 memory_type 区分五种记忆类型：知识、事件、技能、任务、情感。
"""

__version__ = "6.0.0"

# 懒加载：只在实际使用时才导入，避免循环依赖
def __getattr__(name):
    if name == "CognitiveService":
        from .service.cognitive_service import CognitiveService
        return CognitiveService
    elif name == "BaseMemory":
        from .models.data_models import BaseMemory
        return BaseMemory
    elif name == "MemoryType":
        from .models.data_models import MemoryType
        return MemoryType
    elif name == "MemoryError":
        from .models.data_models import MemoryError
        return MemoryError
    elif name == "VectorizationError":
        from .models.data_models import VectorizationError
        return VectorizationError
    elif name == "StorageError":
        from .models.data_models import StorageError
        return StorageError
    elif name == "ValidationError":
        from .models.data_models import ValidationError
        return ValidationError
    elif name == "system_config":
        from .config.system_config import system_config
        return system_config
    elif name == "MemorySystemConfig":
        from .config.system_config import MemorySystemConfig
        return MemorySystemConfig
    elif name == "KNOWLEDGE_CORE_SEPARATOR":
        from .config.system_config import KNOWLEDGE_CORE_SEPARATOR
        return KNOWLEDGE_CORE_SEPARATOR
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # 核心接口
    "CognitiveService",
    "BaseMemory",
    "MemoryType",
    # 异常
    "MemoryError",
    "VectorizationError",
    "StorageError",
    "ValidationError",
    # 配置
    "system_config",
    "MemorySystemConfig",
    "KNOWLEDGE_CORE_SEPARATOR",
]
