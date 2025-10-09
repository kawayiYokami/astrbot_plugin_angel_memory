"""
AstrBot Angel Memory Plugin

基于双层认知架构的AI记忆系统插件，为AstrBot提供记忆能力。
实现观察→回忆→反馈→睡眠的完整认知工作流。
"""

from .main import AngelMemoryPlugin

__version__ = "1.0.0"
__author__ = "Angel Memory Team"

# 导出的主要接口
__all__ = [
    "AngelMemoryPlugin",
    "__version__"
]