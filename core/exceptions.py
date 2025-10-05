"""
记忆系统异常定义

定义记忆系统中的自定义异常类。
"""


class MemorySystemError(Exception):
    """记忆系统基础异常类"""
    pass


class MemoryNotFoundError(MemorySystemError):
    """记忆未找到异常"""
    pass


class MemoryFormatError(MemorySystemError):
    """记忆格式错误异常"""
    pass


class MemoryProcessingError(MemorySystemError):
    """记忆处理错误异常"""
    pass


class ConfigurationError(MemorySystemError):
    """配置错误异常"""
    pass


class SessionError(MemorySystemError):
    """会话相关错误异常"""
    pass