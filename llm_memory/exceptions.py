"""
自定义异常类
"""


class RateLimitExceededError(Exception):
    """速率限制超出异常（429错误重试后仍然失败）"""

    def __init__(self, message: str, attempts: int = 0):
        """
        初始化速率限制异常

        Args:
            message: 错误消息
            attempts: 已尝试的重试次数
        """
        self.attempts = attempts
        super().__init__(message)
