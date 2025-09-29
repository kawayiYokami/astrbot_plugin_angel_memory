"""
全局 logger 容器

这个模块提供一个全局的 logger 容器，避免子模块直接导入 astrbot.api。
只有 main.py 在初始化时设置真正的 logger。
"""

class _LoggerContainer:
    """Logger 容器，存储全局 logger 实例"""

    def __init__(self):
        self._logger = None

    def set_logger(self, logger):
        """设置 logger（仅由 main.py 调用）"""
        self._logger = logger

    def get_logger(self):
        """获取 logger"""
        if self._logger is None:
            # 测试环境下返回一个默认的 logger
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
        return self._logger

# 全局实例
_container = _LoggerContainer()

def set_logger(logger):
    """设置全局 logger（main.py 调用）"""
    _container.set_logger(logger)

def get_logger():
    """获取全局 logger（子模块调用）"""
    return _container.get_logger()
