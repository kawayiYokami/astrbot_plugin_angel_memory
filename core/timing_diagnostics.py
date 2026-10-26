"""
Timing诊断工具 - 用于定位性能瓶颈

使用方法：
from core.timing_diagnostics import timing_log

with timing_log("操作名称"):
    # 你的代码
    pass
"""

import time
import threading
from contextlib import contextmanager

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@contextmanager
def timing_log(operation_name: str, threshold_ms: float = 100.0):
    """
    记录操作耗时的上下文管理器

    Args:
        operation_name: 操作名称
        threshold_ms: 超过此阈值才记录warning（毫秒）
    """
    thread_name = threading.current_thread().name
    start_time = time.time()

    logger.debug(f"[{thread_name}] {operation_name} - 开始")

    try:
        yield
    finally:
        elapsed_ms = (time.time() - start_time) * 1000

        if elapsed_ms > threshold_ms:
            logger.warning(
                f"[{thread_name}] {operation_name} - 完成，耗时: {elapsed_ms:.2f}ms"
            )
        else:
            logger.debug(
                f"[{thread_name}] {operation_name} - 完成，耗时: {elapsed_ms:.2f}ms"
            )


def log_checkpoint(checkpoint_name: str):
    """记录检查点（用于追踪执行流程）"""
    thread_name = threading.current_thread().name
    logger.debug(f"[{thread_name}] 检查点: {checkpoint_name}")
