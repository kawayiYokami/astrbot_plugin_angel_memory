"""异步记忆反馈处理器（优化版）。

直接异步执行 `memory_system.feedback`，无后台线程。
优化版本：移除了不必要的线程和队列，改为直接异步调用。
"""

from __future__ import annotations

import threading  # 单例模式需要锁
from typing import Any, Callable, Dict, Optional, Union

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


FeedbackTask = Union[Dict[str, Any], None]


class FeedbackQueue:
    """异步反馈处理器（无后台线程）。"""

    def __init__(
        self,
        worker_name: str = "MemoryFeedbackWorker",
        flush_interval: float = 0.2,
        batch_threshold: int = 50,
    ):
        # 保留参数以兼容旧代码，但不再使用
        logger.info("反馈处理器已初始化（异步模式，无后台线程）")

    async def submit(self, task: FeedbackTask):
        """直接异步处理任务（无队列，无缓冲）"""
        if task is None:
            logger.warning("反馈处理器收到空任务，拒绝处理")
            return None

        feedback_fn: Optional[Callable] = task.get("feedback_fn")
        session_id: Any = task.get("session_id", "unknown")
        payload: Dict[str, Any] = task.get("payload", {})

        if not feedback_fn:
            logger.warning(f"反馈任务缺少 feedback_fn - 会话ID: {session_id}")
            return None

        logger.info(
            f"[异步反馈] 立即处理任务 - 会话ID: {session_id}, 任务类型: {feedback_fn.__name__}"
        )

        try:
            # 直接异步调用，无后台线程
            result = await feedback_fn(**payload)
            logger.debug(f"[异步反馈] session={session_id} 完成")
            return result
        except Exception as exc:
            logger.error(
                f"[异步反馈] session={session_id} 执行失败: {exc}",
                exc_info=True,
            )
            return None

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """停止处理器（异步模式无需停止操作）"""
        logger.info("反馈处理器已停止（异步模式）")


_queue_instance: Optional[FeedbackQueue] = None
_instance_lock = threading.Lock()


def get_feedback_queue() -> FeedbackQueue:
    global _queue_instance
    if _queue_instance is None:
        with _instance_lock:
            if _queue_instance is None:
                _queue_instance = FeedbackQueue()
    return _queue_instance


def stop_feedback_queue(timeout: Optional[float] = 5.0) -> None:
    global _queue_instance
    with _instance_lock:
        if _queue_instance is not None:
            _queue_instance.stop(timeout=timeout)
            _queue_instance = None
