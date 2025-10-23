"""异步记忆反馈队列。

提供线程安全的任务调度，脱离事件主流程执行 `memory_system.feedback`。
"""

from __future__ import annotations

import threading
import queue
from typing import Callable, Optional, Dict, Any, List, Union

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


FeedbackTask = Union[Dict[str, Any], None]


class FeedbackQueue:
    """后台任务队列，支持按 session 聚合反馈以减少频繁写入。"""

    def __init__(
        self,
        worker_name: str = "MemoryFeedbackWorker",
        flush_interval: float = 0.2,
        batch_threshold: int = 50,
    ):
        self._queue: "queue.Queue[FeedbackTask]" = queue.Queue()
        self._stop_event = threading.Event()

        self.flush_interval = flush_interval
        self.batch_threshold = batch_threshold

        self._buffer_lock = threading.Lock()
        self._buffers: Dict[str, Dict[str, Any]] = {}
        self._scheduled: Dict[str, threading.Timer] = {}

        self._thread = threading.Thread(
            target=self._worker_loop, name=worker_name, daemon=True
        )
        self._thread.start()
        logger.info("反馈队列线程已启动: %s", worker_name)

    def submit(self, task: FeedbackTask) -> None:
        # 检查任务是否为None
        if task is None:
            logger.warning("反馈队列收到空任务，拒绝处理")
            return

        if self._stop_event.is_set():
            logger.warning("反馈队列已停止，拒绝新任务: %s", task.get("session_id"))
            return

        # --- 开始新增的逻辑 ---
        # 通过检查任务的 payload 是否包含 'event_data' 来识别新型异步分析任务
        payload = task.get("payload", {})
        is_analysis_task = "event_data" in payload

        if is_analysis_task:
            session_id = task.get("session_id", "unknown")
            logger.info(
                f"[反馈队列] 收到[异步分析]任务，直接入队 - 会话ID: {session_id}"
            )
            # 对于新任务，绕过缓冲，直接放入执行队列
            self._queue.put(task)
            return
        # --- 新增逻辑结束 ---

        session_id = task.get("session_id", "unknown")
        useful_ids: List[str] = task.get("useful_memory_ids", [])
        new_memories: List[Dict[str, Any]] = task.get("new_memories", [])
        merge_groups: List[List[str]] = task.get("merge_groups", [])

        logger.info(f"[反馈队列] 收到[记忆反馈]任务，进入缓冲 - 会话ID: {session_id}")
        logger.debug(
            f"[反馈队列] 收到任务提交，会话ID: {session_id}，有用记忆数: {len(useful_ids)}，新记忆数: {len(new_memories)}，合并组数: {len(merge_groups)}"
        )

        # 添加更多调试信息
        if payload:
            logger.debug(f"[反馈队列] 任务负载信息 - 会话ID: {session_id}")
            logger.debug(f"  负载键: {list(payload.keys())}")

        with self._buffer_lock:
            buffer = self._buffers.setdefault(
                session_id,
                {
                    "feedback_fn": task.get("feedback_fn"),
                    "useful_ids": set(),
                    "new_memories": [],
                    "merge_groups": [],
                },
            )

            if buffer.get("feedback_fn") is None:
                buffer["feedback_fn"] = task.get("feedback_fn")

            if useful_ids:
                buffer["useful_ids"].update(useful_ids)
            if new_memories:
                buffer["new_memories"].extend(new_memories)
            if merge_groups:
                buffer["merge_groups"].extend(merge_groups)

            should_flush = (
                len(buffer["useful_ids"]) >= self.batch_threshold
                or len(buffer["new_memories"]) >= self.batch_threshold
                or len(buffer["merge_groups"]) >= self.batch_threshold
            )

            logger.debug(
                f"[反馈队列] 会话 {session_id} 缓冲区状态 - 有用记忆: {len(buffer['useful_ids'])}, 新记忆: {len(buffer['new_memories'])}, 合并组: {len(buffer['merge_groups'])}"
            )

            if should_flush:
                logger.debug(f"[反馈队列] 会话 {session_id} 达到批处理阈值，立即刷新")
                self._enqueue_buffer_locked(session_id)
            elif session_id not in self._scheduled:
                logger.debug(
                    f"[反馈队列] 会话 {session_id} 设置定时刷新，间隔: {self.flush_interval}秒"
                )
                timer = threading.Timer(
                    self.flush_interval, self._flush_session, args=[session_id]
                )
                timer.daemon = True
                self._scheduled[session_id] = timer
                timer.start()
            else:
                logger.debug(f"[反馈队列] 会话 {session_id} 已有定时任务，等待执行")

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self.flush_all()
        self._stop_event.set()
        self._queue.put(None)  # sentinel
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("反馈队列线程已停止")

    def flush_all(self) -> None:
        with self._buffer_lock:
            session_ids = list(self._buffers.keys())
            for session_id in session_ids:
                self._enqueue_buffer_locked(session_id)

    def wait_for_idle(self) -> None:
        self.flush_all()
        self._queue.join()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get()
                if task is None:  # sentinel for shutdown
                    break
                self._process_task(task)
            except Exception as exc:  # noqa: BLE001
                logger.error("反馈队列执行任务失败: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()

    def _process_task(self, task: FeedbackTask) -> None:
        # 检查任务是否为None
        if task is None:
            logger.warning("反馈队列收到空任务，跳过处理")
            return

        feedback_fn: Callable[..., None] = task["feedback_fn"]
        session_id: str = task.get("session_id", "unknown")
        payload = task.get("payload", {})

        logger.info(
            f"[后台工人] 开始处理任务 - 会话ID: {session_id}, 任务类型: {feedback_fn.__name__}"
        )

        try:
            feedback_fn(**payload)
            logger.debug("[feedback_queue] session=%s 完成", session_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[feedback_queue] session=%s 执行失败: %s",
                session_id,
                exc,
                exc_info=True,
            )

    def _flush_session(self, session_id: str) -> None:
        with self._buffer_lock:
            self._enqueue_buffer_locked(session_id)

    def _enqueue_buffer_locked(self, session_id: str) -> None:
        buffer = self._buffers.pop(session_id, None)
        timer = self._scheduled.pop(session_id, None)
        if timer:
            timer.cancel()

        if not buffer:
            logger.debug(f"[反馈队列] 会话 {session_id} 没有缓冲区数据，跳过")
            return

        useful_ids = list(buffer["useful_ids"])
        new_memories = buffer["new_memories"]
        merge_groups = buffer["merge_groups"]

        if not useful_ids and not new_memories and not merge_groups:
            logger.debug(f"[反馈队列] 会话 {session_id} 没有需要处理的数据，跳过")
            return

        aggregated_task = {
            "feedback_fn": buffer["feedback_fn"],
            "session_id": session_id,
            "payload": {
                "useful_memory_ids": useful_ids,
                "new_memories": new_memories,
                "merge_groups": merge_groups,
                "session_id": session_id,
            },
        }
        self._queue.put(aggregated_task)
        logger.debug(
            f"[反馈队列] 会话 {session_id} 任务已加入队列，有用记忆: {len(useful_ids)}, 新记忆: {len(new_memories)}, 合并组: {len(merge_groups)}"
        )

        # 添加更多调试信息
        if new_memories:
            logger.debug(f"[反馈队列] 会话 {session_id} 新记忆类型统计:")
            type_count = {}
            for memory in new_memories:
                mem_type = memory.get("type", "unknown")
                type_count[mem_type] = type_count.get(mem_type, 0) + 1
            logger.debug(f"  {type_count}")


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
