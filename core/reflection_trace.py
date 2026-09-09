"""反思归纳过程追踪缓冲。

在内存中保留最近 N 次反思分析的提示词与 LLM 返回结果，供 WebUI 查看与
反馈排查。纯内存、线程安全、重启即清空，不落盘。
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional


class ReflectionTraceBuffer:
    """线程安全的环形缓冲，记录反思分析的提示词、返回结果与成败状态。"""

    def __init__(self, maxlen: int = 10):
        self._maxlen = max(1, int(maxlen))
        self._buffer: deque = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()
        self._next_id = 0

    def record_start(
        self,
        session_id: str,
        provider_id: str,
        prompt: str,
        system_prompt: str = "",
    ) -> Optional[int]:
        """提示词构造完成后入档，返回 trace id 供结果回填；异常时返回 None。

        prompt 只存本次的用户提示词，固定指南另存 system_prompt。
        """
        try:
            trace_id = self._next_id
            self._next_id += 1
            entry: Dict[str, Any] = {
                "id": trace_id,
                "timestamp": time.time(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "session_id": str(session_id or ""),
                "provider_id": str(provider_id or ""),
                "status": "pending",
                "prompt": str(prompt or ""),
                "system_prompt": str(system_prompt or ""),
                "response": "",
                "result": None,
                "error": "",
            }
            with self._lock:
                self._buffer.append(entry)
            return trace_id
        except Exception:
            return None

    def complete(
        self,
        trace_id: Optional[int],
        *,
        status: str,
        response: str = "",
        result: Any = None,
        error: str = "",
    ) -> None:
        """回填结果：status 传 success 或 failed。"""
        if trace_id is None:
            return
        try:
            normalized_status = "success" if status == "success" else "failed"
            with self._lock:
                for entry in self._buffer:
                    if entry["id"] == trace_id:
                        entry["status"] = normalized_status
                        if response:
                            entry["response"] = str(response)
                        if result is not None:
                            entry["result"] = result
                        if error:
                            entry["error"] = str(error)
                        return
        except Exception:
            pass

    def snapshot(self) -> List[Dict[str, Any]]:
        """返回从旧到新的记录深拷贝，不暴露内部可变对象。"""
        try:
            with self._lock:
                return copy.deepcopy(list(self._buffer))
        except Exception:
            return []