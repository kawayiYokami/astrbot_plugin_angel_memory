"""
反思过程追踪 API。

返回内存中最近 N 次反思分析的提示词、LLM 返回与成败状态，仅本机 WebUI 使用。
"""

from __future__ import annotations

from typing import Any, Optional

from quart import jsonify


class TraceAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_trace_buffer(self) -> Optional[Any]:
        try:
            deepmind = self.plugin_context.get_component("deepmind")
        except Exception:
            deepmind = None
        if deepmind is None:
            return None
        return getattr(deepmind, "reflection_trace", None)

    def get_records(self):
        """返回最近 N 次反思记录（内存副本，不落盘）。"""
        buffer = self._get_trace_buffer()
        if buffer is None:
            return jsonify({"ok": True, "records": [], "note": "deepmind 组件未初始化"})
        try:
            records = buffer.snapshot()
        except Exception as e:
            return jsonify({"ok": False, "records": [], "note": f"读取反思记录失败: {e}"})
        return jsonify({"ok": True, "records": records})