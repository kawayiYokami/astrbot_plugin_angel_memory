"""
过程追踪 API。

返回反思与睡眠的最近若干次执行记录（持久化 JSONL + 内存中尚未完成的反思），
并支持由用户手动删除指定历史记录。仅本机 WebUI 使用。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from quart import jsonify, request


class TraceAPI:
    TRACE_LIMIT = 20

    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_deepmind(self) -> Optional[Any]:
        try:
            return self.plugin_context.get_component("deepmind")
        except Exception:
            return None

    async def get_records(self):
        """返回最近若干次反思与睡眠记录（持久化 + 内存中未完成的反思）。"""
        deepmind = self._get_deepmind()
        if deepmind is None:
            return jsonify({"ok": True, "records": [], "note": "deepmind 组件未初始化"})
        try:
            records = await self._collect_records(deepmind)
        except Exception as e:
            return jsonify({"ok": False, "records": [], "note": f"读取追踪记录失败: {e}"})
        return jsonify({"ok": True, "records": records})

    async def delete_records(self):
        """删除指定 id 的历史记录，仅作用于持久化文件。"""
        deepmind = self._get_deepmind()
        if deepmind is None:
            return jsonify({"ok": False, "deleted": 0, "note": "deepmind 组件未初始化"})

        store = getattr(deepmind, "trace_store", None)
        if store is None:
            return jsonify({"ok": False, "deleted": 0, "note": "追踪存储不可用"})

        payload = await request.get_json(silent=True) or {}
        raw_ids = payload.get("ids")
        if not isinstance(raw_ids, list):
            return jsonify({"ok": False, "deleted": 0, "note": "缺少 ids 列表"})

        try:
            deleted = await store.delete_records(raw_ids)
        except Exception as e:
            return jsonify({"ok": False, "deleted": 0, "note": f"删除记录失败: {e}"})
        return jsonify({"ok": True, "deleted": deleted})

    async def _collect_records(self, deepmind) -> List[Dict[str, Any]]:
        store = getattr(deepmind, "trace_store", None)
        persisted = (
            await store.read_recent(limit=self.TRACE_LIMIT) if store is not None else []
        )

        # 内存中仍处于 pending 的反思记录尚未落盘，一并展示
        pending: List[Dict[str, Any]] = []
        buffer = getattr(deepmind, "reflection_trace", None)
        if buffer is not None:
            for entry in buffer.snapshot():
                if entry.get("status") != "pending":
                    continue
                item = dict(entry)
                item["type"] = "reflection"
                pending.append(item)

        merged = list(persisted) + pending
        merged.sort(key=self._sort_key)
        return merged[-self.TRACE_LIMIT :]

    @staticmethod
    def _sort_key(record: Dict[str, Any]) -> float:
        try:
            timestamp = float(record.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        if timestamp > 0:
            return timestamp
        try:
            return time.mktime(
                time.strptime(str(record.get("time", "")), "%Y-%m-%d %H:%M:%S")
            )
        except (TypeError, ValueError):
            return 0.0
