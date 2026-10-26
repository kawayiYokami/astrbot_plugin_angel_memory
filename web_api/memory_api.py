"""
记忆相关 API（p-memory 版：浏览/删除/总览全部走新库，不再读旧 SQL 迁移源）
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _to_seconds(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    # p-memory 用微秒，旧库用秒：大于 1e11 视为微秒
    if number > 1e11:
        return number / 1_000_000.0
    return number if number > 0 else 0.0


def _record_to_item(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("state") or {}
    tags = record.get("tags") or []
    if isinstance(tags, str):
        tags_text = tags
    else:
        tags_text = ", ".join(str(t) for t in tags if str(t).strip())
    created = record.get("created_at_us", record.get("created_at", 0))
    updated = record.get("updated_at_us", record.get("updated_at", created))
    return {
        "id": record.get("id"),
        "memory_type": str(record.get("memory_type") or "知识记忆"),
        "judgment": str(record.get("judgment") or ""),
        "reasoning": str(record.get("reasoning") or ""),
        "strength": int(state.get("strength", 1)),
        "is_active": bool(state.get("pinned", False)),
        "memory_scope": str(record.get("scope") or "public"),
        "created_at": _to_seconds(created),
        "updated_at": _to_seconds(updated),
        "tags": tags_text,
    }


class MemoryAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _memory_store(self):
        try:
            return self.plugin_context.get_component("memory_store")
        except Exception:
            return None

    async def get_overview(self):
        """总览统计：记忆/标签/scopes 走 p-memory，笔记计数走 note_index。"""
        result: Dict[str, Any] = {
            "provider_id": self.plugin_context.get_embedding_provider_id() or "(未配置)",
            "llm_provider_id": self.plugin_context.get_llm_provider_id() or "(未配置)",
            "has_providers": self.plugin_context.has_providers(),
            "memory_count": 0,
            "global_tag_count": 0,
            "note_index_count": 0,
            "scopes": [],
            "index_dir": str(self.plugin_context.get_index_dir()),
        }

        try:
            store = self._memory_store()
            if store is not None:
                import asyncio

                records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
                result["memory_count"] = len(records)
                result["pmemory_count"] = len(records)
                tag_set = set()
                scopes = set()
                for record in records:
                    scope = str(record.get("scope") or "").strip()
                    if scope:
                        scopes.add(scope)
                    tags = record.get("tags") or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(",") if t.strip()]
                    for tag in tags:
                        name = str(tag).strip()
                        if name:
                            tag_set.add(name)
                result["global_tag_count"] = len(tag_set)
                result["scopes"] = sorted(scopes)
        except Exception as e:
            result["error"] = str(e)

        try:
            note_index = self.plugin_context.get_component("note_index")
            if note_index is not None:
                status = note_index.get_status()
                result["note_index_count"] = int(status.get("tracked_files", 0) or 0)
        except Exception:
            pass

        return jsonify(result)

    async def browse_memories(self):
        """分页浏览记忆（p-memory）。"""
        scope = request.args.get("scope", "").strip()
        keyword = request.args.get("keyword", "").strip().lower()
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        store = self._memory_store()
        if store is None:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})

        try:
            import asyncio

            records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
            filtered: List[Dict[str, Any]] = []
            for record in records:
                item = _record_to_item(record)
                if scope and item["memory_scope"] != scope:
                    continue
                if keyword:
                    hay = f"{item['judgment']} {item['reasoning']} {item['tags']}".lower()
                    if keyword not in hay:
                        continue
                filtered.append(item)

            filtered.sort(key=lambda x: (x["created_at"] or 0), reverse=True)
            total = len(filtered)
            page_items = filtered[offset:offset + page_size]
            return jsonify({
                "items": page_items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(total / page_size) if total > 0 else 1,
            })
        except Exception as e:
            return jsonify({"error": str(e), "items": [], "total": 0}), 500

    async def delete_memory(self):
        """删除记忆（p-memory 整数 id）。"""
        data = await request.get_json()
        raw_id = str((data or {}).get("id", "")).strip()
        if not raw_id:
            return jsonify({"success": False, "error": "缺少 id 参数"}), 400
        try:
            memory_id = int(raw_id)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "仅支持新库整数 id"}), 400

        store = self._memory_store()
        if store is None:
            return jsonify({"success": False, "error": "记忆组件不可用"}), 500

        try:
            import asyncio

            deleted = await asyncio.to_thread(store.delete, [memory_id])
            return jsonify({"success": bool(deleted), "id": memory_id})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
