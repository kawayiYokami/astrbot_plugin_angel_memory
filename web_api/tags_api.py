"""
标签相关 API（p-memory 版：标签从记忆记录上聚合，不再读旧 SQL 关系表）
"""

from __future__ import annotations

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
    if number > 1e11:
        return number / 1_000_000.0
    return number if number > 0 else 0.0


def _record_tags(record: Dict[str, Any]) -> List[str]:
    tags = record.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return [str(t).strip() for t in tags if str(t).strip()]


def _record_scope(record: Dict[str, Any]) -> str:
    return str(record.get("scope") or "public")


def aggregate_tags(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按标签名聚合引用计数，引用多者在前，同计数按名字排序。"""
    counts: Dict[str, int] = {}
    for record in records or []:
        for tag in _record_tags(record):
            counts[tag] = counts.get(tag, 0) + 1
    items = [
        {"id": name, "name": name, "memory_refs": count, "note_refs": 0}
        for name, count in counts.items()
    ]
    items.sort(key=lambda x: (-x["memory_refs"], x["name"]))
    return items


def match_tag_names(records: List[Dict[str, Any]], query_text: str) -> List[str]:
    """标签命中：标签名出现在查询文本里即命中（与旧 SQL 版口径一致）。"""
    query = str(query_text or "").strip()
    if not query:
        return []
    names = {tag for record in (records or []) for tag in _record_tags(record)}
    return sorted(name for name in names if name and name in query)


def _scope_visible(record_scope: str, scope: str) -> bool:
    if not scope:
        return True
    if scope == "public":
        return record_scope == "public"
    return record_scope in (scope, "public")


def _hit_item(record: Dict[str, Any], hit_count: int) -> Dict[str, Any]:
    state = record.get("state") or {}
    tags = _record_tags(record)
    created = record.get("created_at_us", record.get("created_at", 0))
    updated = record.get("updated_at_us", record.get("updated_at", created))
    return {
        "id": record.get("id"),
        "hit_count": int(hit_count),
        "memory_type": str(record.get("memory_type") or "知识记忆"),
        "judgment": str(record.get("judgment") or ""),
        "reasoning": str(record.get("reasoning") or ""),
        "strength": int(state.get("strength", 1)),
        "is_active": bool(state.get("pinned", False)),
        "memory_scope": _record_scope(record),
        "created_at": _to_seconds(created),
        "updated_at": _to_seconds(updated),
        "tags": ", ".join(tags),
    }


def search_memories_by_tags(
    records: List[Dict[str, Any]],
    matched: List[str],
    scope: str = "",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """含命中标签的记忆，按命中标签数、强度、创建时间排序。"""
    wanted = {str(t) for t in (matched or []) if str(t).strip()}
    if not wanted:
        return []
    hits = []
    for record in records or []:
        if not _scope_visible(_record_scope(record), str(scope or "").strip()):
            continue
        tags = set(_record_tags(record))
        hit_count = len(tags & wanted)
        if hit_count:
            hits.append((record, hit_count))
    state_strength = lambda record: int((record.get("state") or {}).get("strength", 1) or 0)
    created_ts = lambda record: _to_seconds(
        record.get("created_at_us", record.get("created_at", 0))
    )
    hits.sort(key=lambda pair: (-pair[1], -state_strength(pair[0]), -created_ts(pair[0])))
    return [_hit_item(record, count) for record, count in hits[:max(1, limit)]]


class TagsAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _memory_store(self):
        try:
            return self.plugin_context.get_component("memory_store")
        except Exception:
            return None

    async def get_tags(self):
        """全局标签列表（从 p-memory 聚合）"""
        keyword = request.args.get("keyword", "").strip()
        limit = min(500, max(1, int(request.args.get("limit", 200))))
        offset = max(0, int(request.args.get("offset", 0)))

        store = self._memory_store()
        if store is None:
            return jsonify({"tags": []})

        try:
            import asyncio

            records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
            items = aggregate_tags(records)
            if keyword:
                items = [t for t in items if keyword in t["name"]]
            return jsonify({"tags": items[offset:offset + limit]})
        except Exception as e:
            return jsonify({"tags": [], "error": str(e)}), 500

    async def hit_search(self):
        """标签命中搜索（从 p-memory 聚合）"""
        data = await request.get_json()
        query_text = str((data or {}).get("query", "")).strip()
        scope = str((data or {}).get("scope", "")).strip()
        limit = min(200, max(1, int((data or {}).get("limit", 50))))

        if not query_text:
            return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})

        store = self._memory_store()
        if store is None:
            return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})

        try:
            import asyncio

            records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
            matched = match_tag_names(records, query_text)
            if not matched:
                return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})
            hits = search_memories_by_tags(records, matched, scope, limit)
            return jsonify({
                "matched_tags": matched,
                "matched_tag_ids": [],
                "memory_hits": hits,
            })
        except Exception as e:
            return jsonify({
                "matched_tags": [], "matched_tag_ids": [],
                "memory_hits": [], "error": str(e),
            }), 500
