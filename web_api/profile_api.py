"""用户画像 API（p-memory 版：画像记忆从新库拉，账本仍读 SQL 账本表）

直接复用 llm_memory.utils.user_profile 的识别逻辑，
确保 WebUI 展示与运行时行为一致。
"""

from __future__ import annotations

from typing import Any, Dict, List

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..llm_memory.utils.user_profile import (
    PROFILE_ATTRIBUTE_TAGS,
    extract_profile_attribute_from_tags,
    extract_user_ids_from_tags,
    is_user_id_tag,
    is_user_profile_tags,
)


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


def _record_item(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("state") or {}
    created = record.get("created_at_us", record.get("created_at", 0))
    updated = record.get("updated_at_us", record.get("updated_at", created))
    return {
        "id": record.get("id"),
        "judgment": str(record.get("judgment") or ""),
        "reasoning": str(record.get("reasoning") or ""),
        "strength": int(state.get("strength", 1)),
        "is_active": bool(state.get("pinned", False)),
        "memory_scope": str(record.get("scope") or "public"),
        "created_at": _to_seconds(created),
        "updated_at": _to_seconds(updated),
        "tags": ", ".join(_record_tags(record)),
    }


def profile_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只留画像属性记忆（标签里带属性词），置顶者在前。"""
    kept = []
    for record in records or []:
        tags = _record_tags(record)
        if not tags:
            continue
        if not (set(tags) & set(PROFILE_ATTRIBUTE_TAGS)):
            continue
        kept.append(record)
    kept.sort(
        key=lambda r: (
            not bool((r.get("state") or {}).get("pinned", False)),
            -_to_seconds(r.get("updated_at_us", r.get("updated_at", 0))),
        )
    )
    return kept


def group_users(
    records: List[Dict[str, Any]],
    ledger_users: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把画像记忆按用户 ID 归组，再合并账本里的平台与昵称。"""
    known_user_ids = [u["user_id"] for u in ledger_users or []]
    user_data: Dict[str, Dict[str, Any]] = {}

    for record in records or []:
        tags = _record_tags(record)
        if not is_user_profile_tags(tags, known_user_ids=known_user_ids):
            continue
        user_ids = extract_user_ids_from_tags(tags, known_user_ids=known_user_ids)
        if not user_ids:
            continue
        for user_id in user_ids:
            bucket = user_data.setdefault(
                user_id,
                {
                    "user_id": user_id,
                    "platforms": [],
                    "nickname": "",
                    "memory_count": 0,
                    "attributes": {},
                },
            )
            bucket["memory_count"] += 1

            attr = extract_profile_attribute_from_tags(tags)
            if attr:
                bucket["attributes"][attr] = bucket["attributes"].get(attr, 0) + 1

            if not bucket["nickname"]:
                for tag in tags:
                    if (
                        tag != user_id
                        and tag not in PROFILE_ATTRIBUTE_TAGS
                        and not is_user_id_tag(tag, known_user_ids=known_user_ids)
                    ):
                        bucket["nickname"] = tag
                        break

    for ledger_user in ledger_users or []:
        bucket = user_data.get(ledger_user["user_id"])
        if bucket is None:
            continue
        platform = ledger_user.get("platform")
        if platform and platform not in bucket["platforms"]:
            bucket["platforms"].append(platform)
        if not bucket["nickname"] and ledger_user.get("nickname"):
            bucket["nickname"] = ledger_user["nickname"]

    return sorted(user_data.values(), key=lambda u: u["memory_count"], reverse=True)


def user_profile_memories(
    records: List[Dict[str, Any]],
    user_id: str,
) -> List[Dict[str, Any]]:
    """某个用户的画像记忆明细。"""
    uid = str(user_id or "").strip()
    if not uid:
        return []
    items = []
    for record in records or []:
        tags = _record_tags(record)
        if uid not in tags:
            continue
        if not is_user_profile_tags(tags, known_user_ids=[uid]):
            continue
        item = _record_item(record)
        item["attribute"] = extract_profile_attribute_from_tags(tags)
        items.append(item)
    return items


class ProfileAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _memory_store(self):
        try:
            return self.plugin_context.get_component("memory_store")
        except Exception:
            return None

    def _ledger(self):
        try:
            return self.plugin_context.get_component("memory_sql_manager")
        except Exception:
            return None

    async def list_users(self):
        """列出所有可识别用户：画像记忆来自新库，账本提供平台与昵称。"""
        store = self._memory_store()
        ledger = self._ledger()
        if store is None:
            return jsonify({"users": [], "groups": []})

        try:
            import asyncio

            records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
            ledger_users: List[Dict[str, Any]] = []
            groups: List[Dict[str, Any]] = []
            if ledger is not None:
                ledger_users = await asyncio.to_thread(ledger.list_ledger_users)
                groups = await asyncio.to_thread(ledger.list_ledger_groups)

            users = group_users(profile_records(records), ledger_users)
            return jsonify({"users": users, "groups": groups})
        except Exception as e:
            logger.error(f"[WebUI] 用户画像列表查询失败: {e}", exc_info=True)
            return jsonify({"users": [], "groups": [], "error": str(e)}), 500

    async def get_user_profile(self):
        """获取指定用户的完整画像"""
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"error": "缺少 user_id 参数"}), 400

        store = self._memory_store()
        if store is None:
            return jsonify({"error": "记忆组件不可用"}), 500

        try:
            import asyncio

            records = await asyncio.to_thread(store.list_records, None, all_scopes=True)
            return jsonify({
                "user_id": user_id,
                "memories": user_profile_memories(records, user_id),
            })
        except Exception as e:
            logger.error(f"[WebUI] 用户画像详情查询失败: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
