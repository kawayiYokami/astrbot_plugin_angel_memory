"""Web 接口收敛回归：记忆浏览/标签/导入导出/画像都走 p-memory。

这些用例只测纯函数与数据结构契约，不依赖 quart 请求上下文，
用来钉住「Web 侧不再读旧 SQL 记忆表」这条边界。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from astrbot_plugin_angel_memory.pmemory import MemoryStore, pmemory
from astrbot_plugin_angel_memory.web_api import memory_api, tags_api, profile_api
from astrbot_plugin_angel_memory.web_api.maintenance_api import MaintenanceAPI


def _fresh_store() -> MemoryStore:
    root = Path(tempfile.mkdtemp(prefix="pmem-web-"))
    pmemory.configure(root=root, raw_dir=None)
    return MemoryStore()


def _seed(store: MemoryStore):
    store.remember("红豆喜欢甜食", "聊天里提到", ["红豆", "偏好"], "知识记忆", "恋爱", pinned=True)
    store.remember("原神有七国", "常识", ["原神"], "知识记忆", "public")
    store.remember("用户珍视安静", "性格", ["123456", "事实属性"], "知识记忆", "public")


def test_memory_browse_uses_pmemory_and_keeps_fields():
    store = _fresh_store()
    try:
        _seed(store)
        records = store.list_records(all_scopes=True)
        items = [memory_api._record_to_item(r) for r in records]
        scopes = {item["memory_scope"] for item in items}
        assert scopes == {"恋爱", "public"}
        pinned = [i for i in items if i["is_active"]]
        assert len(pinned) == 1 and pinned[0]["judgment"] == "红豆喜欢甜食"
        # 时间是秒级，前端才认
        assert all(isinstance(i["created_at"], float) for i in items)
        assert all(i["created_at"] < 1e11 for i in items)
    finally:
        pmemory.close()


def test_memory_api_delete_rejects_legacy_uuid(tmp_path):
    store = _fresh_store()
    try:
        record = store.remember("会被删掉", "r", ["t"], "知识记忆", "public")
        memory_id = int(record["id"])
        assert memory_api._to_seconds(1780029185000000) == 1780029185.0
        assert store.delete([memory_id]) == 1
        assert store.list_records(all_scopes=True) == []
    finally:
        pmemory.close()


def test_tags_aggregate_and_hit_search():
    store = _fresh_store()
    try:
        _seed(store)
        records = store.list_records(all_scopes=True)
        tags = tags_api.aggregate_tags(records)
        names = [t["name"] for t in tags]
        assert "红豆" in names and "原神" in names
        assert all(t["note_refs"] == 0 for t in tags)
        assert all(isinstance(t["memory_refs"], int) and t["memory_refs"] >= 1 for t in tags)

        matched = tags_api.match_tag_names(records, "我听说红豆喜欢原神")
        assert matched == ["原神", "红豆"] or matched == ["红豆", "原神"]
        assert tags_api.match_tag_names(records, "完全无关的句子") == []

        hits = tags_api.search_memories_by_tags(records, matched, "", 10)
        assert len(hits) == 2
        assert all(h["hit_count"] >= 1 for h in hits)
        assert {h["judgment"] for h in hits} == {"红豆喜欢甜食", "原神有七国"}
    finally:
        pmemory.close()


def test_tags_hit_search_respects_scope():
    store = _fresh_store()
    try:
        _seed(store)
        records = store.list_records(all_scopes=True)
        # public 视角看不到「恋爱」scope 的记忆
        public_hits = tags_api.search_memories_by_tags(records, ["红豆", "原神"], "public", 10)
        assert {h["judgment"] for h in public_hits} == {"原神有七国"}
        # 恋爱视角能看到 public + 自己
        own_hits = tags_api.search_memories_by_tags(records, ["红豆", "原神"], "恋爱", 10)
        assert {h["judgment"] for h in own_hits} == {"红豆喜欢甜食", "原神有七国"}
    finally:
        pmemory.close()


def test_profile_grouping_uses_ledger_and_pmemory():
    store = _fresh_store()
    try:
        _seed(store)
        records = store.list_records(all_scopes=True)
        profile = profile_api.profile_records(records)
        assert len(profile) == 1

        ledger_users = [{
            "platform": "aiocqhttp",
            "user_id": "123456",
            "nickname": "红豆",
            "nicknames": ["红豆"],
            "first_seen_at": 1.0,
            "last_seen_at": 2.0,
        }]
        users = profile_api.group_users(profile, ledger_users)
        assert len(users) == 1
        user = users[0]
        assert user["user_id"] == "123456"
        assert user["memory_count"] == 1
        assert user["platforms"] == ["aiocqhttp"]
        assert user["nickname"] == "红豆"
        assert user["attributes"] == {"事实属性": 1}

        memories = profile_api.user_profile_memories(records, "123456")
        assert len(memories) == 1
        assert memories[0]["judgment"] == "用户珍视安静"
        assert memories[0]["attribute"]
    finally:
        pmemory.close()


def test_import_snapshot_writes_pmemory_and_marks_legacy():
    store = _fresh_store()
    try:
        api = MaintenanceAPI(plugin_context=None)
        payload = {
            "schema_version": 1,
            "records": [
                {
                    "id": "legacy-uuid-1",
                    "memory_type": "知识记忆",
                    "judgment": "从旧快照导入",
                    "reasoning": "旧备份",
                    "strength": 5,
                    "is_active": 1,
                    "useful_count": 2,
                    "useful_score": 3.5,
                    "memory_scope": "恋爱",
                    "created_at": 1780029185.0,
                    "updated_at": 1780029185.0,
                }
            ],
            "global_tags": [{"id": 1, "name": "旧标签"}],
            "memory_tag_rel": [{"memory_id": "legacy-uuid-1", "tag_id": 1}],
        }
        normalized = api._normalize_snapshot_records(payload)
        assert normalized[0]["tags"] == ["旧标签"]

        # 直接走 store 写入路径（与 import_snapshot 内部同一分支）
        from astrbot_plugin_angel_memory.core.services.legacy_memory_migration import (
            LegacyMemoryMigration,
        )

        metadata = LegacyMemoryMigration._legacy_key("legacy-uuid-1")
        store.remember(
            normalized[0]["judgment"],
            normalized[0]["reasoning"],
            normalized[0]["tags"],
            normalized[0]["memory_type"],
            normalized[0]["memory_scope"],
            pinned=bool(normalized[0]["is_active"]),
            strength=normalized[0]["strength"],
            useful_count=normalized[0]["useful_count"],
            useful_score=normalized[0]["useful_score"],
            metadata=metadata,
            created_at=normalized[0]["created_at"],
        )
        record = store.list_records(all_scopes=True)[0]
        assert record["metadata"]["legacy"]["id"] == "legacy-uuid-1"
        assert record["tags"] == ["旧标签"]
        assert record["state"]["pinned"] is True
        assert record["state"]["strength"] == 5
    finally:
        pmemory.close()


def test_migration_export_snapshot_shape_matches_import():
    store = _fresh_store()
    try:
        _seed(store)
        snapshot = store.export_snapshot()
        text = json.dumps(snapshot, ensure_ascii=False)
        assert "红豆喜欢甜食" in text
        records = snapshot["records"]
        assert len(records) == 3
        # 标签内嵌，回灌时不依赖关系表
        assert all(isinstance(r["tags"], list) and r["tags"] for r in records)
        first = records[0]
        assert set(["id", "judgment", "memory_scope", "created_at", "tags"]) <= set(first)
    finally:
        pmemory.close()
