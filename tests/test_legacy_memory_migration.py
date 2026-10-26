"""旧 SQL 记忆自动迁移测试。

验证三件事：首次启动补齐字段、重复启动不产生副本、旧库新增条目后只补增量。

旧库由 `tests/legacy_db_fixture.LegacyDb` 直接播种：生产侧已不再写旧库，
迁移只负责把旧库读出来。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from astrbot_plugin_angel_memory.core.services.legacy_memory_migration import (
    LegacyMemoryMigration,
)
from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import (
    MemorySqlManager,
)
from astrbot_plugin_angel_memory.pmemory import MemoryStore, pmemory

from legacy_db_fixture import LegacyDb


def _make_source():
    """返回 (迁移源管理器, 旧库播种器)，两者指向同一个 SQL 文件。"""
    path = Path(tempfile.mkdtemp()) / "legacy.db"
    return MemorySqlManager(path), LegacyDb(path)


def _migrate(sql_manager, state_dir: Path, backup_dir: Path | None = None) -> dict:
    migration = LegacyMemoryMigration(
        memory_store=MemoryStore(),
        memory_sql_manager=sql_manager,
        state_dir=state_dir,
        backup_dir=backup_dir,
    )
    return asyncio.run(migration.run_if_needed())


def _setup_env(backup_dir: Path | None = None):
    root = Path(tempfile.mkdtemp(prefix="pmem-mig-"))
    pmemory.configure(root=root, raw_dir=None)
    return root, MemoryStore()


def _write_backup(backup_dir: Path, records, tags=None, rel=None, name="memory_backup_19700101.json"):
    backup_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "exported_at": 0,
        "records": records,
        "global_tags": tags or [],
        "memory_tag_rel": rel or [],
    }
    path = backup_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_first_run_copies_fields_and_tags():
    root, store = _setup_env()
    sql, db = _make_source()
    legacy_id = db.add_memory(
        "红豆喜欢甜食",
        reasoning="聊天里提到",
        tags=["红豆", "偏好"],
        strength=7,
        useful_score=4.5,
        useful_count=3,
        memory_scope="恋爱",
    )

    result = _migrate(sql, root)
    assert (result["status"], result["written"]) == ("success", 1)
    assert result["source"] == "旧SQL"

    records = store.list_records(all_scopes=True)
    assert len(records) == 1
    record = records[0]
    assert record["judgment"] == "红豆喜欢甜食"
    assert record["reasoning"] == "聊天里提到"
    assert record["scope"] == "恋爱"
    assert sorted(record["tags"]) == ["偏好", "红豆"]
    assert int(record["state"]["strength"]) == 7
    assert float(record["state"]["useful_score"]) == 4.5
    assert int(record["state"]["useful_count"]) == 3
    assert record["metadata"]["legacy"]["id"] == legacy_id
    pmemory.close()


def test_second_run_is_idempotent():
    root, store = _setup_env()
    sql, db = _make_source()
    db.add_memory("第一条")
    db.add_memory("第二条")

    first = _migrate(sql, root)
    assert (first["status"], first["written"]) == ("success", 2)
    second = _migrate(sql, root)
    assert (second["status"], second["written"]) == ("skipped", 0)
    assert len(store.list_records(all_scopes=True)) == 2
    pmemory.close()


def test_increment_only_writes_new_rows():
    root, store = _setup_env()
    sql, db = _make_source()
    db.add_memory("第一条")

    first = _migrate(sql, root)
    assert (first["status"], first["written"]) == ("success", 1)

    db.add_memory("第二条")
    second = _migrate(sql, root)
    assert (second["status"], second["written"]) == ("success", 1)
    assert sorted(r["judgment"] for r in store.list_records(all_scopes=True)) == [
        "第一条",
        "第二条",
    ]
    pmemory.close()


def test_active_flags_become_pinned():
    root, store = _setup_env()
    sql, db = _make_source()
    db.add_memory("用户偏好", is_active=True)

    _migrate(sql, root)
    record = store.list_records(all_scopes=True)[0]
    assert record["state"]["pinned"] is True
    pmemory.close()


def test_empty_legacy_library_skips():
    root, _ = _setup_env()
    sql, _db = _make_source()

    result = _migrate(sql, root)
    assert result["status"] == "skipped"
    assert result["written"] == 0
    pmemory.close()


def test_backup_is_used_when_sql_is_empty():
    root, store = _setup_env()
    sql, _db = _make_source()
    backup_dir = Path(tempfile.mkdtemp(prefix="pmem-bak-"))
    _write_backup(
        backup_dir,
        records=[
            {
                "id": "backup-1",
                "memory_type": "知识记忆",
                "judgment": "备份里的记忆",
                "reasoning": "冷备份来源",
                "strength": 5,
                "is_active": 0,
                "useful_count": 2,
                "useful_score": 6.0,
                "last_recalled_at": 0.0,
                "last_decay_at": 0.0,
                "memory_scope": "public",
                "created_at": 1780029185.0,
                "updated_at": 1780029185.0,
            }
        ],
        tags=[{"id": 1, "name": "备份标签"}],
        rel=[{"memory_id": "backup-1", "tag_id": 1}],
    )

    result = _migrate(sql, root, backup_dir)
    assert result["status"] == "success"
    assert result["written"] == 1
    assert "JSON备份" in result["source"]

    records = store.list_records(all_scopes=True)
    assert len(records) == 1
    assert records[0]["judgment"] == "备份里的记忆"
    assert records[0]["tags"] == ["备份标签"]
    pmemory.close()


def test_backup_only_fills_rows_missing_from_sql():
    root, store = _setup_env()
    sql, db = _make_source()
    from_sql = db.add_memory("SQL里的记忆")
    backup_dir = Path(tempfile.mkdtemp(prefix="pmem-bak-"))
    _write_backup(
        backup_dir,
        records=[
            {
                "id": from_sql,
                "judgment": "SQL里的记忆（备份副本）",
                "memory_type": "知识记忆",
                "reasoning": "",
                "strength": 1,
                "is_active": 0,
                "useful_count": 0,
                "useful_score": 0.0,
                "memory_scope": "public",
                "created_at": 1780029185.0,
                "updated_at": 1780029185.0,
            },
            {
                "id": "backup-only",
                "memory_type": "知识记忆",
                "judgment": "只在备份里的记忆",
                "reasoning": "",
                "strength": 1,
                "is_active": 0,
                "useful_count": 0,
                "useful_score": 0.0,
                "memory_scope": "public",
                "created_at": 1780029185.0,
                "updated_at": 1780029185.0,
            },
        ],
    )

    result = _migrate(sql, root, backup_dir)
    assert result["written"] == 2
    judgments = sorted(r["judgment"] for r in store.list_records(all_scopes=True))
    assert judgments == ["SQL里的记忆", "只在备份里的记忆"]
    pmemory.close()


def test_latest_backup_file_wins():
    root, store = _setup_env()
    sql, _db = _make_source()
    backup_dir = Path(tempfile.mkdtemp(prefix="pmem-bak-"))
    base = {
        "memory_type": "知识记忆",
        "reasoning": "",
        "strength": 1,
        "is_active": 0,
        "useful_count": 0,
        "useful_score": 0.0,
        "memory_scope": "public",
        "created_at": 1780029185.0,
        "updated_at": 1780029185.0,
    }
    _write_backup(backup_dir, [{"id": "old", "judgment": "旧备份", **base}], name="memory_backup_20260101.json")
    _write_backup(backup_dir, [{"id": "new", "judgment": "新备份", **base}], name="memory_backup_20260102.json")

    _migrate(sql, root, backup_dir)
    assert [r["judgment"] for r in store.list_records(all_scopes=True)] == ["新备份"]
    pmemory.close()


def test_self_backup_round_trips_inline_tags():
    """p-memory 自己导出的备份（标签内嵌）能被回灌。"""
    root, store = _setup_env()
    store.remember("能回去的记忆", "自备份", ["标签甲", "标签乙"], "知识记忆", "恋爱")
    snapshot = store.export_snapshot()
    pmemory.close()

    backup_dir = Path(tempfile.mkdtemp(prefix="pmem-bak-"))
    (backup_dir / "memory_backup_20260101.json").write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )

    root2, store2 = _setup_env()
    sql, _db = _make_source()
    result = _migrate(sql, root2, backup_dir)
    assert (result["status"], result["written"]) == ("success", 1)

    restored = store2.list_records(all_scopes=True)[0]
    assert restored["judgment"] == "能回去的记忆"
    assert restored["scope"] == "恋爱"
    assert sorted(restored["tags"]) == ["标签乙", "标签甲"]
    pmemory.close()
