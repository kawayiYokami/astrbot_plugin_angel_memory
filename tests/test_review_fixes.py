"""review 返修点的回归测试：直读越权、假成功、迁移导出。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from astrbot_plugin_angel_memory.core.services.legacy_memory_migration import (
    EXPORT_FILE,
    LegacyMemoryMigration,
)
from astrbot_plugin_angel_memory.core.services.note_index_service import NoteIndexService
from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import (
    MemorySqlManager,
)
from astrbot_plugin_angel_memory.pmemory import MemoryStore, pmemory

from legacy_db_fixture import LegacyDb


class _FakeContext:
    def __init__(self, note_index, visible):
        self._note_index = note_index
        self._visible = list(visible)

    def get_component(self, name):
        return self._note_index if name == "note_index" else None

    def visible_domains(self, group_id):
        return list(self._visible)


class _FakeEvent:
    def __init__(self, plugin_context, group_id="42"):
        self.plugin_context = plugin_context
        self._group_id = group_id

    def get_group_id(self):
        return self._group_id


def _setup_notes(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "原神").mkdir(parents=True, exist_ok=True)
    (raw / "绝区零").mkdir(parents=True, exist_ok=True)
    (raw / "原神" / "角色.md").write_text("# 角色\n原神正文 hidden测试", encoding="utf-8")
    (raw / "绝区零" / "代理人.md").write_text("# 代理人\n绝区零正文 hidden测试", encoding="utf-8")
    pmemory.configure(root=tmp_path / "pmemory", raw_dir=raw)
    service = NoteIndexService(raw)
    assert service.upsert_file(raw / "原神" / "角色.md")
    assert service.upsert_file(raw / "绝区零" / "代理人.md")
    gi_id = service._notes.resolve_note_id("gi", raw / "原神" / "角色.md")
    zzz_id = service._notes.resolve_note_id("zzz", raw / "绝区零" / "代理人.md")
    assert gi_id and zzz_id
    return service, gi_id, zzz_id


def test_note_read_hidden_returns_not_found(tmp_path):
    from astrbot_plugin_angel_memory.tools.angel_note_read import NoteRecallTool

    service, gi_id, zzz_id = _setup_notes(tmp_path)
    try:
        # 本群只允许 gi：读 gi 成功，读 zzz 与不存在统一报未找到
        ctx = _FakeContext(service, ["gi"])
        event = _FakeEvent(ctx)
        tool = NoteRecallTool()

        hidden = asyncio.run(tool.run(event, note_id=int(zzz_id)))
        assert "未找到" in hidden
        assert "绝区零正文" not in hidden

        visible = asyncio.run(tool.run(event, note_id=int(gi_id)))
        assert "未找到" not in visible
        assert "原神正文" in visible

        missing = asyncio.run(tool.run(event, note_id=999999))
        assert "未找到" in missing
        # hidden 与不存在的文案一致，不泄露哪个是隐藏域
        assert hidden.split("note_id=")[0] == missing.split("note_id=")[0]
    finally:
        pmemory.close()


class _EmptyRuntime:
    async def feedback(self, **kwargs):
        return []


class _RuntimeContext:
    def __init__(self, runtime):
        self._runtime = runtime

    def get_component(self, name):
        return self._runtime if name == "memory_runtime" else None

    async def resolve_memory_scope_from_event(self, event):
        return "public"


class _RememberEvent:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context


def test_remember_update_missing_source_is_error():
    from astrbot_plugin_angel_memory.tools.angel_remember import CoreMemoryRememberTool

    tool = CoreMemoryRememberTool()
    event = _RememberEvent(_RuntimeContext(_EmptyRuntime()))
    result = asyncio.run(
        tool.run(
            event,
            action="update",
            memory={"judgment": "新判断", "memory_type": "knowledge", "tags": ["标签"]},
            source_memory_ids=["999"],
        )
    )
    assert "源记忆不存在" in result
    assert "已更新" not in result


def test_merge_with_missing_source_creates_nothing(tmp_path):
    from astrbot_plugin_angel_memory.core.memory_runtime.pmemory_runtime import (
        PMemoryMemoryRuntime,
    )

    pmemory.configure(root=tmp_path / "pmemory", raw_dir=None)
    try:
        store = MemoryStore()
        runtime = PMemoryMemoryRuntime(store)
        first = asyncio.run(
            runtime.remember(
                memory_type="knowledge", judgment="第一条",
                reasoning="", tags=["甲"], memory_scope="public",
            )
        )
        before = len(store.list_records(all_scopes=True))
        result = runtime._apply_action(
            {
                "action": "merge",
                "source_memory_ids": [str(first), "999999"],
                "memory": {"type": "knowledge", "judgment": "合并", "tags": ["甲"]},
            },
            "public",
        )
        assert result is None
        assert len(store.list_records(all_scopes=True)) == before
    finally:
        pmemory.close()


def _make_source():
    path = Path(tempfile.mkdtemp()) / "legacy.db"
    return MemorySqlManager(path), LegacyDb(path)


def test_migration_exports_json_then_imports(tmp_path):
    root = Path(tempfile.mkdtemp(prefix="pmem-mig-"))
    pmemory.configure(root=root, raw_dir=None)
    try:
        sql, db = _make_source()
        db.add_memory(
            "导出导入的记忆",
            reasoning="背景",
            tags=["导出标签"],
            created_at=time.time(),
        )

        state_dir = tmp_path / "center"
        migration = LegacyMemoryMigration(
            memory_store=MemoryStore(),
            memory_sql_manager=sql,
            state_dir=state_dir,
        )
        result = asyncio.run(migration.run_if_needed())
        assert result["status"] == "success"
        assert result["written"] == 1

        export_path = state_dir / EXPORT_FILE
        assert export_path.exists()
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert len(payload["records"]) == 1
        assert payload["records"][0]["judgment"] == "导出导入的记忆"

        # 重复启动零副本：导出仍在，导入跳过
        second = asyncio.run(migration.run_if_needed())
        assert second["status"] == "skipped"
        assert second["written"] == 0
    finally:
        pmemory.close()
