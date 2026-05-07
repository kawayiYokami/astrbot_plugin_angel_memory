from __future__ import annotations

import asyncio
import logging
import sys
import threading
import types
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")
    event = types.ModuleType("astrbot.api.event")

    class AstrMessageEvent:
        pass

    event.AstrMessageEvent = AstrMessageEvent

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

CORE_PACKAGE = f"{PACKAGE_NAME}.core"
if CORE_PACKAGE not in sys.modules:
    package = types.ModuleType(CORE_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "core")]
    sys.modules[CORE_PACKAGE] = package

CORE_UTILS_PACKAGE = f"{CORE_PACKAGE}.utils"
if CORE_UTILS_PACKAGE not in sys.modules:
    package = types.ModuleType(CORE_UTILS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "core" / "utils")]
    sys.modules[CORE_UTILS_PACKAGE] = package

LLM_MEMORY_PACKAGE = f"{PACKAGE_NAME}.llm_memory"
if LLM_MEMORY_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory")]
    sys.modules[LLM_MEMORY_PACKAGE] = package

LLM_MEMORY_MODELS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.models"
if LLM_MEMORY_MODELS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_MODELS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "models")]
    sys.modules[LLM_MEMORY_MODELS_PACKAGE] = package

LLM_MEMORY_SERVICE_PACKAGE = f"{LLM_MEMORY_PACKAGE}.service"
if LLM_MEMORY_SERVICE_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_SERVICE_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "service")]
    sys.modules[LLM_MEMORY_SERVICE_PACKAGE] = package

LLM_MEMORY_COMPONENTS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.components"
if LLM_MEMORY_COMPONENTS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_COMPONENTS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "components")]
    sys.modules[LLM_MEMORY_COMPONENTS_PACKAGE] = package

from astrbot_plugin_angel_memory.core.utils.memory_id_resolver import MemoryIDResolver
from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager
from astrbot_plugin_angel_memory.llm_memory.models.data_models import BaseMemory, MemoryType
from astrbot_plugin_angel_memory.llm_memory.service.memory_manager import MemoryManager


class _FakeCollection:
    def __init__(self, records: dict[str, dict]):
        self.records = records
        self.deleted_ids: list[str] = []

    def get(self, ids=None, **kwargs):
        if not ids:
            return {"metadatas": [], "ids": []}
        memory_id = ids[0]
        metadata = self.records.get(memory_id)
        if metadata is None:
            return {"metadatas": [], "ids": []}
        return {"metadatas": [metadata], "ids": [memory_id]}

    def delete(self, ids):
        self.deleted_ids = list(ids or [])
        for memory_id in self.deleted_ids:
            self.records.pop(memory_id, None)


class _FakeStore:
    def __init__(self):
        self.saved_memories = []

    async def remember(self, collection, memory):
        self.saved_memories.append(memory)
        return memory


def test_normalize_memory_actions_deduplicates_merge_sources():
    normalized = MemoryIDResolver.normalize_memory_actions_format(
        [
            {
                "action": "create",
                "source_memory_ids": ["should-be-ignored"],
                "memory": {
                    "type": "knowledge",
                    "judgment": "用户常用 Python。",
                    "reasoning": "用户明确说明。",
                    "tags": ["Python"],
                },
            },
            {
                "action": "merge",
                "source_memory_ids": ["31", "31", "44", ""],
                "memory": {
                    "type": "skill",
                    "judgment": "用户偏好分步执行复杂任务。",
                    "reasoning": "多轮对话一致体现该偏好。",
                    "tags": ["任务偏好"],
                },
            },
        ]
    )

    assert normalized[0]["action"] == "create"
    assert "source_memory_ids" not in normalized[0]
    assert normalized[1]["source_memory_ids"] == ["31", "44"]


def test_normalize_memory_actions_accepts_updata():
    normalized = MemoryIDResolver.normalize_memory_actions_format(
        [
            {
                "action": "updata",
                "source_memory_ids": ["52", "52"],
                "memory": {
                    "type": "knowledge",
                    "judgment": "用户要求所有记忆反馈统一采用 memory_actions 协议。",
                    "reasoning": "用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
                    "tags": ["memory_actions", "反馈协议", "结构约束"],
                },
            }
        ]
    )

    assert len(normalized) == 1
    assert normalized[0]["action"] == "updata"
    assert normalized[0]["source_memory_ids"] == ["52"]


def test_normalize_memory_actions_rejects_multi_source_updata():
    normalized = MemoryIDResolver.normalize_memory_actions_format(
        [
            {
                "action": "updata",
                "source_memory_ids": ["52", "53"],
                "memory": {
                    "type": "knowledge",
                    "judgment": "用户要求所有记忆反馈统一采用 memory_actions 协议。",
                    "reasoning": "用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
                    "tags": ["memory_actions", "反馈协议", "结构约束"],
                },
            }
        ]
    )

    assert normalized == []


def test_merge_memories_allows_single_source_replacement():
    collection = _FakeCollection(
        {
            "m1": {
                "memory_type": MemoryType.KNOWLEDGE.value,
                "judgment": "用户使用旧版反馈协议。",
                "reasoning": "旧记忆。",
                "tags": ["旧协议", "反馈结构"],
                "strength": 3,
                "is_active": False,
                "created_at": 1_700_000_000.0,
                "memory_scope": "public",
                "useful_count": 0,
                "useful_score": 0.0,
                "last_recalled_at": 0.0,
            }
        }
    )
    store = _FakeStore()
    manager = MemoryManager(main_collection=collection, vector_store=store)

    merged = asyncio.run(
        manager.merge_memories(
            memories_to_merge_ids=["m1"],
            new_memory_type="knowledge",
            new_judgment="用户要求所有记忆反馈统一采用 memory_actions 协议。",
            new_reasoning="用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
            new_tags=["memory_actions", "反馈协议", "结构约束"],
        )
    )

    assert merged.judgment == "用户要求所有记忆反馈统一采用 memory_actions 协议。"
    assert merged.strength == 3
    assert collection.deleted_ids == ["m1"]
    assert len(store.saved_memories) == 1


def test_sql_merge_action_allows_single_source_replacement():
    manager = MemorySqlManager.__new__(MemorySqlManager)
    manager._lock = threading.RLock()
    manager._tag_names = set()

    lookup_calls = []
    sync_calls = {}

    def fake_get_memories(source_ids):
        lookup_calls.append(list(source_ids))
        return [
            BaseMemory(
                id="m1",
                memory_type=MemoryType.KNOWLEDGE,
                judgment="用户使用旧版反馈协议。",
                reasoning="旧记忆。",
                tags=["旧协议", "反馈结构"],
                strength=2,
                created_at=1_700_000_000.0,
                memory_scope="public",
            )
        ]

    class _FakeConnection:
        def execute(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    manager._get_memories_by_ids_sync = fake_get_memories
    manager._upsert_tags_and_bind = lambda conn, memory_id, tags: None
    manager._sync_memory_fts_batch_sync = (
        lambda upsert_ids=None, delete_ids=None: sync_calls.update(
            {"upsert_ids": upsert_ids, "delete_ids": delete_ids}
        )
    )
    manager._connect = lambda: _FakeConnection()

    merged = manager._merge_action_sync(
        ["m1"],
        {
            "type": "knowledge",
            "judgment": "用户要求所有记忆反馈统一采用 memory_actions 协议。",
            "reasoning": "用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
            "tags": ["memory_actions", "反馈协议", "结构约束"],
        },
        "public",
    )

    assert lookup_calls == [["m1"]]
    assert merged is not None
    assert merged.strength == 2
    assert merged.judgment == "用户要求所有记忆反馈统一采用 memory_actions 协议。"
    assert sync_calls["delete_ids"] == ["m1"]


def test_sql_updata_action_reuses_merge_path():
    manager = MemorySqlManager.__new__(MemorySqlManager)
    manager._lock = threading.RLock()
    manager._tag_names = set()
    manager.remember = None
    manager.reinforce_memories = None
    manager.decay_recalled_but_useless = None

    called = {}

    def fake_merge_action_sync(source_memory_ids, memory_data, resolved_scope):
        called["source_memory_ids"] = source_memory_ids
        called["memory_data"] = memory_data
        called["resolved_scope"] = resolved_scope
        return BaseMemory(
            id="new1",
            memory_type=MemoryType.KNOWLEDGE,
            judgment=str(memory_data.get("judgment") or ""),
            reasoning=str(memory_data.get("reasoning") or ""),
            tags=list(memory_data.get("tags") or []),
            strength=1,
            memory_scope=resolved_scope,
        )

    manager._normalize_scope = lambda scope: str(scope or "").strip() or "public"
    manager._merge_action_sync = fake_merge_action_sync

    created = asyncio.run(
        manager.process_feedback(
            useful_memory_ids=[],
            recalled_memory_ids=[],
            memory_actions=[
                {
                    "action": "updata",
                    "source_memory_ids": ["m1"],
                    "memory": {
                        "type": "knowledge",
                        "judgment": "用户要求所有记忆反馈统一采用 memory_actions 协议。",
                        "reasoning": "用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
                        "tags": ["memory_actions", "反馈协议", "结构约束"],
                    },
                }
            ],
            memory_scope="public",
        )
    )

    assert len(created) == 1
    assert called["source_memory_ids"] == ["m1"]
    assert called["resolved_scope"] == "public"
