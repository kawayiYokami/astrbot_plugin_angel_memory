from __future__ import annotations

import logging
import sys
import time
import types
from pathlib import Path

import pytest


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

    provider = types.ModuleType("astrbot.api.provider")

    class ProviderRequest:
        def __init__(self):
            self.contexts = []
            self.extra_user_content_parts = []

    provider.ProviderRequest = ProviderRequest

    core_pkg = types.ModuleType("astrbot.core")
    agent_pkg = types.ModuleType("astrbot.core.agent")
    message = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text: str = ""):
            self.text = text

    message.TextPart = TextPart

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.provider"] = provider
    sys.modules["astrbot.core"] = core_pkg
    sys.modules["astrbot.core.agent"] = agent_pkg
    sys.modules["astrbot.core.agent.message"] = message


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.core.services.injection_service import (
    DeepMindInjectionService,
)
from astrbot_plugin_angel_memory.core.memory_runtime.simple_memory_runtime import (
    SimpleMemoryRuntime,
)
from astrbot_plugin_angel_memory.core.session_memory import SessionMemoryManager
from astrbot_plugin_angel_memory.core.utils.memory_injector import MemoryInjector
from astrbot_plugin_angel_memory.llm_memory.models.data_models import (
    BaseMemory,
    MemoryType,
)


def _memory(
    memory_id: str,
    judgment: str,
    tags: list[str] | None = None,
    memory_type: MemoryType = MemoryType.KNOWLEDGE,
    created_at: float = 1_700_000_000.0,
    memory_scope: str = "public",
) -> BaseMemory:
    return BaseMemory(
        id=memory_id,
        memory_type=memory_type,
        judgment=judgment,
        reasoning="长期库中的最新依据。",
        tags=tags or [],
        created_at=created_at,
        memory_scope=memory_scope,
    )


class _Runtime:
    def __init__(self, memories: list[BaseMemory]):
        self.memories = {memory.id: memory for memory in memories}

    async def get_memories_by_ids(self, memory_ids, memory_scope=None):
        result = []
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory is None:
                continue
            if memory_scope == "public" and memory.memory_scope != "public":
                continue
            if memory_scope and memory_scope != "public" and memory.memory_scope not in {
                memory_scope,
                "public",
            }:
                continue
            result.append(memory)
        return result


class _Config:
    enable_soul_system = False


class _DeepMind:
    def __init__(self, runtime):
        self.session_memory_manager = SessionMemoryManager()
        self.memory_system = runtime
        self.memory_injector = MemoryInjector()
        self.config = _Config()
        self.soul = None
        self.user_profile_service = None
        self.logger = logging.getLogger("session-memory-test")


def test_session_memory_only_stores_reference_metadata():
    manager = SessionMemoryManager()
    manager.add_memories_to_session(
        "s1",
        [_memory("m1", "旧文本。", ["小明", "123456", "技能树", "python"])],
    )

    refs = manager.get_session_memories("s1")
    assert len(refs) == 1
    ref = refs[0]
    assert ref.id == "m1"
    assert ref.memory_type == "knowledge"
    assert ref.user_id == "123456"
    assert ref.added_at > 0
    assert not hasattr(ref, "judgment")
    assert not hasattr(ref, "reasoning")
    assert not hasattr(ref, "tags")
    assert not hasattr(ref, "strength")


@pytest.mark.asyncio
async def test_injection_hydrates_latest_memory_and_removes_missing_refs():
    latest = _memory(
        "m1",
        "小明（123456）现在主要使用 Python 写工具。",
        ["小明", "123456", "技能树", "python"],
        created_at=time.time(),
    )
    deepmind = _DeepMind(_Runtime([latest]))
    deepmind.session_memory_manager.add_memories_to_session(
        "s1",
        [
            _memory("m1", "旧文本。", ["小明", "123456", "技能树", "python"]),
            _memory("gone", "已经失效的旧记忆。", ["旧记忆"]),
        ],
    )

    request = types.SimpleNamespace(contexts=[], extra_user_content_parts=[])
    await DeepMindInjectionService(deepmind).inject_memories_to_request(
        request=request,
        session_id="s1",
        note_context="",
        memory_scope="public",
    )

    injected = request.contexts[0]["content"]
    assert "小明（123456）现在主要使用 Python 写工具。" in injected
    assert "旧文本。" not in injected
    assert "已经失效的旧记忆。" not in injected
    assert request.contexts[0]["_no_save"] is True
    assert request.extra_user_content_parts == []
    assert deepmind.session_memory_manager.get_session_memory_ids("s1") == ["m1"]


@pytest.mark.asyncio
async def test_injection_always_uses_no_save_contexts_even_with_secretary_decision():
    latest = _memory("m1", "这是最新记忆。", ["测试"])
    deepmind = _DeepMind(_Runtime([latest]))
    deepmind.session_memory_manager.add_memories_to_session("s1", [latest])

    request = types.SimpleNamespace(contexts=[], extra_user_content_parts=[])
    await DeepMindInjectionService(deepmind).inject_memories_to_request(
        request=request,
        session_id="s1",
        note_context="",
        has_secretary_decision=True,
        memory_scope="public",
    )

    assert len(request.contexts) == 1
    assert request.contexts[0]["_no_save"] is True
    assert "这是最新记忆。" in request.contexts[0]["content"]
    assert request.extra_user_content_parts == []


def test_capacity_cleanup_still_uses_reference_metadata():
    manager = SessionMemoryManager()
    manager.capacity_config.event = 1
    manager.add_memories_to_session(
        "s1",
        [
            _memory("e1", "事件一。", ["事件"], memory_type=MemoryType.EVENT),
            _memory("e2", "事件二。", ["事件"], memory_type=MemoryType.EVENT),
        ],
    )

    refs = manager.get_session_memories("s1")
    assert [ref.id for ref in refs] == ["e2"]


@pytest.mark.asyncio
async def test_simple_runtime_get_memories_by_ids_keeps_order_and_scope():
    public = _memory("public", "公共记忆。")
    private = _memory("private", "私有记忆。", memory_scope="scope-a")

    class Manager:
        async def get_memories_by_ids(self, memory_ids):
            return [private, public]

    runtime = SimpleMemoryRuntime(Manager())

    assert [
        memory.id
        for memory in await runtime.get_memories_by_ids(
            ["public", "private"],
            memory_scope="scope-a",
        )
    ] == ["public", "private"]
    assert [
        memory.id
        for memory in await runtime.get_memories_by_ids(
            ["public", "private"],
            memory_scope="public",
        )
    ] == ["public"]
