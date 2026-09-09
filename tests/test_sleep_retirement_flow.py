"""睡眠回收编排测试：衰减 -> 取候选 -> 审查 -> 应用裁决。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_astrbot_stubs() -> None:
    if "astrbot" not in sys.modules:
        sys.modules["astrbot"] = types.ModuleType("astrbot")

    api = sys.modules.get("astrbot.api")
    if api is None:
        api = types.ModuleType("astrbot.api")
        sys.modules["astrbot.api"] = api
    if not hasattr(api, "logger"):
        api.logger = logging.getLogger("astrbot-test")

    if "astrbot.api.event" not in sys.modules:
        event = types.ModuleType("astrbot.api.event")

        class AstrMessageEvent:
            pass

        event.AstrMessageEvent = AstrMessageEvent
        sys.modules["astrbot.api.event"] = event

    if "astrbot.api.provider" not in sys.modules:
        provider_module = types.ModuleType("astrbot.api.provider")

        class ProviderRequest:
            pass

        provider_module.ProviderRequest = ProviderRequest
        sys.modules["astrbot.api.provider"] = provider_module

    if "astrbot.core.agent.message" not in sys.modules:
        sys.modules.setdefault("astrbot.core", types.ModuleType("astrbot.core"))
        sys.modules.setdefault(
            "astrbot.core.agent", types.ModuleType("astrbot.core.agent")
        )
        message = types.ModuleType("astrbot.core.agent.message")

        class TextPart:
            def __init__(self, text: str = ""):
                self.text = text

        message.TextPart = TextPart
        sys.modules["astrbot.core.agent.message"] = message


_install_astrbot_stubs()

# 被测单元不依赖 tantivy；环境缺少该可选依赖时补一个占位模块，避免收集失败
try:
    import tantivy  # noqa: F401
except ImportError:
    sys.modules["tantivy"] = types.ModuleType("tantivy")

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

SERVICES_PACKAGE = f"{CORE_PACKAGE}.services"
if SERVICES_PACKAGE not in sys.modules:
    package = types.ModuleType(SERVICES_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "core" / "services")]
    sys.modules[SERVICES_PACKAGE] = package

from astrbot_plugin_angel_memory.core.services.sleep_service import DeepMindSleepService
from astrbot_plugin_angel_memory.llm_memory.models.data_models import (
    BaseMemory,
    MemoryType,
)

_SHORT_ID_PATTERN = re.compile(r"- id: `([^`]+)`")


class _FakeProvider:
    """按提示词里的清单编号动态生成裁决，避免测试依赖短 id 的具体形态。"""

    def __init__(self, responder=None):
        self.responder = responder

    async def text_chat(self, prompt: str, system_prompt: str = None):
        text = self.responder(prompt) if self.responder else ""
        return SimpleNamespace(completion_text=text)


class _FakeContext:
    def __init__(self, provider):
        self.provider = provider

    def get_provider_by_id(self, provider_id: str):
        return self.provider


class _FakeMemorySystem:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.decay_calls = 0
        self.list_calls = 0
        self.applied = []

    async def decay_tier0_memories(self):
        self.decay_calls += 1
        return len(self.candidates)

    async def list_retirement_candidates(self):
        self.list_calls += 1
        return list(self.candidates)

    async def apply_retirement_decisions(self, delete_ids, keep_ids):
        self.applied.append((list(delete_ids), list(keep_ids)))
        return {"deleted": len(delete_ids), "promoted": len(keep_ids)}


class _FakeTraceStore:
    def __init__(self):
        self.records = []

    async def append(self, record):
        self.records.append(record)
        return record


class _FakeDeepMind:
    def __init__(self, memory_system, provider):
        self.memory_system = memory_system
        self.logger = logging.getLogger("sleep-flow-test")
        self.provider_id = "provider-a"
        self.context = _FakeContext(provider)
        self.trace_store = _FakeTraceStore()
        self.plugin_context = None

    def is_enabled(self):
        return True


def _memory(memory_id: str) -> BaseMemory:
    return BaseMemory(
        memory_type=MemoryType.KNOWLEDGE,
        judgment=f"判断-{memory_id}",
        reasoning="依据",
        tags=["测试"],
        id=memory_id,
        strength=0,
        is_active=False,
    )


def _verdict_responder(verdicts):
    def responder(prompt: str) -> str:
        listing = prompt.split("# 记忆清单", 1)[-1]
        ids = _SHORT_ID_PATTERN.findall(listing)
        decisions = [
            {"id": ids[index], "verdict": verdict}
            for index, verdict in enumerate(verdicts)
            if verdict and index < len(ids)
        ]
        return json.dumps({"decisions": decisions}, ensure_ascii=False)

    return responder


def _run(candidates, verdicts=()):
    memory_system = _FakeMemorySystem(candidates)
    deepmind = _FakeDeepMind(memory_system, _FakeProvider(_verdict_responder(verdicts)))
    service = DeepMindSleepService(deepmind)
    result = asyncio.run(service._recycle_memories())
    return memory_system, deepmind, result


def test_no_candidates_skips_review_and_apply():
    memory_system, _, result = _run([])

    assert memory_system.decay_calls == 1
    assert memory_system.list_calls == 1
    assert memory_system.applied == []
    assert result["candidates"] == 0


def test_keep_verdict_promotes_candidate():
    memory_system, _, result = _run([_memory("mem-0")], ["keep"])

    assert memory_system.applied == [([], ["mem-0"])]
    assert result["promoted"] == 1
    assert result["deleted"] == 0


def test_delete_verdict_removes_candidate():
    memory_system, _, result = _run([_memory("mem-0")], ["delete"])

    assert memory_system.applied == [(["mem-0"], [])]
    assert result["deleted"] == 1


def test_omitted_candidate_stays_untouched():
    memory_system, _, result = _run([_memory("mem-0"), _memory("mem-1")], ["keep", None])

    assert memory_system.applied == [([], ["mem-0"])]
    assert result["undecided"] == 1


def test_review_failure_skips_whole_batch():
    memory_system = _FakeMemorySystem([_memory("mem-0"), _memory("mem-1")])
    deepmind = _FakeDeepMind(memory_system, None)
    service = DeepMindSleepService(deepmind)

    result = asyncio.run(service._recycle_memories())

    assert memory_system.applied == []
    assert result["skipped"] is True
    assert result["deleted"] == 0
    assert result["promoted"] == 0


def test_empty_response_skips_whole_batch():
    memory_system = _FakeMemorySystem([_memory("mem-0")])
    deepmind = _FakeDeepMind(memory_system, _FakeProvider())
    service = DeepMindSleepService(deepmind)

    result = asyncio.run(service._recycle_memories())

    assert memory_system.applied == []
    assert result["skipped"] is True


def test_sleep_records_trace_with_summary():
    memory_system = _FakeMemorySystem([_memory("mem-0")])
    deepmind = _FakeDeepMind(
        memory_system, _FakeProvider(_verdict_responder(["delete"]))
    )
    service = DeepMindSleepService(deepmind)
    result = asyncio.run(service._recycle_memories())

    asyncio.run(service._record_sleep_trace(result, start_time=0.0))

    record = deepmind.trace_store.records[0]
    assert record["type"] == "sleep"
    assert record["status"] == "success"
    assert record["summary"]["candidates"] == 1
    assert record["summary"]["deleted"] == 1
    assert "记忆清单" in record["prompt"]
    assert record["system_prompt"]
    assert "判断-mem-0" not in record["system_prompt"]
