"""记忆淘汰审查器测试。"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.llm_memory.models.data_models import (
    BaseMemory,
    MemoryType,
)
from astrbot_plugin_angel_memory.llm_memory.service.memory_retirement_reviewer import (
    MemoryRetirementReviewer,
    RetirementReviewError,
)

ID_TOKEN = "test"


class _FakeProvider:
    def __init__(self, text: str = "", error: Exception = None, delay: float = 0.0):
        self.text = text
        self.error = error
        self.delay = delay
        self.calls = 0
        self.prompts = []
        self.system_prompts = []

    async def text_chat(self, prompt: str, system_prompt: str = None):
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(completion_text=self.text)


class _FakeContext:
    def __init__(self, provider):
        self.provider = provider

    def get_provider_by_id(self, provider_id: str):
        return self.provider


def _memory(memory_id: str, judgment: str = "判断内容") -> BaseMemory:
    return BaseMemory(
        memory_type=MemoryType.KNOWLEDGE,
        judgment=judgment,
        reasoning="依据内容",
        tags=["外貌", "自我设定"],
        id=memory_id,
        strength=0,
        is_active=False,
        created_at=0.0,
    )


def _reviewer(provider) -> MemoryRetirementReviewer:
    return MemoryRetirementReviewer(
        context=_FakeContext(provider),
        provider_id="provider-a",
        id_token=ID_TOKEN,
    )


def _short(index: int) -> str:
    return f"m{index}-{ID_TOKEN}"


def _response(*pairs) -> str:
    decisions = [{"id": _short(index), "verdict": verdict} for index, verdict in pairs]
    return json.dumps({"decisions": decisions}, ensure_ascii=False)


def test_review_parses_keep_and_delete():
    provider = _FakeProvider(text=_response((0, "keep"), (1, "delete")))
    candidates = [_memory("mem-0"), _memory("mem-1")]

    outcome = asyncio.run(_reviewer(provider).review(candidates))

    assert outcome.decisions == {"mem-0": "keep", "mem-1": "delete"}
    assert outcome.candidate_count == 2


def test_review_omitted_candidate_absent_from_decisions():
    provider = _FakeProvider(text=_response((0, "keep")))

    outcome = asyncio.run(
        _reviewer(provider).review([_memory("mem-0"), _memory("mem-1")])
    )

    assert outcome.decisions == {"mem-0": "keep"}
    assert "mem-1" not in outcome.decisions


def test_review_ignores_unknown_id_and_invalid_verdict():
    provider = _FakeProvider(
        text=json.dumps(
            {
                "decisions": [
                    {"id": "m9-test", "verdict": "keep"},
                    {"id": _short(0), "verdict": "maybe"},
                    {"id": _short(1), "verdict": "delete"},
                ]
            },
            ensure_ascii=False,
        )
    )

    outcome = asyncio.run(
        _reviewer(provider).review([_memory("mem-0"), _memory("mem-1")])
    )

    assert outcome.decisions == {"mem-1": "delete"}


def test_review_extracts_json_from_wrapped_text():
    provider = _FakeProvider(text=f"这是我的判断：\n```json\n{_response((0, 'keep'))}\n```")

    outcome = asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert outcome.decisions == {"mem-0": "keep"}


def test_prompt_contains_candidate_fields():
    provider = _FakeProvider(text=_response())

    outcome = asyncio.run(
        _reviewer(provider).review([_memory("mem-0", judgment="我的外貌是银色长发")])
    )

    assert "我的外貌是银色长发" in outcome.prompt
    assert f"`{_short(0)}`" in outcome.prompt
    assert "`外貌`" in outcome.prompt
    assert "`依据内容`" in outcome.prompt


def test_repeated_example_json_does_not_map_to_real_candidate():
    """模型复述提示词里的示例（id 为 m0/m1）不得落到真实候选上。"""
    example = json.dumps(
        {
            "decisions": [
                {"id": "m0", "verdict": "keep"},
                {"id": "m1", "verdict": "delete"},
            ]
        },
        ensure_ascii=False,
    )
    provider = _FakeProvider(text=f"{example}\n{_response((1, 'delete'))}")

    outcome = asyncio.run(
        _reviewer(provider).review([_memory("mem-0"), _memory("mem-1")])
    )

    # 示例编号与本次候选不同域，解析到示例时映射为空，结果是悬而不决而非误删
    assert "mem-1" not in outcome.decisions
    assert outcome.decisions == {}


def test_short_ids_are_unique_per_run():
    first = _reviewer(_FakeProvider())._build_short_ids(2)
    second = MemoryRetirementReviewer(
        context=_FakeContext(None), provider_id="p"
    )._build_short_ids(2)

    assert first == ["m0-test", "m1-test"]
    assert second[0].startswith("m0-")
    assert second != first


def test_review_raises_when_provider_missing():
    reviewer = MemoryRetirementReviewer(context=_FakeContext(None), provider_id="p")

    with pytest.raises(RetirementReviewError) as excinfo:
        asyncio.run(reviewer.review([_memory("mem-0")]))

    assert "找不到提供者" in str(excinfo.value)
    assert excinfo.value.prompt


def test_review_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(MemoryRetirementReviewer, "REVIEW_TIMEOUT_SECONDS", 0.01)
    provider = _FakeProvider(text="{}", delay=0.2)

    with pytest.raises(RetirementReviewError) as excinfo:
        asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert "超时" in str(excinfo.value)


def test_review_raises_on_invalid_json():
    provider = _FakeProvider(text="我无法完成这个请求。")

    with pytest.raises(RetirementReviewError) as excinfo:
        asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert "无法解析" in str(excinfo.value)
    assert excinfo.value.response == "我无法完成这个请求。"


def test_review_raises_on_empty_response():
    provider = _FakeProvider(text="   ")

    with pytest.raises(RetirementReviewError) as excinfo:
        asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert "空响应" in str(excinfo.value)


def test_review_raises_when_decisions_missing():
    provider = _FakeProvider(text='{"result": "ok"}')

    with pytest.raises(RetirementReviewError) as excinfo:
        asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert "decisions" in str(excinfo.value)


def test_rules_go_to_system_prompt_and_listing_to_user_prompt():
    """system 只放固定规则、user 只放本次清单，便于命中提示词缓存。"""
    candidates = [_memory("mem-0", judgment="雪豹数据光体")]
    reviewer = _reviewer(_FakeProvider(text="{}"))

    system_prompt = reviewer.build_system_prompt()
    user_prompt = reviewer.build_user_prompt(candidates, [_short(0)])

    assert "# 记忆清单" not in system_prompt
    assert "雪豹数据光体" not in system_prompt
    assert user_prompt.startswith("# 记忆清单")
    assert "雪豹数据光体" in user_prompt
    assert _short(0) in user_prompt


def test_review_sends_fixed_rules_as_system_prompt():
    provider = _FakeProvider(text=_response((0, "keep")))

    outcome = asyncio.run(_reviewer(provider).review([_memory("mem-0")]))

    assert provider.system_prompts[0] == _reviewer(provider).build_system_prompt()
    assert provider.prompts[0].startswith("# 记忆清单")
    assert "# 记忆清单" not in provider.system_prompts[0]
    # trace 记录的用户提示词只放本次清单，固定规则另存 system_prompt
    assert outcome.prompt == provider.prompts[0]
    assert outcome.system_prompt == provider.system_prompts[0]
