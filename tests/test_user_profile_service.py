from __future__ import annotations

import logging
import sys
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

    provider = types.ModuleType("astrbot.api.provider")

    class ProviderRequest:
        pass

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

from astrbot_plugin_angel_memory.core.services.user_profile_service import UserProfileService
from astrbot_plugin_angel_memory.llm_memory.models.data_models import BaseMemory, MemoryType
from astrbot_plugin_angel_memory.llm_memory.utils.user_profile import is_user_profile_tags


def _memory(memory_id: str, judgment: str, tags: list[str]) -> BaseMemory:
    return BaseMemory(
        id=memory_id,
        memory_type=MemoryType.KNOWLEDGE,
        judgment=judgment,
        reasoning="用户在对话中明确说明。",
        tags=tags,
        is_active=True,
        created_at=1_700_000_000.0,
    )


def test_user_profile_tags_require_user_id_and_attribute():
    assert is_user_profile_tags(["小明", "123456", "用户别名"])
    assert not is_user_profile_tags(["小明", "用户别名"])
    assert not is_user_profile_tags(["小明", "123456", "项目"])
    assert not is_user_profile_tags(["小明", "123456", "654321", "关系图谱"])


def test_extract_current_user_ids_deduplicates_latest_batch():
    records = [
        {"role": "user", "sender_id": "123456", "content": "a"},
        {"role": "assistant", "sender_id": "assistant", "content": "b"},
        {"role": "user", "sender_id": "123456", "content": "c"},
        {"role": "user", "sender_id": "654321", "content": "d"},
    ]

    assert UserProfileService.extract_current_user_ids(records) == [
        "123456",
        "654321",
    ]


def test_format_profiles_includes_reasoning_and_filters_regular_duplicates():
    service = UserProfileService()
    service._session_user_ids["s1"] = ["123456"]
    profile = _memory(
        "p1",
        "小明（123456）希望被称呼为阿明。",
        ["小明", "123456", "用户别名"],
    )
    service._session_profiles["s1"] = [profile]

    formatted = service.format_session_profiles("s1")
    assert "[用户画像]" in formatted
    assert "[小明（123456）]" in formatted
    assert "[用户别名]" not in formatted
    assert "——因为用户在对话中明确说明。" in formatted

    regular = [
        _memory("p1", "小明（123456）希望被称呼为阿明。", ["小明", "123456", "用户别名"]),
        _memory("m2", "普通记忆。", ["普通"]),
    ]
    filtered = service.filter_regular_memories("s1", regular)
    assert [memory.id for memory in filtered] == ["m2"]
