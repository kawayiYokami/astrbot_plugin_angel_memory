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
from astrbot_plugin_angel_memory.llm_memory.utils.user_profile import (
    extract_user_id_from_tags,
    is_user_id_tag,
    is_user_profile_tags,
)


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


def test_is_user_id_tag_distinguishes_ids_from_descriptive_tags():
    # 真实平台 ID：纯数字、数字+字母、含符号的平台标识符
    assert is_user_id_tag("1023456789")
    assert is_user_id_tag("o9cq809xxxx@im.wechat")
    assert is_user_id_tag("wxid_abc123")
    # 伪 ID：连字符/下划线词形用户名、中文日期、英文短语、纯中文、过短
    assert not is_user_id_tag("Test-Bot")
    assert not is_user_id_tag("weixin_oc_adapter")
    assert not is_user_id_tag("1998年10月19日")
    assert not is_user_id_tag("docker restart")
    assert not is_user_id_tag("小明")
    assert not is_user_id_tag("abcde")
    assert not is_user_id_tag("12345")


def test_is_user_id_tag_ledger_hit_wins_over_shape_guess():
    # 账本命中：即使是形态上会被排除的词形用户名，只要账本里有就判定为 ID
    assert is_user_id_tag("Test-Bot", known_user_ids=["Test-Bot", "1023456789"])
    assert is_user_id_tag("wxid_abcdef", known_user_ids=["wxid_abcdef"])
    assert is_user_id_tag("10000", known_user_ids=["10000"])
    # 账本未命中：回退形态判定
    assert not is_user_id_tag("Test-Bot", known_user_ids=["1023456789"])
    assert is_user_id_tag("1023456789", known_user_ids=[])


def test_extract_user_id_with_ledger_prefers_known_id():
    # issue #73 复现场景：排除法已不把 Test-Bot 当 ID，无账本也能提取
    assert (
        extract_user_id_from_tags(["Test-Bot", "1023456789", "事实属性"])
        == "1023456789"
    )
    # 账本命中优先：即使形态上像伪 ID 的 tag，账本有即认定
    assert (
        extract_user_id_from_tags(
            ["Test-Bot", "1023456789", "事实属性"],
            known_user_ids=["1023456789"],
        )
        == "1023456789"
    )
    # 账本里不存在的形态疑似 ID 不影响提取真实 ID
    assert (
        extract_user_id_from_tags(
            ["weixin_oc_adapter", "1023456789", "事实属性"],
            known_user_ids=["1023456789"],
        )
        == "1023456789"
    )
    # 多个真实 ID（多用户画像）无账本时仍返回空
    assert extract_user_id_from_tags(["1023456789", "654321", "事实属性"]) == ""


def test_is_user_profile_tags_with_ledger():
    # 排除法修复后：Test-Bot 不再被误判，画像判定通过
    assert is_user_profile_tags(["Test-Bot", "1023456789", "事实属性"])
    # 账本命中：同样通过
    assert is_user_profile_tags(
        ["Test-Bot", "1023456789", "事实属性"],
        known_user_ids=["1023456789"],
    )
    # 缺属性标签仍判定失败
    assert not is_user_profile_tags(
        ["Test-Bot", "1023456789"],
        known_user_ids=["1023456789"],
    )
    # 多个真实 ID 且无账本：判定失败（无法区分归属）
    assert not is_user_profile_tags(["1023456789", "654321", "事实属性"])


def test_extract_current_user_ids_deduplicates_latest_batch():
    records = [
        {"role": "user", "sender_id": "123456", "sender_name": "小明", "content": "a"},
        {"role": "assistant", "sender_id": "assistant", "content": "b"},
        {"role": "user", "sender_id": "123456", "sender_name": "明仔", "content": "c"},
        {"role": "user", "sender_id": "654321", "sender_name": "小红", "content": "d"},
    ]

    assert UserProfileService.extract_current_user_ids(records) == [
        "123456",
        "654321",
    ]
    user_ids, user_names = UserProfileService.extract_current_users(records)
    assert user_ids == ["123456", "654321"]
    assert user_names == {"123456": "明仔", "654321": "小红"}


def test_format_profiles_includes_reasoning_and_filters_regular_duplicates():
    service = UserProfileService()
    service._session_user_ids["s1"] = ["123456"]
    service._session_user_names["s1"] = {"123456": "当前昵称"}
    profile = _memory(
        "p1",
        "小明（123456）希望被称呼为阿明。",
        ["历史昵称", "123456", "用户别名"],
    )
    service._session_profiles["s1"] = [profile]

    formatted = service.format_session_profiles("s1")
    assert "[用户画像]" in formatted
    assert "[当前昵称（123456）]" in formatted
    assert "[历史昵称（123456）]" not in formatted
    assert "[用户别名]" not in formatted
    assert "——因为用户在对话中明确说明。" in formatted

    regular = [
        _memory("p1", "小明（123456）希望被称呼为阿明。", ["小明", "123456", "用户别名"]),
        _memory("m2", "普通记忆。", ["普通"]),
    ]
    filtered = service.filter_regular_memories("s1", regular)
    assert [memory.id for memory in filtered] == ["m2"]
