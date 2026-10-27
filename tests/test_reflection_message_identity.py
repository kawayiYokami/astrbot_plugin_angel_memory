from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace


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

from astrbot_plugin_angel_memory.core.deepmind import DeepMind
from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)


class _ReflectionHarness:
    logger = logging.getLogger("reflection-identity-test")
    prompt_builder = SmallModelPromptBuilder

    _is_missing_user_identity = staticmethod(DeepMind._is_missing_user_identity)
    _is_non_reflection_message = DeepMind._is_non_reflection_message
    NON_REFLECTION_KINDS = DeepMind.NON_REFLECTION_KINDS
    _get_event_sender_identity = DeepMind._get_event_sender_identity
    _log_missing_user_identity = DeepMind._log_missing_user_identity
    _dedupe_and_sort_chat_records = DeepMind._dedupe_and_sort_chat_records
    _build_reflection_records_for_turn = DeepMind._build_reflection_records_for_turn

    def _extract_message_text(self, event) -> str:
        return event.get_message_outline()


class _NativeEvent:
    unified_msg_origin = "aiocqhttp:group:100"

    def __init__(self, sender_id: str, sender_name: str, text: str):
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._text = text

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_sender_name(self) -> str:
        return self._sender_name

    def get_message_outline(self) -> str:
        return self._text


def test_native_reflection_records_use_event_getters():
    harness = _ReflectionHarness()
    event = _NativeEvent("123456", "小明", "我在写 AstrBot 插件。")

    records = harness._build_reflection_records_for_turn(
        event,
        SimpleNamespace(completion_text="收到。"),
    )

    user_record = next(record for record in records if record["role"] == "user")
    assert user_record["sender_id"] == "123456"
    assert user_record["sender_name"] == "小明"


def test_native_reflection_skips_user_message_without_identity(caplog):
    harness = _ReflectionHarness()
    event = _NativeEvent("", "", "我在写 AstrBot 插件。")

    with caplog.at_level(logging.ERROR, logger="reflection-identity-test"):
        records = harness._build_reflection_records_for_turn(
            event,
            SimpleNamespace(completion_text="收到。"),
        )

    assert all(record["role"] != "user" for record in records)
    assert "检测到异常消息，该消息不存在用户和ID" in caplog.text


def test_angelheart_reflection_skips_user_message_without_identity(caplog):
    harness = _ReflectionHarness()
    event = SimpleNamespace(
        angelheart_context=json.dumps(
            {
                "chat_records": [
                    {
                        "role": "user",
                        "content": "异常消息",
                        "sender_id": "user",
                        "sender_name": "用户",
                        "timestamp": 1.0,
                        "source_message_id": "m1",
                    },
                    {
                        "role": "user",
                        "content": "有效消息",
                        "sender_id": "123456",
                        "sender_name": "小明",
                        "timestamp": 2.0,
                        "source_message_id": "m2",
                    },
                ]
            },
            ensure_ascii=False,
        )
    )

    with caplog.at_level(logging.ERROR, logger="reflection-identity-test"):
        records = harness._build_reflection_records_for_turn(
            event,
            SimpleNamespace(completion_text="收到。"),
        )

    assert [record["content"] for record in records] == ["有效消息"]
    assert "检测到异常消息，该消息不存在用户和ID" in caplog.text


def _make_angelheart_event(chat_records):
    return SimpleNamespace(
        angelheart_context=json.dumps(
            {"chat_records": chat_records, "secretary_decision": {}},
            ensure_ascii=False,
        )
    )


def test_angelheart_uses_whole_snapshot_without_is_processed():
    """天使之心不再提供 is_processed，整份快照都应进入反思。"""
    harness = _ReflectionHarness()
    event = _make_angelheart_event(
        [
            {"role": "user", "content": "甲", "sender_id": "1",
             "sender_name": "甲", "timestamp": 1.0, "source_message_id": "m1"},
            {"role": "assistant", "content": "乙", "sender_id": "assistant",
             "sender_name": "助理", "timestamp": 2.0, "source_message_id": "m2"},
            {"role": "user", "content": "丙", "sender_id": "2",
             "sender_name": "丙", "timestamp": 3.0, "source_message_id": "m3"},
        ]
    )

    records = harness._build_reflection_records_for_turn(
        event, SimpleNamespace(completion_text="收到。")
    )

    assert [record["content"] for record in records] == ["甲", "乙", "丙"]


def test_summary_and_system_messages_are_excluded():
    """摘要与系统消息是压缩产物，不能作为反思素材。"""
    harness = _ReflectionHarness()
    event = _make_angelheart_event(
        [
            {"role": "system", "content": "系统提示", "sender_id": "system",
             "sender_name": "system", "timestamp": 0.5, "source_message_id": "s1"},
            {"role": "user", "content": "[当前摘要]\n旧事", "sender_id": "system",
             "sender_name": "context_summary", "kind": "context_summary",
             "timestamp": 1.0, "source_message_id": "s2"},
            {"role": "user", "content": "真实发言", "sender_id": "1",
             "sender_name": "甲", "timestamp": 2.0, "source_message_id": "m1"},
        ]
    )

    records = harness._build_reflection_records_for_turn(
        event, SimpleNamespace(completion_text="收到。")
    )

    assert [record["content"] for record in records] == ["真实发言"]


def test_dedupe_prefers_source_message_id():
    """同一 id 在多份快照里内容或时间被改写，仍应只保留一条。"""
    harness = _ReflectionHarness()
    first = harness._dedupe_and_sort_chat_records(
        [{"role": "user", "content": "原文", "sender_id": "1",
          "sender_name": "甲", "timestamp": 1.0, "source_message_id": "m1"}]
    )
    merged = harness._dedupe_and_sort_chat_records(
        first
        + [{"role": "user", "content": "被压缩改写", "sender_id": "1",
            "sender_name": "甲", "timestamp": 9.0, "source_message_id": "m1"}]
    )

    assert len(merged) == 1
    assert merged[0]["content"] == "原文"


def test_dedupe_falls_back_to_fingerprint_without_id():
    """缺少 source_message_id 时，回退到角色+发送者+时间戳+内容。"""
    harness = _ReflectionHarness()
    records = harness._dedupe_and_sort_chat_records(
        [
            {"role": "user", "content": "同一句", "sender_id": "1",
             "sender_name": "甲", "timestamp": 1.0},
            {"role": "user", "content": "同一句", "sender_id": "1",
             "sender_name": "甲", "timestamp": 1.0},
        ]
    )

    assert len(records) == 1


class _AngelHeartNativeEvent:
    """同时带天使之心上下文与原生事件取数能力的消息事件。"""

    unified_msg_origin = "aiocqhttp:group:100"

    def __init__(self, chat_records, text: str):
        self.angelheart_context = json.dumps(
            {"chat_records": chat_records, "secretary_decision": {}},
            ensure_ascii=False,
        )
        self._text = text

    def get_sender_id(self) -> str:
        return "123456"

    def get_sender_name(self) -> str:
        return "小明"

    def get_message_outline(self) -> str:
        return self._text


def test_angelheart_snapshot_all_filtered_falls_back_to_native(caplog):
    """快照里全是摘要与系统消息时，必须降级到原生分支，而不是静默跳过。"""
    harness = _ReflectionHarness()
    event = _AngelHeartNativeEvent(
        [
            {"role": "system", "content": "系统提示", "sender_id": "system",
             "sender_name": "system", "timestamp": 0.5, "source_message_id": "s1"},
            {"role": "user", "content": "[当前摘要]\n旧事", "sender_id": "system",
             "sender_name": "context_summary", "kind": "context_summary",
             "timestamp": 1.0, "source_message_id": "s2"},
        ],
        text="本轮真实发言",
    )

    with caplog.at_level(logging.WARNING, logger="reflection-identity-test"):
        records = harness._build_reflection_records_for_turn(
            event, SimpleNamespace(completion_text="本轮回复")
        )

    assert [record["content"] for record in records] == ["本轮真实发言", "本轮回复"]
    assert "过滤后为空，降级到原生分支" in caplog.text


def test_angelheart_snapshot_partially_usable_keeps_snapshot():
    """只要快照里还有可用记录，就用快照，不降级。"""
    harness = _ReflectionHarness()
    event = _AngelHeartNativeEvent(
        [
            {"role": "system", "content": "系统提示", "sender_id": "system",
             "sender_name": "system", "timestamp": 0.5, "source_message_id": "s1"},
            {"role": "user", "content": "历史发言", "sender_id": "1",
             "sender_name": "甲", "timestamp": 2.0, "source_message_id": "m1"},
        ],
        text="本轮真实发言",
    )

    records = harness._build_reflection_records_for_turn(
        event, SimpleNamespace(completion_text="本轮回复")
    )

    assert [record["content"] for record in records] == ["历史发言"]
