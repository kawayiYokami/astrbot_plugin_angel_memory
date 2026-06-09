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
                        "is_processed": True,
                    },
                    {
                        "role": "user",
                        "content": "有效消息",
                        "sender_id": "123456",
                        "sender_name": "小明",
                        "timestamp": 2.0,
                        "is_processed": False,
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
