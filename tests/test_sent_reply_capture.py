"""发送现场捕获回复文本：优先取发送现场，取不到才回落到 LLM 响应钩子。

框架在 OnAfterMessageSentEvent 钩子之后才 clear_result，
所以钩子执行时 event 的 result 仍然可用。
"""

import json
import logging

import astrbot_plugin_angel_memory.main as plugin_main
from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)


class FakeResult:
    """MessageEventResult 的替身：只提供 get_plain_text()。"""

    def __init__(self, text: str):
        self._text = text

    def get_plain_text(self) -> str:
        return self._text


class FakeEvent:
    def __init__(self, context_json, result=None):
        self.angelmemory_context = context_json
        self._result = result

    def get_result(self):
        return self._result


def _plugin() -> plugin_main.AngelMemoryPlugin:
    plugin = object.__new__(plugin_main.AngelMemoryPlugin)
    plugin.logger = logging.getLogger("test-sent-reply")
    return plugin


# ---------- 一、发送现场优先 ----------

def test_sent_reply_overrides_stored_hook_text():
    """发送现场取到的文本覆盖 on_llm_response 存下的旧值。"""
    plugin = _plugin()
    event = FakeEvent(
        json.dumps(
            {"llm_response": {"completion_text": "钩子存的旧值", "timestamp": 1.0}}
        ),
        result=FakeResult("发送现场的真实回复"),
    )

    plugin._capture_sent_reply(event)

    data = json.loads(event.angelmemory_context)
    assert data["llm_response"]["completion_text"] == "发送现场的真实回复"
    assert data["llm_response"]["source"] == "after_message_sent"
    assert data["llm_response"]["timestamp"] == 1.0, "应沿用钩子记下的时间戳"


def test_sent_reply_fills_missing_context_entry():
    """上下文里根本没有 llm_response 时，发送现场取到的文本要新建条目。"""
    plugin = _plugin()
    event = FakeEvent(json.dumps({"query": "用户原话"}), result=FakeResult("现场回复"))

    plugin._capture_sent_reply(event)

    data = json.loads(event.angelmemory_context)
    assert data["llm_response"]["completion_text"] == "现场回复"
    assert data["query"] == "用户原话", "其它字段不能被冲掉"


# ---------- 二、取不到时回落 ----------

def test_empty_result_keeps_hook_text():
    """发送现场为空时保留钩子存下的那份，且不打 source 标记。"""
    for result in (None, FakeResult("   ")):
        plugin = _plugin()
        event = FakeEvent(
            json.dumps(
                {"llm_response": {"completion_text": "钩子存的旧值", "timestamp": 1.0}}
            ),
            result=result,
        )

        plugin._capture_sent_reply(event)

        data = json.loads(event.angelmemory_context)
        assert data["llm_response"]["completion_text"] == "钩子存的旧值"
        assert "source" not in data["llm_response"]


def test_both_sources_empty_leaves_context_untouched():
    """两边都空时不伪造文本，上下文保持原样。"""
    plugin = _plugin()
    event = FakeEvent(
        json.dumps({"llm_response": {"completion_text": "", "timestamp": 1.0}}),
        result=None,
    )

    plugin._capture_sent_reply(event)

    data = json.loads(event.angelmemory_context)
    assert data["llm_response"]["completion_text"] == ""


def test_unparsable_context_is_left_alone():
    """上下文不是合法 JSON 时不覆盖，交给下游按原有路径处理。"""
    plugin = _plugin()
    event = FakeEvent("not-json", result=FakeResult("现场回复"))

    plugin._capture_sent_reply(event)

    assert event.angelmemory_context == "not-json"


# ---------- 三、两边都空时提示词要写清缺失 ----------

def test_missing_reply_marks_absence_in_prompt():
    """两边都空时，提示词里明确写出「本轮没有回复记录」。"""
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="用户：你好",
        main_llm_response="",
        raw_memories=[],
    )

    assert "## 你的最新回复" in user_prompt, "该段不能整段消失"
    assert "本轮没有回复记录" in user_prompt
    assert "不代表你没有说话" in user_prompt


def test_present_reply_does_not_claim_absence():
    """有回复时不能出现缺失说明。"""
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="用户：你好",
        main_llm_response="我很好",
        raw_memories=[],
    )

    assert "## 你的最新回复" in user_prompt
    assert "我很好" in user_prompt
    assert "没有回复记录" not in user_prompt
