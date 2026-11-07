"""反思输入：一次请求 = 一份 (req, resp) 记录，反思取最近 N 份。

旧口径的「攒记录 + 去重 + 排序」已经取消，
所以这里不再断言消息身份过滤、摘要排除、去重键这些东西——
它们对应的实现已经不存在了。
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import astrbot_plugin_angel_memory.main as plugin_main
from astrbot_plugin_angel_memory.core import deepmind as deepmind_module
from astrbot_plugin_angel_memory.core.deepmind import DeepMind
from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)


class _StubPluginContext:
    """只提供反思输入装配所需要的那几个口子。"""

    def __init__(self, config=None):
        self.config = dict(config or {})

    def get_config(self, key=None, default=None, *, umo=None):
        if key is None:
            return dict(self.config)
        return self.config.get(key, default)

    def get_llm_provider_id(self):
        return self.config.get("provider_id", "")

    async def resolve_memory_scope_from_event(self, event):
        return "public"

    async def get_event_persona_name(self, event):
        return "测试人格"


class _StubEvent:
    def __init__(self, extras=None, angelheart_context=None, angelmemory_context=None):
        self._extras = dict(extras or {})
        if angelheart_context is not None:
            self.angelheart_context = angelheart_context
        if angelmemory_context is not None:
            self.angelmemory_context = angelmemory_context

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value


class _FakeQueue:
    """只记账，不真的执行反思任务。"""

    def __init__(self):
        self.calls = []

    async def submit(self, item):
        self.calls.append(item)
        return True


class _Req:
    """ProviderRequest 的最小替身：只带 prompt、contexts 与额外内容块。"""

    def __init__(self, prompt="", contexts=None, extra_user_content_parts=None):
        self.prompt = prompt
        self.contexts = list(contexts or [])
        self.extra_user_content_parts = list(extra_user_content_parts or [])


def _deepmind(threshold=3):
    deepmind = object.__new__(DeepMind)
    deepmind.plugin_context = _StubPluginContext(
        {"memory_behavior": {"reflection_turn_threshold": threshold}}
    )
    deepmind.config = SimpleNamespace()
    deepmind.logger = logging.getLogger("reflection-turn-test")
    deepmind._provider_id_fallback = ""
    deepmind._reflection_state_lock = asyncio.Lock()
    deepmind._reflection_states = {}
    deepmind.prompt_builder = SmallModelPromptBuilder()
    return deepmind


def _turn_event(req=None, resp="", when=None, sender_id="", sender_name="", **kwargs):
    return _StubEvent(
        extras={
            "angel_reflection_turn": {
                "req": req,
                "resp": resp,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "timestamp": time.time() if when is None else when,
            }
        },
        **kwargs,
    )


def _record(
    deepmind,
    session_id,
    req=None,
    resp="",
    when=None,
    sender_id="",
    sender_name="",
    **kwargs,
):
    event = _turn_event(
        req=req,
        resp=resp,
        when=when,
        sender_id=sender_id,
        sender_name=sender_name,
        **kwargs,
    )
    asyncio.run(deepmind._record_reflection_turn(event, None, session_id))
    return event


# ---------- 一、成对记录 ----------

def test_turn_is_recorded_as_pair():
    """一次请求写一份记录，req 与 resp 一起存。"""
    deepmind = _deepmind()

    _record(deepmind, "s1", req=_Req("用户原话"), resp="我的回复", when=1000.0)

    state = deepmind._reflection_states["s1"]
    assert state["pending_turns"] == 1
    assert len(state["turns"]) == 1
    turn = state["turns"][0]
    assert turn["session_id"] == "s1"
    assert turn["req"].prompt == "用户原话"
    assert turn["resp"] == "我的回复"
    assert turn["timestamp"] == 1000.0


def test_turn_with_only_one_side_is_still_recorded():
    """只有 req 或只有 resp 也照记：这一轮确实发生过请求或回复。"""
    deepmind = _deepmind()

    _record(deepmind, "s1", req=_Req("只有提问"), resp="")
    _record(deepmind, "s1", req=None, resp="只有回复")

    turns = deepmind._reflection_states["s1"]["turns"]
    assert len(turns) == 2
    assert turns[0]["req"].prompt == "只有提问"
    assert turns[1]["resp"] == "只有回复"


def test_turn_with_neither_side_is_not_recorded():
    """req 与 resp 都空时整份丢弃，不占窗口、不计轮次。"""
    deepmind = _deepmind()

    _record(deepmind, "s1", req=None, resp="")

    state = deepmind._reflection_states.get("s1")
    assert state is None or state["turns"] == []
    assert (state or {}).get("pending_turns", 0) == 0


# ---------- 二、取最近 N 份 ----------

def test_reflection_takes_latest_n_turns_in_order():
    """只取最后 N 份，且保持写入顺序（按时间升序）。"""
    queue = _FakeQueue()
    origin = deepmind_module.get_feedback_queue
    deepmind_module.get_feedback_queue = lambda: queue
    try:
        deepmind = _deepmind(threshold=2)
        for index in range(4):
            _record(
                deepmind,
                "s1",
                req=_Req(f"提问{index}"),
                resp=f"回答{index}",
                when=1000.0 + index,
            )

        assert asyncio.run(deepmind._trigger_reflection_if_needed("s1", "count")) is True
        assert len(queue.calls) == 1

        text = queue.calls[0]["payload"]["historical_chat_text_override"]
        assert "提问2" in text and "回答2" in text
        assert "提问3" in text and "回答3" in text
        assert "提问0" not in text and "提问1" not in text
        assert text.index("回答2") < text.index("提问3"), "轮次之间必须按时间升序"
    finally:
        deepmind_module.get_feedback_queue = origin


# ---------- 三、用户输入取法 ----------

def test_user_text_prefers_prompt():
    deepmind = _deepmind()
    req = _Req("prompt 里的原话", [{"role": "user", "content": "contexts 里的话"}])

    assert deepmind._request_user_text(req) == "prompt 里的原话"


def test_user_text_falls_back_to_extra_content_parts():
    """prompt 为空时看 extra_user_content_parts（纯媒体轮的文字描述在那里）。"""
    deepmind = _deepmind()
    req = _Req(
        "",
        extra_user_content_parts=[
            {"type": "text", "text": "图片里的文字描述"},
            {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
        ],
    )

    assert deepmind._request_user_text(req) == "图片里的文字描述"


def test_user_text_accepts_content_part_objects():
    """extra_user_content_parts 也可能是 ContentPart 对象而不是 dict。"""
    deepmind = _deepmind()
    req = _Req(
        "",
        extra_user_content_parts=[SimpleNamespace(type="text", text="对象式描述")],
    )

    assert deepmind._request_user_text(req) == "对象式描述"


def test_user_text_ignores_contexts_history():
    """纯媒体轮 prompt 为空时，不能把上一轮 user 的话当成本轮输入。"""
    deepmind = _deepmind()
    req = _Req(
        "",
        [
            {"role": "user", "content": "上一轮那个人说的话"},
            {"role": "assistant", "content": "上一轮的回复"},
        ],
    )

    assert deepmind._request_user_text(req) == "", "contexts 是历史，不能当本轮输入"


def test_user_text_is_empty_without_request():
    deepmind = _deepmind()

    assert deepmind._request_user_text(None) == ""


# ---------- 三之二、摊平后保留发送者 ----------

def test_different_senders_keep_their_own_headers():
    """同一会话里两个不同发送者，摊平后各自署名，不能都变成「群友」。"""
    deepmind = _deepmind(threshold=2)
    _record(
        deepmind, "s1", req=_Req("我先说"), resp="答一", when=1000.0,
        sender_id="1", sender_name="甲",
    )
    _record(
        deepmind, "s1", req=_Req("我后说"), resp="答二", when=1001.0,
        sender_id="2", sender_name="乙",
    )

    records = deepmind._turns_to_chat_records(deepmind._reflection_states["s1"]["turns"])
    users = [record for record in records if record["role"] == "user"]
    assert len(users) == 2
    assert (users[0]["sender_id"], users[0]["sender_name"]) == ("1", "甲")
    assert (users[1]["sender_id"], users[1]["sender_name"]) == ("2", "乙")

    text, _ = SmallModelPromptBuilder.format_chat_records(records)
    assert "甲（1）" in text
    assert "乙（2）" in text
    assert "群友" not in text


def test_turn_without_user_text_only_emits_assistant():
    """取不到用户文本时只产出回复那条，不伪造一条空用户消息。"""
    deepmind = _deepmind()
    records = deepmind._turns_to_chat_records(
        [
            {
                "session_id": "s1",
                "timestamp": 1000.0,
                "req": _Req(""),
                "resp": "只有回复",
                "sender_id": "1",
                "sender_name": "甲",
            }
        ]
    )

    assert [record["role"] for record in records] == ["assistant"]
    assert records[0]["content"] == "只有回复"


# ---------- 四、回复文本取法 ----------

def test_response_prefers_completion_text():
    run_context = SimpleNamespace(
        messages=[SimpleNamespace(role="assistant", content="messages 里的兜底")]
    )
    response = SimpleNamespace(completion_text="completion_text 里的回复")

    assert (
        plugin_main.AngelMemoryPlugin._resolve_agent_response_text(run_context, response)
        == "completion_text 里的回复"
    )


def test_response_falls_back_to_run_context_messages():
    """completion_text 为空时，从 run_context.messages 里最后一条 assistant 兜底。"""
    run_context = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="提问"),
            SimpleNamespace(role="assistant", content="先前的回复"),
            SimpleNamespace(role="assistant", content="最后一条回复"),
        ]
    )
    response = SimpleNamespace(completion_text="")

    assert (
        plugin_main.AngelMemoryPlugin._resolve_agent_response_text(run_context, response)
        == "最后一条回复"
    )


def test_response_is_empty_when_both_sides_missing():
    run_context = SimpleNamespace(messages=[SimpleNamespace(role="user", content="提问")])

    assert plugin_main.AngelMemoryPlugin._resolve_agent_response_text(run_context, None) == ""


# ---------- 五、请求快照复制 ----------

def test_provider_request_snapshot_copies_contexts_list():
    """快照必须复制顶层 list：contexts 是共享对象，后续轮次还会往里 append。"""
    request = _Req("提问", [{"role": "user", "content": "旧消息"}])
    event = _StubEvent()

    plugin_main.AngelMemoryPlugin._remember_provider_request(event, request)

    snapshot = event.get_extra("angel_provider_request")
    assert snapshot is not request
    assert snapshot.contexts == request.contexts
    assert snapshot.contexts is not request.contexts

    # 原对象后续增长不影响快照
    request.contexts.append({"role": "user", "content": "后加的消息"})
    assert len(snapshot.contexts) == 1


# ---------- 六、窗口上限 ----------

def test_window_limit_has_floor_and_scales_with_threshold():
    assert _deepmind(threshold=3)._reflection_window_limit() == 20
    assert _deepmind(threshold=15)._reflection_window_limit() == 30


def test_window_truncates_to_limit():
    """超出上限时丢最旧的，保留最近的。"""
    deepmind = _deepmind(threshold=3)
    limit = deepmind._reflection_window_limit()

    for index in range(limit + 5):
        _record(
            deepmind,
            "s1",
            req=_Req(f"提问{index}"),
            resp=f"回答{index}",
            when=1000.0 + index,
        )

    turns = deepmind._reflection_states["s1"]["turns"]
    assert len(turns) == limit
    assert turns[0]["req"].prompt == "提问5"
    assert turns[-1]["req"].prompt == f"提问{limit + 4}"


# ---------- 七、天使之心快照不再进反思 ----------

def test_reflection_text_ignores_angelheart_snapshot():
    queue = _FakeQueue()
    origin = deepmind_module.get_feedback_queue
    deepmind_module.get_feedback_queue = lambda: queue
    try:
        deepmind = _deepmind(threshold=1)
        _record(
            deepmind,
            "s1",
            req=_Req("只有这一句"),
            resp="只有这一答",
            when=1000.0,
            angelheart_context=json.dumps(
                {"chat_records": [{"role": "user", "content": "天使之心快照里的话"}]}
            ),
        )

        assert asyncio.run(deepmind._trigger_reflection_if_needed("s1", "count")) is True

        text = queue.calls[0]["payload"]["historical_chat_text_override"]
        assert "只有这一句" in text
        assert "只有这一答" in text
        assert "天使之心快照里的话" not in text
    finally:
        deepmind_module.get_feedback_queue = origin


# ---------- 八、提示词侧兜底措辞 ----------

def test_missing_reply_marks_absence_in_prompt():
    """两边都缺时，提示词里明确写出「本轮没有回复记录」。"""
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="用户：你好",
        main_llm_response="",
        raw_memories=[],
    )

    assert "## 你的最新回复" in user_prompt
    assert "本轮没有回复记录" in user_prompt
    assert "不代表你没有说话" in user_prompt


def test_present_reply_does_not_claim_absence():
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="用户：你好",
        main_llm_response="我很好",
        raw_memories=[],
    )

    assert "## 你的最新回复" in user_prompt
    assert "我很好" in user_prompt
    assert "没有回复记录" not in user_prompt


# ---------- 九、每份记录带上那一轮检索出的记忆 ----------

TZ = timezone(timedelta(hours=8))


def _memories_json(*memories):
    return json.dumps({"raw_memories": list(memories)})


def test_turn_records_that_round_memories():
    """每份记录单独存自己那轮注入的记忆批次。"""
    deepmind = _deepmind()

    _record(
        deepmind,
        "s1",
        req=_Req("问1"),
        resp="答1",
        when=1000.0,
        angelmemory_context=_memories_json(
            {"id": "m1", "judgment": "甲"}, {"id": "m2", "judgment": "乙"}
        ),
    )

    turn = deepmind._reflection_states["s1"]["turns"][0]
    assert [memory["id"] for memory in turn["memories"]] == ["m1", "m2"]


def test_memories_merged_by_id_across_turns():
    """多轮合并按 id 去重，保持首次出现顺序。"""
    deepmind = _deepmind(threshold=3)

    _record(
        deepmind,
        "s1",
        req=_Req("问1"),
        resp="答1",
        when=1000.0,
        angelmemory_context=_memories_json(
            {"id": "m1", "judgment": "甲"}, {"id": "m2", "judgment": "乙"}
        ),
    )
    _record(
        deepmind,
        "s1",
        req=_Req("问2"),
        resp="答2",
        when=1001.0,
        angelmemory_context=_memories_json(
            {"id": "m2", "judgment": "乙"}, {"id": "m3", "judgment": "丙"}
        ),
    )

    merged = deepmind._merge_turn_memories(deepmind._reflection_states["s1"]["turns"])
    assert [memory["id"] for memory in merged] == ["m1", "m2", "m3"]


def test_reflection_payload_carries_merged_memories_not_last_turn_only():
    """反思拿到的是多轮合并结果，不是最后一轮那一批。"""
    queue = _FakeQueue()
    origin = deepmind_module.get_feedback_queue
    deepmind_module.get_feedback_queue = lambda: queue
    try:
        deepmind = _deepmind(threshold=2)
        _record(
            deepmind,
            "s1",
            req=_Req("问1"),
            resp="答1",
            when=1000.0,
            angelmemory_context=_memories_json({"id": "m1", "judgment": "甲的旧事"}),
        )
        _record(
            deepmind,
            "s1",
            req=_Req("问2"),
            resp="答2",
            when=1001.0,
            angelmemory_context=_memories_json({"id": "m2", "judgment": "乙的新事"}),
        )

        assert asyncio.run(deepmind._trigger_reflection_if_needed("s1", "count")) is True

        payload = queue.calls[0]["payload"]
        ids = [memory["id"] for memory in payload["raw_memories_override"]]
        assert ids == ["m1", "m2"], "前几轮检索出来的记忆不能被丢掉"
    finally:
        deepmind_module.get_feedback_queue = origin


# ---------- 十、记忆列表带创建时间 ----------

def test_memory_created_at_formats_to_minute():
    stamp = datetime(2026, 11, 7, 14, 7, 33, tzinfo=TZ).timestamp()

    assert (
        SmallModelPromptBuilder._format_memory_created_at(stamp) == "2026-11-07 14:07"
    ), "精确到分钟，秒要截掉"


def test_memory_created_at_keeps_year_across_new_year():
    stamp = datetime(2025, 12, 31, 23, 59, 59, tzinfo=TZ).timestamp()

    assert SmallModelPromptBuilder._format_memory_created_at(stamp) == "2025-12-31 23:59"


def test_memory_created_at_missing_returns_empty():
    """取不到有效时间时返回空串，不能退化成 1970-01-01。"""
    for value in (None, 0, "", "不是时间", -5):
        assert SmallModelPromptBuilder._format_memory_created_at(value) == ""


def test_memory_list_line_carries_created_at():
    stamp = datetime(2026, 11, 7, 14, 7, tzinfo=TZ).timestamp()
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="历史",
        main_llm_response="回复",
        raw_memories=[
            SimpleNamespace(id="1265468", judgment="喜欢喝豆浆", created_at=stamp)
        ],
    )

    assert "id:1265468 2026-11-07 14:07 - 喜欢喝豆浆" in user_prompt


def test_memory_list_line_omits_stamp_when_created_at_missing():
    _, user_prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
        historical_query="历史",
        main_llm_response="回复",
        raw_memories=[SimpleNamespace(id="1265468", judgment="没有时间", created_at=0)],
    )

    assert "id:1265468 - 没有时间" in user_prompt
    assert "1970-01-01" not in user_prompt
