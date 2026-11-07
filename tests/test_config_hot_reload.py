"""配置热生效：提供商与反思调度参数按次现读，保存后无需重载插件。"""

import asyncio
import logging
import time
from types import SimpleNamespace

from astrbot_plugin_angel_memory.core import deepmind as deepmind_module
from astrbot_plugin_angel_memory.core.deepmind import DeepMind


class _StubPluginContext:
    """只提供 DeepMind 现读配置所需要的口子。"""

    def __init__(self, config=None):
        self.config = dict(config or {})

    def get_config(self, key=None, default=None, *, umo=None):
        if key is None:
            return dict(self.config)
        return self.config.get(key, default)

    def get_llm_provider_id(self):
        return self.config.get("provider_id", "")


class _FakePromptBuilder:
    def format_chat_records(self, records):
        return "", []


class _FakeQueue:
    """只记账，不真的执行反思任务。"""

    def __init__(self):
        self.calls = []

    async def submit(self, item):
        self.calls.append(item)
        return True


def _bare_deepmind(config=None, provider_id=""):
    """跳过重量级初始化，只装配现读相关的那几个属性。"""
    deepmind = object.__new__(DeepMind)
    deepmind.plugin_context = _StubPluginContext(config)
    deepmind.config = SimpleNamespace()
    deepmind.logger = logging.getLogger("config-hot-reload-test")
    deepmind._provider_id_fallback = provider_id
    return deepmind


def test_provider_id_reads_live_config():
    deepmind = _bare_deepmind({"provider_id": "provider-a"}, provider_id="provider-a")

    assert deepmind.provider_id == "provider-a"

    # 模拟 WebUI 保存新配置：PluginContext.config 原地更新
    deepmind.plugin_context.config["provider_id"] = "provider-b"

    assert deepmind.provider_id == "provider-b"


def test_provider_id_cleared_means_disabled():
    """清空提供商就是要关掉记忆整理，不能回落到启动时的旧值。"""
    deepmind = _bare_deepmind({"provider_id": "provider-a"}, provider_id="provider-a")

    deepmind.plugin_context.config["provider_id"] = ""

    assert deepmind.provider_id == ""


def test_provider_id_falls_back_without_plugin_context():
    deepmind = _bare_deepmind(provider_id="provider-a")
    deepmind.plugin_context = None

    assert deepmind.provider_id == "provider-a"


def test_reflection_turn_threshold_reads_live_config():
    deepmind = _bare_deepmind({"memory_behavior": {"reflection_turn_threshold": 6}})

    assert deepmind._reflection_turn_threshold == 6

    deepmind.plugin_context.config["memory_behavior"]["reflection_turn_threshold"] = 3

    assert deepmind._reflection_turn_threshold == 3


def test_reflection_params_fall_back_when_missing():
    deepmind = _bare_deepmind()

    assert deepmind._reflection_turn_threshold == 6
    assert deepmind._reflection_idle_seconds == 600
    assert deepmind._reflection_tick_seconds == 600


def test_reflection_params_reject_illegal_values():
    deepmind = _bare_deepmind(
        {
            "memory_behavior": {
                "reflection_turn_threshold": 0,
                "reflection_idle_seconds": "不是数字",
                "reflection_tick_seconds": 3,
            }
        }
    )

    assert deepmind._reflection_turn_threshold == 6
    assert deepmind._reflection_idle_seconds == 600
    # tick 有下限，3 会被抬到 10
    assert deepmind._reflection_tick_seconds == 10


def _prime_reflection_state(deepmind, session_id, pending_turns):
    deepmind.prompt_builder = _FakePromptBuilder()
    deepmind._reflection_state_lock = asyncio.Lock()
    deepmind._reflection_states = {
        session_id: {
            "pending_turns": pending_turns,
            "turns": [],
            "last_activity_at": time.time(),
            "latest_input": {"message": "在吗"},
            "processing": False,
        }
    }


def test_reflection_trigger_uses_live_threshold():
    """阈值改小后，同样的积压轮次应当立刻达到触发线。"""
    queue = _FakeQueue()
    origin = deepmind_module.get_feedback_queue
    deepmind_module.get_feedback_queue = lambda: queue
    try:
        deepmind = _bare_deepmind({"memory_behavior": {"reflection_turn_threshold": 6}})
        _prime_reflection_state(deepmind, "s1", pending_turns=2)

        # 旧阈值 6 时，两轮不够
        assert asyncio.run(deepmind._trigger_reflection_if_needed("s1", "count")) is False

        deepmind.plugin_context.config["memory_behavior"]["reflection_turn_threshold"] = 2
        _prime_reflection_state(deepmind, "s2", pending_turns=2)

        # 保存后不重载，同样两轮立刻触发
        assert asyncio.run(deepmind._trigger_reflection_if_needed("s2", "count")) is True
        assert len(queue.calls) == 1
    finally:
        deepmind_module.get_feedback_queue = origin


def test_min_message_length_reads_live_config():
    deepmind = _bare_deepmind({"memory_behavior": {"min_message_length": 5}})

    assert deepmind.min_message_length == 5

    deepmind.plugin_context.config["memory_behavior"]["min_message_length"] = 12

    assert deepmind.min_message_length == 12


def test_min_message_length_tolerates_missing_and_illegal():
    deepmind = _bare_deepmind({"memory_behavior": {"min_message_length": None}})

    assert deepmind.min_message_length == 5

    deepmind.plugin_context.config["memory_behavior"]["min_message_length"] = -3

    assert deepmind.min_message_length == 0


def test_note_top_k_reads_live_config():
    deepmind = _bare_deepmind({"note_assistant": {"top_k": 3}})

    assert deepmind.note_inject_top_k == 3
    assert deepmind.note_candidate_top_k == 21

    deepmind.plugin_context.config["note_assistant"]["top_k"] = 5

    assert deepmind.note_inject_top_k == 5
    assert deepmind.note_candidate_top_k == 35


def test_note_recall_switch_reads_live_config():
    deepmind = _bare_deepmind({"note_assistant": {"enable_recall": True}})

    assert deepmind.note_recall_enabled is True

    deepmind.plugin_context.config["note_assistant"]["enable_recall"] = False

    assert deepmind.note_recall_enabled is False


def test_note_defaults_when_group_missing():
    deepmind = _bare_deepmind()

    assert deepmind.note_inject_top_k == 3
    assert deepmind.note_candidate_top_k == 21
    assert deepmind.note_recall_enabled is True
