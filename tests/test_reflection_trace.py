"""ReflectionTraceBuffer 环形缓冲测试。"""

from __future__ import annotations

import logging
import sys
import threading
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

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event


_install_astrbot_stubs()

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

from astrbot_plugin_angel_memory.core.reflection_trace import ReflectionTraceBuffer


def test_rolling_window_keeps_latest():
    buf = ReflectionTraceBuffer(maxlen=10)
    for i in range(30):
        buf.record_start(f"session-{i}", "provider-a", f"prompt-{i}")
    records = buf.snapshot()
    assert len(records) == 10
    # 保留的是最新的 10 条（id 从 20 到 29）
    assert [r["id"] for r in records] == list(range(20, 30))
    # 最新一条内容完整
    assert records[-1]["prompt"] == "prompt-29"
    assert records[-1]["status"] == "pending"


def test_success_complete_fields():
    buf = ReflectionTraceBuffer(maxlen=10)
    trace_id = buf.record_start("s1", "provider-a", "prompt", "固定规则")
    buf.complete(
        trace_id,
        status="success",
        response="raw llm text",
        result={"feedback_data": {"memory_actions": []}},
    )
    record = buf.snapshot()[0]
    assert record["status"] == "success"
    assert record["response"] == "raw llm text"
    assert record["result"]["feedback_data"]["memory_actions"] == []
    assert record["error"] == ""
    # 固定规则与用户提示词分字段存放，前端只展示后者
    assert record["prompt"] == "prompt"
    assert record["system_prompt"] == "固定规则"


def test_failed_complete_fields():
    buf = ReflectionTraceBuffer(maxlen=10)
    trace_id = buf.record_start("s1", "provider-a", "prompt")
    buf.complete(trace_id, status="failed", error="LLM调用失败: timeout")
    record = buf.snapshot()[0]
    assert record["status"] == "failed"
    assert record["error"] == "LLM调用失败: timeout"
    assert record["result"] is None


def test_complete_unknown_id_is_noop():
    buf = ReflectionTraceBuffer(maxlen=10)
    buf.record_start("s1", "provider-a", "prompt")
    buf.complete(999, status="success", response="x")
    records = buf.snapshot()
    assert len(records) == 1
    assert records[0]["status"] == "pending"


def test_snapshot_is_deep_copy():
    buf = ReflectionTraceBuffer(maxlen=10)
    trace_id = buf.record_start("s1", "provider-a", "prompt")
    buf.complete(trace_id, status="success", result={"k": [1, 2]})
    snapshot = buf.snapshot()
    snapshot[0]["result"]["k"].append(3)
    snapshot[0]["status"] = "failed"
    # 改动快照不影响内部缓冲
    fresh = buf.snapshot()
    assert fresh[0]["status"] == "success"
    assert fresh[0]["result"]["k"] == [1, 2]


def test_concurrent_writes():
    buf = ReflectionTraceBuffer(maxlen=10)

    def worker(offset: int):
        for i in range(20):
            trace_id = buf.record_start(f"session-{offset}", "provider-a", f"prompt-{offset}-{i}")
            buf.complete(trace_id, status="success", response=f"resp-{offset}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = buf.snapshot()
    assert len(records) == 10
    # 并发下不允许损坏条目（每条 id/status/response 齐全）
    for r in records:
        assert r["status"] == "success"
        assert r["response"].startswith("resp-")
        assert r["prompt"].startswith("prompt-")


def test_record_start_never_raises():
    buf = ReflectionTraceBuffer(maxlen=10)
    # 任意输入值都不应抛异常
    assert buf.record_start(None, None, None) is not None
    assert buf.record_start("", "", "") is not None
    assert buf.record_start("s", "p", "x") is not None