"""TraceStore JSONL 持久化测试。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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

from astrbot_plugin_angel_memory.core.trace_store import TraceStore


def test_empty_store_returns_empty(tmp_path):
    store = TraceStore(tmp_path)
    assert store.read_recent_sync(limit=5) == []


def test_read_recent_returns_latest_in_order(tmp_path):
    store = TraceStore(tmp_path)
    for i in range(5):
        store.append_sync({"type": "reflection", "session_id": f"s{i}", "prompt": f"prompt-{i}"})

    records = store.read_recent_sync(limit=3)

    assert [r["session_id"] for r in records] == ["s2", "s3", "s4"]


def test_read_recent_supports_offset(tmp_path):
    store = TraceStore(tmp_path)
    for i in range(5):
        store.append_sync({"type": "reflection", "session_id": f"s{i}"})

    records = store.read_recent_sync(limit=2, offset=2)

    assert [r["session_id"] for r in records] == ["s1", "s2"]


def test_long_text_compressed_on_disk_and_restored(tmp_path):
    store = TraceStore(tmp_path)
    text = "记忆内容" * 4000
    store.append_sync(
        {"type": "sleep", "prompt": text, "system_prompt": "固定规则", "response": "模型返回"}
    )

    raw_line = store.path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw_line)
    assert isinstance(payload["prompt"], dict)
    assert "__lzma__" in payload["prompt"]
    assert text not in raw_line

    record = store.read_recent_sync(limit=1)[0]
    assert record["prompt"] == text
    assert record["system_prompt"] == "固定规则"
    assert record["response"] == "模型返回"


def test_extra_fields_are_preserved(tmp_path):
    store = TraceStore(tmp_path)
    store.append_sync(
        {
            "type": "sleep",
            "status": "failed",
            "summary": {"candidates": 3, "deleted": 1},
            "error": "审查调用超时",
        }
    )

    record = store.read_recent_sync(limit=1)[0]

    assert record["type"] == "sleep"
    assert record["status"] == "failed"
    assert record["summary"]["candidates"] == 3
    assert record["error"] == "审查调用超时"


def test_delete_records_keeps_others(tmp_path):
    store = TraceStore(tmp_path)
    ids = [
        store.append_sync({"type": "reflection", "session_id": f"s{i}"})["id"]
        for i in range(3)
    ]

    removed = store.delete_records_sync([ids[1]])

    assert removed == 1
    remaining = store.read_recent_sync(limit=10)
    assert [r["session_id"] for r in remaining] == ["s0", "s2"]


def test_delete_unknown_id_is_noop(tmp_path):
    store = TraceStore(tmp_path)
    store.append_sync({"type": "reflection", "session_id": "s0"})

    assert store.delete_records_sync(["missing-id"]) == 0
    assert len(store.read_recent_sync(limit=10)) == 1


def test_records_survive_new_instance(tmp_path):
    TraceStore(tmp_path).append_sync({"type": "reflection", "prompt": "跨进程读取"})

    reopened = TraceStore(tmp_path)

    assert reopened.read_recent_sync(limit=1)[0]["prompt"] == "跨进程读取"
