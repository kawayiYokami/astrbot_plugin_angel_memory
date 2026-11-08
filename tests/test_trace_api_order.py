"""过程追踪接口的展示顺序：最新一条排在最前面。"""

import asyncio

from web_api.trace_api import TraceAPI


class _FakeStore:
    def __init__(self, records):
        self._records = list(records)

    async def read_recent(self, limit=20):
        return self._records[-limit:]


class _FakeBuffer:
    def snapshot(self):
        return []


class _FakeDeepMind:
    def __init__(self, records):
        self.trace_store = _FakeStore(records)
        self.reflection_trace = _FakeBuffer()


class _FakeContext:
    def __init__(self, deepmind):
        self._deepmind = deepmind

    def get_component(self, name):
        return self._deepmind


def _record(rid, timestamp):
    return {
        "id": str(rid),
        "timestamp": float(timestamp),
        "time": "",
        "type": "reflection",
        "status": "success",
    }


def _collect(records):
    deepmind = _FakeDeepMind(records)
    api = TraceAPI(_FakeContext(deepmind))
    return asyncio.run(api._collect_records(deepmind))


def test_records_are_newest_first():
    got = _collect([_record(i, i) for i in range(5)])

    assert [item["id"] for item in got] == ["4", "3", "2", "1", "0"]


def test_cap_keeps_the_latest_records():
    got = _collect([_record(i, i) for i in range(30)])

    assert len(got) == 20
    assert got[0]["id"] == "29"
    assert got[-1]["id"] == "10"
