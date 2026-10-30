"""记忆检索的分数量规：自动召回按重排分留最高分一半以上，最多 10 条；其余路径不限。"""

import asyncio

from astrbot_plugin_angel_memory.core.memory_runtime.pmemory_runtime import (
    PMemoryMemoryRuntime,
)
from astrbot_plugin_angel_memory.pmemory import MemoryStore


class _FakeKB:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(kwargs)
        return {"hits": self._hits}


class _FakeStore:
    def __init__(self, hits):
        self.scopes = ["public"]
        self.space_id = None
        self.kb = _FakeKB(hits)


def _hit(memory_id, rerank_score, fused_score=0.0, memory_type="知识"):
    return {
        "rerank_score": rerank_score,
        "score": fused_score,
        "record": {
            "id": memory_id,
            "judgment": f"记忆{memory_id}",
            "memory_type": memory_type,
        },
    }


def test_plain_search_keeps_every_hit():
    hits = [_hit(1, 10.0), _hit(2, 4.0), _hit(3, 0.2)]
    store = MemoryStore(_FakeStore(hits))

    assert [item["id"] for item in store.search("查询", "public")] == [1, 2, 3]


def test_ratio_drops_hits_below_half_of_top_rerank_score():
    hits = [_hit(1, 10.0), _hit(2, 6.0), _hit(3, 4.9), _hit(4, 0.2)]
    store = MemoryStore(_FakeStore(hits))

    got = store.search("查询", "public", score_ratio=0.5)
    assert [item["id"] for item in got] == [1, 2]


def test_ratio_falls_back_to_fused_score_when_rerank_missing():
    hits = [_hit(1, None, 1.0), _hit(2, None, 0.6), _hit(3, None, 0.4)]
    store = MemoryStore(_FakeStore(hits))

    got = store.search("查询", "public", score_ratio=0.5)
    assert [item["id"] for item in got] == [1, 2]


def test_ratio_uses_fused_score_when_any_hit_lacks_rerank_score():
    hits = [_hit(1, 100.0, 1.0), _hit(2, None, 0.6), _hit(3, 0.1, 0.2)]
    store = MemoryStore(_FakeStore(hits))

    got = store.search("查询", "public", score_ratio=0.5)
    assert [item["id"] for item in got] == [1, 2]


def test_max_results_caps_below_limit():
    hits = [_hit(index, 10.0) for index in range(1, 16)]
    store = MemoryStore(_FakeStore(hits))

    got = store.search("查询", "public", limit=50, score_ratio=0.5, max_results=10)
    assert len(got) == 10


def test_type_filter_applies_before_score_threshold():
    hits = [
        _hit(1, 1.0, memory_type="知识"),
        _hit(2, 0.9, memory_type="技能"),
        _hit(3, 0.5, memory_type="技能"),
    ]
    store = MemoryStore(_FakeStore(hits))

    got = store.search("查询", "public", memory_type="技能", score_ratio=0.5)
    assert [item["id"] for item in got] == [2, 3]


def test_auto_recall_filters_but_manual_recall_does_not():
    hits = [_hit(1, 10.0), _hit(2, 9.0), _hit(3, 1.0), _hit(4, 0.5), _hit(5, 0.2)]
    runtime = PMemoryMemoryRuntime(MemoryStore(_FakeStore(hits)))

    auto = asyncio.run(runtime.chained_recall("查询", entities=[], final_limit=5))
    manual = asyncio.run(runtime.comprehensive_recall("查询", limit=5))

    assert len(auto) == 2
    assert len(manual) == 5
