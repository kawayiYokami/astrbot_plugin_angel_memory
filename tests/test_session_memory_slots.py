"""短期记忆：按类型配额先进先出淘汰，召回刷新时间与来源，来源消息离场即清。"""

from types import SimpleNamespace

from astrbot_plugin_angel_memory.core.config import MemoryCapacityConfig, MemoryConfig
from astrbot_plugin_angel_memory.core.session_memory import SessionMemory


def _memory(memory_id, memory_type="事件记忆", created_at=0.0):
    return SimpleNamespace(
        id=memory_id,
        memory_type=memory_type,
        created_at=created_at,
        judgment=memory_id,
        reasoning="",
        tags=[],
    )


def _session(multiplier=1.0, config=None):
    return SessionMemory(
        "sess",
        capacity_config=config or MemoryCapacityConfig(),
        capacity_provider=lambda: multiplier,
    )


def test_slot_capacity_drops_the_oldest_first():
    session = _session()

    for index in range(1, 6):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    assert [item.id for item in session.get_memories()] == ["e3", "e4", "e5"]


def test_recall_refreshes_time_and_source_message():
    session = _session()
    session.add_memories([_memory("a")], recalled_message_id="m1", recalled_message_text="第一句")
    first = session.get_memories()[0]
    added_at = first.added_at

    session.add_memories([_memory("a")], recalled_message_id="m9", recalled_message_text="第九句")

    items = session.get_memories()
    assert len(items) == 1
    assert items[0].added_at == added_at
    assert items[0].recalled_at >= added_at
    assert items[0].recalled_message_id == "m9"
    assert items[0].recalled_message_text == "第九句"


def test_recently_recalled_memory_survives_eviction():
    """最久没被召回的先走，刚被召回的那条留在表里。"""
    session = _session()
    for index in range(1, 4):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    session.add_memories([_memory("e1")], recalled_message_id="m2", recalled_message_text="又提到它")
    session.add_memories([_memory("e4", created_at=4.0)])

    assert sorted(item.id for item in session.get_memories()) == ["e1", "e3", "e4"]


def test_capacity_multiplier_widens_every_slot():
    session = _session(multiplier=2.0)

    for index in range(1, 6):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    assert len(session.get_memories()) == 5


def test_departed_message_takes_the_memory_with_it():
    session = _session()
    session.add_memories([_memory("a")], recalled_message_id="m1", recalled_message_text="第一句")
    session.add_memories([_memory("b")], recalled_message_id="m2", recalled_message_text="第二句")

    departed = session.prune_departed(["第二句"])

    assert [item.id for item in departed] == ["a"]
    assert [item.id for item in session.get_memories()] == ["b"]


def test_duplicate_text_keeps_the_memory_instead_of_killing_it():
    """两条消息正文一样时都会判「还在」，结果是多留一条，不会误杀。"""
    session = _session()
    session.add_memories([_memory("a")], recalled_message_id="m1", recalled_message_text="好")

    assert session.prune_departed(["好", "别的话"]) == []
    assert [item.id for item in session.get_memories()] == ["a"]


def test_short_bound_text_is_not_matched_by_substring():
    """「好」这类短正文只做相等判定，否则任何一条消息都把它判成活。"""
    session = _session()
    session.add_memories([_memory("a")], recalled_message_text="好")

    departed = session.prune_departed(["好的我看看", "今天别的话"])

    assert [item.id for item in departed] == ["a"]


def test_short_bound_text_still_matches_an_exact_hit():
    session = _session()
    session.add_memories([_memory("a")], recalled_message_text="好")

    assert session.prune_departed(["好"]) == []


def test_memory_without_source_text_only_leaves_by_capacity():
    session = _session()
    session.add_memories([_memory("a")], recalled_message_id="m1", recalled_message_text="")

    assert session.prune_departed([]) == []
    assert [item.id for item in session.get_memories()] == ["a"]


def test_stats_report_slot_usage_by_type():
    session = _session()
    session.add_memories([_memory("k1", memory_type="知识记忆")])

    stats = session.get_stats()

    assert stats["capacity_multiplier"] == 1.0
    assert stats["by_type"]["knowledge"] == {"current": 1, "capacity": 14}


def test_config_reads_the_multiplier_from_memory_behavior():
    assert MemoryConfig({}).short_term_memory_capacity == 1.0

    config = MemoryConfig({"memory_behavior": {"short_term_memory_capacity": 3}})

    assert config.short_term_memory_capacity == 3.0
    assert config.to_dict()["short_term_memory_capacity"] == 3.0
    assert config.get_capacity_config().event == 3
