"""短期记忆槽：按类型配额与生命值淘汰，叠加消息离场无条件清理。"""

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


def test_slot_capacity_drops_the_oldest_when_life_points_are_equal():
    session = _session()

    for index in range(1, 6):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    assert [item.id for item in session.get_memories()] == ["e3", "e4", "e5"]


def test_slot_capacity_drops_low_life_points_before_old_ones():
    session = _session()
    for index in range(1, 4):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    weak = next(item for item in session.get_memories() if item.id == "e1")
    weak.life_points = 1

    session.add_memories([_memory("e4", created_at=4.0)])

    assert [item.id for item in session.get_memories()] == ["e2", "e3", "e4"]


def test_capacity_multiplier_widens_every_slot():
    session = _session(multiplier=2.0)

    for index in range(1, 6):
        session.add_memories([_memory(f"e{index}", created_at=float(index))])

    assert len(session.get_memories()) == 5


def test_life_points_go_up_for_useful_and_down_for_the_rest():
    session = _session()
    session.add_memories(
        [_memory("a", created_at=1.0), _memory("b", created_at=2.0)]
    )

    session.evaluate_life_points(["a"])

    points = {item.id: item.life_points for item in session.get_memories()}
    assert points == {"a": 4, "b": 2}


def test_memory_dies_when_life_points_run_out():
    session = _session()
    session.add_memories(
        [_memory("a", created_at=1.0), _memory("b", created_at=2.0)]
    )

    dead = session.evaluate_life_points(["a"])
    dead = session.evaluate_life_points([])
    dead = session.evaluate_life_points([])

    assert [item.id for item in dead] == ["b"]
    assert [item.id for item in session.get_memories()] == ["a"]


def test_departed_message_kills_the_memory_regardless_of_life_points():
    session = _session()
    session.add_memories(
        [_memory("a")], bound_message_id="m1", bound_message_text="第一句"
    )
    session.add_memories(
        [_memory("b")], bound_message_id="m2", bound_message_text="第二句"
    )
    next(item for item in session.get_memories() if item.id == "a").life_points = 99

    departed = session.prune_absent_messages({"m2"})

    assert [item.id for item in departed] == ["a"]
    assert [item.id for item in session.get_memories()] == ["b"]


def test_entering_again_does_not_refresh_life_points():
    session = _session()
    session.add_memories([_memory("a")], bound_message_id="m1")
    session.evaluate_life_points([])

    session.add_memories([_memory("a")], bound_message_id="m2")

    items = session.get_memories()
    assert len(items) == 1
    assert items[0].life_points == 2
    assert items[0].bound_message_id == "m1"


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
