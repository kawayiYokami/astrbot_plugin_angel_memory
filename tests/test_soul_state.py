"""灵魂情绪轴的回归测试。"""

import math

import pytest

from astrbot_plugin_angel_memory.core.soul import soul_state as soul_state_module
from astrbot_plugin_angel_memory.core.soul.soul_state import SoulState


def test_initial_state_is_neutral():
    soul = SoulState()
    assert soul.get_scores() == {
        "joy": 0.5,
        "anger": 0.5,
        "surprise": 0.5,
        "disgust": 0.5,
    }


def test_reflect_moves_axes_toward_scores():
    soul = SoulState()
    soul.reflect({"joy": 0.9, "anger": 0.1, "surprise": 0.8, "disgust": 0.2})
    scores = soul.get_scores()
    assert scores["joy"] > 0.5
    assert scores["anger"] < 0.5
    assert scores["surprise"] > 0.5
    assert scores["disgust"] < 0.5


def test_scores_are_passed_through_without_inversion():
    """读数直接注入，不再做反向换算，两端含义由面板标签决定。"""
    soul = SoulState()
    soul.reflect({"surprise": 1.0, "disgust": 1.0})
    scores = soul.get_scores()
    assert scores["surprise"] > 0.5
    assert scores["disgust"] > 0.5


def test_scores_are_clamped_to_unit_range():
    soul = SoulState()
    soul.reflect({"joy": 5.0, "anger": -3.0})
    scores = soul.get_scores()
    assert 0.0 <= scores["joy"] <= 1.0
    assert 0.0 <= scores["anger"] <= 1.0


def test_invalid_score_is_skipped_without_raising():
    soul = SoulState()
    soul.reflect({"joy": "不是数字", "anger": 0.9})
    scores = soul.get_scores()
    assert scores["joy"] == 0.5
    assert scores["anger"] > 0.5


def test_non_finite_score_is_rejected():
    soul = SoulState()
    soul.reflect({"joy": float("nan"), "anger": float("inf"), "surprise": 0.9})
    scores = soul.get_scores()
    assert scores["joy"] == 0.5
    assert scores["anger"] == 0.5
    assert scores["surprise"] > 0.5


def test_unknown_keys_do_not_silently_pass():
    soul = SoulState()
    module_logger = soul_state_module.logger
    records = []
    original = module_logger.warning
    module_logger.warning = lambda message, *args, **kwargs: records.append(str(message))
    try:
        soul.reflect({"Joy": 0.9, "喜": 0.8})
    finally:
        module_logger.warning = original
    assert soul.get_scores()["joy"] == 0.5
    assert any("未命中任何情绪轴" in message for message in records)


def test_resonate_weights_by_memory_strength():
    soul = SoulState()
    soul.resonate(
        [
            {"scores": {"joy": 1.0}, "strength": 10},
            {"scores": {"joy": 0.0}, "strength": 1},
        ]
    )
    # 强记忆把状态明显拉向喜，而不是被两条记忆拉回中线
    assert soul.get_scores()["joy"] > 0.5


def test_resonate_without_scores_is_noop():
    soul = SoulState()
    soul.resonate([])
    soul.resonate([{"scores": {}, "strength": 5}])
    assert soul.get_scores()["joy"] == 0.5


def test_elasticity_resists_runaway_deviation():
    strong = SoulState()
    for _ in range(20):
        strong.reflect({"joy": 1.0})
    mild = SoulState()
    for _ in range(20):
        mild.reflect({"joy": 0.6})
    assert strong.get_scores()["joy"] <= mild.get_scores()["joy"] + 0.35


def test_state_description_lists_all_axes():
    soul = SoulState()
    description = soul.get_state_description()
    for axis in ("joy", "anger", "surprise", "disgust"):
        assert axis in description
