"""p-memory 接线层的最小回归：衰减策略映射、知识域解析、笔记域归属。"""

from astrbot_plugin_angel_memory.core.component_factory import ComponentFactory
from astrbot_plugin_angel_memory.core.services.note_index_service import (
    NoteIndexService,
)
from astrbot_plugin_angel_memory.pmemory.config import (
    normalize_domain,
    read_scopes,
)


def test_decay_policy_matches_builtin_three_tier_semantics():
    policy = ComponentFactory._build_decay_policy()
    assert policy["tier0_threshold"] == 3.0
    assert policy["tier1_threshold"] == 10.0
    assert policy["useful_score_boost"] == 2.5
    assert policy["strength_boost"] == 1
    assert policy["tier0_cycle_days"] == 3
    assert policy["tier1_threshold"] > policy["tier0_threshold"]


def test_read_scopes_matches_legacy_visibility():
    assert read_scopes("public") == ["public"]
    assert read_scopes("恋爱") == ["恋爱", "public"]


def test_normalize_domain_accepts_key_and_chinese_dir():
    assert normalize_domain("gi") == "gi"
    assert normalize_domain("原神") == "gi"
    assert normalize_domain("不存在的域") is None


def test_domain_for_path_groups_by_first_level_directory(tmp_path):
    raw = tmp_path
    original = raw / "原神"
    original.mkdir()
    note = original / "角色.md"
    note.write_text("# 角色", encoding="utf-8")

    assert NoteIndexService.domain_for_path(note, raw) == "gi"
    assert NoteIndexService.domain_for_path(tmp_path / "别处" / "x.md", raw) is None
