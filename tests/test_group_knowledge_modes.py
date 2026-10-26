"""群聊知识库三态：矩阵构建、可见域判定、检索白名单。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from astrbot_plugin_angel_memory.core.services.note_index_service import NoteIndexService
from astrbot_plugin_angel_memory.pmemory import pmemory
from astrbot_plugin_angel_memory.pmemory.config import (
    DOMAIN_DIRS,
    auto_domains_from,
    build_knowledge_matrix,
    normalize_mode,
    visible_domains_from,
)


def test_normalize_mode_rejects_unknown_values():
    assert normalize_mode("AUTO") == "auto"
    assert normalize_mode(" hidden ") == "hidden"
    assert normalize_mode("invisible") is None
    assert normalize_mode(None) is None


def test_matrix_defaults_every_domain_to_manual():
    matrix, errors = build_knowledge_matrix({})
    assert matrix == {}
    assert errors == []
    # 表里没有的群按 manual 处理：能搜、不自动注入
    assert auto_domains_from(matrix, "123") == []
    assert sorted(visible_domains_from(matrix, "123")) == sorted(DOMAIN_DIRS)


def test_legacy_group_domain_map_becomes_auto():
    matrix, errors = build_knowledge_matrix({}, {"123456": "原神"})
    assert errors == []
    assert matrix["123456"]["gi"] == "auto"
    assert matrix["123456"]["hsr"] == "manual"
    assert auto_domains_from(matrix, "123456") == ["gi"]


def test_new_map_overrides_legacy():
    matrix, _ = build_knowledge_matrix(
        {"123456": {"gi": "hidden"}},
        {"123456": "gi"},
    )
    assert matrix["123456"]["gi"] == "hidden"
    assert auto_domains_from(matrix, "123456") == []


def test_invalid_entries_are_reported_not_stored():
    matrix, errors = build_knowledge_matrix(
        {
            "123": {"gi": "invisible", "不存在的域": "auto"},
            "": {"gi": "auto"},
            "456": "gi",
        }
    )
    assert matrix["123"]["gi"] == "manual"
    assert len(errors) == 4
    assert "123" in matrix and "456" not in matrix


def test_hidden_domain_is_not_searchable_but_auto_and_manual_are():
    matrix, _ = build_knowledge_matrix(
        {"42": {"gi": "auto", "hsr": "manual", "zzz": "hidden"}}
    )
    assert auto_domains_from(matrix, "42") == ["gi"]
    visible = visible_domains_from(matrix, "42")
    assert "zzz" not in visible
    assert "gi" in visible and "hsr" in visible


def test_private_chat_sees_every_domain_but_never_auto():
    matrix, _ = build_knowledge_matrix({"42": {"gi": "auto"}})
    assert visible_domains_from(matrix, "") == list(DOMAIN_DIRS)
    assert auto_domains_from(matrix, "") == []


def test_note_search_respects_domain_whitelist(tmp_path):
    raw = tmp_path / "raw"
    for directory in ("原神", "绝区零"):
        (raw / directory).mkdir(parents=True, exist_ok=True)
    (raw / "原神" / "角色.md").write_text("# 角色\n共同关键词 原神笔记正文", encoding="utf-8")
    (raw / "绝区零" / "代理人.md").write_text("# 代理人\n共同关键词 绝区零笔记正文", encoding="utf-8")

    pmemory.configure(root=tmp_path / "pmemory", raw_dir=raw)
    service = NoteIndexService(raw)
    assert service.upsert_file(raw / "原神" / "角色.md")
    assert service.upsert_file(raw / "绝区零" / "代理人.md")

    only_gi = service.search_notes("共同关键词", max_results=20, domains=["gi"])
    assert only_gi, "白名单内的域应该有命中"
    assert {item["metadata"]["domain"] for item in only_gi} == {"gi"}

    hidden_zzz = service.search_notes("共同关键词", max_results=20, domains=["gi", "hsr"])
    assert {item["metadata"]["domain"] for item in hidden_zzz} == {"gi"}

    both = service.search_notes("共同关键词", max_results=20, domains=["gi", "zzz"])
    assert {item["metadata"]["domain"] for item in both} == {"gi", "zzz"}

    pmemory.close()


class _FakeLedger:
    def __init__(self, groups):
        self._groups = groups

    def get_known_group_ids(self):
        return list(self._groups)

    def get_group_name(self, group_id):
        return self._groups.get(group_id, "")


class _FakeContext:
    """只实现群列表接口需要的几个方法。"""

    def __init__(self, groups, modes):
        self._ledger = _FakeLedger(groups)
        self._matrix = modes
        self.config = {}

    def get_component(self, name):
        return self._ledger if name == "memory_sql_manager" else None

    def get_group_knowledge_map(self):
        return {group: dict(value) for group, value in self._matrix.items()}

    def list_knowledge_modes(self, group_id):
        modes = self._matrix.get(str(group_id or "").strip()) or {}
        return {domain: modes.get(domain, "manual") for domain in DOMAIN_DIRS}


def test_group_list_merges_ledger_and_config():
    from astrbot_plugin_angel_memory.web_api.group_knowledge_api import (
        GroupKnowledgeAPI,
    )

    api = GroupKnowledgeAPI(
        _FakeContext(
            groups={"1001": "群一", "1002": "群二"},
            modes={"1002": {"gi": "auto"}, "1003": {"hsr": "hidden"}},
        )
    )
    payload = api.get_groups()
    body = payload.body if hasattr(payload, "body") else payload
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    data = json.loads(body)["data"] if isinstance(body, str) else body
    groups = {row["group_id"]: row for row in data["groups"]}

    # 账本里的群 + 只在配置里出现的群都要列出
    assert set(groups) == {"1001", "1002", "1003"}
    assert groups["1002"]["group_name"] == "群二"
    assert groups["1002"]["modes"]["gi"] == "auto"
    assert groups["1002"]["modes"]["hsr"] == "manual"
    assert groups["1003"]["modes"]["hsr"] == "hidden"
    # 每个群都补齐全部域，避免前端拿到残缺对象
    assert set(groups["1001"]["modes"]) == set(DOMAIN_DIRS)


def test_normalize_groups_fills_missing_domains_and_rejects_bad_values():
    from astrbot_plugin_angel_memory.web_api.group_knowledge_api import (
        GroupKnowledgeAPI,
    )

    table, error = GroupKnowledgeAPI.normalize_groups(
        [{"group_id": "9", "modes": {"gi": "auto"}}]
    )
    assert error == ""
    assert table["9"]["gi"] == "auto"
    assert table["9"]["zzz"] == "manual"

    _, error = GroupKnowledgeAPI.normalize_groups(
        [{"group_id": "9", "modes": {"gi": "visible"}}]
    )
    assert "非法项" in error

    _, error = GroupKnowledgeAPI.normalize_groups([{"modes": {"gi": "auto"}}])
    assert "group_id" in error
