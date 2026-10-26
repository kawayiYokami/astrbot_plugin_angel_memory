"""工具测试探针回归：可见域、自动域、scope 来源、文本计数。

只测纯逻辑与形状契约，不依赖 quart 请求上下文与真实事件。
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from astrbot_plugin_angel_memory.core.memory_runtime.pmemory_runtime import (
    to_base_memory,
)
from astrbot_plugin_angel_memory.pmemory import MemoryStore, pmemory
from astrbot_plugin_angel_memory.pmemory.config import (
    DOMAIN_DIRS,
    auto_domains_from,
    build_knowledge_matrix,
    visible_domains_from,
)
from astrbot_plugin_angel_memory.web_api.tool_test_api import (
    DEFAULT_UMO,
    PreviewEvent,
    ToolTestAPI,
    _MEMORY_COUNT_RE,
    _NOTE_COUNT_RE,
)


def _fresh_store() -> MemoryStore:
    root = Path(tempfile.mkdtemp(prefix="pmem-tooltest-"))
    pmemory.configure(root=root, raw_dir=None)
    return MemoryStore()


def test_hidden_domain_never_visible_or_auto():
    matrix, _ = build_knowledge_matrix({"123": {"gi": "auto", "hsr": "hidden"}})
    assert auto_domains_from(matrix, "123") == ["gi"]
    visible = visible_domains_from(matrix, "123")
    assert "hsr" not in visible
    assert "gi" in visible
    # 未配置的域默认 manual：能搜、不自动注入
    assert "zzz" in visible
    assert sorted(visible) == sorted([d for d in DOMAIN_DIRS if d != "hsr"])


def test_private_chat_sees_all_domains_but_auto_is_empty():
    matrix, _ = build_knowledge_matrix({"123": {"gi": "hidden"}})
    assert auto_domains_from(matrix, "") == []
    assert sorted(visible_domains_from(matrix, "")) == sorted(DOMAIN_DIRS)


def test_count_extracts_tool_headers():
    raw = "[记忆检索结果 (12条)]\nfoo\n\n[笔记检索结果 (3个文件)]\nbar"
    assert ToolTestAPI._count(_MEMORY_COUNT_RE, raw) == 12
    assert ToolTestAPI._count(_NOTE_COUNT_RE, raw) == 3


def test_count_returns_none_when_no_result_section():
    """没命中时不返回 0，避免把「没搜到」和「没有这段」混为一谈。"""
    empty = "[记忆检索结果]\n无相关记忆。\n\n[笔记检索结果]\n无相关笔记。"
    assert ToolTestAPI._count(_MEMORY_COUNT_RE, empty) is None
    assert ToolTestAPI._count(_NOTE_COUNT_RE, empty) is None


def test_memory_items_keep_scope_and_type():
    store = _fresh_store()
    try:
        store.remember("红豆喜欢甜食", "聊天里提到", ["红豆"], "知识记忆", "恋爱", pinned=True)
        record = store.list_records(all_scopes=True)[0]
        items = ToolTestAPI._memory_items([to_base_memory(record)])
        assert len(items) == 1
        assert items[0]["judgment"] == "红豆喜欢甜食"
        assert items[0]["memory_scope"] == "恋爱"
        assert items[0]["memory_type"] == "知识记忆"
        assert items[0]["is_active"] is True
        assert items[0]["tags"] == ["红豆"]
    finally:
        pmemory.close()


def test_note_items_accept_pmemory_shape_and_truncate():
    long_content = "x" * 500
    notes = [
        {
            "content": long_content,
            "similarity": 0.82,
            "metadata": {
                "source_file_path": "原神/钟离.md",
                "note_id": 7,
                "line_start": 3,
                "line_end": 9,
                "domain": "gi",
            },
        },
        "不是字典，应被跳过",
    ]
    items = ToolTestAPI._note_items(notes)
    assert len(items) == 1
    assert items[0]["path"] == "原神/钟离.md"
    assert items[0]["note_id"] == 7
    assert len(items[0]["content"]) == 200


def test_normalize_falls_back_to_default_umo():
    payload = ToolTestAPI._normalize({"query": "  钟离  ", "umo": "", "group_id": None})
    assert payload["query"] == "钟离"
    assert payload["group_id"] == ""
    assert payload["umo"] == DEFAULT_UMO


@pytest.mark.asyncio
async def test_scope_info_falls_back_to_public_when_unmapped():
    """替身事件只带必要字段时，scope 解析必须回落 public 而不是抛异常。"""

    class _Ctx:
        def get_event_conversation_id(self, event):
            return event.unified_msg_origin

        async def get_event_persona_name(self, event):
            del event
            return ""

        def resolve_memory_scope_with_source(self, conversation_id, persona_name=""):
            del conversation_id, persona_name
            return "public", "default", "public"

    event = PreviewEvent("aiocqhttp:GroupMessage:0", "", _Ctx())
    api = ToolTestAPI(_Ctx())
    info = await api._scope_info(event)
    assert info["memory_scope"] == "public"
    assert info["matched_by"] == "default"
    assert info["conversation_id"] == "aiocqhttp:GroupMessage:0"


@pytest.mark.asyncio
async def test_scope_info_with_real_event_shape():
    """真实 PluginContext 的 scope 解析对替身事件可用。"""

    class _Ctx:
        def get_event_conversation_id(self, event):
            return event.unified_msg_origin

        async def get_event_persona_name(self, event):
            del event
            return ""

        def resolve_memory_scope_with_source(self, conversation_id, persona_name=""):
            del conversation_id, persona_name
            return "恋爱", "conversation", "aiocqhttp:GroupMessage:830624502"

    event = PreviewEvent("aiocqhttp:GroupMessage:830624502", "830624502", _Ctx())
    api = ToolTestAPI(_Ctx())
    info = await api._scope_info(event)
    assert info["memory_scope"] == "恋爱"
    assert info["matched_by"] == "conversation"
    assert info["matched_key"] == "aiocqhttp:GroupMessage:830624502"


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self, default=None):
        return self._body if self._body is not None else default


def _patch_request(monkeypatch, body):
    monkeypatch.setattr(
        "astrbot_plugin_angel_memory.web_api.tool_test_api.request",
        _FakeRequest(body),
    )


def _body(resp):
    """starlette 的 JSONResponse 只能从 body 取。"""
    import json as _json

    return _json.loads(resp.body)


class _FakeRuntime:
    def __init__(self, memories):
        self._memories = memories
        self.calls = []

    async def chained_recall(self, **kwargs):
        self.calls.append(kwargs)
        return self._memories


class _FakeQueryProcessor:
    def process_query_for_memory(self, query, event):
        del event
        return f"{query} 预处理"

    def process_query_for_notes(self, query, event):
        del event
        return f"{query} 预处理"

    def extract_rag_fields(self, event):
        del event
        return {"entities": [], "facts": [], "keywords": []}


class _FakeSoul:
    def __init__(self):
        self.resonated = 0

    def get_value(self, dimension):
        del dimension
        return 7.0

    def resonate(self, snapshots):
        del snapshots
        self.resonated += 1


class _FakeDeepmind:
    def __init__(self, memories, notes):
        import logging

        from astrbot_plugin_angel_memory.core.services.retrieval_service import (
            DeepMindRetrievalService,
        )

        self.logger = logging.getLogger("test-tool-test")
        self.memory_system = _FakeRuntime(memories)
        self.note_service = None
        self.query_processor = _FakeQueryProcessor()
        self.soul = _FakeSoul()
        self.note_recall_enabled = True
        self.note_inject_top_k = 3
        self.note_candidate_top_k = 21
        self.CHAINED_RECALL_PER_TYPE_LIMIT = 7
        self.retrieval_service = DeepMindRetrievalService(self)
        self._notes = notes

    async def note_search(self, **kwargs):
        del kwargs
        return self._notes


class _FakeNoteService:
    def __init__(self, notes=None):
        self._notes = notes or []

    async def search_notes_by_top_k(self, **kwargs):
        del kwargs
        return list(self._notes)


class _ProbeCtx:
    def __init__(self, deepmind, auto=None, visible=None):
        self._deepmind = deepmind
        self._auto = auto or []
        self._visible = visible or list(DOMAIN_DIRS)

    def get_component(self, name):
        return self._deepmind if name == "deepmind" else None

    def get_event_conversation_id(self, event):
        return event.unified_msg_origin

    async def get_event_persona_name(self, event):
        del event
        return ""

    def resolve_memory_scope_with_source(self, conversation_id, persona_name=""):
        del conversation_id, persona_name
        return "public", "default", "public"

    async def resolve_memory_scope_from_event(self, event):
        del event
        return "public"

    def auto_domains(self, group_id):
        del group_id
        return list(self._auto)

    def visible_domains(self, group_id):
        del group_id
        return list(self._visible)


@pytest.mark.asyncio
async def test_auto_rag_is_readonly_and_reports_scope(monkeypatch):
    """探针必须不触发灵魂共鸣：这是「只读」的硬边界。"""
    store = _fresh_store()
    try:
        store.remember("红豆喜欢甜食", "r", ["红豆"], "知识记忆", "public")
        record = store.list_records(all_scopes=True)[0]
        memory = to_base_memory(record)
    finally:
        pmemory.close()

    deepmind = _FakeDeepmind([memory], [])
    deepmind.note_service = _FakeNoteService()
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "钟离", "group_id": "830624502", "umo": ""})

    resp = await api.auto_rag()
    body = _body(resp)
    data = body["data"]

    assert data["error"] is None
    assert data["memory_scope"] == "public"
    assert data["processed_query"] == "钟离 预处理"
    assert data["auto_domains"] == ["gi"]
    assert len(data["memories"]) == 1
    assert data["memories"][0]["judgment"] == "红豆喜欢甜食"
    # 只读：共鸣没被触发
    assert deepmind.soul.resonated == 0


@pytest.mark.asyncio
async def test_auto_rag_rejects_empty_query(monkeypatch):
    api = ToolTestAPI(_ProbeCtx(_FakeDeepmind([], [])))
    _patch_request(monkeypatch, {"query": "   "})
    resp = await api.auto_rag()
    body = _body(resp)
    assert body.get("status") != "ok"
