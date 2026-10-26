"""工具测试探针回归：文本输出、错误上报、可见域、只读边界。

不依赖 quart 请求上下文与真实事件，只钉形状契约与边界。
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from astrbot_plugin_angel_memory.core.memory_runtime.pmemory_runtime import (
    to_base_memory,
)
from astrbot_plugin_angel_memory.core.services.retrieval_service import (
    DeepMindRetrievalService,
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
)


# === 替身 ===


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
    return json.loads(resp.body)


class _FakeQueryProcessor:
    def __init__(self, boom=False):
        self.boom = boom

    def process_query_for_memory(self, query, event):
        del event
        if self.boom:
            raise RuntimeError("预处理炸了")
        return f"{query} 预处理"

    def process_query_for_notes(self, query, event):
        return self.process_query_for_memory(query, event)

    def extract_rag_fields(self, event):
        del event
        return {"entities": [], "facts": [], "keywords": []}


class _FakeRuntime:
    def __init__(self, memories, boom=False):
        self._memories = memories
        self.boom = boom

    async def chained_recall(self, **kwargs):
        del kwargs
        if self.boom:
            raise RuntimeError("召回炸了")
        return self._memories

    async def comprehensive_recall(self, query, limit=None, event=None, memory_scope="public"):
        del query, limit, event, memory_scope
        return list(self._memories)


class _FakeNoteIndex:
    def __init__(self, notes=None):
        self._notes = notes or []

    async def search_notes_async(self, query, max_results=50, domains=None):
        del query, max_results, domains
        return list(self._notes)


class _FakeSoul:
    def __init__(self):
        self.resonated = 0
        self.energy = {"RecallDepth": 0.0}

    def get_value(self, dimension):
        del dimension
        return 7.0

    def resonate(self, snapshots):
        del snapshots
        self.resonated += 1


class _FakeNoteService:
    def __init__(self, notes=None):
        self._notes = notes or []

    async def search_notes_by_top_k(self, **kwargs):
        del kwargs
        return list(self._notes)


class _FakeDeepmind:
    def __init__(self, memories, notes=None, query_boom=False, recall_boom=False):
        self.logger = logging.getLogger("test-tool-test")
        self.memory_system = _FakeRuntime(memories, boom=recall_boom)
        self.note_service = _FakeNoteService(notes)
        self.query_processor = _FakeQueryProcessor(boom=query_boom)
        self.soul = _FakeSoul()
        self.note_recall_enabled = True
        self.note_inject_top_k = 3
        self.note_candidate_top_k = 21
        self.CHAINED_RECALL_PER_TYPE_LIMIT = 7
        self.retrieval_service = DeepMindRetrievalService(self)


class _ProbeCtx:
    def __init__(self, deepmind=None, auto=None, visible=None, scope_boom=False):
        self._deepmind = deepmind
        self._auto = auto or []
        self._visible = visible if visible is not None else list(DOMAIN_DIRS)
        self._scope_boom = scope_boom

    def get_component(self, name):
        if self._deepmind is None:
            return None
        if name == "deepmind":
            return self._deepmind
        if name == "memory_runtime":
            return self._deepmind.memory_system
        if name == "note_index":
            return _FakeNoteIndex()
        return None

    def get_event_conversation_id(self, event):
        return event.unified_msg_origin

    async def get_event_persona_name(self, event):
        del event
        return ""

    def resolve_memory_scope_with_source(self, conversation_id, persona_name=""):
        del conversation_id, persona_name
        if self._scope_boom:
            raise RuntimeError("scope 解析炸了")
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


def _fresh_store() -> MemoryStore:
    root = Path(tempfile.mkdtemp(prefix="pmem-tooltest-"))
    pmemory.configure(root=root, raw_dir=None)
    return MemoryStore()


def _seed_memory() -> object:
    store = _fresh_store()
    try:
        store.remember("红豆喜欢甜食", "聊天里提到", ["红豆"], "知识记忆", "public", pinned=True)
        record = store.list_records(all_scopes=True)[0]
        return to_base_memory(record)
    finally:
        pmemory.close()


# === 域可见性 ===


def test_hidden_domain_never_visible_or_auto():
    matrix, _ = build_knowledge_matrix({"123": {"gi": "auto", "hsr": "hidden"}})
    assert auto_domains_from(matrix, "123") == ["gi"]
    visible = visible_domains_from(matrix, "123")
    assert "hsr" not in visible
    assert "gi" in visible
    assert "zzz" in visible
    assert sorted(visible) == sorted([d for d in DOMAIN_DIRS if d != "hsr"])


def test_private_chat_sees_all_domains_but_auto_is_empty():
    matrix, _ = build_knowledge_matrix({"123": {"gi": "hidden"}})
    assert auto_domains_from(matrix, "") == []
    assert sorted(visible_domains_from(matrix, "")) == sorted(DOMAIN_DIRS)


# === 输入规整 ===


def test_normalize_falls_back_to_default_umo():
    payload = ToolTestAPI._normalize({"query": "  洛伊  ", "umo": "", "group_id": None})
    assert payload["query"] == "洛伊"
    assert payload["group_id"] == ""
    assert payload["umo"] == DEFAULT_UMO


@pytest.mark.asyncio
async def test_empty_query_reports_error(monkeypatch):
    api = ToolTestAPI(_ProbeCtx(_FakeDeepmind([])))
    _patch_request(monkeypatch, {"query": "   "})
    body = _body(await api.run())
    assert body["text"] == ""
    assert body["errors"] == ["查询词不能为空"]


# === 正常路径 ===


@pytest.mark.asyncio
async def test_run_outputs_plain_text_for_both_chains(monkeypatch):
    memory = _seed_memory()
    deepmind = _FakeDeepmind([memory])
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759", "umo": ""})

    body = _body(await api.run())
    text = body["text"]

    assert body["errors"] == []
    assert "【自动 RAG】" in text
    assert "【主动搜索】" in text
    assert "预处理后查询词：洛伊 预处理" in text
    assert "记忆 1 条" in text
    assert "红豆喜欢甜食" in text
    # 主动搜索走真实工具，替身事件应能跑通并给出记忆段
    assert "[记忆检索结果" in text
    # 只读：灵魂共鸣没被触发
    assert deepmind.soul.resonated == 0


@pytest.mark.asyncio
async def test_no_auto_domain_marks_notes_as_config_issue(monkeypatch):
    deepmind = _FakeDeepmind([])
    ctx = _ProbeCtx(deepmind, auto=[])
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759", "umo": ""})

    text = _body(await api.run())["text"]
    assert "自动域=（无）" in text
    assert "没有标记为「自动」的知识域" in text
    assert "（无相关笔记）" in text


# === 错误上报 ===


@pytest.mark.asyncio
async def test_memory_recall_failure_is_reported_not_swallowed(monkeypatch):
    deepmind = _FakeDeepmind([], recall_boom=True)
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.run())
    assert any("召回炸了" in e for e in body["errors"])
    assert "[错误]" in body["text"]
    # 别的段落仍要跑完，不能一段炸全停
    assert "【主动搜索】" in body["text"]


@pytest.mark.asyncio
async def test_query_preprocess_failure_falls_back_and_reports(monkeypatch):
    deepmind = _FakeDeepmind([], query_boom=True)
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.run())
    assert any("预处理炸了" in e for e in body["errors"])
    # 降级用原查询词，仍然照跑
    assert "预处理后查询词：洛伊" in body["text"]


@pytest.mark.asyncio
async def test_scope_failure_is_reported(monkeypatch):
    deepmind = _FakeDeepmind([])
    ctx = _ProbeCtx(deepmind, scope_boom=True)
    deepmind.plugin_context = ctx
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.run())
    assert any("scope 解析炸了" in e for e in body["errors"])
    assert "[错误] 解析记忆域失败" in body["text"]


@pytest.mark.asyncio
async def test_missing_deepmind_component_is_reported(monkeypatch):
    api = ToolTestAPI(_ProbeCtx(None))
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.run())
    assert any("潜意识组件不可用" in e for e in body["errors"])


# === 替身事件 ===


@pytest.mark.asyncio
async def test_scope_line_uses_event_origin():
    class _Ctx(_ProbeCtx):
        def resolve_memory_scope_with_source(self, conversation_id, persona_name=""):
            del persona_name
            assert conversation_id == "aiocqhttp:GroupMessage:830624502"
            return "恋爱", "conversation", conversation_id

    event = PreviewEvent("aiocqhttp:GroupMessage:830624502", "830624502", _Ctx())
    lines: list[str] = []
    scope, error = await ToolTestAPI(_Ctx())._scope_line(event, lines)
    assert error == ""
    assert scope == "恋爱"
    assert "恋爱" in lines[0]
