"""工具测试探针回归：两条链路各自独立、原文输出、错误上抛。

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
    PREVIEW_SESSION_ID,
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

    async def get_memories_by_ids(self, memory_ids, memory_scope=None):
        del memory_scope
        return [m for m in self._memories if str(getattr(m, "id", "")) in set(memory_ids)]


class _FakeSoul:
    def __init__(self):
        self.resonated = 0
        self.axes = {"joy": 0.5, "anger": 0.5, "surprise": 0.5, "disgust": 0.5}

    def get_scores(self):
        return dict(self.axes)

    def resonate(self, entries):
        del entries
        self.resonated += 1


class _FakeNoteService:
    def __init__(self, notes=None):
        self._notes = notes or []

    async def search_notes_by_top_k(self, **kwargs):
        del kwargs
        return list(self._notes)


class _FakeSessionMemory:
    def __init__(self):
        self.store = {}
        self.cleared = []

    def add_memories_to_session(self, session_id, memories):
        self.store.setdefault(session_id, []).extend(memories)

    def get_session_memory_ids(self, session_id):
        return [str(getattr(m, "id", "")) for m in self.store.get(session_id, [])]

    def get_session_memories(self, session_id):
        return list(self.store.get(session_id, []))

    def remove_memory_ids_from_session(self, session_id, memory_ids):
        del session_id, memory_ids
        return 0

    def clear_session(self, session_id):
        self.cleared.append(session_id)
        self.store.pop(session_id, None)


class _FakeInjector:
    @staticmethod
    def format_session_memories_for_prompt(memories, short_id_registry=None):
        del short_id_registry
        if not memories:
            return ""
        return "\n".join(str(getattr(m, "judgment", "")) for m in memories)


class _FakeInjectionService:
    def __init__(self, boom=False):
        self.boom = boom

    async def inject_memories_to_request(
        self,
        request,
        session_id,
        note_context,
        soul_state_values=None,
        has_secretary_decision=False,
        memory_scope="public",
    ):
        del soul_state_values, has_secretary_decision, memory_scope
        if self.boom:
            raise RuntimeError("注入炸了")
        parts = ["<instruction>固定说明</instruction>"]
        memories = self.deepmind.session_memory_manager.get_session_memories(session_id)
        memory_text = self.deepmind.memory_injector.format_session_memories_for_prompt(memories)
        if memory_text:
            parts.append(f"<memories>\n{memory_text}\n</memories>")
        if note_context:
            parts.append(f"<notes>\n{note_context}\n</notes>")
        request.contexts.append({"role": "user", "content": "\n\n".join(parts), "_no_save": True})


class _FakeDeepmind:
    def __init__(self, memories, notes=None, recall_boom=False, inject_boom=False):
        self.logger = logging.getLogger("test-tool-test")
        self.memory_system = _FakeRuntime(memories, boom=recall_boom)
        self.note_service = _FakeNoteService(notes)
        self.query_processor = _FakeQueryProcessor()
        self.soul = _FakeSoul()
        self.note_recall_enabled = True
        self.note_inject_top_k = 3
        self.note_candidate_top_k = 21
        self.CHAINED_RECALL_PER_TYPE_LIMIT = 7
        self.session_memory_manager = _FakeSessionMemory()
        self.memory_injector = _FakeInjector()
        self.retrieval_service = DeepMindRetrievalService(self)
        self.injection_service = _FakeInjectionService(boom=inject_boom)
        self.injection_service.deepmind = self


class _ProbeCtx:
    def __init__(self, deepmind=None, auto=None, visible=None, scope_boom=False):
        self._deepmind = deepmind
        self._auto = auto or []
        self._visible = visible if visible is not None else list(DOMAIN_DIRS)
        self._scope_boom = scope_boom
        if deepmind is not None:
            # 检索链路会经 deepmind.plugin_context 读域配置
            deepmind.plugin_context = self

    def get_component(self, name):
        if self._deepmind is None:
            return None
        if name == "deepmind":
            return self._deepmind
        if name == "memory_runtime":
            return self._deepmind.memory_system
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
        if self._scope_boom:
            raise RuntimeError("scope 解析炸了")
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


def _seed_memory():
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
    assert sorted(visible) == sorted([d for d in DOMAIN_DIRS if d != "hsr"])


def test_private_chat_sees_all_domains_but_auto_is_empty():
    matrix, _ = build_knowledge_matrix({"123": {"gi": "hidden"}})
    assert auto_domains_from(matrix, "") == []
    assert sorted(visible_domains_from(matrix, "")) == sorted(DOMAIN_DIRS)


# === 入参 ===


def test_normalize_falls_back_to_default_umo():
    payload = ToolTestAPI._normalize({"query": "  洛伊  ", "umo": "", "group_id": None})
    assert payload["query"] == "洛伊"
    assert payload["group_id"] == ""
    assert payload["umo"] == DEFAULT_UMO


@pytest.mark.asyncio
async def test_empty_query_rejected_on_both_endpoints(monkeypatch):
    api = ToolTestAPI(_ProbeCtx(_FakeDeepmind([])))
    _patch_request(monkeypatch, {"query": "   "})
    for handler in (api.auto_rag, api.active_recall):
        body = _body(await handler())
        assert body["text"] == ""
        assert body["errors"] == ["查询词不能为空"]


# === 两条链路互不混合 ===


@pytest.mark.asyncio
async def test_auto_rag_returns_injection_text_only(monkeypatch):
    """自动 RAG 只返回注入上下文原文，不夹带任何自造说明。"""
    memory = _seed_memory()
    deepmind = _FakeDeepmind([memory])
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759", "umo": ""})

    body = _body(await api.auto_rag())
    text = body["text"]

    assert body["errors"] == []
    assert text.startswith("<instruction>")
    assert "<memories>" in text
    assert "红豆喜欢甜食" in text
    # 不得出现自造标题、耗时、域统计这类包装内容
    for banned in ("【自动 RAG】", "耗时=", "自动域=", "预处理后查询词", "注入预览"):
        assert banned not in text
    # 主动搜索的内容不能混进来
    assert "[记忆检索结果" not in text


@pytest.mark.asyncio
async def test_active_recall_returns_tool_text_only(monkeypatch):
    """主动搜索只返回工具原文，不夹带自动 RAG 的内容。"""
    memory = _seed_memory()
    deepmind = _FakeDeepmind([memory])
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759", "umo": ""})

    body = _body(await api.active_recall())
    text = body["text"]

    assert "[记忆检索结果" in text
    assert "[笔记检索结果" in text
    for banned in ("【主动搜索】", "耗时=", "可见域=", "<instruction>", "<memories>"):
        assert banned not in text


@pytest.mark.asyncio
async def test_preview_session_is_isolated_and_cleaned(monkeypatch):
    """注入预览用独立会话，跑完必须清掉，不污染真实会话缓存。"""
    memory = _seed_memory()
    deepmind = _FakeDeepmind([memory])
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    await api.auto_rag()

    assert PREVIEW_SESSION_ID in deepmind.session_memory_manager.cleared
    assert PREVIEW_SESSION_ID not in deepmind.session_memory_manager.store


# === 错误上报 ===


@pytest.mark.asyncio
async def test_recall_failure_is_reported(monkeypatch):
    deepmind = _FakeDeepmind([], recall_boom=True)
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.auto_rag())
    assert any("召回炸了" in e for e in body["errors"])
    # 召回失败后注入仍按真实流程跑完，只是没有记忆块：显示的就是真实注入内容
    assert "<instruction>" in body["text"]
    assert "<memories>" not in body["text"]


@pytest.mark.asyncio
async def test_injection_failure_is_reported(monkeypatch):
    memory = _seed_memory()
    deepmind = _FakeDeepmind([memory], inject_boom=True)
    ctx = _ProbeCtx(deepmind, auto=["gi"])
    api = ToolTestAPI(ctx)
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})

    body = _body(await api.auto_rag())
    assert any("注入炸了" in e for e in body["errors"])
    # 出错也必须清掉预览会话
    assert PREVIEW_SESSION_ID in deepmind.session_memory_manager.cleared


@pytest.mark.asyncio
async def test_missing_deepmind_component_is_reported(monkeypatch):
    api = ToolTestAPI(_ProbeCtx(None))
    _patch_request(monkeypatch, {"query": "洛伊", "group_id": "327313759"})
    body = _body(await api.auto_rag())
    assert any("潜意识组件不可用" in e for e in body["errors"])


# === 替身事件 ===


def test_preview_event_exposes_only_needed_fields():
    event = PreviewEvent("aiocqhttp:GroupMessage:830624502", "830624502", None)
    assert event.unified_msg_origin == "aiocqhttp:GroupMessage:830624502"
    assert event.get_group_id() == "830624502"
    assert event.get_platform_name() == ""
    # 不提供天使之心上下文：检索侧按 hasattr 自然跳过
    assert not hasattr(event, "angelheart_context")
