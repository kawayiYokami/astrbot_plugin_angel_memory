"""记忆注入：短期记忆全量注入到上下文末尾一条，不往历史消息里挂。"""

from types import SimpleNamespace

import pytest
from astrbot_plugin_angel_memory.core.deepmind import DeepMind
from astrbot_plugin_angel_memory.core.services.injection_service import (
    DeepMindInjectionService,
)
from astrbot_plugin_angel_memory.core.session_memory import SessionMemory

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart


class _FakeMemoryInjector:
    @staticmethod
    def format_session_memories_for_prompt(memories, short_id_registry=None):
        ids = [str(getattr(memory, "id", "")) for memory in memories or []]
        return "\n".join(f"- 记忆{mid}" for mid in ids)


class _FakeSessionMemoryManager:
    """只保留注入服务需要的那几个口子。"""

    def __init__(self, memories=None):
        self.items = list(memories or [])

    def get_session_memories(self, session_id):
        return list(self.items)

    def prune_departed_from_session(self, session_id, alive_texts):
        alive = {str(text).strip() for text in (alive_texts or [])}
        departed = [
            item
            for item in self.items
            if item.bound_message_text and item.bound_message_text not in alive
        ]
        departed_ids = {item.id for item in departed}
        self.items = [item for item in self.items if item.id not in departed_ids]
        return departed


class _FakeMemorySystem:
    def __init__(self, memories=None):
        self.memories = list(memories or [])

    async def get_memories_by_ids(self, memory_ids, memory_scope="public"):
        wanted = {str(mid) for mid in memory_ids}
        return [m for m in self.memories if str(getattr(m, "id", "")) in wanted]


class _FakeDeepMind:
    def __init__(self, memories=None):
        self.config = SimpleNamespace(enable_soul_system=False)
        self.memory_system = _FakeMemorySystem(memories)
        self.session_memory_manager = _FakeSessionMemoryManager()
        self.user_profile_service = None
        self.memory_injector = _FakeMemoryInjector()


def _memory(memory_id):
    return SimpleNamespace(id=memory_id)


def _session(memory_id, bound_message_text="", bound_message_id=""):
    return SimpleNamespace(
        id=memory_id,
        bound_message_id=bound_message_id,
        bound_message_text=bound_message_text,
        memory=_memory(memory_id),
    )


def _text_of(request):
    for block in reversed(request.contexts or []):
        content = block.get("content") if isinstance(block, dict) else ""
        if isinstance(content, str) and "<system_reminder>" in content:
            return content
    return ""


@pytest.mark.asyncio
async def test_block_carries_every_memory_in_the_session():
    deepmind = _FakeDeepMind([_memory("a"), _memory("b")])
    deepmind.session_memory_manager.items = [_session("a", "第一句"), _session("b", "第二句")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(request, "sess", "")

    text = _text_of(request)
    assert "记忆a" in text
    assert "记忆b" in text
    assert text.startswith("<system_reminder>")
    assert 'id="' not in text
    # 行为守则进系统提示词，不跟着对话消息走
    assert "<instruction>" not in text
    assert "【关于你记得的事】" in (request.system_prompt or "")
    # 注入块追加在上下文末尾，带 _no_save 不落库
    assert len(request.contexts) == 1
    assert request.contexts[0]["_no_save"] is True
    assert request.contexts[0]["role"] == "user"


@pytest.mark.asyncio
async def test_nothing_to_inject_means_no_block():
    deepmind = _FakeDeepMind([])
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(request, "sess", "")

    assert request.contexts == []
    assert request.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_soul_state_keeps_block_alive_without_memories():
    deepmind = _FakeDeepMind([])
    deepmind.config = SimpleNamespace(enable_soul_system=True)
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request,
        "sess",
        "",
        soul_state_values={"joy": 0.6, "surprise": 0.5, "anger": 0.4, "disgust": 0.7},
    )

    text = _text_of(request)
    assert "<soul_state>" in text
    assert "<memories>" not in text


@pytest.mark.asyncio
async def test_memories_come_before_soul_state():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "第一句")]
    deepmind.config = SimpleNamespace(enable_soul_system=True)
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request,
        "sess",
        "",
        soul_state_values={"joy": 0.6, "surprise": 0.5, "anger": 0.4, "disgust": 0.7},
    )

    text = _text_of(request)
    assert text.index("<memories>") < text.index("<soul_state>")
    # 状态是最后一块，紧挨着收尾标签
    assert text.index("</soul_state>") + len("</soul_state>\n") == text.index(
        "</system_reminder>"
    )


@pytest.mark.asyncio
async def test_same_request_is_not_injected_twice():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "第一句")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    for _ in range(3):
        await service.inject_memories_to_request(request, "sess", "")

    assert len(request.contexts) == 1


@pytest.mark.asyncio
async def test_injected_block_is_not_persisted_into_history():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "第一句")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(request, "sess", "")

    assert request.contexts[-1].get("_no_save") is True


@pytest.mark.asyncio
async def test_inject_returns_every_memory_it_rendered():
    deepmind = _FakeDeepMind()
    deepmind.session_memory_manager.items = [
        _session("a", "第一句"),
        _session("b", "第二句"),
    ]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest(contexts=[{"role": "user", "content": "第一句"}])

    attached = await service.inject_memories_to_request(request, "sess", "")

    assert [memory.id for memory in attached] == ["a", "b"]


def test_context_user_texts_reads_user_blocks_only():
    deepmind = object.__new__(DeepMind)
    deepmind.injection_service = SimpleNamespace(
        _plain_text=DeepMindInjectionService._plain_text
    )
    request = ProviderRequest(
        contexts=[
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "回应"},
            {"role": "user", "content": [TextPart(text="第二句")]},
            {"role": "user", "content": "   "},
        ]
    )

    assert deepmind._context_user_texts(request) == ["第一句", "第二句"]


def test_departed_message_takes_the_memory_with_it():
    deepmind = object.__new__(DeepMind)
    deepmind.injection_service = SimpleNamespace(
        _plain_text=DeepMindInjectionService._plain_text
    )
    session = SessionMemory("sess")
    session.add_memories([_base_memory("a")], bound_message_text="第一句")
    session.add_memories([_base_memory("b")], bound_message_text="第二句")

    departed = session.prune_departed(deepmind._context_user_texts(
        ProviderRequest(contexts=[{"role": "user", "content": "第二句"}])
    ))

    assert [item.id for item in departed] == ["a"]
    assert [item.bound_message_text for item in departed] == ["第一句"]
    assert [item.id for item in session.get_memories()] == ["b"]


def test_reflection_entry_binds_the_message_text_too():
    """反思入仓必须连正文一起绑定，否则条目离场时那次反思没有对话可看。"""
    deepmind = object.__new__(DeepMind)
    deepmind.injection_service = SimpleNamespace(
        _plain_text=DeepMindInjectionService._plain_text
    )

    message_id, text = deepmind._latest_user_message(
        {
            "raw_chat_records": [
                {"role": "user", "content": "第一句", "source_message_id": "m1"},
                {"role": "assistant", "content": "回应"},
                {"role": "user", "content": "第二句", "source_message_id": "m2"},
            ]
        }
    )

    assert message_id == "m2"
    assert text == "第二句"


def test_reflection_entry_falls_back_when_no_user_message():
    deepmind = object.__new__(DeepMind)
    deepmind.injection_service = SimpleNamespace(
        _plain_text=DeepMindInjectionService._plain_text
    )

    assert deepmind._latest_user_message({"raw_chat_records": []}) == ("", "")


def test_history_blocks_follow_their_own_message_even_with_same_text():
    """挂回历史消息的备用路径：重复短句也按位置各挂各的。"""
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(
        contexts=[
            {"role": "user", "content": "好的"},
            {"role": "assistant", "content": "回应"},
            {"role": "user", "content": "好的"},
        ]
    )

    attached = service.attach_history_blocks(
        request,
        [("m1", "好的"), ("m2", "好的"), ("m3", "当前")],
        {"m1": [_memory("a")], "m2": [_memory("b")]},
        current_message_id="m3",
    )

    assert [memory.id for memory in attached] == ["a", "b"]
    assert "记忆a" in request.contexts[1]["content"]
    assert "记忆b" in request.contexts[4]["content"]
    assert request.contexts[1]["content"].startswith("<system_reminder>")
    assert 'id="' not in request.contexts[1]["content"]


def test_truncated_old_message_does_not_steal_the_surviving_one():
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(contexts=[{"role": "user", "content": "好的"}])

    attached = service.attach_history_blocks(
        request,
        [("m1", "好的"), ("m2", "好的"), ("m3", "当前")],
        {"m1": [_memory("a")], "m2": [_memory("b")]},
        current_message_id="m3",
    )

    assert [memory.id for memory in attached] == ["b"]
    assert len(request.contexts) == 2
    assert "记忆b" in request.contexts[1]["content"]
    assert "记忆a" not in request.contexts[1]["content"]


def test_history_blocks_are_not_attached_twice():
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(contexts=[{"role": "user", "content": "第一句"}])
    order = [("m1", "第一句")]
    blocks = {"m1": [_memory("a")]}

    service.attach_history_blocks(request, order, blocks, current_message_id="m9")
    service.attach_history_blocks(request, order, blocks, current_message_id="m9")

    assert len(request.contexts) == 2


@pytest.mark.asyncio
async def test_history_path_splits_current_and_history_memories(monkeypatch):
    """开关打开时，历史消息各挂自己的记忆，当前消息的记忆进末尾块。"""
    monkeypatch.setattr(DeepMindInjectionService, "INJECT_HISTORY_BLOCKS", True)
    deepmind = _FakeDeepMind()
    deepmind.session_memory_manager.items = [
        _session("a", "第一句", bound_message_id="m1"),
        _session("c", "当前", bound_message_id="m9"),
    ]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest(contexts=[{"role": "user", "content": "第一句"}])

    attached = await service.inject_memories_to_request(
        request,
        "sess",
        "",
        message_id="m9",
        message_order=[("m1", "第一句"), ("m9", "当前")],
    )

    assert [memory.id for memory in attached] == ["a", "c"]
    assert "记忆a" in request.contexts[1]["content"]
    tail = _text_of(request)
    assert "记忆c" in tail
    assert "记忆a" not in tail


@pytest.mark.asyncio
async def test_history_path_is_off_by_default():
    """默认只注入末尾一条，历史消息不带记忆块。"""
    deepmind = _FakeDeepMind()
    deepmind.session_memory_manager.items = [
        _session("a", "第一句", bound_message_id="m1"),
        _session("c", "当前", bound_message_id="m9"),
    ]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest(contexts=[{"role": "user", "content": "第一句"}])

    attached = await service.inject_memories_to_request(
        request,
        "sess",
        "",
        message_id="m9",
        message_order=[("m1", "第一句"), ("m9", "当前")],
    )

    assert [memory.id for memory in attached] == ["a", "c"]
    assert len(request.contexts) == 2
    assert "记忆a" in request.contexts[1]["content"]


def test_bound_text_comes_from_request_prompt():
    """当前发言取请求体里那份，跟上下文同一个口径。"""
    deepmind = object.__new__(DeepMind)
    deepmind.injection_service = SimpleNamespace(
        _plain_text=DeepMindInjectionService._plain_text
    )
    request = ProviderRequest(prompt="[群友: 红豆 (ID: 289104862)] (2026-11-09 09:03): 在吗")

    assert deepmind._request_prompt_text(request) == (
        "[群友: 红豆 (ID: 289104862)] (2026-11-09 09:03): 在吗"
    )


def test_bound_text_leaves_when_context_no_longer_contains_it():
    session = SessionMemory("sess")
    session.add_memories([_base_memory("a")], bound_message_text="在吗")

    departed = session.prune_departed(["别的话"])

    assert [item.id for item in departed] == ["a"]


def test_bound_text_matches_the_same_request_text():
    """写入和比对都是请求体里那份文本时，全等就能认出来。"""
    text = "[群友: 红豆 (ID: 289104862)] (2026-11-09 09:03): 在吗"
    session = SessionMemory("sess")
    session.add_memories([_base_memory("a")], bound_message_text=text)

    assert session.prune_departed([text]) == []


def _base_memory(memory_id):
    return SimpleNamespace(id=memory_id, memory_type="event", tags=[], judgment="")
