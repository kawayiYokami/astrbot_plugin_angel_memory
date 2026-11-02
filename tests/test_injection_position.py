"""记忆注入：短期记忆按消息绑定、进入侧去重、消息离场即离开、按绑定挂位置。"""

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

    def prune_absent_messages_from_session(self, session_id, alive_message_ids):
        alive = {str(mid) for mid in (alive_message_ids or [])}
        departed = [
            item
            for item in self.items
            if not item.bound_message_id or item.bound_message_id not in alive
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


def _session(memory_id, bound_message_id):
    return SimpleNamespace(
        id=memory_id,
        bound_message_id=bound_message_id,
        bound_message_text="",
        memory=_memory(memory_id),
    )


def _text_of(request):
    return "\n".join(
        str(getattr(part, "text", "")) for part in request.extra_user_content_parts
    )


@pytest.mark.asyncio
async def test_current_block_carries_memories_bound_to_current_message():
    deepmind = _FakeDeepMind([_memory("a"), _memory("z")])
    deepmind.session_memory_manager.items = [_session("a", "msg-1"), _session("z", "msg-0")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request, "sess", "", message_id="msg-1"
    )

    text = _text_of(request)
    assert "记忆a" in text
    assert "记忆z" not in text
    assert text.startswith("<system_reminder>")
    assert 'id="' not in text
    # 行为守则进系统提示词，不跟着对话消息走
    assert "<instruction>" not in text
    assert "【关于你记得的事】" in (request.system_prompt or "")
    assert request.contexts == []


@pytest.mark.asyncio
async def test_nothing_to_inject_means_no_block():
    deepmind = _FakeDeepMind([])
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request, "sess", "", message_id="msg-1"
    )

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
        message_id="msg-1",
    )

    text = _text_of(request)
    assert "<soul_state>" in text
    assert "<memories>" not in text


@pytest.mark.asyncio
async def test_memories_come_before_soul_state():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "msg-1")]
    deepmind.config = SimpleNamespace(enable_soul_system=True)
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request,
        "sess",
        "",
        soul_state_values={"joy": 0.6, "surprise": 0.5, "anger": 0.4, "disgust": 0.7},
        message_id="msg-1",
    )

    text = _text_of(request)
    assert text.index("<memories>") < text.index("<soul_state>")
    # 状态是最后一块，紧挨着收尾标签
    assert text.index("</soul_state>") + len("</soul_state>\n") == text.index(
        "</system_reminder>"
    )


@pytest.mark.asyncio
async def test_same_message_is_not_injected_twice():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "msg-1")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    for _ in range(3):
        await service.inject_memories_to_request(
            request, "sess", "", message_id="msg-1"
        )

    assert len(request.extra_user_content_parts) == 1


@pytest.mark.asyncio
async def test_injected_block_is_not_persisted_into_history():
    deepmind = _FakeDeepMind([_memory("a")])
    deepmind.session_memory_manager.items = [_session("a", "msg-1")]
    service = DeepMindInjectionService(deepmind)
    request = ProviderRequest()

    await service.inject_memories_to_request(
        request, "sess", "", message_id="msg-1"
    )

    block = request.extra_user_content_parts[0]
    assert block.model_dump_for_context().get("_no_save") is True


def test_history_blocks_follow_their_own_message_even_with_same_text():
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

    assert len(attached) == 2
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


def test_history_block_matches_context_whose_content_is_part_objects():
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(
        contexts=[{"role": "user", "content": [TextPart(text="好的")]}]
    )

    attached = service.attach_history_blocks(
        request,
        [("m1", [TextPart(text="好的")])],
        {"m1": [_memory("a")]},
        current_message_id="m9",
    )

    assert [memory.id for memory in attached] == ["a"]
    assert len(request.contexts) == 2
    assert "记忆a" in request.contexts[1]["content"]


def test_history_blocks_are_not_attached_twice():
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(contexts=[{"role": "user", "content": "第一句"}])
    order = [("m1", "第一句")]
    blocks = {"m1": [_memory("a")]}

    service.attach_history_blocks(request, order, blocks, current_message_id="m9")
    service.attach_history_blocks(request, order, blocks, current_message_id="m9")

    assert len(request.contexts) == 2


@pytest.mark.asyncio
async def test_inject_returns_only_the_memories_actually_attached():
    deepmind = _FakeDeepMind()
    deepmind.session_memory_manager.items = [
        _session("a", "m1"),
        _session("b", "m2"),
        _session("c", "m9"),
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


def test_history_block_is_dropped_when_bound_message_is_gone():
    service = DeepMindInjectionService(_FakeDeepMind())
    request = ProviderRequest(contexts=[{"role": "user", "content": "还在的消息"}])

    attached = service.attach_history_blocks(
        request,
        [("m-old", "已经丢失的消息")],
        {"m-old": [_memory("a")]},
        current_message_id="m-new",
    )

    assert attached == []
    assert len(request.contexts) == 1


def test_entering_again_does_not_rebind_the_memory():
    session = SessionMemory("sess")

    session.add_memories([_base_memory("a")], bound_message_id="m1", bound_message_text="第一句")
    session.add_memories([_base_memory("a")], bound_message_id="m2", bound_message_text="第二句")

    items = session.get_memories()
    assert [item.id for item in items] == ["a"]
    assert items[0].bound_message_id == "m1"
    assert items[0].bound_message_text == "第一句"


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


def test_prune_removes_memories_whose_message_is_gone():
    session = SessionMemory("sess")
    session.add_memories([_base_memory("a")], bound_message_id="m1", bound_message_text="第一句")
    session.add_memories([_base_memory("b")], bound_message_id="m2", bound_message_text="第二句")

    departed = session.prune_absent_messages({"m2"})

    assert [item.id for item in departed] == ["a"]
    assert [item.bound_message_text for item in departed] == ["第一句"]
    assert [item.memory.id for item in departed] == ["a"]
    assert [item.id for item in session.get_memories()] == ["b"]


def _base_memory(memory_id):
    return SimpleNamespace(id=memory_id, memory_type="event", tags=[], judgment="")
