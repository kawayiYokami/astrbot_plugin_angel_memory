"""没配核心模型（provider_id）时，召回与注入照跑，只有需要模型的整理步骤被跳过。

背景：原先 organize_and_inject_memories 在检索之前就因 provider_id 为空而 return，
把不需要模型的召回、记忆注入、笔记注入、灵魂注入、会话缓存一起停掉了。
"""

import asyncio
import json
import logging
from types import SimpleNamespace

from astrbot_plugin_angel_memory.core.deepmind import DeepMind


class _StubSqlManager:
    def upsert_user(self, *args, **kwargs):
        return None

    def upsert_group(self, *args, **kwargs):
        return None

    def build_id_mapping(self, memory_ids):
        return {str(mid): str(mid) for mid in memory_ids}


class _StubPluginContext:
    """只提供 organize_and_inject_memories 真正用到的那几个口子。"""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self._memory_sql = _StubSqlManager()
        self._memory_store = object()

    def get_config(self, key=None, default=None, *, umo=None):
        if key is None:
            return dict(self.config)
        return self.config.get(key, default)

    def get_llm_provider_id(self):
        return self.config.get("provider_id", "")

    def get_component(self, name):
        return {
            "memory_sql_manager": self._memory_sql,
            "memory_store": self._memory_store,
        }.get(name)

    async def resolve_memory_scope_from_event(self, event):
        return "public"


class _StubPromptBuilder:
    def format_chat_records(self, records):
        return "", []


class _StubRetrievalService:
    """检索链的替身：记下被调用，并返回一条记忆与一条笔记。"""

    def __init__(self):
        self.calls = []

    async def retrieve_memories_and_notes(self, event, query):
        self.calls.append({"event": event, "query": query})
        memory = SimpleNamespace(id="mem-1", to_dict=lambda: {"id": "mem-1"})
        return {
            "long_term_memories": [memory],
            "candidate_notes": [
                {
                    "content": "笔记正文",
                    "metadata": {"note_id": 7, "source_file_path": "note.md"},
                }
            ],
            "note_id_mapping": {},
            "memory_id_mapping": {},
            "secretary_decision": {},
            "core_topic": "话题",
            "errors": [],
        }


class _StubInjectionService:
    """注入链的替身：记下真正传给注入的内容。"""

    def __init__(self):
        self.calls = []

    @staticmethod
    def _plain_text(content):
        return str(content or "")

    async def inject_memories_to_request(
        self,
        request,
        session_id,
        note_context,
        soul_state_values=None,
        message_id="",
        message_order=None,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "note_context": note_context,
                "soul_state_values": soul_state_values,
                "message_id": message_id,
                "message_order": message_order,
            }
        )
        return []


class _StubSessionMemoryManager:
    def __init__(self, departed=None):
        self.departed = list(departed or [])
        self.added = []
        self.pruned = []
        self.get_calls = 0

    def add_memories_to_session(self, session_id, memories, **kwargs):
        self.added.append((session_id, list(memories)))

    def prune_departed_from_session(self, session_id, alive_texts):
        self.pruned.append((session_id, list(alive_texts or [])))
        return list(self.departed)

    def get_session_memories(self, session_id):
        self.get_calls += 1
        return []


class _StubUserProfileService:
    def __init__(self):
        self.refreshed = []
        self.store_set = 0

    def set_memory_store(self, store):
        self.store_set += 1

    async def refresh_session_profiles(self, **kwargs):
        self.refreshed.append(kwargs)
        return []


class _StubSoul:
    def get_scores(self):
        return {"joy": 0.6, "anger": 0.4, "surprise": 0.5, "disgust": 0.5}


def _event(text="今天聊了什么", message_id="msg-1", sender_id="user-1"):
    return SimpleNamespace(
        unified_msg_origin="group:123",
        message_obj=SimpleNamespace(message_id=message_id),
        get_message_outline=lambda: text,
        sender_id=sender_id,
        sender_name="红豆",
        get_platform_name=lambda: "qq",
        get_group_id=lambda: "123",
    )


def _deepmind(provider_id="", departed=None):
    """跳过重量级初始化，只装配 organize_and_inject_memories 用到的属性。"""
    deepmind = object.__new__(DeepMind)
    deepmind.plugin_context = _StubPluginContext({"provider_id": provider_id})
    deepmind.config = SimpleNamespace()
    deepmind.logger = logging.getLogger("recall-without-provider-test")
    deepmind._provider_id_fallback = provider_id
    deepmind.prompt_builder = _StubPromptBuilder()
    deepmind.retrieval_service = _StubRetrievalService()
    deepmind.injection_service = _StubInjectionService()
    deepmind.session_memory_manager = _StubSessionMemoryManager(departed)
    deepmind.user_profile_service = _StubUserProfileService()
    deepmind.soul = _StubSoul()
    deepmind.memory_system = object()
    deepmind.reflection_trace = None
    deepmind._reflection_tick_task = None
    return deepmind


def test_recall_and_injection_run_without_provider():
    """没配 provider_id 时，检索、记忆注入、笔记注入、灵魂注入、会话缓存全部照常执行。"""
    deepmind = _deepmind(provider_id="")
    event = _event()
    request = SimpleNamespace(
        prompt="今天聊了什么",
        contexts=[{"role": "user", "content": "早先的一句话"}],
    )

    asyncio.run(deepmind.organize_and_inject_memories(event, request))

    assert deepmind.retrieval_service.calls, "检索没被执行"
    assert deepmind.injection_service.calls, "注入没被执行"
    assert deepmind.session_memory_manager.added, "记忆没进会话缓存"

    # 当前这条发言不在 contexts 里，必须单独补进存活集合，否则刚绑上的记忆会被自己清掉
    alive = deepmind.session_memory_manager.pruned[0][1]
    assert "今天聊了什么" in alive, "当前发言没进存活集合"
    assert "早先的一句话" in alive, "上下文里的用户消息没进存活集合"

    injected = deepmind.injection_service.calls[0]
    assert "笔记切片" in injected["note_context"], "笔记上下文没进注入"
    assert injected["soul_state_values"] == {
        "joy": 0.6,
        "anger": 0.4,
        "surprise": 0.5,
        "disgust": 0.5,
    }, "灵魂状态没进注入"
    context = json.loads(event.angelmemory_context)
    assert context["recall_query"], "召回查询没写进上下文"
    assert context["core_topic"] == "话题"


def test_model_dependent_steps_skipped_without_provider():
    """没配 provider_id 时，需要模型的整理步骤被跳过，且不抛异常。"""
    departed = [SimpleNamespace(memory=SimpleNamespace(id="mem-1"))]
    deepmind = _deepmind(provider_id="", departed=departed)
    event = _event()

    reflect_calls = []

    async def _fake_reflect(*args, **kwargs):
        reflect_calls.append(args)

    deepmind._reflect_departed_memories = _fake_reflect

    asyncio.run(deepmind.organize_and_inject_memories(event, SimpleNamespace()))
    assert not reflect_calls, "没配模型时不该提交离场反思"
    assert deepmind.session_memory_manager.pruned, "短期记忆清理不该被一起跳过"

    buffered = []

    async def _fake_buffer(*args, **kwargs):
        buffered.append("buffer")

    async def _fake_trigger(*args, **kwargs):
        buffered.append("trigger")

    deepmind._record_reflection_turn = _fake_buffer
    deepmind._trigger_reflection_if_needed = _fake_trigger

    asyncio.run(
        deepmind.async_analyze_and_update_memory(event, SimpleNamespace(completion_text=""))
    )
    assert not buffered, "没配模型时不该进入记忆整理链"


def test_behaviour_unchanged_with_provider():
    """配了 provider_id 时，行为与改动前一致：召回照跑，整理链也照走。"""
    departed = [SimpleNamespace(memory=SimpleNamespace(id="mem-1"))]
    deepmind = _deepmind(provider_id="provider-a", departed=departed)
    event = _event()

    reflect_calls = []

    async def _fake_reflect(*args, **kwargs):
        reflect_calls.append(args)

    deepmind._reflect_departed_memories = _fake_reflect

    asyncio.run(deepmind.organize_and_inject_memories(event, SimpleNamespace()))
    assert deepmind.retrieval_service.calls, "配了模型时检索照常"
    assert deepmind.injection_service.calls, "配了模型时注入照常"
    assert reflect_calls, "配了模型时离场反思照常提交"

    buffered = []

    async def _fake_buffer(*args, **kwargs):
        buffered.append("buffer")

    async def _fake_trigger(*args, **kwargs):
        buffered.append("trigger")

    async def _fake_tick_task(*args, **kwargs):
        return None

    deepmind._record_reflection_turn = _fake_buffer
    deepmind._trigger_reflection_if_needed = _fake_trigger
    deepmind._ensure_reflection_tick_task = _fake_tick_task

    asyncio.run(
        deepmind.async_analyze_and_update_memory(event, SimpleNamespace(completion_text=""))
    )
    assert buffered == ["buffer", "trigger"], "配了模型时整理链照常执行"
