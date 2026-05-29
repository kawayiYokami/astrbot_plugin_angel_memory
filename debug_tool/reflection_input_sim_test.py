"""
反思解耦最小模拟测试

目标：
1. 直接构造 ReflectionInput。
2. 不依赖 AstrMessageEvent。
3. 调用 _execute_async_analysis_task 并验证可成功返回。

运行：
python debug_tool/reflection_input_sim_test.py
"""

from __future__ import annotations

import asyncio
import sys
from types import MethodType
from pathlib import Path
from typing import Any, Dict
import types

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 本地模拟环境可能没有 astrbot，注入最小 stub 以便导入 core.deepmind
if "astrbot" not in sys.modules:
    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    provider_mod = types.ModuleType("astrbot.api.provider")

    class _StubLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    class _StubAstrMessageEvent:
        pass

    class _StubProviderRequest:
        pass

    api_mod.logger = _StubLogger()
    event_mod.AstrMessageEvent = _StubAstrMessageEvent
    provider_mod.ProviderRequest = _StubProviderRequest

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.provider"] = provider_mod

try:
    from core.deepmind import DeepMind, ReflectionInput  # noqa: E402
except Exception as import_error:  # pragma: no cover - 调试脚本降级路径
    print(
        "SKIP: 当前运行环境无法直接导入 core.deepmind。"
        f"请在插件加载上下文中运行。详情: {import_error}"
    )
    raise SystemExit(0)


class _DummyLogger:
    def info(self, msg: str, *args, **kwargs):
        print(msg)

    def warning(self, msg: str, *args, **kwargs):
        print(msg)

    def error(self, msg: str, *args, **kwargs):
        print(msg)

    def debug(self, msg: str, *args, **kwargs):
        pass


class _DummyProviderResp:
    def __init__(self, text: str):
        self.completion_text = text


class _DummyProvider:
    async def text_chat(self, prompt: str):
        # 返回最小可解析 JSON
        return _DummyProviderResp(
            '{"feedback_data":{"useful_memory_ids":[],"new_memories":{},"merge_groups":[]}}'
        )


class _DummyContext:
    def get_provider_by_id(self, provider_id: str):
        return _DummyProvider()


class _DummyJsonParser:
    def extract_json(self, text: str) -> Dict[str, Any]:
        # 直接返回固定结构，避免耦合解析细节
        return {"feedback_data": {"useful_memory_ids": [], "new_memories": {}, "merge_groups": []}}


class _DummyPluginContext:
    def resolve_memory_scope(self, session_id: str) -> str:
        return "public"


class _DummySessionMemoryManager:
    def add_memories_to_session(self, session_id: str, memories):
        return None


class _DummyPromptBuilder:
    def format_chat_records(self, chat_records):
        return ("[用户] 测试消息\n[助理] 测试回复", [])


class _DummySoul:
    def adjust(self, state_code: str, mode: str = "reflect"):
        return None

    def get_state_description(self) -> str:
        return "dummy"


class _FakeDeepMind:
    pass


async def main() -> None:
    # 构建一个“只含反思所需属性”的轻量对象，并绑定 DeepMind 的执行方法
    dm = _FakeDeepMind()
    dm.logger = _DummyLogger()
    dm.prompt_builder = _DummyPromptBuilder()
    dm.json_parser = _DummyJsonParser()
    dm.context = _DummyContext()
    dm.provider_id = "dummy_provider"
    dm.config = {}
    dm.soul = _DummySoul()
    dm.memory_system = None
    dm.plugin_context = _DummyPluginContext()
    dm.session_memory_manager = _DummySessionMemoryManager()

    # 绑定真实实现方法（只验证“入口参数解耦”能力）
    dm._execute_async_analysis_task = MethodType(DeepMind._execute_async_analysis_task, dm)

    reflection_input = ReflectionInput(
        session_id="sim_session_1",
        memory_scope="public",
        latest_user_text="今天我想系统复习线代",
        latest_assistant_text="你可以先从特征值和特征向量开始。",
        secretary_decision={"topic": "线性代数复习"},
        chat_records=[
            {
                "role": "user",
                "content": "今天我想系统复习线代",
                "sender_id": "u1",
                "sender_name": "用户",
                "timestamp": 1000.0,
                "is_processed": False,
                "is_structured_toolcall": False,
            },
            {
                "role": "assistant",
                "content": "你可以先从特征值和特征向量开始。",
                "sender_id": "assistant",
                "sender_name": "助理",
                "timestamp": 1001.0,
                "is_processed": False,
                "is_structured_toolcall": False,
            },
        ],
        memory_context={
            "query": "线代 复习",
            "raw_chat_records": [],
            "raw_memories": [],
            "core_topic": "线性代数复习",
            "memory_id_mapping": {},
        },
    )

    ok = await dm._execute_async_analysis_task(
        reflection_input=reflection_input,
        historical_chat_text_override="[用户] 今天我想系统复习线代\n[助理] 你可以先从特征值和特征向量开始。",
    )
    assert ok is True, "反思执行应返回 True"
    print("PASS: ReflectionInput 模拟测试通过")


if __name__ == "__main__":
    asyncio.run(main())
