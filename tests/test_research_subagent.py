from __future__ import annotations

import sys
import types

from tools.research_subagent import build_research_handoff_tool


class FakeAgent:
    def __init__(self, name, instructions=None, tools=None):
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.begin_dialogs = None


class FakeHandoffTool:
    def __init__(self, agent, tool_description=None):
        self.name = f"transfer_to_{agent.name}"
        self.agent = agent
        self.description = tool_description
        self.provider_id = None


class FakePersonaManager:
    def __init__(self, persona):
        self.persona = persona
        self.personas_v3 = [persona]

    def get_persona_v3_by_id(self, persona_id):
        if persona_id == self.persona.get("id"):
            return self.persona
        return None


class FakeContext:
    def __init__(self, persona):
        self.persona_manager = FakePersonaManager(persona)


class FakePluginContext:
    def __init__(self, config):
        self.config = config

    def get_all_config(self):
        return self.config


def install_fake_handoff_modules(monkeypatch):
    agent_module = types.ModuleType("astrbot.core.agent.agent")
    agent_module.Agent = FakeAgent
    handoff_module = types.ModuleType("astrbot.core.agent.handoff")
    handoff_module.HandoffTool = FakeHandoffTool

    monkeypatch.setitem(sys.modules, "astrbot.core.agent.agent", agent_module)
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.handoff", handoff_module)


def test_build_research_handoff_uses_provider_persona_and_tools(monkeypatch):
    install_fake_handoff_modules(monkeypatch)
    persona = {
        "id": "researcher-persona",
        "name": "研究员",
        "prompt": "只相信可验证资料。",
        "tools": ["web_search", "note_recall"],
        "_begin_dialogs_processed": [{"role": "assistant", "content": "ready"}],
    }
    plugin_context = FakePluginContext(
        {
            "research_assistant": {
                "persona_id": "researcher-persona",
                "provider_id": "deep-research-model",
            }
        }
    )

    tool = build_research_handoff_tool(FakeContext(persona), plugin_context)

    assert tool.name == "transfer_to_researcher"
    assert tool.provider_id == "deep-research-model"
    assert tool.agent.name == "researcher"
    assert tool.agent.tools == ["web_search", "note_recall"]
    assert "只相信可验证资料。" in tool.agent.instructions
    assert tool.agent.begin_dialogs == [{"role": "assistant", "content": "ready"}]


def test_build_research_handoff_skips_when_upstream_handoff_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.agent", None)
    monkeypatch.setitem(sys.modules, "astrbot.core.agent.handoff", None)

    tool = build_research_handoff_tool(
        FakeContext({}),
        FakePluginContext({"research_assistant": {}}),
    )

    assert tool is None
