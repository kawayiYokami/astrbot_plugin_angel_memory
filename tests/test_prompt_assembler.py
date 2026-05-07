from __future__ import annotations

import logging
import sys
import types
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")
    event = types.ModuleType("astrbot.api.event")

    class AstrMessageEvent:
        pass

    event.AstrMessageEvent = AstrMessageEvent

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

LLM_MEMORY_PACKAGE = f"{PACKAGE_NAME}.llm_memory"
if LLM_MEMORY_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory")]
    sys.modules[LLM_MEMORY_PACKAGE] = package

LLM_MEMORY_PROMPTS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.prompts"
if LLM_MEMORY_PROMPTS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_PROMPTS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "prompts")]
    sys.modules[LLM_MEMORY_PROMPTS_PACKAGE] = package

LLM_MEMORY_UTILS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.utils"
if LLM_MEMORY_UTILS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_UTILS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "utils")]
    sys.modules[LLM_MEMORY_UTILS_PACKAGE] = package

from astrbot_plugin_angel_memory.llm_memory.prompts.prompt_assembler import PromptAssembler


def test_prompt_assembler_contains_critical_sections():
    prompt = PromptAssembler.build_memory_system_guide()

    assert "memory_actions" in prompt
    assert "updata" in prompt
    assert "人物画像优先规则" in prompt
    assert "禁止画像内容" in prompt
    assert prompt.count("```json") == 1


def test_prompt_assembler_reads_all_declared_sections():
    section_paths = PromptAssembler.get_section_paths()

    assert len(section_paths) == len(PromptAssembler.SECTION_FILES)
    assert all(path.exists() for path in section_paths)
