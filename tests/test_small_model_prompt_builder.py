"""反思提示词拆分测试：system 放固定指南、user 放本次数据。"""

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

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)


def test_reflection_prompts_split_guide_and_data():
    system_prompt, user_prompt = (
        SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
            historical_query="红豆：你今天好吗",
            main_llm_response="我很好",
            raw_memories=[],
            memory_id_mapping={},
        )
    )

    assert system_prompt.strip()
    assert "红豆：你今天好吗" not in system_prompt
    assert "# 本次对话数据" not in system_prompt

    assert user_prompt.startswith("# 本次对话数据")
    assert "红豆：你今天好吗" in user_prompt
    assert "我很好" in user_prompt
    assert "现在请开始分析并输出JSON结果。" in user_prompt
