from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def build_research_handoff_tool(context: Any, plugin_context: Any):
    """Build the upstream HandoffTool for the research assistant."""
    try:
        from astrbot.core.agent.agent import Agent
        from astrbot.core.agent.handoff import HandoffTool
    except Exception as e:
        logger.warning(f"[研究子代理] 跳过 注册原因=当前 AstrBot 不支持 Handoff 子代理 错误={e}")
        return None

    prompt = _load_research_prompt()
    if not prompt:
        logger.warning("[研究子代理] 跳过 注册原因=研究员提示词为空")
        return None

    research_config = _get_research_config(plugin_context)
    persona_id = str(research_config.get("persona_id", "") or "").strip()
    provider_id = str(research_config.get("provider_id", "") or "").strip()

    tools = None
    begin_dialogs = None
    if persona_id:
        persona = _find_persona(context, persona_id)
        if persona:
            persona_prompt = str(persona.get("prompt", "") or "").strip()
            if persona_prompt:
                prompt = f"{prompt}\n\n<用户补充提示>\n{persona_prompt}\n</用户补充提示>"

            tools = _extract_persona_tools(persona)
            begin_dialogs = copy.deepcopy(persona.get("_begin_dialogs_processed"))
            tool_desc = "所有激活工具" if tools is None else f"{len(tools)} 个指定工具"
            logger.info(f"[研究子代理] 使用研究人格 人格={persona_id} 工具={tool_desc}")
        else:
            logger.warning(f"[研究子代理] 研究人格不存在，将使用所有激活工具 人格={persona_id}")

    agent = Agent(
        name="researcher",
        instructions=prompt,
        tools=tools,
    )
    agent.begin_dialogs = begin_dialogs

    handoff = HandoffTool(
        agent=agent,
        tool_description=(
            "将复杂研究、资料查找、长链路分析和需要多步工具调用的任务委派给研究员子代理，"
            "由它完成调查并返回研究报告。"
        ),
    )
    handoff.provider_id = provider_id or None

    provider_desc = provider_id if provider_id else "跟随当前会话"
    logger.info(
        f"[研究子代理] 完成 注册工具=transfer_to_researcher 模型={provider_desc}"
    )
    return handoff


def _get_research_config(plugin_context: Any) -> dict[str, Any]:
    try:
        config = plugin_context.get_all_config()
        research_config = config.get("research_assistant", {})
        return research_config if isinstance(research_config, dict) else {}
    except Exception as e:
        logger.warning(f"[研究子代理] 读取配置失败，将使用默认配置 错误={e}")
        return {}


def _find_persona(context: Any, persona_id: str) -> dict[str, Any] | None:
    persona_manager = getattr(context, "persona_manager", None)
    if persona_manager is None:
        return None

    getter = getattr(persona_manager, "get_persona_v3_by_id", None)
    if callable(getter):
        try:
            persona = getter(persona_id)
            if persona:
                return persona
        except Exception:
            pass

    try:
        personas = getattr(persona_manager, "personas_v3", []) or []
        for persona in personas:
            if not isinstance(persona, dict):
                continue
            candidates = {
                str(persona.get("id", "") or ""),
                str(persona.get("name", "") or ""),
            }
            if persona_id in candidates:
                return persona
    except Exception as e:
        logger.warning(f"[研究子代理] 查找研究人格失败 人格={persona_id} 错误={e}")

    return None


def _extract_persona_tools(persona: dict[str, Any]) -> list[str] | None:
    tools = persona.get("tools")
    if tools is None:
        return None
    if isinstance(tools, list):
        return [str(tool).strip() for tool in tools if str(tool).strip()]
    return []


def _load_research_prompt() -> str:
    prompt_file = Path(__file__).parent / "research_fellow_prompt.md"
    try:
        return prompt_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.error(f"[研究子代理] 提示词文件不存在 路径={prompt_file}")
    except Exception as e:
        logger.error(f"[研究子代理] 加载提示词失败 路径={prompt_file} 错误={e}")
    return ""
