import random
from typing import Optional
from pathlib import Path
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.star import Context
from astrbot.core.agent.tool import ToolSet
from dataclasses import dataclass, field

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ResearchTool(FunctionTool):
    """
    智能研究助手工具

    该工具通过调用 tool_loop_agent 启动一个独立的研究 Agent，
    使用用户预先配置的研究人格和工具集进行深度研究。
    """

    name: str = "research_topic"
    description: str = (
        "如果你想要学习或者研究某个论题，请使用我，生成一个探索任务。我将启动一个研究Agent，它会一边学习一边研究，最后生成一个探索报告。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "详细的研究论题描述。必须包含：\n"
                        "- 论题标题\n"
                        "- 背景信息（研究对象是什么，为什么重要）\n"
                        "- 方法建议（如何查找相关资料，使用什么工具）\n"
                        "- 研究目标（需要弄清楚什么）\n"
                        "- 报告要求（期望包含哪些内容）"
                    ),
                },
                "complexity": {
                    "type": "string",
                    "description": (
                        "任务复杂度评估，决定研究深度：\n"
                        "- 'normal': 通用任务，标准深度研究（默认）\n"
                        "- 'complex': 复杂任务，需要深度全面研究"
                    ),
                    "enum": ["normal", "complex"],
                    "default": "normal"
                },
            },
            "required": ["topic"],
        }
    )

    # 不定义 __init__，使用 dataclass 的自动初始化
    # 在实例创建后通过外部设置 context
    context: Context = field(init=False, repr=False, default=None)

    def set_context(self, context: Context):
        """
        设置 context（在创建实例后调用）

        Args:
            context: AstrBot context对象
        """
        self.context = context
        self.logger = logger

    async def run(self, event: AstrMessageEvent, topic: str, complexity: str = "normal") -> str:
        """
        执行研究任务

        Args:
            event: 消息事件对象
            topic: 研究论题
            complexity: 任务复杂度 (simple/normal/complex)

        Returns:
            str: 研究报告或错误信息
        """
        # 根据复杂度映射 max_steps
        complexity_map = {
            "normal": 30,
            "complex": 100
        }
        max_steps = complexity_map.get(complexity, 30)

        logger.info(f"ResearchTool: 开始研究论题 - {topic} (复杂度: {complexity}, max_steps: {max_steps})")

        try:
            # 0. 发送启动消息
            plugin_context = event.plugin_context
            if plugin_context:
                config = plugin_context.get_all_config()
                research_config = config.get('research_assistant', {})
                start_messages_str = research_config.get('start_messages', '')
                if start_messages_str:
                    messages = [msg.strip() for msg in start_messages_str.split('|') if msg.strip()]
                    if messages:
                        message_to_send = random.choice(messages)
                        try:
                            message_chain = MessageChain().message(message_to_send)
                            await self.context.send_message(event.unified_msg_origin, message_chain)
                            logger.debug(f"ResearchTool: 已发送启动消息 - '{message_to_send}'")
                        except Exception as e:
                            logger.warning(f"ResearchTool: 发送启动消息失败 - {e}")

            # 1. 获取插件配置
            if not hasattr(event, 'plugin_context') or event.plugin_context is None:
                logger.error("ResearchTool: 无法从事件中获取 plugin_context")
                return "❌ 错误：内部服务错误，无法获取插件上下文。"

            plugin_context = event.plugin_context
            config = plugin_context.get_all_config()
            research_config = config.get('research_assistant', {})

            # 2. 加载系统提示词
            system_prompt = self._load_system_prompt()
            if not system_prompt:
                logger.error("ResearchTool: 无法加载系统提示词")
                return "❌ 错误：无法加载研究员系统提示词文件。"

            # 3. 检查并合并人格配置
            persona_id = research_config.get('persona_id', '').strip()
            if persona_id:
                # 获取人格信息
                persona = await self._get_persona(persona_id)
                if persona:
                    persona_prompt = persona.get("prompt", "")
                    if persona_prompt:
                        # 将人格提示词补充到系统提示词中
                        system_prompt = f"{system_prompt}\n\n<用户补充提示>\n{persona_prompt}\n</用户补充提示>"
                        logger.info(f"ResearchTool: 已补充人格 '{persona_id}' 的提示词")

                    # 使用人格配置的工具集
                    persona_toolset = await self._get_persona_toolset(persona)
                    if persona_toolset and len(persona_toolset.tools) > 0:
                        active_tools = persona_toolset
                        logger.info(f"ResearchTool: 使用人格 '{persona_id}' 的工具集 ({len(active_tools.tools)} 个工具)")
                    else:
                        logger.warning(f"ResearchTool: 人格 '{persona_id}' 未配置工具，将使用全局工具")
                        active_tools = await self._get_global_tools()
                else:
                    logger.warning(f"ResearchTool: 找不到人格 '{persona_id}'，将使用默认配置")
                    active_tools = await self._get_global_tools()
            else:
                # 未配置人格，使用全局工具集
                active_tools = await self._get_global_tools()

            # 4. 确定使用的 Provider ID
            configured_provider = research_config.get('provider_id', '').strip()
            if configured_provider:
                provider_id = configured_provider
                logger.info(f"ResearchTool: 使用配置的提供商 '{provider_id}'")
            else:
                # 使用当前会话的 Provider ID
                umo = event.unified_msg_origin
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                logger.info(f"ResearchTool: 使用当前会话提供商 '{provider_id}'")

            # 5. 注入步数限制提示
            step_limit_prompt = f"""
<步数与效率>
## 步数限制与效率意识 (非常重要)
**你必须在 {max_steps} 次工具调用之内完成所有研究和最终汇报。** 这是一个硬性指标。
<思考>
- **每次调用工具前**：在心中默念：“这是第 X 次调用，我还有 Y 次机会。”
- **资源规划**：你的总步数是 {max_steps} 步，你可能只有几次搜索和阅读的机会，剩下的需要用来记忆和总结。
- **效率是关键**：不要浪费任何一次调用。如果一次搜索没有好的结果，立即调整策略，而不是盲目重复。
</思考>
</步数与效率>
"""
            system_prompt = f"{system_prompt}\n{step_limit_prompt}"

            # 6. 启动 Tool Loop Agent
            logger.info("ResearchTool: 启动研究 Agent 循环...")

            llm_response = await self.context.tool_loop_agent(
                event=event,
                chat_provider_id=provider_id,
                prompt=f"请研究以下论题：{topic}",
                system_prompt=system_prompt,
                tools=active_tools,
                max_steps=max_steps,
            )

            logger.info("ResearchTool: 研究任务完成")

            # 添加系统提示，提醒 LLM 需要转述给用户
            system_reminder = (
                "<system_reminder>\n"
                "⚠️ 重要提示：工具调用结果用户是看不到的！\n"
                "请你根据以下研究报告和上下文，用清晰、简洁的方式转述给用户。\n"
                "你可以：\n"
                "- 提取关键信息和核心发现\n"
                "- 总结主要结论\n"
                "- 重新组织信息使其更易理解\n"
                "- 根据用户的原始问题，突出最相关的内容\n"
                "</system_reminder>\n\n"
                "=== 研究报告 ===\n"
            )

            return f"{system_reminder}{llm_response.completion_text}"

        except Exception as e:
            logger.error(f"ResearchTool: 执行过程中发生错误 - {e}", exc_info=True)
            return f"❌ 研究任务执行失败：{str(e)}"

    async def _get_persona(self, persona_id: str) -> Optional[dict]:
        """
        获取指定人格信息

        Args:
            persona_id: 人格名称

        Returns:
            Optional[dict]: 人格信息，如果不存在则返回 None
        """
        try:
            personas = self.context.persona_manager.personas_v3
            persona = next((p for p in personas if p["name"] == persona_id), None)
            return persona
        except Exception as e:
            logger.error(f"ResearchTool: 获取人格信息失败 - {e}")
            return None

    async def _get_persona_toolset(self, persona: dict) -> ToolSet:
        """
        获取人格配置的工具集

        Args:
            persona: 人格信息

        Returns:
            ToolSet: 人格的工具集
        """
        try:
            tool_manager = self.context.get_llm_tool_manager()

            if persona.get("tools") is None:
                # None 表示使用所有激活的工具
                toolset = tool_manager.get_full_tool_set()
                # 过滤未激活的工具
                for tool in toolset.tools[:]:
                    if not tool.active:
                        toolset.remove_tool(tool.name)
            elif isinstance(persona["tools"], list) and len(persona["tools"]) > 0:
                # 使用人格配置的特定工具列表
                toolset = ToolSet()
                for tool_name in persona["tools"]:
                    tool = tool_manager.get_func(tool_name)
                    if tool and tool.active:
                        toolset.add_tool(tool)
            else:
                # 空列表表示不使用任何工具
                toolset = ToolSet()

            return toolset

        except Exception as e:
            logger.error(f"ResearchTool: 获取工具集失败 - {e}")
            return ToolSet()

    async def _get_global_tools(self) -> ToolSet:
        """
        获取全局工具集

        Returns:
            ToolSet: 全局激活的工具集
        """
        try:
            tool_manager = self.context.get_llm_tool_manager()
            all_tools = tool_manager.get_full_tool_set()
            # 过滤未激活的工具
            active_tools = ToolSet()
            for tool in all_tools.tools:
                if tool.active:
                    active_tools.add_tool(tool)

            logger.info(f"ResearchTool: 获取到 {len(active_tools.tools)} 个全局工具")
            return active_tools
        except Exception as e:
            logger.error(f"ResearchTool: 获取全局工具集失败 - {e}")
            return ToolSet()

    def _load_system_prompt(self) -> Optional[str]:
        """
        从文件加载系统提示词

        Returns:
            Optional[str]: 系统提示词内容，如果加载失败则返回 None
        """
        try:
            # 获取当前文件所在目录
            current_dir = Path(__file__).parent
            prompt_file = current_dir / "research_fellow_prompt.md"

            if not prompt_file.exists():
                logger.error(f"ResearchTool: 提示词文件不存在: {prompt_file}")
                return None

            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.debug(f"ResearchTool: 成功加载系统提示词 ({len(content)} 字符)")
            return content

        except Exception as e:
            logger.error(f"ResearchTool: 加载系统提示词失败 - {e}")
            return None
