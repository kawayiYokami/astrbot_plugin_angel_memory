from typing import List
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

# 导入必要的服务组件和模型
from ..llm_memory.service.cognitive_service import CognitiveService

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CoreMemoryRememberTool(FunctionTool):
    name: str = "core_memory_remember"
    description: str = "当你认为接收到的信息**极其重要、具有长期价值**，且**不应被遗忘**时，调用此工具将其永久保存为主动记忆。例如，用户反复强调的偏好、核心事实、关键原则，或你自身推导出的长期有效结论。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "description": "核心论断，50字符内，不换行。例：'用户对猫毛过敏'",
                },
                "reasoning": {
                    "type": "string",
                    "description": "论证或背景，50字符内，不换行。例：'用户明确提及过敏史'",
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "分类标签，至少一个。例：['用户偏好', '健康']",
                    "minItems": 1
                },
                "memory_type": {
                    "type": "string",
                    "description": "记忆类型：knowledge(知识)/event(事件)/skill(技能)",
                    "enum": ["knowledge", "event", "skill"],
                    "default": "knowledge"
                },
                "strength": {
                    "type": "integer",
                    "description": "重要性评分(1-100)，影响回忆概率",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["judgment", "reasoning", "tags", "strength"],
        }
    )

    def __post_init__(self):
        # 初始化日志记录器
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        judgment: str,
        reasoning: str,
        tags: List[str],
        strength: int,
        memory_type: str = "knowledge",
    ):
        self.logger.debug(f"{self.name} - LLM 调用: judgment='{judgment[:50]}...', strength={strength}")

        # --- 验证逻辑 ---
        MAX_LENGTH = 50
        if len(judgment) > MAX_LENGTH:
            return f"错误：`judgment` (核心论断)超过 {MAX_LENGTH} 字符的长度限制。请精简你的论断。"
        if '\n' in judgment:
            return "错误：`judgment` (核心论断)不允许包含换行符。请将信息压缩到一行。"
        if len(reasoning) > MAX_LENGTH:
            return f"错误：`reasoning` (论证或背景)超过 {MAX_LENGTH} 字符的长度限制。请提供更简短的解释。"
        if '\n' in reasoning:
            return "错误：`reasoning` (论证或背景)不允许包含换行符。请将信息压缩到一行。"
        for tag in tags:
            if len(tag) > MAX_LENGTH:
                return f"错误：标签 '{tag}' 超过 {MAX_LENGTH} 字符的长度限制。请使用更短的标签。"
            if '\n' in tag:
                return f"错误：标签 '{tag}' 不允许包含换行符。"

        # --- 获取服务 ---
        if not hasattr(event, 'plugin_context') or event.plugin_context is None:
            self.logger.error(f"{self.name}: 无法从事件中获取 plugin_context。")
            return "错误：内部服务错误，无法获取插件上下文。"

        plugin_context = event.plugin_context

        try:
            cognitive_service: CognitiveService = plugin_context.get_component("cognitive_service")
            if not cognitive_service:
                raise ValueError("CognitiveService 未在 PluginContext 中注册。")
            conversation_id = plugin_context.get_event_conversation_id(event)
            memory_scope = plugin_context.resolve_memory_scope(conversation_id)
        except Exception as e:
            self.logger.error(f"{self.name}: 无法获取上下文信息或 CognitiveService 实例: {e}")
            return "错误：无法确定当前会话ID，记忆写入已拒绝（严格隔离模式）。"

        # --- 调用服务 ---
        try:
            memory_id = await cognitive_service.remember(
                memory_type=memory_type,
                judgment=judgment,
                reasoning=reasoning,
                tags=tags,
                strength=strength,
                is_active=True,
                memory_scope=memory_scope,
            )
            self.logger.info(f"{self.name}: 成功铭记记忆。ID: {memory_id}, judgment='{judgment[:50]}...', strength={strength}")
            return f"好的，我已将关于「{judgment}」的论断铭记于心，重要性设为 {strength}。记忆ID: {memory_id}"
        except Exception as e:
            self.logger.error(f"{self.name}: 调用认知服务铭记记忆失败: {e}", exc_info=True)
            return f"铭记记忆失败：{str(e)}。请稍后再试或检查输入。"
