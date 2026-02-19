import json
from typing import List
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

# 导入必要的服务组件
from ..core.utils.note_context_builder import NoteContextBuilder

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ExpandNoteContextTool(FunctionTool):
    name: str = "note_recall"
    description: str = "回忆笔记的完整内容。当你想起某个笔记片段，想要查看更多详细内容时使用。请提供笔记ID（如'N001'）。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "note_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "你想查看的笔记ID（如 ['N001']）",
                    "minItems": 1,
                    "maxItems": 1
                },
                "token_budget": {
                    "type": "integer",
                    "description": "返回内容的token预算限制（默认2000）。系统会在此预算内尽可能多地返回笔记内容。",
                    "default": 2000,
                    "minimum": 100,
                    "maximum": 10000
                }
            },
            "required": ["note_ids"]
        }
    )

    def __post_init__(self):
        # 初始化日志记录器
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        note_ids: List[str],
        token_budget: int = 2000,
    ) -> str:
        """
        执行笔记上下文扩展

        Args:
            event: 消息事件（包含上下文信息）
            note_ids: 笔记短ID列表
            token_budget: token预算限制

        Returns:
            扩展后的完整笔记上下文
        """
        self.logger.debug(f"{self.name} - LLM 调用: note_ids={note_ids}, token_budget={token_budget}")

        # --- 参数验证 ---
        if not note_ids or len(note_ids) == 0:
            return "错误：必须提供至少一个笔记ID。"

        # --- 获取服务 ---
        if not hasattr(event, 'plugin_context') or event.plugin_context is None:
            self.logger.error(f"{self.name}: 无法从事件中获取 plugin_context。")
            return "错误：内部服务错误，无法获取插件上下文。"

        plugin_context = event.plugin_context

        try:
            if plugin_context.get_config("enable_simple_memory", False):
                return (
                    "当前处于简化记忆模式（enable_simple_memory=true），"
                    "笔记检索功能不可用。请关闭简化记忆模式后再使用该工具。"
                )
            # 获取 note_service
            note_service = plugin_context.get_component("note_service")
            if not note_service:
                raise ValueError("NoteService 未在 PluginContext 中注册。")
        except Exception as e:
            self.logger.error(f"{self.name}: 无法获取 NoteService 实例: {e}")
            return "错误：内部服务错误，无法初始化笔记系统。"

        # --- 获取 note_id_mapping ---
        note_id_mapping = None
        try:
            if hasattr(event, 'angelmemory_context') and event.angelmemory_context:
                context_data = json.loads(event.angelmemory_context)
                note_id_mapping = context_data.get("note_id_mapping", {})

                if not note_id_mapping:
                    self.logger.warning(f"{self.name}: angelmemory_context 中缺少 note_id_mapping")
                    return (
                        "错误：无法获取笔记ID映射信息。这可能是因为：\n"
                        "1. 当前对话中没有检索到任何笔记\n"
                        "2. 系统内部状态异常\n\n"
                        "建议：请先发起一个需要检索笔记的查询，再调用此工具。"
                    )
            else:
                self.logger.error(f"{self.name}: event 中缺少 angelmemory_context")
                return (
                    "错误：无法获取记忆上下文信息。这可能是因为：\n"
                    "1. 当前对话尚未进行记忆检索\n"
                    "2. 记忆系统未正确初始化\n\n"
                    "建议：请先与我进行正常对话，让系统检索相关记忆后，再使用此工具。"
                )
        except json.JSONDecodeError as e:
            self.logger.error(f"{self.name}: 解析 angelmemory_context 失败: {e}")
            return "错误：解析记忆上下文数据失败，请稍后再试。"
        except Exception as e:
            self.logger.error(f"{self.name}: 获取 note_id_mapping 失败: {e}")
            return f"错误：获取笔记映射信息失败：{str(e)}"

        # --- 调用核心服务 ---
        try:
            expanded_context = NoteContextBuilder.expand_context_from_note_ids(
                note_ids=note_ids,
                note_service=note_service,
                total_token_budget=token_budget,
                note_id_mapping=note_id_mapping
            )

            if not expanded_context:
                # 检查是否是ID不存在的问题
                available_ids = list(note_id_mapping.keys()) if note_id_mapping else []
                return (
                    f"未能获取到笔记内容。可能的原因：\n"
                    f"1. 提供的ID不存在：{note_ids}\n"
                    f"2. 当前可用的笔记ID：{available_ids[:10]}{'...' if len(available_ids) > 10 else ''}\n\n"
                    f"建议：请检查提供的ID是否正确，或使用当前对话中出现的笔记ID。"
                )

            # 成功返回
            result_header = (
                f"已为你展开 {len(note_ids)} 个笔记的完整上下文（token预算：{token_budget}）：\n\n"
                f"{'='*60}\n\n"
            )

            self.logger.info(
                f"{self.name}: 成功展开 {len(note_ids)} 个笔记的上下文，"
                f"token预算={token_budget}"
            )

            return result_header + expanded_context

        except Exception as e:
            self.logger.error(f"{self.name}: 展开笔记上下文失败: {e}", exc_info=True)
            return f"展开笔记上下文失败：{str(e)}。请稍后再试或检查提供的笔记ID是否有效。"
