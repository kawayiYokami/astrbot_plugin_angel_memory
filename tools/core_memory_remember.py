from typing import List, Optional
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CoreMemoryRememberTool(FunctionTool):
    name: str = "core_memory_remember"
    description: str = (
        "保存与用户相关、长期有价值的记忆。"
        "支持新增(create)、更新(update)、合并(merge)三种动作。"
        "禁止保存密码、密钥等敏感信息。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "merge"],
                    "description": (
                        "记忆动作。create=新增一条记忆；"
                        "update=更新一条已有记忆；"
                        "merge=把多条旧记忆合并为一条新记忆。"
                    ),
                },
                "memory": {
                    "type": "object",
                    "description": (
                        "目标记忆内容。create 时是新记忆；"
                        "update 时是 source_memory_ids[0] 的新版本；"
                        "merge 时是多条源记忆合并后的结果。"
                    ),
                    "properties": {
                        "judgment": {
                            "type": "string",
                            "description": "记忆本体。用一句独立、清楚、可检索的判断句写出真正要记住的内容。",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["knowledge", "skill", "emotional", "event"],
                            "description": (
                                "记忆类型。knowledge=稳定认知或事实，"
                                "skill=做事方法或能力，"
                                "emotional=稳定情绪偏好或态度，"
                                "event=发生过的事件。"
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "支撑 judgment 的依据或背景，可为空。只写理由、证据、来源。",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "检索锚点列表。每项必须是独立、紧凑、稳定、可检索的词元。",
                            "minItems": 1,
                        },
                    },
                    "required": ["judgment", "memory_type", "tags"],
                },
                "source_memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "源记忆短 ID（记忆看板里的数字编号）。"
                        "create 时传空数组或省略；update 必须正好 1 个；merge 至少 2 个。"
                    ),
                },
            },
            "required": ["action", "memory"],
        }
    )

    def __post_init__(self):
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        action: str,
        memory: dict,
        source_memory_ids: Optional[List[str]] = None,
    ):
        action = str(action or "").strip().lower()
        source_memory_ids = source_memory_ids or []

        # --- 参数验证 ---
        if action not in ("create", "update", "merge"):
            return "错误：action 必须是 create、update 或 merge。"

        if not isinstance(memory, dict):
            return "错误：memory 必须是一个对象。"

        judgment = str(memory.get("judgment", "") or "").strip()
        reasoning = str(memory.get("reasoning", "") or "").strip()
        memory_type = str(memory.get("memory_type", "knowledge") or "knowledge").strip().lower()
        tags = memory.get("tags", [])

        if not judgment:
            return "错误：memory.judgment 不能为空。"
        if not tags or not isinstance(tags, list):
            return "错误：memory.tags 必须是非空数组。"
        if memory_type not in ("knowledge", "skill", "emotional", "event"):
            return f"错误：memory.memory_type 不支持「{memory_type}」，可选：knowledge/skill/emotional/event。"

        # 动作与 source_memory_ids 的约束
        if action == "create" and source_memory_ids:
            return "错误：create 动作不需要 source_memory_ids。"
        if action == "update":
            if len(source_memory_ids) != 1:
                return "错误：update 动作的 source_memory_ids 必须正好 1 个。"
        if action == "merge":
            if len(source_memory_ids) < 2:
                return "错误：merge 动作的 source_memory_ids 至少需要 2 个。"

        # --- 获取服务 ---
        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            self.logger.error(f"{self.name}: 无法从事件中获取 plugin_context。")
            return "错误：内部服务错误，无法获取插件上下文。"

        plugin_context = event.plugin_context

        try:
            memory_runtime = plugin_context.get_component("memory_runtime")
            if not memory_runtime:
                raise ValueError("memory_runtime 未在 PluginContext 中注册。")
            memory_scope = await plugin_context.resolve_memory_scope_from_event(event)
            if not isinstance(memory_scope, str):
                memory_scope = str(memory_scope)
        except Exception as e:
            self.logger.error(f"{self.name}: 无法获取上下文信息: {e}")
            return "错误：无法确定当前会话，记忆写入已拒绝。"

        # --- 短 ID 翻译为完整 ID ---
        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        resolved_source_ids: List[str] = []
        if source_memory_ids:
            if not memory_sql_manager:
                return "错误：记忆管理器不可用，无法解析源记忆 ID。"
            for short_id in source_memory_ids:
                full_id = memory_sql_manager.get_full_id(str(short_id).strip())
                if not full_id:
                    return f"错误：未找到短 ID「{short_id}」对应的记忆，请确认 ID 正确。"
                resolved_source_ids.append(full_id)

        # --- 执行动作 ---
        try:
            if action == "create":
                memory_id = await memory_runtime.remember(
                    memory_type=memory_type,
                    judgment=judgment,
                    reasoning=reasoning,
                    tags=tags,
                    strength=50,
                    is_active=True,
                    memory_scope=memory_scope,
                )
                short_id = ""
                if memory_sql_manager:
                    short_id = memory_sql_manager.get_short_id(str(memory_id))
                self.logger.info(
                    f"{self.name}: create 成功 short_id={short_id} judgment=「{judgment[:30]}」"
                )
                return f"已记住：「{judgment}」（ID: {short_id or memory_id}）"

            elif action in ("update", "merge"):
                # 构建 memory_actions 格式，复用 feedback 管线
                # 本插件下游动作名：updata（兼容历史拼写）
                downstream_action = "updata" if action == "update" else "merge"
                memory_actions = [
                    {
                        "action": downstream_action,
                        "source_memory_ids": resolved_source_ids,
                        "memory": {
                            "type": memory_type,
                            "judgment": judgment,
                            "reasoning": reasoning,
                            "tags": tags,
                        },
                    }
                ]
                created_memories = await memory_runtime.feedback(
                    useful_memory_ids=[],
                    recalled_memory_ids=[],
                    memory_actions=memory_actions,
                    memory_scope=memory_scope,
                )
                # 为新记忆注册短 ID
                new_short_id = ""
                if created_memories and memory_sql_manager:
                    new_id = str(getattr(created_memories[0], "id", "") or "").strip()
                    if new_id:
                        new_short_id = memory_sql_manager.register_short_id(new_id)

                source_display = ", ".join(source_memory_ids)
                verb = "更新" if action == "update" else "合并"
                self.logger.info(
                    f"{self.name}: {action} 成功 sources=[{source_display}] "
                    f"new_short_id={new_short_id} judgment=「{judgment[:30]}」"
                )
                return (
                    f"已{verb}记忆 [{source_display}] → 新记忆「{judgment}」"
                    f"（ID: {new_short_id or '已生成'}）"
                )

        except Exception as e:
            self.logger.error(f"{self.name}: {action} 失败: {e}", exc_info=True)
            return f"记忆{action}失败：{str(e)}。请稍后再试。"
