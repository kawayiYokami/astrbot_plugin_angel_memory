"""会话分柜映射（conversation_scope_map）的候选名单。

分柜表结构是 `{键: 柜子名}`，键有两种形态，各自对应一种分柜方式：

- 人格键：人格的 `persona_id`（与运行期 `resolve_memory_scope_with_source`
  使用的匹配键是同一个字符串，见 core/plugin_context.py:356）
- 会话键：完整会话 ID `平台ID:GroupMessage:群号`

生效的是哪一类由 `conversation_scope_mode` 决定；另一类键原样留在表里、
不参与匹配，所以两类并存是合法状态，这里不再做混填拦截，只提供候选。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.web import error_response, json_response

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

# 平台适配器实例 id。aiocqhttp 单实例部署下就是 default，
# 会话 ID 形如 default:GroupMessage:830624502（MessageSession.__str__）
DEFAULT_PLATFORM_ID = "default"
GROUP_MESSAGE_TYPE = "GroupMessage"

PUBLIC_SCOPE = "public"


def _OK(data=None, message=""):
    return json_response({"status": "ok", "message": message, "data": data})


def _ERR(message, status_code=400):
    return error_response(message, status_code=status_code)


def build_session_id(group_id: object, platform_id: str = DEFAULT_PLATFORM_ID) -> str:
    """把群号拼成运行期真正参与精确匹配的会话 ID。"""
    return f"{platform_id}:{GROUP_MESSAGE_TYPE}:{str(group_id or '').strip()}"


class ScopeMapAPI:
    """分柜映射编辑所需的候选名单。"""

    def __init__(self, plugin_context: PluginContext, plugin=None):
        self._ctx = plugin_context
        self._plugin = plugin

    async def get_candidates(self):
        """返回人格、群聊、柜子名三份候选，供前端下拉选择。"""
        scope_map = self._ctx.get_conversation_scope_map()
        personas = await self._persona_names()
        groups = self._groups()
        scopes = sorted({PUBLIC_SCOPE, *scope_map.values()})
        return _OK(
            {
                "personas": personas,
                "groups": groups,
                "scopes": [name for name in scopes if name],
            }
        )

    async def _persona_names(self) -> list[str]:
        """人格名即 persona_id：v3 的 name 字段就是它（astrbot/core/persona_mgr.py:445）。"""
        manager = getattr(self._ctx.astrbot_context, "persona_manager", None)
        names: list[str] = []
        if manager is not None:
            try:
                rows = await manager.get_all_personas()
                for row in rows:
                    name = str(
                        getattr(row, "persona_id", None)
                        or getattr(row, "name", None)
                        or (row.get("persona_id") if isinstance(row, dict) else "")
                        or ""
                    ).strip()
                    if name:
                        names.append(name)
            except Exception as e:
                logger.warning(f"AngelMemory: 读取人格列表失败: {e}")
        if not names:
            # 退到已加载的 v3 人格表，避免数据库不可用时下拉空白
            for item in getattr(manager, "personas_v3", None) or []:
                name = str((item or {}).get("name") or "").strip()
                if name:
                    names.append(name)
        return sorted(set(names))

    def _groups(self) -> list[dict[str, str]]:
        """群列表复用群聊知识库那份实现（群账本 + 配置里出现过的群）。"""
        from .group_knowledge_api import GroupKnowledgeAPI

        try:
            known = GroupKnowledgeAPI(self._ctx)._known_groups()
        except Exception as e:
            logger.warning(f"AngelMemory: 读取群列表失败: {e}")
            known = []
        return [
            {
                "group_id": str(group.get("group_id") or ""),
                "group_name": str(group.get("group_name") or ""),
                "session_id": build_session_id(group.get("group_id")),
            }
            for group in known
            if str(group.get("group_id") or "").strip()
        ]


__all__ = [
    "ScopeMapAPI",
    "build_session_id",
]
