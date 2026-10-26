"""群聊知识库可见性 API。

每个群对五个知识域各有一个三态：auto / manual / hidden。数据存在插件配置
`group_knowledge_map` 里，与原生配置页是同一份数据。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

_OK = lambda data=None, message="": json_response(
    {"status": "ok", "message": message, "data": data}
)
_ERR = lambda message, status_code=400: error_response(message, status_code=status_code)

DOMAIN_LABELS = {
    "gi": "原神",
    "hsr": "崩坏星穹铁道",
    "zzz": "绝区零",
    "personal": "海豹的个人信息",
    "angel": "AI 自写笔记",
}


class GroupKnowledgeAPI:
    """群聊知识库三态表读写。"""

    def __init__(self, plugin_context: "PluginContext", plugin=None):
        self._ctx = plugin_context
        self._plugin = plugin
        self._config = plugin_context.config

    # === 读 ===

    def get_groups(self):
        groups = self._known_groups()
        payload = []
        for group in groups:
            group_id = group["group_id"]
            payload.append(
                {
                    "group_id": group_id,
                    "group_name": group["group_name"],
                    # 统一走一次补全，避免配置里只写了部分域时前端拿到残缺对象
                    "modes": self._ctx.list_knowledge_modes(group_id),
                }
            )
        return _OK(
            {
                "domains": [
                    {"key": key, "label": label} for key, label in DOMAIN_LABELS.items()
                ],
                "modes": ["auto", "manual", "hidden"],
                "groups": payload,
            }
        )

    def _known_groups(self) -> List[Dict[str, str]]:
        """群列表来自群账本，再并上配置里出现过、但账本还没登记的群。"""
        rows: Dict[str, str] = {}
        memory_sql_manager = self._ctx.get_component("memory_sql_manager")
        if memory_sql_manager is not None:
            try:
                for group_id in memory_sql_manager.get_known_group_ids():
                    rows[group_id] = memory_sql_manager.get_group_name(group_id)
            except Exception as e:
                logger.warning(f"AngelMemory: 读取群账本失败: {e}")

        for group_id in self._ctx.get_group_knowledge_map():
            rows.setdefault(group_id, "")

        ordered = sorted(rows.items(), key=lambda item: (item[1] or "", item[0]))
        return [
            {"group_id": group_id, "group_name": name} for group_id, name in ordered
        ]

    # === 写 ===

    async def save_groups(self):
        payload = await request.json(default=None)
        groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(groups, list):
            return _ERR("groups 必须是数组")

        table, error = self.normalize_groups(groups)
        if error:
            return _ERR(error)

        self._ctx.update_config({"group_knowledge_map": json.dumps(table, ensure_ascii=False)})
        try:
            self._config.save_config()
        except Exception as e:
            logger.error(f"AngelMemory: 知识可见性落盘失败: {e}", exc_info=True)
            return _ERR(f"配置落盘失败: {e}")

        return _OK(message="已保存并即时生效")

    @staticmethod
    def normalize_groups(groups: List[Any]) -> tuple[Dict[str, Dict[str, str]], str]:
        """校验并补齐前端提交的三态表；返回 (表, 错误信息)。"""
        from ..pmemory.config import DOMAIN_DIRS, normalize_domain, normalize_mode

        table: Dict[str, Dict[str, str]] = {}
        for item in groups:
            if not isinstance(item, dict):
                return {}, "groups 的每一项都必须是对象"
            group_id = str(item.get("group_id") or "").strip()
            if not group_id:
                return {}, "group_id 不能为空"
            raw_modes = item.get("modes")
            if not isinstance(raw_modes, dict):
                return {}, f"{group_id} 的 modes 必须是对象"

            modes: Dict[str, str] = {}
            for raw_domain, raw_mode in raw_modes.items():
                domain = normalize_domain(raw_domain)
                mode = normalize_mode(raw_mode)
                if not domain or mode is None:
                    return {}, f"{group_id} 存在非法项: {raw_domain}={raw_mode}"
                modes[domain] = mode
            # 补齐没提交的域，避免表里出现半套配置
            for domain in DOMAIN_DIRS:
                modes.setdefault(domain, "manual")
            table[group_id] = modes
        return table, ""


__all__ = ["GroupKnowledgeAPI", "DOMAIN_LABELS"]
