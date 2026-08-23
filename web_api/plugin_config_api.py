"""插件全局配置（_conf_schema.json）读取与保存 API。

数据源是宿主的 AstrBotConfig 对象（dict 子类），由 PluginContext 持有同一引用。
保存时经 PluginContext.update_config 原地更新键值并刷新内部缓存，
运行时组件动态读取即时生效，无需重载插件；与原生 dashboard 配置页共存，
互为同一份数据的两个编辑入口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

# schema type -> 校验/转换规则；bool 必须先于 int 判断（bool 是 int 子类）
_SCHEMA_TYPES = {"string", "text", "int", "float", "bool", "list", "object"}

_OK = lambda data=None, message="": json_response(
    {"status": "ok", "message": message, "data": data}
)
_ERR = lambda message, status_code=400: error_response(message, status_code=status_code)


def _convert_value(value, meta, path: str) -> tuple[list[str], Any]:
    """按 schema 元数据校验并转换单个值，返回 (errors, converted)。"""
    t = meta.get("type")
    if t not in _SCHEMA_TYPES:
        return [], value
    if t == "object":
        items = meta.get("items", {})
        if not isinstance(value, dict):
            return [f"{path} 应为对象"], value
        errors: list[str] = []
        converted: dict = {}
        for key, sub_meta in items.items():
            if key not in value:
                continue
            sub_errors, sub_value = _convert_value(value[key], sub_meta, f"{path}.{key}")
            errors.extend(sub_errors)
            converted[key] = sub_value
        return errors, converted
    if t == "bool":
        if not isinstance(value, bool):
            return [f"{path} 应为布尔值"], value
        return [], value
    if t == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return [], int(value)
            return [f"{path} 应为整数"], value
        return [], value
    if t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path} 应为数字"], value
        return [], float(value)
    if t == "list":
        if not isinstance(value, list):
            return [f"{path} 应为列表"], value
        return [], value
    # string / text
    if not isinstance(value, str):
        return [f"{path} 应为字符串"], value
    return [], value


class PluginConfigAPI:
    """插件全局配置读取与保存。"""

    def __init__(self, plugin_context: "PluginContext", plugin=None):
        self._ctx = plugin_context
        self._plugin = plugin
        self._config = plugin_context.config
        self._schema = getattr(plugin_context.config, "schema", None) or {}

    def get_config(self):
        providers: Dict[str, List[str]] = {"chat": [], "embedding": [], "rerank": []}
        try:
            host = self._ctx.astrbot_context
            for p in host.get_all_providers():
                providers["chat"].append(p.meta().id)
            for p in host.get_all_embedding_providers():
                providers["embedding"].append(p.meta().id)
            rerank_getter = getattr(host, "get_all_rerank_providers", None)
            if callable(rerank_getter):
                for p in rerank_getter():
                    providers["rerank"].append(p.meta().id)
        except Exception as e:
            logger.warning(f"AngelMemory: 获取 provider 列表失败: {e}")

        return _OK(
            {
                "schema": self._schema,
                "values": dict(self._config),
                "providers": providers,
            }
        )

    async def save_config(self):
        payload = await request.json(default=None)
        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            return _ERR("values 必须是 JSON 对象")

        errors: list[str] = []
        converted: Dict[str, Any] = {}
        for key, value in values.items():
            meta = self._schema.get(key)
            if not isinstance(meta, dict):
                errors.append(f"未知配置项: {key}")
                continue
            key_errors, key_value = _convert_value(value, meta, key)
            errors.extend(key_errors)
            converted[key] = key_value
        if errors:
            return _ERR("；".join(errors))

        # 经 PluginContext 原地更新顶层键（保持 dict 引用）并刷新 scope map 缓存；
        # 运行时组件动态读取即时生效
        self._ctx.update_config(converted)
        try:
            self._config.save_config()
        except Exception as e:
            logger.error(f"AngelMemory: 配置落盘失败: {e}", exc_info=True)
            return _ERR(f"配置落盘失败: {e}")

        hook = getattr(self._plugin, "on_config_saved", None)
        if callable(hook):
            try:
                hook()
            except Exception as e:
                logger.error(f"AngelMemory: 配置热生效钩子失败: {e}", exc_info=True)
                return _ERR(f"配置已保存，但热生效刷新失败: {e}")

        return _OK(message="已保存并即时生效")
