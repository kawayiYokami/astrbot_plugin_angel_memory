import json
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, base_path: str = None):
        if base_path:
            self.base_path = base_path
        else:
            # Default: relative to this file
            # this file: .../debug_tool/utils/config_loader.py
            # root: .../debug_tool
            self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_config_path(self) -> str:
        # data/plugins/astrbot_plugin_angel_memory/debug_tool -> data/cmd_config.json
        # ../../../cmd_config.json
        return os.path.abspath(os.path.join(self.base_path, "../../../cmd_config.json"))

    def load_config(self) -> Dict[str, Any]:
        config_path = self.get_config_path()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
            return json.load(f)

    def get_embedding_provider(self) -> Optional[Dict[str, Any]]:
        """获取插件配置的 embedding provider"""
        # 1. 先读取插件配置，获取 embedding_provider_id
        plugin_config_path = os.path.abspath(os.path.join(self.base_path, "../../../config/astrbot_plugin_angel_memory_config.json"))
        plugin_config = {}
        if os.path.exists(plugin_config_path):
            try:
                with open(plugin_config_path, 'r', encoding='utf-8-sig') as f:
                    plugin_config = json.load(f)
            except Exception:
                pass

        retrieval = plugin_config.get("retrieval", {}) or {}
        configured_provider_id = retrieval.get("embedding_provider_id", "")

        # 2. 如果配置了 provider_id，在 cmd_config.json 中查找对应的 provider
        if configured_provider_id:
            config = self.load_config()
            providers = config.get("provider", [])
            for p in providers:
                if p.get("id") == configured_provider_id and p.get("enable"):
                    return p
            # 配置了 provider_id 但找不到或未启用，给出警告
            logger.warning(
                f"⚠️ 插件配置了 embedding_provider_id='{configured_provider_id}'，"
                f"但在 cmd_config.json 中未找到或未启用！回退到第一个可用的 embedding provider。"
            )

        # 3. 回退：查找第一个启用的 embedding provider
        config = self.load_config()
        providers = config.get("provider", [])
        fallback_provider = None
        for p in providers:
            if p.get("enable") and (p.get("type") == "openai_embedding" or p.get("provider_type") == "embedding"):
                fallback_provider = p
                break

        if fallback_provider:
            logger.warning(
                f"⚠️ 回退使用 embedding provider: {fallback_provider.get('id')} "
                f"(配置期望: {configured_provider_id or '未配置'})"
            )
            return fallback_provider

        return None

    def get_embedding_provider_status(self) -> Dict[str, Any]:
        """获取 embedding provider 状态信息，用于判断是否回退"""
        # 1. 先读取插件配置，获取 embedding_provider_id
        plugin_config_path = os.path.abspath(os.path.join(self.base_path, "../../../config/astrbot_plugin_angel_memory_config.json"))
        plugin_config = {}
        if os.path.exists(plugin_config_path):
            try:
                with open(plugin_config_path, 'r', encoding='utf-8-sig') as f:
                    plugin_config = json.load(f)
            except Exception:
                pass

        retrieval = plugin_config.get("retrieval", {}) or {}
        configured_provider_id = retrieval.get("embedding_provider_id", "")

        # 2. 如果配置了 provider_id，在 cmd_config.json 中查找对应的 provider
        if configured_provider_id:
            config = self.load_config()
            providers = config.get("provider", [])
            for p in providers:
                if p.get("id") == configured_provider_id and p.get("enable"):
                    return {
                        "is_fallback": False,
                        "configured": configured_provider_id,
                        "actual": configured_provider_id
                    }
            # 配置了但找不到或未启用
            return {
                "is_fallback": True,
                "configured": configured_provider_id,
                "actual": None,
                "warning": f"插件配置了 '{configured_provider_id}'，但未找到或未启用"
            }

        # 3. 回退：查找第一个启用的 embedding provider
        config = self.load_config()
        providers = config.get("provider", [])
        fallback_provider = None
        for p in providers:
            if p.get("enable") and (p.get("type") == "openai_embedding" or p.get("provider_type") == "embedding"):
                fallback_provider = p
                break

        if fallback_provider:
            return {
                "is_fallback": True,
                "configured": None,
                "actual": fallback_provider.get("id"),
                "warning": "未配置 embedding_provider_id，回退使用第一个可用的 provider"
            }

        return {
            "is_fallback": False,
            "configured": None,
            "actual": None
        }

    def get_data_dir(self, provider_id: str) -> str:
        # data/plugins/astrbot_plugin_angel_memory/debug_tool -> data/plugin_data/astrbot_plugin_angel_memory
        # ../../../plugin_data/astrbot_plugin_angel_memory
        base_data_dir = os.path.abspath(os.path.join(self.base_path, "../../../plugin_data/astrbot_plugin_angel_memory"))
        return os.path.join(base_data_dir, f"memory_{provider_id}", "chromadb")

    def get_plugin_data_dir(self) -> str:
        return os.path.abspath(
            os.path.join(self.base_path, "../../../plugin_data/astrbot_plugin_angel_memory")
        )

    def get_memory_center_dir(self) -> str:
        return os.path.join(self.get_plugin_data_dir(), "memory_center")

    def get_memory_center_index_dir(self) -> str:
        return os.path.join(self.get_memory_center_dir(), "index")

    def get_simple_memory_db_path(self) -> str:
        return os.path.join(self.get_memory_center_index_dir(), "simple_memory.db")

    def get_maintenance_state_path(self) -> str:
        return os.path.join(self.get_memory_center_dir(), "maintenance_state.json")

    def get_backup_dir(self) -> str:
        return os.path.join(self.get_memory_center_dir(), "backups")

    def get_raw_notes_dir(self) -> str:
        # data/plugins/astrbot_plugin_angel_memory/debug_tool -> data/plugin_data/astrbot_plugin_angel_memory/raw
        base_data_dir = os.path.abspath(os.path.join(self.base_path, "../../../plugin_data/astrbot_plugin_angel_memory"))
        return os.path.join(base_data_dir, "raw")
