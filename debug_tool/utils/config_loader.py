import json
import os
from typing import Dict, Any, Optional

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
        config = self.load_config()
        providers = config.get("provider", [])

        # 1. Find the enabled embedding provider
        for p in providers:
            # Logic: type is openai_embedding OR provider_type is embedding
            if p.get("enable") and (p.get("type") == "openai_embedding" or p.get("provider_type") == "embedding"):
                return p

        return None

    def get_data_dir(self, provider_id: str) -> str:
        # data/plugins/astrbot_plugin_angel_memory/debug_tool -> data/plugin_data/astrbot_plugin_angel_memory
        # ../../../plugin_data/astrbot_plugin_angel_memory
        base_data_dir = os.path.abspath(os.path.join(self.base_path, "../../../plugin_data/astrbot_plugin_angel_memory"))
        return os.path.join(base_data_dir, f"memory_{provider_id}", "chromadb")

    def get_raw_notes_dir(self) -> str:
        # data/plugins/astrbot_plugin_angel_memory/debug_tool -> data/plugin_data/astrbot_plugin_angel_memory/raw
        base_data_dir = os.path.abspath(os.path.join(self.base_path, "../../../plugin_data/astrbot_plugin_angel_memory"))
        return os.path.join(base_data_dir, "raw")