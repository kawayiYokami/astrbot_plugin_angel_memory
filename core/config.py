"""
配置管理模块

负责从插件配置中读取记忆系统相关参数。
"""

import os
from typing import Dict, Any, Optional
from astrbot.api import AstrBotConfig


class MemoryConfig:
    """记忆系统配置管理类"""

    def __init__(
        self, config: AstrBotConfig, data_dir: str, plugin_name: str = "astrbot_plugin_angel_memory"
    ):
        """
        初始化配置管理器

        Args:
            config: AstrBot配置对象
            data_dir: 插件数据目录
            plugin_name: 插件名称
        """
        self.config = config
        self.plugin_name = plugin_name

        # 直接使用传入的数据目录
        self._data_dir = data_dir

        # 确保目录存在
        os.makedirs(self._data_dir, exist_ok=True)

    def get_data_directory(self) -> str:
        """
        获取插件数据目录路径

        Returns:
            数据目录绝对路径
        """
        return self._data_dir

    def get_storage_path(self, filename: str) -> str:
        """
        获取存储文件的完整路径

        Args:
            filename: 文件名

        Returns:
            文件的完整路径
        """
        return os.path.join(self.get_data_directory(), filename)

    def get_min_message_length(self) -> int:
        """
        获取触发记忆处理的最小消息长度

        Returns:
            最小消息长度
        """
        return self.config.get("min_message_length", 5)

    def get_short_term_memory_capacity(self) -> int:
        """
        获取短期记忆容量倍数

        Returns:
            容量倍数
        """
        return self.config.get("short_term_memory_capacity", 1)

    def get_sleep_interval(self) -> int:
        """
        获取睡眠间隔（秒）

        Returns:
            睡眠间隔秒数
        """
        return self.config.get("sleep_interval", 3600)  # 默认1小时

    def get_provider_id(self) -> str:
        """
        获取记忆整理LLM提供商ID

        Returns:
            提供商ID，留空则跳过记忆整理
        """
        return self.config.get("provider_id", "")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取任意配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)
