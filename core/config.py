"""
记忆插件配置管理

处理插件的配置选项和默认值。
"""

from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class MemoryCapacityConfig:
    """记忆容量配置类"""

    knowledge: int = 14  # 普通知识记忆容量
    knowledge_user_info: int = 14  # 用户信息专用知识记忆容量
    emotional: int = 1
    skill: int = 7
    task: int = 1
    event: int = 3


class MemoryConstants:
    """记忆系统常量定义"""

    MIN_MESSAGE_LENGTH = 5
    SHORT_TERM_MEMORY_CAPACITY = 1.0
    SLEEP_INTERVAL = 3600  # 默认睡眠间隔（秒）
    DEFAULT_DATA_DIR = None  # 数据目录必须由外部传入，不设默认值
    SMALL_MODEL_NOTE_BUDGET = 8000  # 默认小模型笔记Token预算
    LARGE_MODEL_NOTE_BUDGET = 12000  # 默认大模型笔记Token预算

    # 时间常量（秒）
    TIME_SECOND = 1
    TIME_MINUTE = 60
    TIME_HOUR = 3600
    TIME_DAY = 86400

    # 记忆类型映射
    MEMORY_TYPE_NAMES = {
        "knowledge": "知识",
        "event": "事件",
        "skill": "技能",
        "emotional": "情感",
        "task": "任务",
        "meta": "元记忆",
    }

    MEMORY_TYPE_MAPPING = {
        "知识记忆": "knowledge",
        "事件记忆": "event",
        "技能记忆": "skill",
        "任务记忆": "task",
        "情感记忆": "emotional",
        "元记忆": "meta",
    }


class MemoryConfig:
    """记忆插件配置类

    负责管理插件的用户可配置参数，与llm_memory的系统配置形成分层结构：
    - 本类：用户级别的插件配置（min_message_length, sleep_interval等）
    - system_config：系统级别的技术配置（向量存储、嵌入模型等）

    配置验证确保用户输入的参数在合理范围内，防止异常配置导致系统不稳定。
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        data_dir: str = MemoryConstants.DEFAULT_DATA_DIR,
    ):
        """
        初始化配置

        Args:
            config: 配置字典
            data_dir: 数据目录路径
        """
        self.config = config or {}
        self.data_dir = data_dir

        # 预计算常用配置值（使用更紧凑的表达）
        config_get = self.config.get
        self._min_message_length = self._validate_min_message_length(
            config_get("min_message_length", MemoryConstants.MIN_MESSAGE_LENGTH)
        )
        self._short_term_memory_capacity = self._validate_short_term_memory_capacity(
            config_get(
                "short_term_memory_capacity", MemoryConstants.SHORT_TERM_MEMORY_CAPACITY
            )
        )
        self._sleep_interval = self._validate_sleep_interval(
            config_get("sleep_interval", MemoryConstants.SLEEP_INTERVAL)
        )
        self._data_directory = config_get("data_directory", self.data_dir)
        self._provider_id = config_get("provider_id", "")
        self._small_model_note_budget = self._validate_token_budget(
            config_get("small_model_note_budget", 8000), "small_model_note_budget"
        )
        self._large_model_note_budget = self._validate_token_budget(
            config_get("large_model_note_budget", 12000), "large_model_note_budget"
        )
        self._enable_local_embedding = config_get("enable_local_embedding", False)

    @property
    def min_message_length(self) -> int:
        """获取最小消息长度"""
        return self._min_message_length

    @property
    def short_term_memory_capacity(self) -> float:
        """获取短期记忆容量倍数"""
        return self._short_term_memory_capacity

    @property
    def sleep_interval(self) -> int:
        """获取睡眠间隔（秒）"""
        return self._sleep_interval

    @property
    def data_directory(self) -> str:
        """获取数据目录路径"""
        return self._data_directory

    @property
    def provider_id(self) -> str:
        """获取记忆整理LLM提供商ID"""
        return self._provider_id

    @property
    def small_model_note_budget(self) -> int:
        """获取小模型笔记Token预算"""
        return self._small_model_note_budget

    @property
    def large_model_note_budget(self) -> int:
        """获取大模型笔记Token预算"""
        return self._large_model_note_budget
    @property
    def enable_local_embedding(self) -> bool:
        """是否启用本地嵌入模型"""
        return self._enable_local_embedding

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "min_message_length": self.min_message_length,
            "short_term_memory_capacity": self.short_term_memory_capacity,
            "sleep_interval": self.sleep_interval,
            "enable_local_embedding": self.enable_local_embedding,
            "data_directory": self.data_directory,
            "provider_id": self.provider_id,
            "small_model_note_budget": self.small_model_note_budget,
            "large_model_note_budget": self.large_model_note_budget,
        }

    def get_capacity_config(self) -> MemoryCapacityConfig:
        """获取记忆容量配置"""
        return MemoryCapacityConfig()

    def _validate_min_message_length(self, value: Any) -> int:
        """验证最小消息长度"""
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"min_message_length must be a positive integer, got: {value}"
            )
        if value > 500:
            raise ValueError(f"min_message_length too large (max 500), got: {value}")
        return value

    def _validate_short_term_memory_capacity(self, value: Any) -> float:
        """验证短期记忆容量倍数"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"short_term_memory_capacity must be a positive number, got: {value}"
            )
        if value > 10.0:
            raise ValueError(
                f"short_term_memory_capacity too large (max 10.0), got: {value}"
            )
        return float(value)

    def _validate_sleep_interval(self, value: Any) -> int:
        """验证睡眠间隔"""
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"sleep_interval must be a non-negative integer, got: {value}"
            )
        if value > 24 * 60 * 60:  # 超过24小时被视为不合理
            raise ValueError(
                f"sleep_interval too large (max 86400 seconds), got: {value}"
            )
        return value

    def _validate_token_budget(self, value: Any, field_name: str) -> int:
        """验证Token预算"""
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{field_name} must be a non-negative integer, got: {value}"
            )
        if value > 64000:
            raise ValueError(f"{field_name} too large (max 64000), got: {value}")
        return value

    def _validate_bool(self, value: Any, field_name: str) -> bool:
        """验证布尔值"""
        if not isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be a boolean, got: {type(value).__name__}"
            )
        return value
