"""
画像提取规则配置 — 红绿灯三级分类
绿灯直接提取，黄灯可配置开关，红灯禁止提取。
"""
from dataclasses import dataclass, field


@dataclass
class ProfileExtractionConfig:
    """画像提取规则配置"""

    # 绿灯：始终提取，全场景可见
    green_tags: list[str] = field(default_factory=lambda: [
        "preferred_name",       # 称呼偏好 "叫我小貔貅"
        "communication_style",  # 交流风格 "别废话"
        "public_identity",      # 公开身份 "做后端的"
        "interest_domain",      # 兴趣领域 "在搞Rust"
        "bot_feedback",          # 对bot评价 "上次回答很好"
    ])

    # 黄灯：默认关闭，部署者可开
    yellow_tags: list[str] = field(default_factory=list)  # 默认空
    _available_yellow: tuple = (
        "personal_experience",  # 个人经历 "去过冰岛"
        "emotional_state",      # 情绪表达 "心情很差"
    )

    # 红灯关键词（包含任一即跳过整条消息）
    red_keywords: tuple = (
        "密码", "token", "api_key", "secret", "私钥",
        "别跟别人说", "不要说出去", "保密", "不要告诉别人",
        "电话", "手机号", "身份证", "地址",
    )

    # 提取参数
    extraction_interval: int = 5       # 每N次记忆落库触发一次
    max_profile_items: int = 20        # 单用户画像条目上限
    dedup_threshold: float = 0.85      # 同类型内去重阈值
    min_message_length: int = 6        # 过短消息不提取
    max_tag_length: int = 30           # 单条标签最大字符数

    # 时态标记
    mark_temporal: bool = True         # 区分当前状态/历史陈述

    @property
    def all_yellow(self) -> tuple:
        return self._available_yellow

    def is_yellow_enabled(self, tag: str) -> bool:
        return tag in self.yellow_tags

    def is_green(self, tag: str) -> bool:
        return tag in self.green_tags

    def contains_red(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.red_keywords)
