from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryDecayConfig:
    tier0_threshold: float = 3.0
    tier1_threshold: float = 10.0
    consolidate_speed: float = 2.5
    cycle_tier0_days: int = 3
    forget_speed: float = 1.0
    tier0_forget_speed: float = 1.0


class MemoryDecayPolicy:
    """
    三档记忆衰减策略：
    - T0: 时间自然遗忘
    - T1: 仅在“被召回且无用”时遗忘
    - T2: 永不遗忘
    """

    def __init__(self, config: MemoryDecayConfig | None = None):
        self.config = config or MemoryDecayConfig()

    def tier_of(self, useful_score: float) -> int:
        score = float(useful_score or 0.0)
        if score >= self.config.tier1_threshold:
            return 2
        if score >= self.config.tier0_threshold:
            return 1
        return 0

    def useful_score_after_useful(self, useful_score: float) -> float:
        return float(useful_score or 0.0) + float(self.config.consolidate_speed)

    def should_natural_decay(self, useful_score: float) -> bool:
        return self.tier_of(useful_score) == 0

    def should_decay_on_useless_recall(self, useful_score: float) -> bool:
        return self.tier_of(useful_score) == 1

    def is_immortal(self, useful_score: float) -> bool:
        return self.tier_of(useful_score) == 2

    def tier0_decay_cycle_days(self) -> int:
        base = max(1.0, float(self.config.cycle_tier0_days))
        speed = max(0.01, float(self.config.forget_speed)) * max(
            0.01, float(self.config.tier0_forget_speed)
        )
        adjusted = int(round(base / speed))
        return max(1, adjusted)


def build_decay_config(plugin_config: Dict[str, Any] | None) -> MemoryDecayConfig:
    """
    从插件配置构建衰减参数。

    规则：
    - 默认使用系统内置参数（经过测试）。
    - 仅当 memory_behavior.decay_policy_override.enabled=true 时才覆盖默认值。
    """
    defaults = MemoryDecayConfig()
    cfg = plugin_config or {}
    memory_behavior = cfg.get("memory_behavior", {}) or {}
    override = memory_behavior.get("decay_policy_override", {}) or {}
    if not isinstance(override, dict) or not bool(override.get("enabled", False)):
        return defaults

    def _to_float(key: str, default: float) -> float:
        raw = override.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "记忆衰减参数无效，已回退默认值: %s=%r (default=%s)",
                key,
                raw,
                default,
            )
            return float(default)

    def _to_int(key: str, default: int) -> int:
        raw = override.get(key, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "记忆衰减参数无效，已回退默认值: %s=%r (default=%s)",
                key,
                raw,
                default,
            )
            return int(default)

    tier0_threshold = _to_float("tier0_threshold", defaults.tier0_threshold)
    tier1_threshold = _to_float("tier1_threshold", defaults.tier1_threshold)
    if tier1_threshold < tier0_threshold:
        tier1_threshold = tier0_threshold

    return MemoryDecayConfig(
        tier0_threshold=tier0_threshold,
        tier1_threshold=tier1_threshold,
        consolidate_speed=max(
            0.01,
            _to_float("consolidate_speed", defaults.consolidate_speed),
        ),
        cycle_tier0_days=max(1, _to_int("cycle_tier0_days", defaults.cycle_tier0_days)),
        forget_speed=max(0.01, _to_float("forget_speed", defaults.forget_speed)),
        tier0_forget_speed=max(
            0.01, _to_float("tier0_forget_speed", defaults.tier0_forget_speed)
        ),
    )
