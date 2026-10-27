import math
import threading
from typing import Dict, List, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SoulState:
    """
    灵魂状态管理器

    以四对互斥情绪为轴，记录 AI 当前的情绪读数：
    - 喜 ↔ 悲
    - 怒 ↔ 恐
    - 意料中 ↔ 出乎意料
    - 厌恶 ↔ 认可

    每条轴取值 0.0 ~ 1.0，0.5 表示中性。状态只描述情绪本身，
    注入上下文时直接使用这份读数，由面板标签决定两端的含义。

    核心特性：
    - 弹性系统：越远离中线越难继续偏离，松手自动回中
    - 双轨调整：主动反思（强度大）+ 被动共鸣（按记忆强度加权）
    """

    # 情绪轴，固定顺序，与提示词的 soul_state 字段一一对应
    AXES = ["joy", "anger", "surprise", "disgust"]

    AXIS_MID = 0.5
    AXIS_MIN = 0.0
    AXIS_MAX = 1.0

    # 弹性参数
    ELASTICITY_FACTOR = 4.0   # 离中线越远，可移动幅度衰减越快
    REGRESSION_FACTOR = 0.1   # 自然回归中线的力度

    # 强度参数
    REFLECT_STRENGTH = 1.0    # 主动反思强度
    RESONATE_STRENGTH = 0.3   # 被动共鸣强度

    def __init__(self, config=None):
        """
        初始化灵魂状态

        状态仅存在于内存，重启后回到四轴居中的中性状态。
        """
        self._lock = threading.RLock()
        self.axes: Dict[str, float] = {axis: self.AXIS_MID for axis in self.AXES}

    # ===== 核心算法模块（可独立测试） =====

    def _clamp(self, value: float) -> float:
        """把数值收进 0~1。"""
        return max(self.AXIS_MIN, min(self.AXIS_MAX, value))

    def _parse_score(self, axis: str, raw) -> Optional[float]:
        """把一条读数转成合法分值，非法读数返回 None。"""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(f"情绪读数无效: {axis}={raw!r}，跳过")
            return None
        if not math.isfinite(value):
            logger.warning(f"情绪读数不是有限数: {axis}={raw!r}，跳过")
            return None
        return self._clamp(value)

    def _apply(self, targets: Dict[str, float], strength: float, mode: str) -> None:
        """
        把当前状态朝目标读数推进一步。

        弹性的作用是让情绪有惯性：已经偏离很远的轴，再推也推不动多少，
        而一旦停止施力，又会被回归力慢慢拉回中线。
        """
        if not targets:
            return
        with self._lock:
            changes = []
            for axis in self.AXES:
                if axis not in targets:
                    continue
                target = self._parse_score(axis, targets[axis])
                if target is None:
                    continue
                current = self.axes[axis]
                elasticity = math.exp(
                    -self.ELASTICITY_FACTOR * abs(current - self.AXIS_MID)
                )
                updated = current + (target - current) * strength * elasticity
                updated += (self.AXIS_MID - updated) * self.REGRESSION_FACTOR
                updated = self._clamp(updated)
                self.axes[axis] = updated
                changes.append(f"{axis}: {current:.2f}->{updated:.2f}")
            if changes:
                logger.debug(f"灵魂调整 [{mode}]: {', '.join(changes)}")
            else:
                unknown = sorted(str(key) for key in targets if key not in self.AXES)
                if unknown:
                    logger.warning(
                        f"灵魂调整 [{mode}] 未命中任何情绪轴，已跳过；"
                        f"收到的键={unknown}，可用键={self.AXES}"
                    )
                else:
                    logger.warning(f"灵魂调整 [{mode}] 所有读数均无效，已跳过")

    # ===== 公共接口 =====

    def reflect(self, scores: Dict[str, float]) -> None:
        """按本轮情绪读数修正当前状态。"""
        self._apply(scores, self.REFLECT_STRENGTH, "reflect")

    def resonate(self, entries: List[dict]) -> None:
        """
        旧记忆的情绪牵引当前状态。

        Args:
            entries: 每项形如 {"scores": {轴: 值}, "strength": 记忆强度}
                     记忆强度越大，它的情绪对当前状态影响越大。
        """
        weighted: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            scores = entry.get("scores") or {}
            if not isinstance(scores, dict):
                continue
            try:
                weight = float(entry.get("strength", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            weight = max(0.0, weight)
            if weight <= 0:
                continue
            for axis in self.AXES:
                if axis not in scores:
                    continue
                try:
                    value = self._clamp(float(scores[axis]))
                except (TypeError, ValueError):
                    continue
                weighted[axis] = weighted.get(axis, 0.0) + value * weight
                weights[axis] = weights.get(axis, 0.0) + weight

        if not weighted:
            logger.debug("灵魂共鸣: 无可用的记忆情绪，跳过")
            return

        targets = {axis: weighted[axis] / weights[axis] for axis in weighted}
        logger.debug(f"灵魂共鸣: 由 {len(entries)} 条记忆牵引 {sorted(targets)}")
        self._apply(targets, self.RESONATE_STRENGTH, "resonate")

    def get_scores(self) -> Dict[str, float]:
        """获取四条情绪轴的当前读数，注入与写入记忆都用这份数据。"""
        with self._lock:
            return {axis: round(value, 2) for axis, value in self.axes.items()}

    def get_state_description(self) -> str:
        """获取当前状态的文本描述（用于调试）。"""
        scores = self.get_scores()
        parts = [f"{axis}={value:.2f}" for axis, value in scores.items()]
        return " | ".join(parts)
