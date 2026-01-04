import math
import threading
from typing import Dict, List

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class SoulState:
    """
    灵魂状态管理器 (Soul State Manager)

    管理 AI 的核心精神状态（4维能量槽），并通过橡皮筋算法（Tanh）将其映射为具体的行为参数。
    实现了类似人类的"情绪惯性"和"创伤应激"机制。

    核心特性：
    - 弹性系统：能量值倾向回归mid（默认值），离mid越远越难继续偏离
    - 双轨调整：主动反思（强度大）+ 被动共鸣（强度小）
    - 原子化接口：所有调整都通过4位二进制代码统一处理
    """

    # 维度名称列表（固定顺序）
    DIMENSIONS = ["RecallDepth", "ImpressionDepth", "ExpressionDesire", "Creativity"]

    # 弹性参数
    ELASTICITY_FACTOR = 0.1  # 弹性系数，决定离mid越远时衰减速度
    REGRESSION_FACTOR = 0.1  # 自然回归系数，决定向mid回归的力度

    # 强度参数
    REFLECT_STRENGTH = 1.0   # 主动反思强度
    RESONATE_STRENGTH = 0.3  # 被动共鸣强度

    # 能量范围
    ENERGY_SOFT_LIMIT = 20.0  # 能量值软限制

    def __init__(self, config=None):
        """
        初始化灵魂状态

        注意：状态仅在内存中维护，重启插件后会重置为中庸状态(0.0)
        """
        self._lock = threading.RLock() # 线程锁

        # 能量池：累积历史刺激，初始为0（中庸），范围软限制 [-20, 20]
        self.energy = {
            "RecallDepth":      0.0, # 回忆量倾向：决定检索量 (RAG Top_K)
            "ImpressionDepth":  0.0, # 记住量倾向：决定记忆生成数量 (Memory Generation Limit)
            "ExpressionDesire": 0.0, # 发言长度倾向：决定发言长度 (Max Tokens)
            "Creativity":       0.0  # 思维发散倾向：决定温度 (Temperature)
        }

        # 从配置中读取回归值，min/max值在此处硬编码以符合群聊场景
        self.config = {
            "RecallDepth": {
                "min": 1,
                "mid": getattr(config, "soul_recall_depth_mid", 7),
                "max": 20
            },
            "ImpressionDepth": {
                "min": 1,
                "mid": getattr(config, "soul_impression_depth_mid", 3),
                "max": 10
            },
            "ExpressionDesire": {
                "min": 0.0,
                "mid": getattr(config, "soul_expression_desire_mid", 0.5),
                "max": 1.0
            },
            "Creativity": {
                "min": 0.0,
                "mid": getattr(config, "soul_creativity_mid", 0.7),
                "max": 1.0
            }
        }

    # ===== 核心算法模块（可独立测试） =====

    def _calculate_elastic_delta(
        self,
        current_energy: float,
        direction: int,
        mid: float,
        base_strength: float
    ) -> float:
        """
        计算弹性变动量（核心算法，可独立测试）

        核心思想：
        1. 离mid越远，弹性系数越小（越难移动）
        2. 总是有向mid回归的自然力
        3. 方向和强度共同决定最终变动

        Args:
            current_energy: 当前能量值
            direction: 变动方向（+1=增加，-1=减少）
            mid: 默认值（回归中心）
            base_strength: 基础变动强度

        Returns:
            实际变动量（已考虑弹性和回归力）
        """
        # 1. 计算到mid的距离
        distance_to_mid = abs(current_energy - mid)

        # 2. 弹性系数：使用指数衰减
        # 距离越远，系数越小（越难移动）
        elasticity = math.exp(-self.ELASTICITY_FACTOR * distance_to_mid)

        # 3. 计算方向性变动
        directional_delta = direction * base_strength * elasticity

        # 4. 自然回归力（总是向mid靠拢）
        # 回归力与距离成正比
        regression_force = (mid - current_energy) * self.REGRESSION_FACTOR

        # 5. 合成最终变动量
        total_delta = directional_delta + regression_force

        return total_delta

    def _generate_resonate_code(self, snapshots: List[Dict[str, float]]) -> str:
        """
        从多个记忆快照生成4位共鸣代码（核心算法，可独立测试）

        算法：
        1. 计算所有快照的平均值（每个维度）
        2. 与当前能量值对比
        3. 平均值 > 当前值 → 1（倾向增加）
        4. 平均值 <= 当前值 → 0（倾向减少）

        Args:
            snapshots: 多个记忆的状态快照列表

        Returns:
            4位二进制字符串，如"1011"
        """
        if not snapshots:
            return "0000"  # 没有快照，不调整

        # 计算每个维度的平均值
        avg_energy = {}
        for dim in self.DIMENSIONS:
            values = [s.get(dim, 0.0) for s in snapshots if dim in s]
            if values:
                avg_energy[dim] = sum(values) / len(values)
            else:
                avg_energy[dim] = 0.0

        # 生成4位代码
        code = ""
        with self._lock:
            for dim in self.DIMENSIONS:
                current_value = self.energy[dim]
                avg_value = avg_energy[dim]
                # 平均值 > 当前值 → 倾向增加 → 1
                code += '1' if avg_value > current_value else '0'

        return code

    # ===== 公共接口 =====

    def get_value(self, dimension: str) -> float:
        """
        核心算法：橡皮筋阻尼映射 (Tanh)
        将无界的状态能量值映射到有界的物理参数区间。

        公式：
        y = mid + (max - mid) * tanh(k * x)  if x >= 0
        y = mid + (mid - min) * tanh(k * x)  if x < 0
        """
        with self._lock:
            if dimension not in self.energy or dimension not in self.config:
                logger.warning(f"未知维度: {dimension}，返回默认值 0")
                return 0.0

            E = self.energy[dimension]
            cfg = self.config[dimension]
            k = 0.3  # 敏感度系数，决定了能量变化的响应速度

            if E >= 0:
                val = cfg['mid'] + (cfg['max'] - cfg['mid']) * math.tanh(k * E)
            else:
                val = cfg['mid'] + (cfg['mid'] - cfg['min']) * math.tanh(k * E)

            # 强制截断在 min-max 范围内（虽然 tanh 不会越界，但浮点运算可能微小溢出）
            val = max(cfg['min'], min(cfg['max'], val))

        # 对于整数类型的参数（如Top_K），进行取整
        if dimension in ["RecallDepth", "ImpressionDepth"]:
            return int(round(val))
        # 对于归一化参数（如ExpressionDesire, Creativity），保留两位小数
        return round(val, 2)

    def adjust(self, code: str, mode: str = "reflect"):
        """
        统一的原子化调整接口（新接口）

        这是所有状态调整的统一入口，支持主动反思和被动共鸣两种模式。

        Args:
            code: 4位二进制字符串，如"1011"
                  每一位对应一个维度的增减方向
                  1=增加该维度，0=减少该维度
            mode: 调整模式
                - "reflect": 主动反思（强度1.0）
                - "resonate": 被动共鸣（强度0.3）

        Raises:
            ValueError: 如果code格式不正确
        """
        # 1. 验证code格式
        if len(code) != 4 or not all(c in '01' for c in code):
            raise ValueError(f"Invalid code: {code}, must be 4-bit binary string like '1011'")

        # 2. 根据mode确定强度
        if mode == "reflect":
            base_strength = self.REFLECT_STRENGTH
        elif mode == "resonate":
            base_strength = self.RESONATE_STRENGTH
        else:
            logger.warning(f"Unknown mode: {mode}, using 'reflect'")
            base_strength = self.REFLECT_STRENGTH

        # 3. 对4个维度依次调整
        with self._lock:
            changes = []
            for i, dim in enumerate(self.DIMENSIONS):
                # 解析方向：1=增加，0=减少
                direction = +1 if code[i] == '1' else -1

                # 获取配置
                mid = self.config[dim]['mid']
                current_energy = self.energy[dim]

                # 计算弹性变动量
                delta = self._calculate_elastic_delta(
                    current_energy,
                    direction,
                    mid,
                    base_strength
                )

                # 应用变动
                self.energy[dim] += delta

                # 软限制
                self.energy[dim] = max(-self.ENERGY_SOFT_LIMIT,
                                      min(self.ENERGY_SOFT_LIMIT, self.energy[dim]))

                changes.append(f"{dim}: {current_energy:.2f}->{self.energy[dim]:.2f} (Δ{delta:+.2f})")

            mode_emoji = "🧘" if mode == "reflect" else "🎼"
            logger.debug(f"{mode_emoji} Soul Adjust [{mode}] ({code}): {', '.join(changes)}")

    def resonate(self, snapshots: List[Dict[str, float]]):
        """
        共鸣机制：多个旧记忆状态影响当前状态（新接口）

        工作流程：
        1. 计算所有快照的平均能量值
        2. 与当前状态对比生成4位代码
        3. 使用resonate模式调用adjust()

        Args:
            snapshots: 多个记忆的状态快照列表
        """
        if not snapshots:
            logger.debug("🎼 Soul Resonate: No snapshots, skipping")
            return

        # 生成共鸣代码
        code = self._generate_resonate_code(snapshots)

        logger.debug(f"🎼 Soul Resonate: Generated code={code} from {len(snapshots)} snapshots")

        # 使用resonate模式调整
        self.adjust(code, mode="resonate")

    def update_energy(self, dimension: str, delta: float, decay: float = 0.0):
        """
        [已弃用] 更新能量状态（保留用于向后兼容）

        建议使用新的adjust()接口替代。

        Args:
            dimension: 维度名称
            delta: 变化量（可正可负）
            decay: 自然衰减系数 (0.0 - 1.0)
        """
        logger.warning(
            f"update_energy() is deprecated, please use adjust() instead. "
            f"Called with dimension={dimension}, delta={delta}, decay={decay}"
        )

        with self._lock:
            if dimension not in self.energy:
                return

            original_val = self.energy[dimension]

            # 1. 自然衰减 (回归中庸)
            if decay > 0:
                self.energy[dimension] *= (1.0 - decay)
                if abs(self.energy[dimension]) < 0.1:
                    self.energy[dimension] = 0.0

            # 2. 施加刺激
            self.energy[dimension] += delta

            # 3. 软限制
            self.energy[dimension] = max(-self.ENERGY_SOFT_LIMIT,
                                        min(self.ENERGY_SOFT_LIMIT, self.energy[dimension]))

            new_val = self.energy[dimension]
            logger.debug(f"🔋 Soul Update [{dimension}]: {original_val:.2f} -> {new_val:.2f} (Delta={delta}, Decay={decay})")
    def get_snapshot(self) -> Dict[str, float]:
        """获取当前状态快照（用于存入新记忆）"""
        with self._lock:
            return self.energy.copy()

    def get_state_description(self) -> str:
        """获取当前状态的文本描述（用于调试或Prompt注入）"""
        with self._lock:
            v_recall = self.get_value('RecallDepth')
            v_impress = self.get_value('ImpressionDepth')
            v_express = self.get_value('ExpressionDesire')
            v_create = self.get_value('Creativity')

        desc = []
        desc.append(f"🧠 回忆倾向: {v_recall}条")
        desc.append(f"📝 记住倾向: {v_impress}条")
        desc.append(f"🗣️ 表达欲望: {v_express:.2f}")
        desc.append(f"🎨 思维发散: {v_create:.2f}")
        return " | ".join(desc)

    # 移除 save 和 load 方法，因为不需要持久化了