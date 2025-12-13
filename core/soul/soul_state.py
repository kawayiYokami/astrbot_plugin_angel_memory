import math
import threading
from typing import Dict

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class SoulState:
    """
    灵魂状态管理器 (Soul State Manager)

    管理 AI 的核心精神状态（4维能量槽），并通过橡皮筋算法（Tanh）将其映射为具体的行为参数。
    实现了类似人类的“情绪惯性”和“创伤应激”机制。
    """

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

        # 移除自动加载逻辑
        # if self.storage_path and os.path.exists(self.storage_path):
        #     self.load()

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

        # 对于整数类型的参数（如Top_K, Tokens），进行取整
        if dimension in ["RecallDepth", "ImpressionDepth", "ExpressionDesire"]:
            return int(round(val))
        return round(val, 2)

    def update_energy(self, dimension: str, delta: float, decay: float = 0.0):
        """
        更新能量状态 (线程安全)

        Args:
            dimension: 维度名称
            delta: 变化量（可正可负）
            decay: 自然衰减系数 (0.0 - 1.0)，每轮更新前先让当前能量衰减
        """
        with self._lock:
            if dimension not in self.energy:
                return

            original_val = self.energy[dimension]

            # 1. 自然衰减 (回归中庸)
            if decay > 0:
                self.energy[dimension] *= (1.0 - decay)
                # 如果能量非常小，直接归零，避免无限逼近
                if abs(self.energy[dimension]) < 0.1:
                    self.energy[dimension] = 0.0

            # 2. 施加刺激
            self.energy[dimension] += delta

            # 3. 软限制 (可选，防止数值溢出，Tanh本身能处理大数值，但保持在[-10, 10]比较合理)
            self.energy[dimension] = max(-20.0, min(20.0, self.energy[dimension]))

            new_val = self.energy[dimension]
            logger.debug(f"🔋 Soul Update [{dimension}]: {original_val:.2f} -> {new_val:.2f} (Delta={delta}, Decay={decay})")

        # 4. 不再自动保存到文件
        # self.save()

    def resonate(self, snapshot: Dict[str, float], intensity: float = 0.1):
        """
        共鸣机制：让旧记忆的状态快照冲击当前状态 (线程安全)

        Args:
            snapshot: 记忆中的状态快照 {"RecallDepth": 1.5, ...}
            intensity: 共鸣强度系数 (0.0 - 1.0)
        """
        if not snapshot:
            return

        if not 0.0 <= intensity <= 1.0:
            logger.warning(f"intensity 参数超出范围 [0.0, 1.0]: {intensity}，将被截断")
            intensity = max(0.0, min(1.0, intensity))

        with self._lock:
            changes = []
            for dim, val in snapshot.items():
                if dim in self.energy:
                    original_val = self.energy[dim]
                    # 简单累加共鸣
                    delta = val * intensity
                    self.energy[dim] += delta
                    # 应用软限制，与 update_energy 保持一致
                    self.energy[dim] = max(-20.0, min(20.0, self.energy[dim]))
                    changes.append(f"{dim}: {original_val:.1f}->{self.energy[dim]:.1f}")

            if changes:
                logger.debug(f"🎼 Soul Resonate: {', '.join(changes)}")

        # 不再自动保存到文件
        # self.save()
    def get_snapshot(self) -> Dict[str, float]:
        """获取当前状态快照（用于存入新记忆）"""
        with self._lock:
            return self.energy.copy()

    def get_state_description(self) -> str:
        """获取当前状态的文本描述（用于调试或Prompt注入）"""
        with self._lock:
            # Capture values inside lock for consistency
            v_recall = self.get_value('RecallDepth')
            v_impress = self.get_value('ImpressionDepth')
            v_express = self.get_value('ExpressionDesire')
            v_create = self.get_value('Creativity')

            e_recall = self.energy['RecallDepth']
            e_impress = self.energy['ImpressionDepth']
            e_express = self.energy['ExpressionDesire']
            e_create = self.energy['Creativity']

        desc = []
        desc.append(f"🧠 回忆倾向(Recall): {v_recall}条 (E={e_recall:.1f})")
        desc.append(f"📝 记住倾向(Impression): {v_impress}条 (E={e_impress:.1f})")
        desc.append(f"🗣️ 表达欲望(Expression): {v_express} Tokens (E={e_express:.1f})")
        desc.append(f"🎨 思维发散(Creativity): {v_create} Temp (E={e_create:.1f})")
        return " | ".join(desc)

    # 移除 save 和 load 方法，因为不需要持久化了