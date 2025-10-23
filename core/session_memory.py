"""
会话短期记忆管理模块

负责管理所有会话的短期记忆，支持并发和线程安全。
实现FIFO策略的记忆通道管理。
"""

import threading
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from ..llm_memory.models.data_models import BaseMemory
from .config import MemoryConstants, MemoryCapacityConfig


@dataclass
class MemoryItem:
    """记忆项数据结构"""

    id: str
    memory_type: str
    judgment: str
    reasoning: str
    tags: List[str]
    strength: int = 0
    life_points: int = 3  # 生命值，新记忆默认3点
    created_at: float = 0.0  # 创建时间戳


class SessionMemory:
    """单个会话的短期记忆管理"""

    def __init__(
        self,
        session_id: str,
        capacity_config: MemoryCapacityConfig,
        capacity_multiplier: float = 1.0,
    ):
        """
        初始化会话记忆

        Args:
            session_id: 会话ID
            capacity_config: 容量配置
            capacity_multiplier: 容量倍数
        """
        self.session_id = session_id
        self.capacity_config = capacity_config
        self.capacity_multiplier = capacity_multiplier
        self.lock = threading.Lock()

        # 使用普通列表存储记忆（移除FIFO机制）
        self.memories = []  # List[MemoryItem]

        # 记忆ID到记忆项的映射（用于快速查找）
        self.memory_map = {}  # Dict[str, MemoryItem]

    def add_memories(self, memories: List[BaseMemory]) -> None:
        """
        添加记忆到会话中

        Args:
            memories: 记忆列表
        """
        with self.lock:
            for memory in memories:
                # 获取记忆类型字符串（处理枚举类型）
                memory_type_str = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, "value")
                    else str(memory.memory_type)
                )

                # 将中文枚举值转换为英文键
                memory_type_key = MemoryConstants.MEMORY_TYPE_MAPPING.get(
                    memory_type_str, memory_type_str.lower()
                )

                memory_item = MemoryItem(
                    id=memory.id,
                    memory_type=memory_type_key,
                    judgment=getattr(memory, "judgment", ""),
                    reasoning=getattr(memory, "reasoning", ""),
                    tags=getattr(memory, "tags", []),
                    strength=getattr(memory, "strength", 0),
                    life_points=3,  # 新记忆默认3点生命值
                    created_at=time.time(),  # 记录创建时间
                )

                # 如果记忆已存在，先移除旧的
                if memory.id in self.memory_map:
                    old_memory = self.memory_map[memory.id]
                    if old_memory in self.memories:
                        self.memories.remove(old_memory)

                # 添加新记忆
                self.memories.append(memory_item)
                self.memory_map[memory.id] = memory_item

                # 检查容量并智能清理
                self._cleanup_by_capacity(memory_type_key)

    def _cleanup_by_capacity(self, memory_type: str) -> None:
        """
        根据容量清理记忆：删除该类型中生命值最低且最旧的记忆

        Args:
            memory_type: 记忆类型
        """
        # 获取该类型的记忆
        type_memories = [
            memory for memory in self.memories if memory.memory_type == memory_type
        ]

        # 获取该类型的容量限制
        base_capacity = getattr(self.capacity_config, memory_type, 0)
        capacity = int(base_capacity * self.capacity_multiplier)

        # 如果容量为0或未超限，不需要清理
        if capacity <= 0 or len(type_memories) <= capacity:
            return

        # 计算需要删除的数量
        excess_count = len(type_memories) - capacity

        # 按生命值升序，时间戳升序排序（生命值最低且最旧的排在前面）
        type_memories.sort(key=lambda x: (x.life_points, x.created_at))

        # 删除最差的记忆
        for i in range(excess_count):
            memory_to_remove = type_memories[i]
            if memory_to_remove in self.memories:
                self.memories.remove(memory_to_remove)
            if memory_to_remove.id in self.memory_map:
                del self.memory_map[memory_to_remove.id]

    def get_memories(self) -> List[MemoryItem]:
        """
        获取会话中的所有记忆

        Returns:
            记忆列表
        """
        with self.lock:
            return list(self.memories)

    def get_memories_by_type(self, memory_type: str) -> List[MemoryItem]:
        """
        获取指定类型的记忆

        Args:
            memory_type: 记忆类型

        Returns:
            指定类型的记忆列表
        """
        with self.lock:
            return [
                memory for memory in self.memories if memory.memory_type == memory_type
            ]

    def clear(self) -> None:
        """清空会话记忆"""
        with self.lock:
            self.memories.clear()
            self.memory_map.clear()

    def update_memories(
        self, new_memories: List[BaseMemory], useful_memory_ids: List[str]
    ) -> None:
        """
        更新会话记忆：添加新记忆，评估现有记忆，清理死亡记忆

        Args:
            new_memories: 新记忆列表
            useful_memory_ids: 有用记忆ID列表
        """
        with self.lock:
            # 1. 添加新记忆
            for memory in new_memories:
                memory_type_str = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, "value")
                    else str(memory.memory_type)
                )
                memory_type_key = MemoryConstants.MEMORY_TYPE_MAPPING.get(
                    memory_type_str, memory_type_str.lower()
                )

                memory_item = MemoryItem(
                    id=memory.id,
                    memory_type=memory_type_key,
                    judgment=getattr(memory, "judgment", ""),
                    reasoning=getattr(memory, "reasoning", ""),
                    tags=getattr(memory, "tags", []),
                    strength=getattr(memory, "strength", 0),
                    life_points=3,  # 新记忆默认3点生命值
                    created_at=time.time(),  # 记录创建时间
                )

                # 如果记忆已存在，先移除旧的
                if memory.id in self.memory_map:
                    old_memory = self.memory_map[memory.id]
                    if old_memory in self.memories:
                        self.memories.remove(old_memory)

                # 添加新记忆
                self.memories.append(memory_item)
                self.memory_map[memory.id] = memory_item

                # 检查容量并智能清理
                self._cleanup_by_capacity(memory_type_key)

            # 2. 评估现有记忆的生命值
            memories_to_remove = []
            for memory in self.memories:
                if memory.id in useful_memory_ids:
                    # 有用记忆+1生命值
                    memory.life_points += 1
                else:
                    # 其他记忆-1生命值
                    memory.life_points -= 1

                # 记录需要删除的记忆（生命值为0）
                if memory.life_points <= 0:
                    memories_to_remove.append(memory)

            # 3. 清理死亡记忆
            for memory in memories_to_remove:
                if memory in self.memories:
                    self.memories.remove(memory)
                if memory.id in self.memory_map:
                    del self.memory_map[memory.id]

    def get_stats(self) -> Dict[str, Any]:
        """
        获取会话记忆统计信息

        Returns:
            统计信息字典
        """
        with self.lock:
            stats = {
                "session_id": self.session_id,
                "capacity_multiplier": self.capacity_multiplier,
                "total_memories": len(self.memories),
                "by_type": {},
                "life_points_distribution": {
                    "high": 0,  # >5
                    "medium": 0,  # 2-5
                    "low": 0,  # 1
                    "critical": 0,  # =0 (即将死亡)
                },
            }

            # 按类型统计
            for memory in self.memories:
                memory_type = memory.memory_type
                if memory_type not in stats["by_type"]:
                    stats["by_type"][memory_type] = {
                        "current": 0,
                        "capacity": int(
                            getattr(self.capacity_config, memory_type, 0)
                            * self.capacity_multiplier
                        ),
                        "usage": 0.0,
                    }
                stats["by_type"][memory_type]["current"] += 1

                # 生命值分布统计
                if memory.life_points > 5:
                    stats["life_points_distribution"]["high"] += 1
                elif memory.life_points >= 2:
                    stats["life_points_distribution"]["medium"] += 1
                elif memory.life_points == 1:
                    stats["life_points_distribution"]["low"] += 1

            # 计算使用率
            for memory_type, type_stats in stats["by_type"].items():
                capacity = type_stats["capacity"]
                if capacity > 0:
                    type_stats["usage"] = type_stats["current"] / capacity

            return stats


class SessionMemoryManager:
    """会话记忆管理器 - 管理所有会话的短期记忆"""

    def __init__(self, capacity_multiplier: float = 1.0):
        """
        初始化会话记忆管理器

        Args:
            capacity_multiplier: 容量倍数
        """
        self.capacity_multiplier = capacity_multiplier
        self.capacity_config = MemoryCapacityConfig()
        self.lock = threading.Lock()
        self.sessions: Dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """
        获取或创建会话记忆

        Args:
            session_id: 会话ID

        Returns:
            会话记忆对象
        """
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMemory(
                    session_id, self.capacity_config, self.capacity_multiplier
                )
            return self.sessions[session_id]

    def add_memories_to_session(
        self, session_id: str, memories: List[BaseMemory]
    ) -> None:
        """
        向指定会话添加记忆

        Args:
            session_id: 会话ID
            memories: 记忆列表
        """
        session = self.get_or_create_session(session_id)
        session.add_memories(memories)

    def update_session_memories(
        self,
        session_id: str,
        new_memories: List[BaseMemory],
        useful_memory_ids: List[str],
    ) -> None:
        """
        更新会话记忆：添加新记忆，评估现有记忆，清理死亡记忆

        Args:
            session_id: 会话ID
            new_memories: 新记忆列表
            useful_memory_ids: 有用记忆ID列表
        """
        session = self.get_or_create_session(session_id)
        session.update_memories(new_memories, useful_memory_ids)

    def get_session_memories(self, session_id: str) -> List[MemoryItem]:
        """
        获取会话记忆

        Args:
            session_id: 会话ID

        Returns:
            记忆列表
        """
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].get_memories()
            return []

    def clear_session(self, session_id: str) -> None:
        """
        清空指定会话的记忆

        Args:
            session_id: 会话ID
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                del self.sessions[session_id]

    def get_all_session_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有会话的统计信息

        Returns:
            所有会话的统计信息
        """
        with self.lock:
            return {
                session_id: session.get_stats()
                for session_id, session in self.sessions.items()
            }
