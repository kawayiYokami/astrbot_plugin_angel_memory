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
            # 按类型分组记忆
            memories_by_type = {}
            for memory in memories:
                memory_type_str = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, "value")
                    else str(memory.memory_type)
                )
                memory_type_key = MemoryConstants.MEMORY_TYPE_MAPPING.get(
                    memory_type_str, memory_type_str.lower()
                )
                memories_by_type.setdefault(memory_type_key, []).append(memory)

            # 处理每种类型的记忆
            for memory_type, type_memories in memories_by_type.items():
                if memory_type == "knowledge":
                    self._add_knowledge_memories_with_priority(type_memories)
                else:
                    self._add_regular_memories(type_memories, memory_type)

    def _add_knowledge_memories_with_priority(self, memories: List[BaseMemory]) -> None:
        """
        智能添加知识记忆，优先处理昵称/印象记忆

        Args:
            memories: 知识记忆列表
        """
        # 分离昵称/印象记忆和其他用户信息记忆
        nickname_memories = []
        user_info_memories = []
        regular_memories = []

        for memory in memories:
            memory_item = self._create_memory_item(memory, "knowledge")

            if self._is_nickname_memory(memory_item):
                nickname_memories.append(memory_item)
            elif self._is_user_info_memory(memory_item):
                user_info_memories.append(memory_item)
            else:
                regular_memories.append(memory_item)

        # 1. 优先添加昵称记忆（无容量限制）
        for memory_item in nickname_memories:
            self._add_memory_item(memory_item)

        # 2. 智能分配用户信息记忆容量
        if user_info_memories:
            self._add_user_info_memories_with_capacity_management(user_info_memories)

        # 3. 添加普通知识记忆
        for memory_item in regular_memories:
            self._add_memory_item(memory_item)

        # 4. 清理容量
        self._cleanup_by_capacity("knowledge")

    def _is_nickname_memory(self, memory: MemoryItem) -> bool:
        """
        判断是否为昵称/印象记忆

        Args:
            memory: 记忆对象

        Returns:
            是否为昵称/印象记忆
        """
        if not hasattr(memory, "tags") or not isinstance(memory.tags, list):
            return False

        # 检查是否包含"昵称"、"印象"等标签
        nickname_tags = ["昵称", "印象", "称呼", "名字"]
        return any(tag in memory.tags for tag in nickname_tags)

    def _is_user_info_memory(self, memory: MemoryItem) -> bool:
        """
        判断是否为用户相关信息记忆

        Args:
            memory: 记忆对象

        Returns:
            是否为用户相关信息记忆
        """
        if not hasattr(memory, "tags") or not isinstance(memory.tags, list):
            return False

        user_tags = [tag for tag in memory.tags if tag == "用户"]
        id_tags = [tag for tag in memory.tags if tag.isdigit() and len(tag) > 5]

        return len(user_tags) > 0 and len(id_tags) > 0

    def _add_user_info_memories_with_capacity_management(self, user_info_memories: List[MemoryItem]) -> None:
        """
        按用户剩余容量智能分配用户信息记忆

        Args:
            user_info_memories: 用户信息记忆列表
        """
        if not user_info_memories:
            return

        # 获取总容量和当前各用户的使用情况
        total_capacity = int(getattr(self.capacity_config, "knowledge_user_info", 0) * self.capacity_multiplier)
        current_user_memories = self._get_current_user_info_memories()

        # 计算每个用户的剩余容量
        user_remaining_capacity = self._calculate_user_remaining_capacity(current_user_memories, total_capacity)

        # 按强度排序记忆
        user_info_memories.sort(key=lambda x: (-x.strength, x.created_at))

        # 为每个用户分配记忆
        for memory in user_info_memories:
            user_id = self._extract_user_id(memory)
            if user_id and user_remaining_capacity.get(user_id, 0) > 0:
                self._add_memory_item(memory)
                user_remaining_capacity[user_id] -= 1
            # 用户容量已满时直接跳过，不添加记忆

    def _get_current_user_info_memories(self) -> Dict[str, List[MemoryItem]]:
        """
        获取当前各用户的记忆信息

        Returns:
            按用户ID分组的记忆字典
        """
        user_memories = {}
        for memory in self.memories:
            if memory.memory_type == "knowledge" and self._is_user_info_memory(memory):
                user_id = self._extract_user_id(memory)
                if user_id:
                    user_memories.setdefault(user_id, []).append(memory)
        return user_memories

    def _calculate_user_remaining_capacity(self, current_user_memories: Dict[str, List[MemoryItem]], total_capacity: int) -> Dict[str, int]:
        """
        计算每个用户的剩余容量

        Args:
            current_user_memories: 当前用户记忆字典
            total_capacity: 总容量

        Returns:
            每个用户的剩余容量字典
        """
        num_users = len(current_user_memories)
        if num_users == 0:
            return {}

        capacity_per_user = total_capacity // num_users if num_users > 0 else 0
        remaining_capacity = {}

        for user_id, memories in current_user_memories.items():
            used = len(memories)
            remaining_capacity[user_id] = max(0, capacity_per_user - used)

        return remaining_capacity

    

    def _add_regular_memories(self, memories: List[BaseMemory], memory_type: str) -> None:
        """
        添加普通类型的记忆

        Args:
            memories: 记忆列表
            memory_type: 记忆类型
        """
        for memory in memories:
            memory_item = self._create_memory_item(memory, memory_type)
            self._add_memory_item(memory_item)

        # 清理容量
        self._cleanup_by_capacity(memory_type)

    def _create_memory_item(self, memory: BaseMemory, memory_type: str) -> MemoryItem:
        """
        创建记忆项对象

        Args:
            memory: 基础记忆对象
            memory_type: 记忆类型

        Returns:
            记忆项对象
        """
        return MemoryItem(
            id=memory.id,
            memory_type=memory_type,
            judgment=getattr(memory, "judgment", ""),
            reasoning=getattr(memory, "reasoning", ""),
            tags=getattr(memory, "tags", []),
            strength=getattr(memory, "strength", 0),
            life_points=3,  # 新记忆默认3点生命值
            created_at=time.time(),  # 记录创建时间
        )

    def _add_memory_item(self, memory_item: MemoryItem) -> None:
        """
        添加单个记忆项

        Args:
            memory_item: 记忆项对象
        """
        # 如果记忆已存在，先移除旧的
        if memory_item.id in self.memory_map:
            old_memory = self.memory_map[memory_item.id]
            if old_memory in self.memories:
                self.memories.remove(old_memory)

        # 添加新记忆
        self.memories.append(memory_item)
        self.memory_map[memory_item.id] = memory_item

    def _cleanup_by_capacity(self, memory_type: str) -> None:
        """
        根据容量清理记忆：删除该类型中生命值最低且最旧的记忆
        知识记忆类型有独立的用户信息容量管理

        Args:
            memory_type: 记忆类型
        """
        # 获取该类型的记忆
        type_memories = [
            memory for memory in self.memories if memory.memory_type == memory_type
        ]

        # 知识记忆类型需要特殊处理
        if memory_type == "knowledge":
            self._cleanup_knowledge_memories(type_memories)
            return

        # 其他类型按原逻辑处理
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

    def _cleanup_knowledge_memories(self, knowledge_memories: List[MemoryItem]) -> None:
        """
        清理知识记忆，分别管理普通知识记忆和用户信息记忆
        用户信息记忆按用户平分容量

        Args:
            knowledge_memories: 所有知识记忆列表
        """
        # 分离用户信息记忆和普通知识记忆
        user_info_memories = []
        regular_memories = []

        for memory in knowledge_memories:
            if self._is_user_info_memory(memory):
                user_info_memories.append(memory)
            else:
                regular_memories.append(memory)

        # 获取容量限制
        regular_capacity = int(getattr(self.capacity_config, "knowledge", 0) * self.capacity_multiplier)
        user_info_capacity = int(getattr(self.capacity_config, "knowledge_user_info", 0) * self.capacity_multiplier)

        # 清理普通知识记忆
        regular_excess = len(regular_memories) - regular_capacity
        if regular_excess > 0:
            regular_memories.sort(key=lambda x: (x.life_points, x.created_at))
            for i in range(regular_excess):
                memory_to_remove = regular_memories[i]
                if memory_to_remove in self.memories:
                    self.memories.remove(memory_to_remove)
                if memory_to_remove.id in self.memory_map:
                    del self.memory_map[memory_to_remove.id]

        # 按用户分组清理用户信息记忆
        self._cleanup_user_info_memories_by_user(user_info_memories, user_info_capacity)

    def _cleanup_user_info_memories_by_user(self, user_info_memories: List[MemoryItem], total_capacity: int) -> None:
        """
        按用户平分容量清理用户信息记忆

        Args:
            user_info_memories: 用户信息记忆列表
            total_capacity: 用户信息记忆总容量
        """
        if not user_info_memories or total_capacity <= 0:
            return

        # 按用户ID分组
        user_groups = {}
        for memory in user_info_memories:
            user_id = self._extract_user_id(memory)
            if user_id:
                user_groups.setdefault(user_id, []).append(memory)

        if not user_groups:
            return

        # 计算每个用户的容量（平分）
        num_users = len(user_groups)
        capacity_per_user = total_capacity // num_users if num_users > 0 else 0

        # 为每个用户清理记忆
        for user_id, memories in user_groups.items():
            # 按强度排序，优先保留高强度记忆
            memories.sort(key=lambda x: (-x.strength, x.created_at))

            # 删除超出的记忆
            for memory in memories[capacity_per_user:]:
                if memory in self.memories:
                    self.memories.remove(memory)
                if memory.id in self.memory_map:
                    del self.memory_map[memory.id]

    def _extract_user_id(self, memory: MemoryItem) -> str:
        """
        从记忆中提取用户ID

        Args:
            memory: 记忆对象

        Returns:
            用户ID字符串，如果未找到返回空字符串
        """
        if hasattr(memory, "tags") and isinstance(memory.tags, list):
            for tag in memory.tags:
                if tag.isdigit() and len(tag) > 5:
                    return tag
        return ""

    def get_memories(self) -> List[MemoryItem]:
        """
        获取会话中的所有记忆

        Returns:
            记忆列表
        """
        with self.lock:
            return list(self.memories)



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
                # 但保护用户信息记忆，不轻易删除
                if memory.life_points <= 0 and not self._is_user_info_memory(memory):
                    memories_to_remove.append(memory)
                elif memory.life_points <= -2:  # 用户信息记忆最多只能降到-2
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

                # 知识记忆需要特殊处理
                if memory_type == "knowledge":
                    # 判断是否为用户信息记忆
                    if self._is_user_info_memory(memory):
                        sub_type = "knowledge_user_info"
                    else:
                        sub_type = "knowledge_regular"
                else:
                    sub_type = memory_type

                if sub_type not in stats["by_type"]:
                    if sub_type == "knowledge_regular":
                        capacity = int(
                            getattr(self.capacity_config, "knowledge", 0)
                            * self.capacity_multiplier
                        )
                    elif sub_type == "knowledge_user_info":
                        capacity = int(
                            getattr(self.capacity_config, "knowledge_user_info", 0)
                            * self.capacity_multiplier
                        )
                    else:
                        capacity = int(
                            getattr(self.capacity_config, memory_type, 0)
                            * self.capacity_multiplier
                        )

                    stats["by_type"][sub_type] = {
                        "current": 0,
                        "capacity": capacity,
                        "usage": 0.0,
                    }
                stats["by_type"][sub_type]["current"] += 1

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
