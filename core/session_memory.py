"""
会话短期记忆管理模块

负责管理所有会话的短期记忆，支持并发和线程安全。
实现FIFO策略的记忆通道管理。
"""

import threading
from typing import Dict, List, Any
from collections import deque
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


class SessionMemory:
    """单个会话的短期记忆管理"""

    def __init__(self, session_id: str, capacity_config: MemoryCapacityConfig, capacity_multiplier: float = 1.0):
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

        # 初始化各类型记忆通道（使用deque实现FIFO）
        self.memories = {
            memory_type: deque(maxlen=int(getattr(capacity_config, memory_type, 0) * capacity_multiplier))
            for memory_type in ['knowledge', 'emotional', 'skill', 'task', 'event']
        }

        # 记忆ID到记忆项的映射（用于快速查找）
        self.memory_map = {}

    def add_memories(self, memories: List[BaseMemory]) -> None:
        """
        添加记忆到会话中

        Args:
            memories: 记忆列表
        """
        with self.lock:
            for memory in memories:
                # 获取记忆类型字符串（处理枚举类型）
                memory_type_str = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)

                # 将中文枚举值转换为英文键
                memory_type_key = MemoryConstants.MEMORY_TYPE_MAPPING.get(memory_type_str, memory_type_str.lower())

                memory_item = MemoryItem(
                    id=memory.id,
                    memory_type=memory_type_key,
                    judgment=getattr(memory, 'judgment', ''),
                    reasoning=getattr(memory, 'reasoning', ''),
                    tags=getattr(memory, 'tags', []),
                    strength=getattr(memory, 'strength', 0)
                )

                # 如果记忆已存在，先移除旧的
                if memory.id in self.memory_map:
                    old_memory = self.memory_map[memory.id]
                    if old_memory.memory_type in self.memories:
                        try:
                            self.memories[old_memory.memory_type].remove(old_memory)
                        except ValueError:
                            pass  # 记忆可能已经被FIFO移除

                # 添加新记忆
                self.memories[memory_type_key].append(memory_item)
                self.memory_map[memory.id] = memory_item

    def get_memories(self) -> List[MemoryItem]:
        """
        获取会话中的所有记忆

        Returns:
            记忆列表
        """
        with self.lock:
            all_memories = []
            for memory_type, memory_deque in self.memories.items():
                all_memories.extend(list(memory_deque))

            return all_memories

    def get_memories_by_type(self, memory_type: str) -> List[MemoryItem]:
        """
        获取指定类型的记忆

        Args:
            memory_type: 记忆类型

        Returns:
            指定类型的记忆列表
        """
        with self.lock:
            if memory_type in self.memories:
                return list(self.memories[memory_type])
            return []

    def clear(self) -> None:
        """清空会话记忆"""
        with self.lock:
            for memory_deque in self.memories.values():
                memory_deque.clear()
            self.memory_map.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取会话记忆统计信息

        Returns:
            统计信息字典
        """
        with self.lock:
            stats = {
                'session_id': self.session_id,
                'capacity_multiplier': self.capacity_multiplier,
                'total_memories': len(self.memory_map),
                'by_type': {}
            }

            for memory_type, memory_deque in self.memories.items():
                base_capacity = getattr(self.capacity_config, memory_type, 0)
                capacity = int(base_capacity * self.capacity_multiplier)
                stats['by_type'][memory_type] = {
                    'current': len(memory_deque),
                    'capacity': capacity,
                    'usage': len(memory_deque) / capacity if capacity > 0 else 0
                }

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
                    session_id,
                    self.capacity_config,
                    self.capacity_multiplier
                )
            return self.sessions[session_id]

    def add_memories_to_session(self, session_id: str, memories: List[BaseMemory]) -> None:
        """
        向指定会话添加记忆

        Args:
            session_id: 会话ID
            memories: 记忆列表
        """
        session = self.get_or_create_session(session_id)
        session.add_memories(memories)

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