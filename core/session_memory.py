"""
短期记忆模块

负责管理所有会话的短期记忆，支持并发和线程安全。

淘汰有两条路：
一是槽位配额，每种记忆类型各有容量，超出就按生命值低、创建早的先走；
二是消息绑定，带着这条记忆进来的消息离开请求，它就跟着走，不看生命值。
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass

from ..llm_memory.models.data_models import BaseMemory
from .config import MemoryCapacityConfig, MemoryConstants


@dataclass
class MemoryItem:
    """短期记忆里的一条记忆条目。

    记忆内容跟着记忆条目一起存，注入和反思都直接用短期记忆里的这份，不再回查长期库。
    代价是长期记忆被改写后，短期记忆里仍是进入时的文本，下一次进入前保持原样。
    """

    id: str
    memory_type: str
    added_at: float = 0.0  # 进入短期记忆的时间戳
    life_points: int = 3  # 生命值，每被判定有用加一点，每轮没被判定有用减一点
    created_at: float = 0.0  # 记忆本身的创建时间，淘汰时先走更旧的
    bound_message_id: str = ""  # 绑定消息 ID，该消息离场则这条记忆条目随之离场
    bound_message_text: str = ""  # 绑定消息正文，离场时要连它一起送审
    memory: Any = None  # 记忆本体，进入时一并留下


class SessionMemory:
    """单个会话的短期记忆。"""

    def __init__(
        self,
        session_id: str,
        capacity_config: Optional[MemoryCapacityConfig] = None,
        capacity_provider: Optional[Callable[[], float]] = None,
    ):
        """
        初始化会话记忆

        Args:
            session_id: 会话ID
            capacity_config: 按类型的容量配额
            capacity_provider: 容量倍数的现读函数，每次淘汰前取一次，配置改了立刻生效
        """
        self.session_id = session_id
        self.capacity_config = capacity_config or MemoryCapacityConfig()
        self.capacity_provider = capacity_provider
        self.lock = threading.Lock()
        self.memories = []  # List[MemoryItem]
        self.memory_map = {}  # Dict[str, MemoryItem]

    def _current_multiplier(self) -> float:
        """取当前容量倍数。"""
        if self.capacity_provider is None:
            return MemoryConstants.SHORT_TERM_MEMORY_CAPACITY
        try:
            value = float(self.capacity_provider())
        except Exception:
            return MemoryConstants.SHORT_TERM_MEMORY_CAPACITY
        return value if value > 0 else MemoryConstants.SHORT_TERM_MEMORY_CAPACITY

    def _capacity_of(self, memory_type: str, multiplier: float) -> int:
        """某类型当前的槽位数量，未配置的类型不设上限。"""
        base = getattr(self.capacity_config, memory_type, 0) or 0
        return int(base * multiplier)

    def add_memories(
        self,
        memories: List[BaseMemory],
        bound_message_id: str = "",
        bound_message_text: str = "",
    ) -> None:
        """
        将记忆条目放入短期记忆

        Args:
            memories: 长期记忆列表
            bound_message_id: 绑定消息 ID，该消息离场后这条记忆条目随之离场
            bound_message_text: 绑定消息正文，离场时反思要带上
        """
        if not memories:
            return
        multiplier = self._current_multiplier()
        with self.lock:
            for memory in memories:
                self._add_memory_item(
                    self._create_memory_item(
                        memory,
                        bound_message_id=bound_message_id,
                        bound_message_text=bound_message_text,
                    ),
                    multiplier,
                )

    @staticmethod
    def _safe_parse_created_at(value) -> float:
        """把记忆的创建时间统一解析成时间戳，脏数据按当前时间处理。"""
        now = time.time()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return now
            try:
                return float(stripped)
            except ValueError:
                pass
            try:
                from datetime import datetime

                return datetime.fromisoformat(stripped).timestamp()
            except (ValueError, AttributeError):
                pass
        return now

    def _create_memory_item(
        self,
        memory: BaseMemory,
        bound_message_id: str = "",
        bound_message_text: str = "",
    ) -> MemoryItem:
        """创建记忆条目。"""
        raw_memory_type = getattr(memory, "memory_type", "")
        type_value = (
            raw_memory_type.value
            if hasattr(raw_memory_type, "value")
            else str(raw_memory_type or "")
        )
        return MemoryItem(
            id=str(getattr(memory, "id", "") or "").strip(),
            memory_type=MemoryConstants.MEMORY_TYPE_MAPPING.get(
                type_value, type_value.lower()
            ),
            added_at=time.time(),
            life_points=3,
            created_at=self._safe_parse_created_at(
                getattr(memory, "created_at", None)
            ),
            bound_message_id=str(bound_message_id or "").strip(),
            bound_message_text=str(bound_message_text or "").strip(),
            memory=memory,
        )

    def _add_memory_item(self, memory_item: MemoryItem, multiplier: float) -> None:
        """
        添加单个记忆条目，并检查该类型的槽位是否还够

        Args:
            memory_item: 记忆条目
            multiplier: 当前容量倍数
        """
        if not memory_item.id:
            return

        # 进入侧去重：同一条记忆只绑定一次，后来的召回不再刷新它
        if memory_item.id in self.memory_map:
            return

        self.memories.append(memory_item)
        self.memory_map[memory_item.id] = memory_item
        self._cleanup_by_capacity(memory_item.memory_type, multiplier)

    def _cleanup_by_capacity(
        self, memory_type: str, multiplier: float
    ) -> List[MemoryItem]:
        """按类型槽位清理：生命值低的先走，同样低则更旧的先走。"""
        capacity = self._capacity_of(memory_type, multiplier)
        if capacity <= 0:
            return []

        type_memories = [
            memory for memory in self.memories if memory.memory_type == memory_type
        ]
        excess = len(type_memories) - capacity
        if excess <= 0:
            return []

        type_memories.sort(key=lambda memory: (memory.life_points, memory.created_at))
        removed = type_memories[:excess]
        self._remove_items(removed)
        return removed

    def _remove_items(self, items: List[MemoryItem]) -> None:
        """从短期记忆里移除指定条目。"""
        removed_ids = {item.id for item in items}
        self.memories = [
            memory for memory in self.memories if memory.id not in removed_ids
        ]
        for memory_id in removed_ids:
            self.memory_map.pop(memory_id, None)

    def evaluate_life_points(self, useful_memory_ids) -> List[MemoryItem]:
        """按这一轮的有用判定调整生命值，生命值归零的条目当场淘汰。"""
        useful = {
            str(mid).strip() for mid in (useful_memory_ids or []) if str(mid).strip()
        }
        with self.lock:
            dead = []
            for memory in self.memories:
                memory.life_points += 1 if memory.id in useful else -1
                if memory.life_points <= 0:
                    dead.append(memory)
            if dead:
                self._remove_items(dead)
            return dead

    def prune_absent_messages(self, alive_message_ids) -> List[MemoryItem]:
        """移除绑定消息已不在本次请求里的记忆条目，返回被移除的记忆条目本身。"""
        alive = {
            str(mid).strip() for mid in (alive_message_ids or []) if str(mid).strip()
        }
        with self.lock:
            departed = [
                memory
                for memory in self.memories
                if not memory.bound_message_id or memory.bound_message_id not in alive
            ]
            if not departed:
                return []
            self._remove_items(departed)
            return departed

    def get_memories(self) -> List[MemoryItem]:
        """
        获取短期记忆中的所有记忆条目

        Returns:
            记忆条目列表
        """
        with self.lock:
            return list(self.memories)

    def clear(self) -> None:
        """清空短期记忆。"""
        with self.lock:
            self.memories.clear()
            self.memory_map.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取短期记忆统计信息

        Returns:
            统计信息字典
        """
        multiplier = self._current_multiplier()
        with self.lock:
            by_type: Dict[str, Dict[str, Any]] = {}
            for memory in self.memories:
                slot = by_type.setdefault(
                    memory.memory_type,
                    {
                        "current": 0,
                        "capacity": self._capacity_of(memory.memory_type, multiplier),
                    },
                )
                slot["current"] += 1
            return {
                "session_id": self.session_id,
                "capacity_multiplier": multiplier,
                "total_memories": len(self.memories),
                "by_type": by_type,
            }


class SessionMemoryManager:
    """短期记忆管理器，管理所有会话的短期记忆。"""

    def __init__(
        self,
        capacity_config: Optional[MemoryCapacityConfig] = None,
        capacity_provider: Optional[Callable[[], float]] = None,
    ):
        """
        Args:
            capacity_config: 按类型的容量配额
            capacity_provider: 容量倍数的现读函数
        """
        self.capacity_config = capacity_config or MemoryCapacityConfig()
        self.capacity_provider = capacity_provider
        self.lock = threading.Lock()
        self.sessions: Dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """
        获取或创建短期记忆

        Args:
            session_id: 会话ID

        Returns:
            短期记忆对象
        """
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMemory(
                    session_id, self.capacity_config, self.capacity_provider
                )
            return self.sessions[session_id]

    def add_memories_to_session(
        self,
        session_id: str,
        memories: List[BaseMemory],
        bound_message_id: str = "",
        bound_message_text: str = "",
    ) -> None:
        """
        向指定会话的短期记忆添加记忆条目

        Args:
            session_id: 会话ID
            memories: 记忆列表
            bound_message_id: 绑定消息 ID
            bound_message_text: 绑定消息正文
        """
        session = self.get_or_create_session(session_id)
        session.add_memories(memories, bound_message_id, bound_message_text)

    def evaluate_feedback_in_session(
        self, session_id: str, useful_memory_ids
    ) -> List[MemoryItem]:
        """按这一轮的有用判定调整该会话记忆的生命值，返回被淘汰的条目。"""
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.evaluate_life_points(useful_memory_ids)

    def prune_absent_messages_from_session(
        self, session_id: str, alive_message_ids
    ) -> List[MemoryItem]:
        """按本次请求实际存在的消息清理短期记忆，返回被移除的记忆条目本身。"""
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.prune_absent_messages(alive_message_ids)

    def get_session_memories(self, session_id: str) -> List[MemoryItem]:
        """
        获取短期记忆

        Args:
            session_id: 会话ID

        Returns:
            记忆条目列表
        """
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].get_memories()
            return []

    def clear_session(self, session_id: str) -> None:
        """
        清空指定会话的短期记忆

        Args:
            session_id: 会话ID
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                del self.sessions[session_id]

    def get_all_session_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取全部会话的统计信息。"""
        with self.lock:
            sessions = list(self.sessions.items())
        return {session_id: session.get_stats() for session_id, session in sessions}
