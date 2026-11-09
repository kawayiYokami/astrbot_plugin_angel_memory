"""
短期记忆模块

负责管理所有会话的短期记忆，支持并发和线程安全。

一条记录就是一行：记忆本体，加上最后一次召回它的原消息。
淘汰有两条路：
一是槽位配额，每种记忆类型各有容量，超出就先淘汰最久没被召回的那条；
二是来源离场，最后召回它的那条消息已经不在本轮上下文里，它就跟着走。
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass

from ..llm_memory.models.data_models import BaseMemory
from .config import MemoryCapacityConfig, MemoryConstants


def is_source_alive(bound_text: str, alive_texts) -> bool:
    """来源消息还在不在：正文与某条存活消息相同。

    写入与比对取的是同一份请求体文本——当前发言取 `request.prompt`，历史取
    `request.contexts`，两边口径一致，所以按全等比较，不做剥离也不做包含。
    """
    bound = str(bound_text or "").strip()
    if not bound:
        return False
    for text in alive_texts or []:
        if str(text or "").strip() == bound:
            return True
    return False


@dataclass
class MemoryItem:
    """短期记忆里的一条记忆条目。

    记忆内容跟着条目一起存，注入和反思都直接用这里的这份，不再回查长期库。
    recalled_message_* 记的是最后一次召回它的原消息，用来判断这条记忆的来源还在不在。
    """

    id: str
    memory_type: str
    added_at: float = 0.0  # 第一次写进短期记忆的时间，之后不再变
    recalled_at: float = 0.0  # 最后一次被召回的时间，每次召回都覆盖
    recalled_message_id: str = ""  # 最后一次召回它的那条消息 ID
    recalled_message_text: str = ""  # 那条消息的正文，用来跟本轮上下文比对
    memory: Any = None  # 记忆本体

    def recall_again(self, message_id: str, message_text: str, at: float) -> None:
        """同一轮或后续轮次再次召回：刷新召回时间与来源消息，写入时间保持不变。"""
        self.recalled_at = at
        self.recalled_message_id = message_id
        self.recalled_message_text = message_text


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
        recalled_message_id: str = "",
        recalled_message_text: str = "",
    ) -> None:
        """
        把这一轮召回的记忆放进短期记忆

        已经在表里的条目不会新建，只刷新它的召回时间和来源消息。

        Args:
            memories: 本轮召回的记忆列表
            recalled_message_id: 本轮触发召回的消息 ID
            recalled_message_text: 那条消息的正文
        """
        if not memories:
            return
        multiplier = self._current_multiplier()
        now = time.time()
        message_id = str(recalled_message_id or "").strip()
        message_text = str(recalled_message_text or "").strip()
        with self.lock:
            for memory in memories:
                existing = self.memory_map.get(str(getattr(memory, "id", "") or "").strip())
                if existing is not None:
                    existing.recall_again(message_id, message_text, now)
                    existing.memory = memory
                    continue
                self._add_memory_item(
                    self._create_memory_item(memory, message_id, message_text, now),
                    multiplier,
                )

    def _create_memory_item(
        self,
        memory: BaseMemory,
        recalled_message_id: str = "",
        recalled_message_text: str = "",
        at: float = 0.0,
    ) -> MemoryItem:
        """创建记忆条目。"""
        raw_memory_type = getattr(memory, "memory_type", "")
        type_value = (
            raw_memory_type.value
            if hasattr(raw_memory_type, "value")
            else str(raw_memory_type or "")
        )
        moment = at or time.time()
        return MemoryItem(
            id=str(getattr(memory, "id", "") or "").strip(),
            memory_type=MemoryConstants.MEMORY_TYPE_MAPPING.get(
                type_value, type_value.lower()
            ),
            added_at=moment,
            recalled_at=moment,
            recalled_message_id=recalled_message_id,
            recalled_message_text=recalled_message_text,
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
        self.memories.append(memory_item)
        self.memory_map[memory_item.id] = memory_item
        self._cleanup_by_capacity(memory_item.memory_type, multiplier)

    def _cleanup_by_capacity(
        self, memory_type: str, multiplier: float
    ) -> List[MemoryItem]:
        """按类型槽位清理：最久没被召回的先走，同分时更早写入的先走。"""
        capacity = self._capacity_of(memory_type, multiplier)
        if capacity <= 0:
            return []

        type_memories = [
            memory for memory in self.memories if memory.memory_type == memory_type
        ]
        excess = len(type_memories) - capacity
        if excess <= 0:
            return []

        type_memories.sort(key=lambda memory: (memory.recalled_at, memory.added_at))
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

    def prune_departed(self, alive_texts) -> List[MemoryItem]:
        """移除来源消息已经不在本轮上下文里的记忆条目，返回被移除的条目本身。

        `alive_texts` 是本轮上下文里所有用户消息的纯文本。条目存的是它当初作为
        `request.prompt` 的那份文本，与上下文同一口径，所以按全等比较，不做剥离
        也不做包含。没有来源文本的条目不参与判定，只受槽位淘汰。
        """
        alive = [str(text or "").strip() for text in (alive_texts or [])]
        alive = [text for text in alive if text]
        with self.lock:
            departed = [
                memory
                for memory in self.memories
                if memory.recalled_message_text
                and not is_source_alive(memory.recalled_message_text, alive)
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
        recalled_message_id: str = "",
        recalled_message_text: str = "",
    ) -> None:
        """
        向指定会话的短期记忆添加记忆条目

        Args:
            session_id: 会话ID
            memories: 记忆列表
            recalled_message_id: 本轮触发召回的消息 ID
            recalled_message_text: 那条消息的正文
        """
        session = self.get_or_create_session(session_id)
        session.add_memories(memories, recalled_message_id, recalled_message_text)

    def prune_departed_from_session(
        self, session_id: str, alive_texts
    ) -> List[MemoryItem]:
        """按本轮上下文清理来源已离场的记忆，返回被移除的记忆条目本身。"""
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.prune_departed(alive_texts)

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
