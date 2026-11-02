"""
短期记忆模块

负责管理所有会话的短期记忆，支持并发和线程安全。
短期记忆的生命周期只有一条规则：带入这条记忆的消息还在请求里，它就留着；
消息走了，它就离开。不再按容量裁剪。
"""

import threading
import time
from typing import Any, Dict, List

from dataclasses import dataclass

from ..llm_memory.models.data_models import BaseMemory


@dataclass
class MemoryItem:
    """短期记忆里的一条记忆条目。

    记忆内容跟着记忆条目一起存，注入和反思都直接用短期记忆里的这份，不再回查长期库。
    代价是长期记忆被改写后，短期记忆里仍是进入时的文本，下一次进入前保持原样。
    """

    id: str
    memory_type: str
    added_at: float = 0.0  # 进入短期记忆的时间戳
    bound_message_id: str = ""  # 绑定消息 ID，该消息离场则这条记忆条目随之离场
    bound_message_text: str = ""  # 绑定消息正文，离场时要连它一起送审
    memory: Any = None  # 记忆本体，进入时一并留下


class SessionMemory:
    """单个会话的短期记忆。"""

    def __init__(self, session_id: str):
        """
        初始化会话记忆

        Args:
            session_id: 会话ID
        """
        self.session_id = session_id
        self.lock = threading.Lock()
        self.memories = []  # List[MemoryItem]
        self.memory_map = {}  # Dict[str, MemoryItem]

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
        with self.lock:
            for memory in memories:
                self._add_memory_item(
                    self._create_memory_item(
                        memory,
                        bound_message_id=bound_message_id,
                        bound_message_text=bound_message_text,
                    )
                )

    def _create_memory_item(
        self,
        memory: BaseMemory,
        bound_message_id: str = "",
        bound_message_text: str = "",
    ) -> MemoryItem:
        """创建记忆条目。"""
        raw_memory_type = getattr(memory, "memory_type", "")
        return MemoryItem(
            id=str(getattr(memory, "id", "") or "").strip(),
            memory_type=str(
                raw_memory_type.value
                if hasattr(raw_memory_type, "value")
                else raw_memory_type
            ),
            added_at=time.time(),
            bound_message_id=str(bound_message_id or "").strip(),
            bound_message_text=str(bound_message_text or "").strip(),
            memory=memory,
        )

    def _add_memory_item(self, memory_item: MemoryItem) -> None:
        """
        添加单个记忆条目

        Args:
            memory_item: 记忆条目
        """
        if not memory_item.id:
            return

        # 进入侧去重：同一条记忆只绑定一次，后来的召回不再刷新它
        if memory_item.id in self.memory_map:
            return

        self.memories.append(memory_item)
        self.memory_map[memory_item.id] = memory_item

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
            departed_ids = {memory.id for memory in departed}
            self.memories = [
                memory for memory in self.memories if memory.id not in departed_ids
            ]
            for memory_id in departed_ids:
                self.memory_map.pop(memory_id, None)
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
        with self.lock:
            by_type: Dict[str, int] = {}
            for memory in self.memories:
                by_type[memory.memory_type] = by_type.get(memory.memory_type, 0) + 1
            return {
                "session_id": self.session_id,
                "total_memories": len(self.memories),
                "by_type": by_type,
            }


class SessionMemoryManager:
    """短期记忆管理器，管理所有会话的短期记忆。"""

    def __init__(self):
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
                self.sessions[session_id] = SessionMemory(session_id)
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

    def prune_absent_messages_from_session(
        self, session_id: str, alive_message_ids
    ) -> List[MemoryItem]:
        """按本次请求实际存在的消息清理短期记忆，返回被移除的记忆条目本身。"""
        with self.lock:
            if session_id not in self.sessions:
                return []
        session = self.sessions[session_id]
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
