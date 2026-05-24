from __future__ import annotations

from typing import Any, List, Optional

from ...llm_memory.components.memory_sql_manager import MemorySqlManager
from ...llm_memory.models.data_models import BaseMemory


class SimpleMemoryRuntime:
    """基于 SQL + tags 的最简记忆运行时。"""

    def __init__(self, memory_sql_manager: MemorySqlManager):
        self._manager = memory_sql_manager

    async def remember(
        self,
        memory_type: str,
        judgment: str,
        reasoning: str,
        tags: List[str],
        is_active: bool = False,
        strength: Optional[int] = None,
        memory_scope: str = "public",
    ) -> str:
        memory = await self._manager.remember(
            memory_type=memory_type,
            judgment=judgment,
            reasoning=reasoning,
            tags=tags,
            is_active=is_active,
            strength=strength,
            memory_scope=memory_scope,
        )
        return memory.id

    async def recall(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        memories = await self._manager.recall_by_tags(
            query=query,
            limit=limit,
            memory_scope=memory_scope or "public",
        )
        if memory_type:
            return [
                mem
                for mem in memories
                if getattr(mem.memory_type, "value", str(mem.memory_type))
                in {memory_type, self._map_type_to_cn(memory_type)}
            ]
        return memories

    async def comprehensive_recall(
        self,
        query: str,
        limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        limit = int(limit or 10)
        return await self._manager.recall_by_tags(
            query=query,
            limit=limit,
            memory_scope=memory_scope,
        )

    async def chained_recall(
        self,
        query: str,
        entities: List[str],
        per_type_limit: int = 7,
        final_limit: Optional[int] = None,
        vector: Optional[List[float]] = None,
        event: Any = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        merged_query = " ".join([str(query or "").strip(), *[str(e).strip() for e in (entities or []) if str(e).strip()]]).strip()
        limit = int(final_limit or per_type_limit or 7)
        memories = await self._manager.recall_by_tags(
            query=merged_query,
            limit=limit,
            memory_scope=memory_scope,
        )
        return memories

    async def get_memories_by_ids(
        self,
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        memories = await self._manager.get_memories_by_ids(memory_ids)
        return self._filter_and_order_memories(
            memories=memories,
            memory_ids=memory_ids,
            memory_scope=memory_scope,
        )

    async def feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        recalled_memory_ids: Optional[List[str]] = None,
        memory_actions: Optional[List[dict]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        return await self._manager.process_feedback(
            useful_memory_ids=useful_memory_ids,
            recalled_memory_ids=recalled_memory_ids,
            memory_actions=memory_actions,
            memory_scope=memory_scope,
        )

    async def consolidate_memories(self) -> None:
        await self._manager.consolidate_memories()

    async def recall_by_sender_tag(self, tag: str, limit: int = 20):
        """跨 scope 按发送者 tag 召回记忆。"""
        memory_ids = await self._manager.get_memory_ids_by_tag(tag)
        if not memory_ids:
            return []
        memories = await self._manager.get_memories_by_ids(memory_ids)
        # 简单按创建时间倒序
        memories.sort(key=lambda m: getattr(m, 'created_at', 0), reverse=True)
        return memories[:limit]

    def shutdown(self) -> None:
        return None

    @staticmethod
    def _filter_and_order_memories(
        memories: List[BaseMemory],
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        scope = str(memory_scope or "").strip()
        memory_map = {str(mem.id): mem for mem in memories or [] if getattr(mem, "id", None)}
        ordered: List[BaseMemory] = []
        seen = set()
        for raw_id in memory_ids or []:
            memory_id = str(raw_id or "").strip()
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            memory = memory_map.get(memory_id)
            if memory is None:
                continue
            if scope:
                mem_scope = str(getattr(memory, "memory_scope", "public") or "public").strip()
                if scope == "public":
                    if mem_scope != "public":
                        continue
                elif mem_scope not in {scope, "public"}:
                    continue
            ordered.append(memory)
        return ordered

    @staticmethod
    def _map_type_to_cn(memory_type: str) -> str:
        mapping = {
            "knowledge": "知识记忆",
            "event": "事件记忆",
            "skill": "技能记忆",
            "task": "任务记忆",
            "emotional": "情感记忆",
        }
        return mapping.get(str(memory_type or "").strip(), str(memory_type or "").strip())
