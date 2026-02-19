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
        include_consolidated: bool = True,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        memories = await self._manager.recall_by_tags(
            query=query,
            limit=limit,
            memory_scope=memory_scope or "public",
        )
        if not include_consolidated:
            memories = [mem for mem in memories if not self._is_consolidated(mem)]
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
        fresh_limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        limit = int(fresh_limit or 10)
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
        final_limit: int = 7,
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
        passive_ids = [mem.id for mem in memories if not mem.is_active]
        if passive_ids:
            await self._manager.decay_memories(passive_ids, delta=1)
            refreshed = await self._manager.get_memories_by_ids(passive_ids)
            refreshed_by_id = {mem.id: mem for mem in refreshed}
            memories = [refreshed_by_id.get(mem.id, mem) for mem in memories]
        return memories

    async def feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        new_memories: Optional[List[dict]] = None,
        merge_groups: Optional[List[List[str]]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        return await self._manager.process_feedback(
            useful_memory_ids=useful_memory_ids,
            new_memories=new_memories,
            merge_groups=merge_groups,
            memory_scope=memory_scope,
        )

    async def consolidate_memories(self) -> None:
        await self._manager.consolidate_memories()

    def shutdown(self) -> None:
        return None

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

    @staticmethod
    def _is_consolidated(memory: BaseMemory) -> bool:
        if bool(getattr(memory, "is_consolidated", False)):
            return True

        tags = [str(tag).strip().lower() for tag in (getattr(memory, "tags", None) or [])]
        markers = {"consolidated", "merged", "合并记忆", "已巩固"}
        return any(tag in markers for tag in tags)
