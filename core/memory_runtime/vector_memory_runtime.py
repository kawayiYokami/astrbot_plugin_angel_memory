from __future__ import annotations

from typing import Any, List, Optional

from ...llm_memory.models.data_models import BaseMemory
from ...llm_memory.service.cognitive_service import CognitiveService


class VectorMemoryRuntime:
    """向量记忆运行时适配器。"""

    def __init__(self, cognitive_service: CognitiveService):
        self._cognitive_service = cognitive_service

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
        return await self._cognitive_service.remember(
            memory_type=memory_type,
            judgment=judgment,
            reasoning=reasoning,
            tags=tags,
            is_active=is_active,
            strength=strength,
            memory_scope=memory_scope,
        )

    async def recall(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        include_consolidated: bool = True,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        return await self._cognitive_service.recall(
            memory_type=memory_type,
            query=query,
            limit=limit,
            include_consolidated=include_consolidated,
            memory_scope=memory_scope,
        )

    async def comprehensive_recall(
        self,
        query: str,
        fresh_limit: int = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        return await self._cognitive_service.comprehensive_recall(
            query=query,
            fresh_limit=fresh_limit,
            event=event,
            vector=vector,
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
        return await self._cognitive_service.chained_recall(
            query=query,
            entities=entities,
            per_type_limit=per_type_limit,
            final_limit=final_limit,
            vector=vector,
            event=event,
            memory_scope=memory_scope,
        )

    async def feedback(
        self,
        useful_memory_ids: List[str] = None,
        new_memories: List[dict] = None,
        merge_groups: List[List[str]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        return await self._cognitive_service.feedback(
            useful_memory_ids=useful_memory_ids,
            new_memories=new_memories,
            merge_groups=merge_groups,
            memory_scope=memory_scope,
        )

    async def consolidate_memories(self) -> None:
        await self._cognitive_service.consolidate_memories()
