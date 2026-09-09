from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        return await self._cognitive_service.recall(
            memory_type=memory_type,
            query=query,
            limit=limit,
            memory_scope=memory_scope,
        )

    async def comprehensive_recall(
        self,
        query: str,
        limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        return await self._cognitive_service.comprehensive_recall(
            query=query,
            limit=limit,
            event=event,
            vector=vector,
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
        return await self._cognitive_service.chained_recall(
            query=query,
            entities=entities,
            per_type_limit=per_type_limit,
            final_limit=final_limit,
            vector=vector,
            event=event,
            memory_scope=memory_scope,
        )

    async def get_memories_by_ids(
        self,
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        return await self._cognitive_service.get_memories_by_ids(
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
        return await self._cognitive_service.feedback(
            useful_memory_ids=useful_memory_ids,
            recalled_memory_ids=recalled_memory_ids,
            memory_actions=memory_actions,
            memory_scope=memory_scope,
        )

    async def decay_tier0_memories(self) -> int:
        return await self._cognitive_service.decay_tier0_memories()

    async def list_retirement_candidates(self) -> List[BaseMemory]:
        return await self._cognitive_service.list_retirement_candidates()

    async def apply_retirement_decisions(
        self,
        delete_ids: List[str],
        keep_ids: List[str],
    ) -> Dict[str, int]:
        return await self._cognitive_service.apply_retirement_decisions(
            delete_ids=delete_ids,
            keep_ids=keep_ids,
        )

    def shutdown(self) -> None:
        return None
