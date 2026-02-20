from __future__ import annotations

from typing import Any, List, Optional, Protocol

from ...llm_memory.models.data_models import BaseMemory


class MemoryRuntime(Protocol):
    """统一记忆运行时接口。"""

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
        ...

    async def recall(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        include_consolidated: bool = True,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        ...

    async def comprehensive_recall(
        self,
        query: str,
        fresh_limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        ...

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
        ...

    async def feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        recalled_memory_ids: Optional[List[str]] = None,
        new_memories: Optional[List[dict]] = None,
        merge_groups: Optional[List[List[str]]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        ...

    async def consolidate_memories(self) -> None:
        ...
