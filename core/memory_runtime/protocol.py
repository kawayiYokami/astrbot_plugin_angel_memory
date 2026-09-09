from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

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
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        ...

    async def comprehensive_recall(
        self,
        query: str,
        limit: Optional[int] = None,
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
        final_limit: Optional[int] = None,
        vector: Optional[List[float]] = None,
        event: Any = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        ...

    async def get_memories_by_ids(
        self,
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        ...

    async def feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        recalled_memory_ids: Optional[List[str]] = None,
        memory_actions: Optional[List[dict]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        ...

    async def decay_tier0_memories(self) -> int:
        """睡眠第一步：T0 时间衰减，只降强度，不删除。"""
        ...

    async def list_retirement_candidates(self) -> List[BaseMemory]:
        """睡眠第二步：取出强度归零、等待裁决的被动记忆。"""
        ...

    async def apply_retirement_decisions(
        self,
        delete_ids: List[str],
        keep_ids: List[str],
    ) -> Dict[str, int]:
        """睡眠第三步：按裁决执行删除与永久化。"""
        ...
