from __future__ import annotations

from typing import Any, Protocol

from ...llm_memory.models.data_models import BaseMemory


class MemoryRuntime(Protocol):
    """统一记忆运行时接口。"""

    async def remember(
        self,
        memory_type: str,
        judgment: str,
        reasoning: str,
        tags: list[str],
        is_active: bool = False,
        strength: int | None = None,
        memory_scope: str = "public",
    ) -> str:
        ...

    async def comprehensive_recall(
        self,
        query: str,
        limit: int | None = None,
        event: Any = None,
        vector: list[float] | None = None,
        memory_scope: str = "public",
    ) -> list[BaseMemory]:
        ...

    async def chained_recall(
        self,
        query: str,
        entities: list[str],
        vector: list[float] | None = None,
        event: Any = None,
        memory_scope: str = "public",
    ) -> list[BaseMemory]:
        ...

    async def get_memories_by_ids(
        self,
        memory_ids: list[str],
        memory_scope: str | None = None,
    ) -> list[BaseMemory]:
        ...

    async def feedback(
        self,
        useful_memory_ids: list[str] | None = None,
        recalled_memory_ids: list[str] | None = None,
        memory_actions: list[dict] | None = None,
        memory_scope: str = "public",
    ) -> list[BaseMemory]:
        ...

    async def decay_tier0_memories(self) -> int:
        """睡眠第一步：T0 时间衰减，只降强度，不删除。"""
        ...

    async def list_retirement_candidates(self) -> list[BaseMemory]:
        """睡眠第二步：取出强度归零、等待裁决的被动记忆。"""
        ...

    async def apply_retirement_decisions(
        self,
        delete_ids: list[str],
        keep_ids: list[str],
    ) -> dict[str, int]:
        """睡眠第三步：按裁决执行删除与永久化。"""
        ...
