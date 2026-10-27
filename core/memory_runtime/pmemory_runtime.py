"""p-memory 实现的记忆运行时。

对上层（DeepMind、工具）只暴露 `MemoryRuntime` 协议，内部把调用转成
`MemoryStore` 的操作。检索排序、类型分组、衰减规则都交给库；这里只做
「协议形状」与「记忆类型中英转换」两件事。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ...llm_memory.models.data_models import BaseMemory, MemoryType
from ...pmemory import MemoryStore, to_seconds
from .protocol import MemoryRuntime

# 工具层用英文键，库里存中文枚举
TYPE_TO_CN = {
    "knowledge": "知识记忆",
    "event": "事件记忆",
    "skill": "技能记忆",
    "emotional": "情感记忆",
    "task": "任务记忆",
    "meta": "元记忆",
}
CN_TO_TYPE = {value: key for key, value in TYPE_TO_CN.items()}


def _cn_type(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "知识记忆"
    return TYPE_TO_CN.get(text.lower(), text)


def to_base_memory(record: dict) -> BaseMemory:
    state = record.get("state") or {}
    metadata = record.get("metadata") or {}
    cn = str(record.get("memory_type") or "知识记忆")
    memory_type = MemoryType.KNOWLEDGE
    for member in MemoryType:
        if member.value == cn:
            memory_type = member
            break
    return BaseMemory(
        id=str(record.get("id") or ""),
        memory_type=memory_type,
        judgment=str(record.get("judgment") or ""),
        reasoning=str(record.get("reasoning") or ""),
        tags=[str(tag) for tag in (record.get("tags") or [])],
        strength=int(state.get("strength", 1)),
        is_active=bool(state.get("pinned", False)),
        created_at=to_seconds(record.get("created_at_us")),
        soul_state=dict(metadata.get("soul_state") or {}),
        memory_scope=str(record.get("scope") or "public"),
        useful_count=int(state.get("useful_count", 0)),
        useful_score=float(state.get("useful_score", 0.0)),
        last_recalled_at=to_seconds(state.get("last_recalled_at_us")),
    )


class PMemoryMemoryRuntime(MemoryRuntime):
    """唯一实现：全部记忆读写交给 p-memory。"""

    def __init__(self, store: MemoryStore, decay_policy: Optional[dict] = None) -> None:
        self._store = store
        self._decay_policy = decay_policy
        self._decay_candidates: List[dict] = []

    # === 写 ===

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
        record = await asyncio.to_thread(
            self._store.remember,
            judgment,
            reasoning,
            tags,
            _cn_type(memory_type),
            memory_scope,
            pinned=bool(is_active),
            strength=strength,
        )
        return str(record.get("id") or "")

    # === 读 ===

    async def recall(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        records = await asyncio.to_thread(
            self._store.search,
            query,
            memory_scope or "public",
            limit=limit,
            memory_type=_cn_type(memory_type),
        )
        return [to_base_memory(record) for record in records]

    async def comprehensive_recall(
        self,
        query: str,
        limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        del event, vector  # 向量由库内部完成，宿主不再传向量
        records = await asyncio.to_thread(
            self._store.search,
            query,
            memory_scope,
            limit=int(limit or 10),
        )
        return [to_base_memory(record) for record in records]

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
        del vector, event
        merged = " ".join(
            [str(query or "").strip()]
            + [str(entity).strip() for entity in (entities or []) if str(entity).strip()]
        ).strip()
        records = await asyncio.to_thread(
            self._store.search,
            merged,
            memory_scope,
            limit=int(final_limit or per_type_limit or 7),
        )
        return [to_base_memory(record) for record in records]

    async def get_memories_by_ids(
        self,
        memory_ids: List[str],
        memory_scope: Optional[str] = None,
    ) -> List[BaseMemory]:
        records = await asyncio.to_thread(self._get_many, memory_ids, memory_scope)
        return [to_base_memory(record) for record in records]

    def _get_many(self, memory_ids: List[str], memory_scope: Optional[str]) -> List[dict]:
        output: List[dict] = []
        for memory_id in memory_ids:
            record = self._store.get(memory_id, memory_scope or "public")
            if record:
                output.append(record)
        return output

    # === 反馈 ===

    async def feedback(
        self,
        useful_memory_ids: Optional[List[str]] = None,
        recalled_memory_ids: Optional[List[str]] = None,
        memory_actions: Optional[List[dict]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        refreshed: List[dict] = []
        useful = list(useful_memory_ids or [])
        recalled = list(recalled_memory_ids or [])
        if useful or recalled:
            refreshed = await asyncio.to_thread(
                self._store.feedback,
                recalled,
                useful,
                memory_scope,
                self._decay_policy,
            )

        for action in memory_actions or []:
            record = await asyncio.to_thread(self._apply_action, action, memory_scope)
            if record:
                refreshed.append(record)
        return [to_base_memory(record) for record in refreshed]

    def _apply_action(self, action: dict, scope: str) -> Optional[dict]:
        """update 原地更新；merge 新建后删除源记忆。

        删除有意不看 scope：合并允许跨会话收拢记忆，删的是全局 id。
        但新建前必须确认源全部存在，否则就是假成功——缺任一源直接返回 None，
        由工具层报“源记忆不存在”。
        """
        name = str(action.get("action") or "").strip().lower()
        payload = action.get("memory") or {}
        source_ids = [str(item).strip() for item in (action.get("source_memory_ids") or []) if str(item).strip()]
        judgment = str(payload.get("judgment") or "").strip()
        if not judgment:
            return None
        cn_type = _cn_type(payload.get("type") or payload.get("memory_type") or "knowledge")
        tags = payload.get("tags") or []
        reasoning = str(payload.get("reasoning") or "")
        metadata = dict(action.get("metadata") or {})

        if name in ("updata", "update") and source_ids:
            target = source_ids[0]
            record = self._store.get(target, scope)
            if record is None:
                return None
            metadata.setdefault("soul_state", (record.get("metadata") or {}).get("soul_state") or {})
            updated = self._store.update(
                int(target),
                judgment=judgment,
                reasoning=reasoning,
                tags=tags,
                memory_type=cn_type,
                metadata={**(record.get("metadata") or {}), **metadata},
            )
            return updated or None

        if name == "merge" and source_ids:
            for sid in source_ids:
                if self._store.get(sid, all_scopes=True) is None:
                    return None

        created = self._store.remember(
            judgment,
            reasoning,
            tags,
            cn_type,
            scope,
            metadata={**metadata, "source_ids": source_ids},
        )
        if source_ids:
            self._store.delete(source_ids)
        return created or None

    # === 生命周期 ===

    async def decay_tier0_memories(self) -> int:
        report = await asyncio.to_thread(self._store.decay, self._decay_policy)
        self._decay_candidates = list(report.get("candidates") or [])
        return int(report.get("decayed") or 0)

    async def list_retirement_candidates(self) -> List[BaseMemory]:
        records = self._decay_candidates
        if not records:
            records = await asyncio.to_thread(self._list_by_strength_zero)
        return [to_base_memory(record) for record in records]

    def _list_by_strength_zero(self) -> List[dict]:
        candidates: List[dict] = []
        for record in self._store.list_records(all_scopes=True):
            state = record.get("state") or {}
            if int(state.get("strength", 1)) <= 0 and not state.get("pinned", False):
                candidates.append(record)
        return candidates

    async def apply_retirement_decisions(
        self,
        delete_ids: List[str],
        keep_ids: List[str],
    ) -> Dict[str, int]:
        deleted = await asyncio.to_thread(self._store.delete, delete_ids)
        kept = 0
        for memory_id in keep_ids:
            ok = await asyncio.to_thread(self._store.pin, memory_id)
            if ok:
                kept += 1
        self._decay_candidates = [
            record
            for record in self._decay_candidates
            if str(record.get("id")) not in {str(i) for i in delete_ids + keep_ids}
        ]
        return {"deleted": deleted, "kept": kept}

    def shutdown(self) -> None:
        return None
