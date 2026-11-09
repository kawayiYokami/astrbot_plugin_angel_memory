"""p-memory 实现的记忆运行时。

对上层（DeepMind、工具）只暴露 `MemoryRuntime` 协议，内部把调用转成
`MemoryStore` 的操作。检索排序、类型分组、衰减规则都交给库；这里只做
「协议形状」与「记忆类型中英转换」两件事。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from astrbot.api import logger

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


def _normalize_scope(value: object) -> str:
    """判重键的柜子名，与落库口径一致：strip 后空值兜底 public。

    两侧必须走同一套归一，否则传入空串或带空格的柜子名时，
    判重会查不到已有记录，同一句话落成两条。
    """
    return str(value or "").strip() or "public"


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
        """新增一条记忆；同柜子已有同正文时返回那条的 id，不重复写。"""
        existing = await asyncio.to_thread(self._judgment_index, memory_scope)
        hit = existing.get((str(judgment or "").strip(), _normalize_scope(memory_scope)))
        if hit:
            return str(hit.get("id") or "")
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

    async def comprehensive_recall(
        self,
        query: str,
        limit: Optional[int] = None,
        event: Any = None,
        vector: Optional[List[float]] = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        del event, vector  # 向量由库内部完成，宿主不再传向量
        # 主动回忆不重排、不设分数门槛：按库的融合分取前面这些条就够了
        records = await asyncio.to_thread(
            self._store.search,
            query,
            memory_scope,
            limit=int(limit or 50),
            rerank=False,
        )
        return [to_base_memory(record) for record in records]

    async def chained_recall(
        self,
        query: str,
        entities: List[str],
        vector: Optional[List[float]] = None,
        event: Any = None,
        memory_scope: str = "public",
    ) -> List[BaseMemory]:
        del vector, event
        merged = " ".join(
            [str(query or "").strip()]
            + [str(entity).strip() for entity in (entities or []) if str(entity).strip()]
        ).strip()
        # 不设调用方可调的条数：候选、门槛、最终条数都是固定口径
        records = await asyncio.to_thread(
            self._store.search,
            merged,
            memory_scope,
            limit=MemoryStore.CANDIDATE_LIMIT,
            score_ratio=MemoryStore.RERANK_SCORE_RATIO,
            max_results=MemoryStore.MAX_RESULTS,
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

        existing = await asyncio.to_thread(self._judgment_index, memory_scope)
        for action in memory_actions or []:
            record = await asyncio.to_thread(
                self._apply_action, action, memory_scope, existing
            )
            if record:
                refreshed.append(record)
        return [to_base_memory(record) for record in refreshed]

    def _judgment_index(self, scope: str) -> Dict[tuple, dict]:
        """当前可见范围内「(正文, 柜子) → 已有记录」，供写入前判重。

        判重键用正文全等而不是相似度：同一句话在同一个柜子里只该有一条。
        """
        index: Dict[tuple, dict] = {}
        for record in self._store.list_records(scope):
            key = (
                str(record.get("judgment") or "").strip(),
                _normalize_scope(record.get("scope")),
            )
            index.setdefault(key, record)
        return index

    def _apply_action(
        self, action: dict, scope: str, existing: Optional[Dict[tuple, dict]] = None
    ) -> Optional[dict]:
        """update 原地更新；merge 新建后删除源记忆；create 同正文同柜子只留一条。

        删除有意不看 scope：合并允许跨会话收拢记忆，删的是全局 id。
        但新建前必须确认源全部存在，否则就是假成功——缺任一源直接返回 None，
        由工具层报“源记忆不存在”。

        create 判重是幂等的：已有同正文同柜子的记录就直接返回它，不再新建。
        反思可能反复提取同一件事，迁移也可能把新库备份当成旧数据读回来，
        这一层兜住两种重复。
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

        # 只有 create 判重：merge 必须新建结果再删源，不能因为撞上已有正文就跳过。
        key = (judgment, _normalize_scope(scope))
        if name != "merge" and existing is not None:
            hit = existing.get(key)
            if hit:
                logger.debug(
                    "[PMEMORY] 跳过重复记忆 id=%s judgment=「%s」",
                    hit.get("id"),
                    judgment[:20],
                )
                return hit

        created = self._store.remember(
            judgment,
            reasoning,
            tags,
            cn_type,
            scope,
            metadata={**metadata, "source_ids": source_ids},
        )
        if created and existing is not None:
            existing.setdefault(key, created)
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
