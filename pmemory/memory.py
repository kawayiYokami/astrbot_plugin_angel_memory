"""p-memory 记忆适配层。

只做两件事：把 Angel 的字段翻译成 p-memory 的输入形状，以及把库返回的记录
转回普通 dict。检索排序、折叠、衰减规则都交给库，本项目不在此处二次排序。

口径：
- namespace 固定 `angel/memory`；
- scope 沿用现有 `memory_scope`；
- id 直接用库分配的整数。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from astrbot.api import logger
from p_memory.errors import NotFoundError, PMemoryError

from .config import memory_namespace, read_scopes
from .kb import pmemory

SECOND_US = 1_000_000


def _text(value: object) -> str:
    return str(value or "").strip()


def _tags(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def to_us(seconds: object) -> int | None:
    """秒 → 微秒。旧库用秒，p-memory 用 UTC 微秒。"""
    try:
        number = float(seconds or 0)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return int(number * SECOND_US)


def to_seconds(microseconds: object) -> float:
    try:
        number = float(microseconds or 0)
    except (TypeError, ValueError):
        return 0.0
    return number / SECOND_US if number > 0 else 0.0


class MemoryStore:
    """记忆读写。所有方法都是同步的，调用方自行决定是否放到线程里跑。"""

    # 自动召回口径：先按融合分捞 CANDIDATE_LIMIT 条候选，重排后只留不低于最高分
    # RERANK_SCORE_RATIO 的命中，最后取前 MAX_RESULTS 条给模型。
    # 主动回忆走另一条路：不重排、不设门槛，按融合分取前 limit 条。
    CANDIDATE_LIMIT = 50
    RERANK_SCORE_RATIO = 0.5
    MAX_RESULTS = 10

    def __init__(self, store=None) -> None:
        self._store = store or pmemory

    # === 内部 ===

    def _filter(self, scope: str | None, all_scopes: bool = False) -> dict:
        """读过滤。生命周期操作要跨 scope，检索才按会话范围收窄。

        库要求至少一个显式 scope，且没有「全部 scope」通配，跨 scope 只能靠登记名单。
        """
        if all_scopes:
            return {"namespace": memory_namespace(), "scopes": self._store.scopes}
        return {"namespace": memory_namespace(), "scopes": read_scopes(scope)}

    @property
    def space(self) -> str | None:
        return self._store.space_id

    @property
    def scopes(self) -> list[str]:
        return self._store.scopes

    def register_scopes(self, names) -> None:
        self._store.register_scopes(names)

    # === 写 ===

    def remember(
        self,
        judgment: str,
        reasoning: str,
        tags: Iterable[str],
        memory_type: str,
        scope: str | None = None,
        *,
        pinned: bool = False,
        strength: int | None = None,
        useful_count: int | None = None,
        useful_score: float | None = None,
        last_recalled_at: float | None = None,
        last_decay_at: float | None = None,
        metadata: dict | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> dict:
        state: dict[str, Any] = {"pinned": bool(pinned)}
        if strength is not None:
            state["strength"] = int(strength)
        if useful_count is not None:
            state["useful_count"] = int(useful_count)
        if useful_score is not None:
            state["useful_score"] = float(useful_score)
        recalled_us = to_us(last_recalled_at)
        if recalled_us is not None:
            state["last_recalled_at_us"] = recalled_us
        decay_us = to_us(last_decay_at)
        if decay_us is not None:
            state["last_decay_at_us"] = decay_us

        payload: dict[str, Any] = {
            "judgment": _text(judgment),
            "reasoning": _text(reasoning),
            "memory_type": _text(memory_type) or "知识记忆",
            "tags": _tags(tags),
            "namespace": "angel/memory",
            "scope": _text(scope) or "public",
            "state": state,
            "metadata": dict(metadata or {}),
        }
        created_us = to_us(created_at)
        if created_us is not None:
            payload["created_at_us"] = created_us
        updated_us = to_us(updated_at)
        if updated_us is not None:
            payload["updated_at_us"] = updated_us

        # 新出现的 scope 要登记，否则跨 scope 的生命周期操作永远看不见它
        self._store.register_scopes([payload["scope"]])
        receipt = self._store.kb.memories.upsert(payload)
        record = (receipt or {}).get("value") or {}
        self._store.kb.update_index()
        if not record.get("id"):
            # 某些路径只在 receipt 顶层给 id
            record = {"id": (receipt or {}).get("id")}
        return record

    def update(
        self,
        memory_id: int,
        *,
        judgment: str | None = None,
        reasoning: str | None = None,
        tags: Iterable[str] | None = None,
        memory_type: str | None = None,
        scope: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """原地更新正文。upsert 要求整条记忆，缺的字段从现有记录补。"""
        record = self.get(memory_id, all_scopes=True)
        if not record:
            return {}
        payload: dict[str, Any] = {
            "id": int(memory_id),
            "judgment": _text(judgment if judgment is not None else record.get("judgment")),
            "reasoning": _text(reasoning if reasoning is not None else record.get("reasoning")),
            "memory_type": _text(memory_type or record.get("memory_type")) or "知识记忆",
            "tags": _tags(tags if tags is not None else record.get("tags")),
            "namespace": _text(record.get("namespace")) or memory_namespace(),
            "scope": _text(scope or record.get("scope")) or "public",
            "state": None,
            "metadata": dict(metadata if metadata is not None else record.get("metadata") or {}),
        }
        receipt = self._store.kb.memories.upsert(payload)
        self._store.kb.update_index()
        return (receipt or {}).get("value") or {}

    def pin(self, memory_id: int, scope: str | None = None) -> bool:
        """固定保留：upsert 要求整条记忆，所以先读出原文再连同新状态写回。

        固定动作发生在退休裁决里，那时手里只有 id，不知道记忆的 scope，
        因此按全 scope 读取。
        """
        record = self.get(memory_id, scope, all_scopes=True)
        if not record:
            return False
        state = dict(record.get("state") or {})
        state["pinned"] = True
        return self._write_state(int(memory_id), record, state)

    def _write_state(self, memory_id: int, record: dict, state: dict) -> bool:
        payload: dict[str, Any] = {
            "id": memory_id,
            "judgment": _text(record.get("judgment")),
            "reasoning": _text(record.get("reasoning")),
            "memory_type": _text(record.get("memory_type")) or "知识记忆",
            "tags": _tags(record.get("tags")),
            "namespace": _text(record.get("namespace")) or memory_namespace(),
            "scope": _text(record.get("scope")) or "public",
            "state": state,
            "metadata": dict(record.get("metadata") or {}),
        }
        try:
            self._store.kb.memories.upsert(payload)
            return True
        except PMemoryError:
            logger.exception("[PMEMORY] 写入记忆状态失败 id=%s", memory_id)
            return False

    def delete(self, ids: Iterable[int]) -> int:
        values = [int(i) for i in ids if str(i).strip().lstrip("-").isdigit()]
        if not values:
            return 0
        receipt = self._store.kb.memories.delete(values)
        self._store.kb.update_index()
        return int((receipt or {}).get("value") or 0)

    # === 读 ===

    def get(
        self,
        memory_id: object,
        scope: str | None = None,
        *,
        all_scopes: bool = False,
    ) -> dict | None:
        try:
            raw_id = int(str(memory_id))
        except (TypeError, ValueError):
            return None
        try:
            return self._store.kb.memories.get(raw_id, filter=self._filter(scope, all_scopes=all_scopes))
        except NotFoundError:
            return None
        except PMemoryError:
            logger.exception("[PMEMORY] 读取记忆失败 id=%s", memory_id)
            return None

    def search(
        self,
        query: str,
        scope: str | None = None,
        *,
        limit: int = 10,
        score_ratio: float | None = None,
        max_results: int | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """检索记忆。

        分数过滤与条数截断都在拿到结果之后做。
        `rerank` 关掉时按库的融合分排序，不调用重排回调。
        `score_ratio` 给出时只保留不低于本批最高分该比例的命中：整批都有重排分就按
        重排分比较，只要有命中缺重排分就统一退回库的融合分，两种尺子不混比。
        `max_results` 是独立于 `limit` 的最终条数上限，不给就按 `limit` 截断。
        """
        text = _text(query)
        if not text:
            return []
        space = self.space
        want = max(1, int(limit))
        try:
            result = self._store.kb.search(
                text,
                filter=self._filter(scope),
                kinds=["memory"],
                limit=want,
                embed_space=space,
                text=True,
                vector=bool(space),
                rerank=bool(rerank),
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 记忆检索失败 scope=%s", scope)
            return []

        records: list[dict] = []
        rerank_scores: list[float | None] = []
        fused_scores: list[float] = []
        for hit in result.get("hits") or []:
            record = hit.get("record") or {}
            if not record:
                continue
            rerank = hit.get("rerank_score")
            records.append(record)
            rerank_scores.append(float(rerank) if isinstance(rerank, (int, float)) else None)
            fused_scores.append(float(hit.get("score") or 0.0))

        if not records:
            return []

        if score_ratio is None:
            kept = records
        else:
            if rerank_scores and all(score is not None for score in rerank_scores):
                scores = [float(score) for score in rerank_scores]
            else:
                scores = fused_scores
            # 重排分必须为正：负分是供应商吐出来的原始 logit，不做归一化，直接丢掉。
            # 剩下的按最高分的比例留人，负分不参与最高分的计算。
            scored = [(record, score) for record, score in zip(records, scores, strict=False) if score > 0]
            if not scored:
                return []
            threshold = max(score for _, score in scored) * float(score_ratio)
            kept = [record for record, score in scored if score >= threshold]

        cap = want if max_results is None else min(want, int(max_results))
        return kept[:cap]

    def list_records(self, scope: str | None = None, *, all_scopes: bool = False) -> list[dict]:
        """按 scope 翻页取全部记忆；`all_scopes=True` 时跨 scope 取。"""
        records: list[dict] = []
        cursor = None
        while True:
            page = self._store.kb.memories.list(
                filter=self._filter(scope, all_scopes=all_scopes), limit=200, after=cursor
            )
            items = page.get("items") or []
            records.extend(items)
            cursor = page.get("next_cursor")
            if not cursor:
                break
        return records

    # === 生命周期 ===

    def feedback(
        self,
        recalled_ids: Iterable[object] = (),
        useful_ids: Iterable[object] = (),
        scope: str | None = None,
        policy: dict | None = None,
    ) -> list[dict]:
        recalled = [int(i) for i in recalled_ids if str(i).strip().isdigit()]
        useful = [int(i) for i in useful_ids if str(i).strip().isdigit()]
        if not recalled and not useful:
            return []
        if not recalled and useful:
            recalled = list(useful)
        useful = [i for i in useful if i in recalled]

        # 库按 filter 查记录，跨 scope 的 id 会让整次反馈报 not_found，
        # 所以先按当前可见范围过滤一遍。
        visible = [i for i in recalled if self.get(i, scope) is not None]
        useful = [i for i in useful if i in visible]
        if not visible:
            logger.debug("[PMEMORY] 记忆反馈跳过：无可见记录 scope=%s", scope)
            return []
        try:
            self._store.kb.memories.feedback(
                recalled_ids=visible,
                useful_ids=useful,
                filter=self._filter(scope),
                policy=policy or None,
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 记忆反馈失败")
            return []
        refreshed: list[dict] = []
        for memory_id in dict.fromkeys(visible + useful):
            record = self.get(memory_id, scope)
            if record:
                refreshed.append(record)
        return refreshed

    def decay(
        self,
        policy: dict | None = None,
    ) -> dict[str, Any]:
        """时间衰减，只降强度不删；返回降级条数与候选记录。

        衰减作用于全部 scope：每个会话的记忆都该随时间淡去，不能只处理 public。
        """
        try:
            receipt = self._store.kb.memories.decay(filter=self._filter(None, all_scopes=True), policy=policy or None)
        except PMemoryError:
            logger.exception("[PMEMORY] 时间衰减失败")
            return {"decayed": 0, "candidates": []}
        report = (receipt or {}).get("value") or {}
        decayed = int(report.get("decayed") or 0)
        keys = report.get("retirement_candidates") or []
        candidates: list[dict] = []
        for key in keys:
            raw_id = key.get("id") if isinstance(key, dict) else key
            record = self.get(raw_id, all_scopes=True)
            if record:
                candidates.append(record)
        return {"decayed": decayed, "candidates": candidates}

    def retirement_policy(self, config: dict | None = None) -> dict | None:
        """把插件配置翻译成库的衰减策略参数；没有可用项时返回 None 用库默认。"""
        if not isinstance(config, dict):
            return None
        mapping = {
            "tier0_threshold": "tier0_threshold",
            "tier1_threshold": "tier1_threshold",
            "tier0_cycle_days": "tier0_cycle_days",
            "useful_score_boost": "useful_score_boost",
            "strength_boost": "strength_boost",
        }
        policy: dict[str, Any] = {}
        for source, target in mapping.items():
            if source in config and config[source] is not None:
                policy[target] = config[source]
        return policy or None

    def export_snapshot(self) -> dict[str, Any]:
        """导出冷却备份快照：与旧库 export_backup_snapshot 同结构，供失败时回灌。

        标签直接内嵌在每条记录里（`tags` 字段），因为库把标签挂在记录上而不走
        关系表；回灌时优先读这个字段。
        """
        records: list[dict[str, Any]] = []
        for record in self.list_records(all_scopes=True):
            state = record.get("state") or {}
            legacy = (record.get("metadata") or {}).get("legacy") or {}
            records.append(
                {
                    "id": str(legacy.get("id") or record.get("id") or ""),
                    "memory_type": str(record.get("memory_type") or "知识记忆"),
                    "judgment": str(record.get("judgment") or ""),
                    "reasoning": str(record.get("reasoning") or ""),
                    "strength": int(state.get("strength", 1)),
                    "is_active": 1 if state.get("pinned") else 0,
                    "useful_count": int(state.get("useful_count", 0)),
                    "useful_score": float(state.get("useful_score", 0.0)),
                    "last_recalled_at": to_seconds(state.get("last_recalled_at_us")),
                    "last_decay_at": to_seconds(state.get("last_decay_at_us")),
                    "memory_scope": str(record.get("scope") or "public"),
                    "tags": [str(tag) for tag in (record.get("tags") or [])],
                    "created_at": to_seconds(record.get("created_at_us")),
                    "updated_at": to_seconds(record.get("updated_at_us")),
                }
            )
        return {
            "schema_version": 1,
            "exported_at": int(now_seconds()),
            "records": records,
            "global_tags": [],
            "memory_tag_rel": [],
        }


def now_seconds() -> float:
    return time.time()
