"""p-memory 记忆适配层。

只做两件事：把 Angel 的字段翻译成 p-memory 的输入形状，以及把库返回的记录
转回普通 dict。检索排序、折叠、衰减规则都交给库，本项目不在此处二次排序。

口径：
- namespace 固定 `angel/memory`；
- scope 沿用现有 `memory_scope`；
- id 直接用库分配的整数。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from p_memory.errors import NotFoundError, PMemoryError

from .config import memory_namespace, read_scopes
from .kb import pmemory

logger = logging.getLogger(__name__)

SECOND_US = 1_000_000


def _text(value: object) -> str:
    return str(value or "").strip()


def _tags(value: object) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def to_us(seconds: object) -> Optional[int]:
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

    def __init__(self, store=None) -> None:
        self._store = store or pmemory

    # === 内部 ===

    def _filter(self, scope: Optional[str], all_scopes: bool = False) -> dict:
        """读过滤。生命周期操作要跨 scope，检索才按会话范围收窄。

        库要求至少一个显式 scope，且没有「全部 scope」通配，跨 scope 只能靠登记名单。
        """
        if all_scopes:
            return {"namespace": memory_namespace(), "scopes": self._store.scopes}
        return {"namespace": memory_namespace(), "scopes": read_scopes(scope)}

    @property
    def space(self) -> Optional[str]:
        return self._store.space_id

    @property
    def scopes(self) -> List[str]:
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
        scope: Optional[str] = None,
        *,
        pinned: bool = False,
        strength: Optional[int] = None,
        useful_count: Optional[int] = None,
        useful_score: Optional[float] = None,
        last_recalled_at: Optional[float] = None,
        last_decay_at: Optional[float] = None,
        metadata: Optional[dict] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
    ) -> dict:
        state: Dict[str, Any] = {"pinned": bool(pinned)}
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

        payload: Dict[str, Any] = {
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
        judgment: Optional[str] = None,
        reasoning: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        memory_type: Optional[str] = None,
        scope: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """原地更新正文。upsert 要求整条记忆，缺的字段从现有记录补。"""
        record = self.get(memory_id, all_scopes=True)
        if not record:
            return {}
        payload: Dict[str, Any] = {
            "id": int(memory_id),
            "judgment": _text(judgment if judgment is not None else record.get("judgment")),
            "reasoning": _text(
                reasoning if reasoning is not None else record.get("reasoning")
            ),
            "memory_type": _text(memory_type or record.get("memory_type"))
            or "知识记忆",
            "tags": _tags(tags if tags is not None else record.get("tags")),
            "namespace": _text(record.get("namespace")) or memory_namespace(),
            "scope": _text(scope or record.get("scope")) or "public",
            "state": None,
            "metadata": dict(
                metadata if metadata is not None else record.get("metadata") or {}
            ),
        }
        receipt = self._store.kb.memories.upsert(payload)
        self._store.kb.update_index()
        return (receipt or {}).get("value") or {}

    def pin(self, memory_id: int, scope: Optional[str] = None) -> bool:
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
        payload: Dict[str, Any] = {
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
        scope: Optional[str] = None,
        *,
        all_scopes: bool = False,
    ) -> Optional[dict]:
        try:
            raw_id = int(str(memory_id))
        except (TypeError, ValueError):
            return None
        try:
            return self._store.kb.memories.get(
                raw_id, filter=self._filter(scope, all_scopes=all_scopes)
            )
        except NotFoundError:
            return None
        except PMemoryError:
            logger.exception("[PMEMORY] 读取记忆失败 id=%s", memory_id)
            return None

    def search(
        self,
        query: str,
        scope: Optional[str] = None,
        *,
        limit: int = 10,
        memory_type: Optional[str] = None,
        candicate_multiplier: int = 3,
    ) -> List[dict]:
        """检索记忆。类型过滤在拿到结果后做，所以候选窗口按倍数放大。"""
        text = _text(query)
        if not text:
            return []
        space = self.space
        want = max(1, int(limit))
        fetch = want if not memory_type else max(want * max(1, candicate_multiplier), want)
        try:
            result = self._store.kb.search(
                text,
                filter=self._filter(scope),
                kinds=["memory"],
                limit=fetch,
                embed_space=space,
                text=True,
                vector=bool(space),
                rerank=True,
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 记忆检索失败 scope=%s", scope)
            return []
        hits = []
        for hit in result.get("hits") or []:
            record = hit.get("record") or {}
            if not record:
                continue
            if memory_type and _text(record.get("memory_type")) != _text(memory_type):
                continue
            hits.append(record)
            if len(hits) >= want:
                break
        return hits

    def list_records(self, scope: Optional[str] = None, *, all_scopes: bool = False) -> List[dict]:
        """按 scope 翻页取全部记忆；`all_scopes=True` 时跨 scope 取。"""
        records: List[dict] = []
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
        scope: Optional[str] = None,
        policy: Optional[dict] = None,
    ) -> List[dict]:
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
        refreshed: List[dict] = []
        for memory_id in dict.fromkeys(visible + useful):
            record = self.get(memory_id, scope)
            if record:
                refreshed.append(record)
        return refreshed

    def decay(
        self,
        policy: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """时间衰减，只降强度不删；返回降级条数与候选记录。

        衰减作用于全部 scope：每个会话的记忆都该随时间淡去，不能只处理 public。
        """
        try:
            receipt = self._store.kb.memories.decay(
                filter=self._filter(None, all_scopes=True), policy=policy or None
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 时间衰减失败")
            return {"decayed": 0, "candidates": []}
        report = (receipt or {}).get("value") or {}
        decayed = int(report.get("decayed") or 0)
        keys = report.get("retirement_candidates") or []
        candidates: List[dict] = []
        for key in keys:
            raw_id = key.get("id") if isinstance(key, dict) else key
            record = self.get(raw_id, all_scopes=True)
            if record:
                candidates.append(record)
        return {"decayed": decayed, "candidates": candidates}

    def retirement_policy(self, config: Optional[dict] = None) -> Optional[dict]:
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
        policy: Dict[str, Any] = {}
        for source, target in mapping.items():
            if source in config and config[source] is not None:
                policy[target] = config[source]
        return policy or None

    def export_snapshot(self) -> Dict[str, Any]:
        """导出冷却备份快照：与旧库 export_backup_snapshot 同结构，供失败时回灌。

        标签直接内嵌在每条记录里（`tags` 字段），因为库把标签挂在记录上而不走
        关系表；回灌时优先读这个字段。
        """
        records: List[Dict[str, Any]] = []
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
