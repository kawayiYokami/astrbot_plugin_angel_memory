"""存储层睡眠三步接口与 FTS 同步测试。

依赖 tantivy：宿主环境缺 tantivy 时本文件无法收集，需在容器内运行。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import time
import types
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.llm_memory.components.bm25_retriever import (
    TantivyBM25Retriever,
)
from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import (
    MemorySqlManager,
)
from astrbot_plugin_angel_memory.llm_memory.models.data_models import (
    BaseMemory,
    MemoryType,
)

DAY = 86400.0


def _make_manager() -> MemorySqlManager:
    return MemorySqlManager(db_path=Path(tempfile.mkdtemp()) / "test.db")


def _put(
    manager: MemorySqlManager,
    judgment: str,
    *,
    strength: int = 10,
    is_active: bool = False,
    age_days: float = 0.0,
    tags=None,
    reasoning: str = "测试背景",
) -> BaseMemory:
    memory = BaseMemory(
        memory_type=MemoryType.KNOWLEDGE,
        judgment=judgment,
        reasoning=reasoning,
        tags=list(tags or ["测试标签"]),
        strength=strength,
        is_active=is_active,
        created_at=time.time() - age_days * DAY,
    )
    manager._upsert_memory_sync(memory)
    return memory


def _put_candidate(
    manager: MemorySqlManager,
    judgment: str,
    *,
    tags=None,
    is_active: bool = False,
) -> BaseMemory:
    """构造强度已归零的记忆。

    _upsert_memory_sync 会把 strength=0 抬成 1（`int(getattr(...) or 1)`），
    所以先落库再用 SQL 归零，还原衰减到 0 后的真实状态。
    """
    memory = _put(manager, judgment, strength=1, is_active=is_active, tags=tags)
    _set_strength(manager, memory.id, 0)
    return memory


def _row(manager: MemorySqlManager, memory_id: str):
    with manager._connect() as conn:
        return conn.execute(
            "SELECT * FROM memory_records WHERE id = ?", (memory_id,)
        ).fetchone()


def _strength(manager: MemorySqlManager, memory_id: str) -> int:
    row = _row(manager, memory_id)
    return -1 if row is None else int(row["strength"])


def _is_active(manager: MemorySqlManager, memory_id: str):
    row = _row(manager, memory_id)
    return None if row is None else int(row["is_active"])


def _set_strength(manager: MemorySqlManager, memory_id: str, value: int) -> None:
    with manager._connect() as conn:
        conn.execute(
            "UPDATE memory_records SET strength = ? WHERE id = ?", (value, memory_id)
        )
        conn.commit()


def _fts_ids(manager: MemorySqlManager, query: str) -> set:
    """读取 FTS 索引命中集合。

    tantivy 的 reader 在 commit 后按延迟策略刷新，这里显式 reload
    以便立即观察索引落盘结果。
    """
    manager._fts_retriever._memory_index.reload()
    return {
        row["id"]
        for row in manager._fts_retriever.search_memory_bm25_only(
            query=query, limit=20
        )
    }


def _candidates(manager: MemorySqlManager):
    return asyncio.run(manager.list_retirement_candidates())


def _apply(manager: MemorySqlManager, delete_ids=(), keep_ids=()):
    return asyncio.run(
        manager.apply_retirement_decisions(list(delete_ids), list(keep_ids))
    )


def _decay(manager: MemorySqlManager) -> int:
    return asyncio.run(manager.decay_tier0_memories())


def test_decay_reduces_passive_strength_only():
    m = _make_manager()
    passive = _put(m, "被动记忆衰减目标", strength=5, age_days=7)
    active = _put(m, "主动记忆不衰减目标", strength=5, is_active=True, age_days=7)

    assert _decay(m) == 1

    assert _strength(m, passive.id) == 3
    assert _strength(m, active.id) == 5


def test_decay_is_idempotent_within_cycle():
    m = _make_manager()
    memory = _put(m, "同一周期内不重复衰减", strength=5, age_days=7)

    assert _decay(m) == 1
    assert _decay(m) == 0
    assert _strength(m, memory.id) == 3


def test_decay_clamps_to_zero_and_row_survives():
    m = _make_manager()
    memory = _put(m, "强度归零仍需保留", strength=1, age_days=30)

    _decay(m)

    assert _strength(m, memory.id) == 0
    assert _row(m, memory.id) is not None
    assert [item.id for item in _candidates(m)] == [memory.id]


def test_list_candidates_filters_active_and_positive_strength():
    m = _make_manager()
    candidate = _put_candidate(m, "候选目标", tags=["甲标签"])
    _put(m, "强度未归零不进候选", strength=3)
    _put_candidate(m, "主动记忆不进候选", is_active=True)

    candidates = _candidates(m)

    assert [item.id for item in candidates] == [candidate.id]
    assert candidates[0].judgment == "候选目标"
    assert candidates[0].tags == ["甲标签"]


def test_apply_delete_removes_row_and_fts_document():
    m = _make_manager()
    memory = _put_candidate(m, "RetireAlphaDeleteTarget")
    assert memory.id in _fts_ids(m, "RetireAlphaDeleteTarget")

    result = _apply(m, delete_ids=[memory.id])

    assert result == {"deleted": 1, "promoted": 0}
    assert _row(m, memory.id) is None
    assert memory.id not in _fts_ids(m, "RetireAlphaDeleteTarget")


def test_apply_delete_guard_keeps_fts_document():
    """候选在审查期间被强化时删除会被守卫拦下，FTS 文档必须同步保留。"""
    m = _make_manager()
    memory = _put_candidate(m, "RetireBetaGuardTarget")
    _set_strength(m, memory.id, 1)

    result = _apply(m, delete_ids=[memory.id])

    assert result == {"deleted": 0, "promoted": 0}
    assert _row(m, memory.id) is not None
    assert memory.id in _fts_ids(m, "RetireBetaGuardTarget")


def test_apply_keep_promotes_to_permanent():
    m = _make_manager()
    memory = _put_candidate(m, "保留升级目标")

    result = _apply(m, keep_ids=[memory.id])

    assert result == {"deleted": 0, "promoted": 1}
    assert _is_active(m, memory.id) == 1
    assert _candidates(m) == []


def test_apply_keep_on_permanent_memory_counts_nothing():
    m = _make_manager()
    memory = _put_candidate(m, "已是永久记忆", is_active=True)

    result = _apply(m, keep_ids=[memory.id])

    assert result == {"deleted": 0, "promoted": 0}
    assert _is_active(m, memory.id) == 1


def test_apply_without_decision_leaves_candidate_untouched():
    m = _make_manager()
    memory = _put_candidate(m, "悬而未决保持原状")

    result = _apply(m)

    assert result == {"deleted": 0, "promoted": 0}
    assert _is_active(m, memory.id) == 0
    assert _strength(m, memory.id) == 0


def test_apply_mixed_decisions_and_tag_cleanup():
    m = _make_manager()
    doomed = _put_candidate(m, "RetireGammaDoomed", tags=["待删标签"])
    kept = _put_candidate(m, "RetireGammaKept", tags=["保留标签"])

    result = _apply(m, delete_ids=[doomed.id], keep_ids=[kept.id])

    assert result == {"deleted": 1, "promoted": 1}
    assert _row(m, doomed.id) is None
    assert _is_active(m, kept.id) == 1
    assert doomed.id not in _fts_ids(m, "RetireGammaDoomed")
    assert kept.id in _fts_ids(m, "RetireGammaKept")
    with m._connect() as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) AS c FROM memory_tag_rel WHERE memory_id = ?",
            (doomed.id,),
        ).fetchone()
    assert int(orphans["c"]) == 0


def test_index_format_version_reuse_and_rebuild(tmp_path):
    """格式版本一致时复用已有索引，版本不符时清空重建。"""
    db_path = tmp_path / "test.db"

    first = MemorySqlManager(db_path=db_path)
    assert first._fts_retriever.memory_index_reusable is False
    memory = _put(first, "RetireVersionReuse", strength=1)
    assert memory.id in _fts_ids(first, "RetireVersionReuse")

    second = MemorySqlManager(db_path=db_path)
    assert second._fts_retriever.memory_index_reusable is True
    assert memory.id in _fts_ids(second, "RetireVersionReuse")

    version_file = (
        second._fts_retriever._memory_dir / TantivyBM25Retriever._INDEX_FORMAT_FILE
    )
    version_file.write_text("1", encoding="utf-8")

    third = MemorySqlManager(db_path=db_path)
    assert third._fts_retriever.memory_index_reusable is False
    assert memory.id in _fts_ids(third, "RetireVersionReuse")
