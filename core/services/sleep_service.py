from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

from ...llm_memory.models.data_models import BaseMemory
from ...llm_memory.service.memory_retirement_reviewer import (
    MemoryRetirementReviewer,
    RetirementReviewError,
)
from .sleep_maintenance_service import SleepMaintenanceService


class DeepMindSleepService:
    """DeepMind 的睡眠巩固职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind
        self.maintenance_service = SleepMaintenanceService(deepmind)
        self.reviewer = MemoryRetirementReviewer(
            context=getattr(deepmind, "context", None),
            provider_id=getattr(deepmind, "provider_id", ""),
            logger=getattr(deepmind, "logger", None),
        )

    async def sleep(self) -> bool:
        deepmind = self.deepmind
        if not deepmind.is_enabled():
            return False
        if deepmind.memory_system is None:
            deepmind.logger.error("[睡眠] 失败（记忆系统不可用）")
            return False

        start_time = time.time()
        try:
            deepmind.logger.info("[睡眠] 阶段=前置维护 开始")
            await self.maintenance_service.run_pre_consolidate()
            deepmind.logger.info("[睡眠] 阶段=前置维护 完成")

            deepmind.logger.info("[睡眠] 阶段=记忆回收 开始")
            recycle_result = await self._recycle_memories()
            deepmind.logger.info("[睡眠] 阶段=记忆回收 完成")

            cleanup_completed_at = time.time()
            deepmind.logger.info("[睡眠] 阶段=后置维护 开始")
            await self.maintenance_service.run_post_consolidate(
                cleanup_completed_at=cleanup_completed_at
            )
            deepmind.logger.info("[睡眠] 阶段=后置维护 完成")

            await self._record_sleep_trace(recycle_result, start_time)
            return True
        except Exception as e:
            elapsed_time = time.time() - start_time
            deepmind.logger.error(
                f"记忆巩固失败（耗时 {elapsed_time:.2f} 秒）: {e}", exc_info=True
            )
            return False

    async def _recycle_memories(self) -> Dict[str, Any]:
        """衰减 -> 取候选 -> 审查 -> 应用裁决。审查失败时整批跳过。"""
        deepmind = self.deepmind
        await deepmind.memory_system.decay_tier0_memories()

        candidates = await deepmind.memory_system.list_retirement_candidates()
        if not candidates:
            deepmind.logger.info("[淘汰审查] 跳过（无候选）")
            return self._empty_recycle_result()

        deepmind.logger.info(f"[淘汰审查] 开始 候选数={len(candidates)}")
        review_start = time.time()
        try:
            outcome = await self.reviewer.review(candidates)
        except RetirementReviewError as e:
            cost_ms = int((time.time() - review_start) * 1000)
            deepmind.logger.error(
                f"[淘汰审查] 失败 候选数={len(candidates)} "
                f"本次整批跳过，不删除任何记忆 耗时毫秒={cost_ms} 异常={e}"
            )
            return {
                "candidates": len(candidates),
                "deleted": 0,
                "promoted": 0,
                "undecided": 0,
                "skipped": True,
                "cost_ms": cost_ms,
                "prompt": e.prompt,
                "system_prompt": e.system_prompt,
                "response": e.response,
                "error": str(e),
            }

        delete_ids, keep_ids, undecided = self._split_decisions(
            candidates, outcome.decisions
        )
        result = await deepmind.memory_system.apply_retirement_decisions(
            delete_ids, keep_ids
        )
        cost_ms = int((time.time() - review_start) * 1000)
        deepmind.logger.info(
            f"[淘汰审查] 完成 候选数={len(candidates)} 判删={len(delete_ids)} "
            f"判留={len(keep_ids)} 悬而不决={len(undecided)} 耗时毫秒={cost_ms}"
        )
        return {
            "candidates": len(candidates),
            "deleted": int(result.get("deleted", 0) or 0),
            "promoted": int(result.get("promoted", 0) or 0),
            "undecided": len(undecided),
            "skipped": False,
            "cost_ms": cost_ms,
            "prompt": outcome.prompt,
            "system_prompt": outcome.system_prompt,
            "response": outcome.response,
            "error": "",
        }

    @staticmethod
    def _split_decisions(
        candidates: Sequence[BaseMemory],
        decisions: Dict[str, str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """按裁决表拆分候选：删除、永久化、悬而不决。"""
        delete_ids: List[str] = []
        keep_ids: List[str] = []
        undecided: List[str] = []
        for memory in candidates:
            memory_id = str(getattr(memory, "id", "") or "")
            if not memory_id:
                continue
            verdict = decisions.get(memory_id)
            if verdict == "delete":
                delete_ids.append(memory_id)
            elif verdict == "keep":
                keep_ids.append(memory_id)
            else:
                undecided.append(memory_id)
        return delete_ids, keep_ids, undecided

    @staticmethod
    def _empty_recycle_result() -> Dict[str, Any]:
        return {
            "candidates": 0,
            "deleted": 0,
            "promoted": 0,
            "undecided": 0,
            "skipped": False,
            "cost_ms": 0,
            "prompt": "",
            "system_prompt": "",
            "response": "",
            "error": "",
        }

    async def _record_sleep_trace(
        self,
        recycle_result: Dict[str, Any],
        start_time: float,
    ) -> None:
        """把本轮睡眠的审查过程追加到持久化追踪文件。"""
        deepmind = self.deepmind
        trace_store = getattr(deepmind, "trace_store", None)
        if trace_store is None:
            return
        try:
            await trace_store.append(
                {
                    "type": "sleep",
                    "status": "failed" if recycle_result.get("skipped") else "success",
                    "provider_id": getattr(deepmind, "provider_id", ""),
                    "prompt": recycle_result.get("prompt", ""),
                    "system_prompt": recycle_result.get("system_prompt", ""),
                    "response": recycle_result.get("response", ""),
                    "error": recycle_result.get("error", ""),
                    "summary": {
                        "candidates": int(recycle_result.get("candidates", 0) or 0),
                        "deleted": int(recycle_result.get("deleted", 0) or 0),
                        "promoted": int(recycle_result.get("promoted", 0) or 0),
                        "undecided": int(recycle_result.get("undecided", 0) or 0),
                        "review_skipped": bool(recycle_result.get("skipped", False)),
                        "review_cost_ms": int(recycle_result.get("cost_ms", 0) or 0),
                        "sleep_cost_ms": int((time.time() - start_time) * 1000),
                    },
                }
            )
        except Exception as e:
            deepmind.logger.warning(f"[过程追踪] 睡眠记录落盘失败: {e}")
