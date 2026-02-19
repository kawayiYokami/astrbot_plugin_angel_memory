from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Set

from ...llm_memory.components.memory_sql_manager import MemorySqlManager
from ...llm_memory.service.cognitive_service import CognitiveService


class SimpleToVectorSyncService:
    """向量模式启动时：将 simple 库中缺失记忆回灌到向量库。"""

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def _normalize_judgment(value: str) -> str:
        return str(value or "").strip()

    @staticmethod
    def _map_memory_type(raw_type: str) -> str:
        raw = str(raw_type or "").strip()
        mapping = {
            "知识记忆": "knowledge",
            "事件记忆": "event",
            "技能记忆": "skill",
            "任务记忆": "task",
            "情感记忆": "emotional",
            "knowledge": "knowledge",
            "event": "event",
            "skill": "skill",
            "task": "task",
            "emotional": "emotional",
        }
        return mapping.get(raw, "knowledge")

    async def sync_missing_memories(
        self,
        cognitive_service: CognitiveService,
        memory_sql_manager: MemorySqlManager,
        provider_id: str = "",
    ) -> Dict[str, int]:
        start_time = time.time()
        self.logger.info(
            f"[simple_to_vector_sync] start provider={provider_id or 'unknown'}"
        )

        try:
            vector_result = await asyncio.to_thread(
                cognitive_service.main_collection.get, include=["metadatas"]
            )
            vector_metadatas: List[Dict] = list((vector_result or {}).get("metadatas") or [])
        except Exception as e:
            self.logger.error(
                f"[simple_to_vector_sync] failed error=读取向量库失败: {e}",
                exc_info=True,
            )
            return {"sql_total": 0, "vector_total": 0, "missing": 0, "migrated": 0, "failed": 1}

        vector_judgments: Set[str] = set()
        for meta in vector_metadatas:
            if not isinstance(meta, dict):
                continue
            judgment = self._normalize_judgment(meta.get("judgment"))
            if judgment:
                vector_judgments.add(judgment)

        try:
            sql_memories = await memory_sql_manager.list_all_memories_for_vector_sync()
        except Exception as e:
            self.logger.error(
                f"[simple_to_vector_sync] list_all_memories_for_vector_sync failed: {e}",
                exc_info=True,
            )
            sql_memories = []
        missing = [
            item
            for item in sql_memories
            if self._normalize_judgment(item.get("judgment")) not in vector_judgments
        ]

        migrated = 0
        failed = 0
        for item in missing:
            try:
                raw_strength = item.get("strength", 1)
                if raw_strength in (None, ""):
                    safe_strength = 1
                else:
                    try:
                        safe_strength = int(raw_strength)
                    except (TypeError, ValueError):
                        safe_strength = 1
                await cognitive_service.remember(
                    memory_type=self._map_memory_type(item.get("memory_type")),
                    judgment=str(item.get("judgment") or "").strip(),
                    reasoning=str(item.get("reasoning") or "").strip(),
                    tags=item.get("tags") or [],
                    is_active=bool(item.get("is_active", False)),
                    strength=safe_strength,
                    memory_scope=str(item.get("memory_scope") or "public").strip() or "public",
                )
                migrated += 1
            except Exception as e:
                failed += 1
                self.logger.warning(
                    f"[simple_to_vector_sync] migrate_failed judgment={item.get('judgment')} error={e}"
                )

        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[simple_to_vector_sync] done "
            f"sql_total={len(sql_memories)} "
            f"vector_total={len(vector_judgments)} "
            f"missing={len(missing)} "
            f"migrated={migrated} "
            f"failed={failed} "
            f"cost_ms={cost_ms}"
        )
        return {
            "sql_total": len(sql_memories),
            "vector_total": len(vector_judgments),
            "missing": len(missing),
            "migrated": migrated,
            "failed": failed,
        }
