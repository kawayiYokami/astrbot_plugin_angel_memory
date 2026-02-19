from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from ...llm_memory.components.memory_sql_manager import MemorySqlManager


class SimpleMemoryBackupService:
    """向量记忆 -> SimpleMemory 的备份服务。"""

    def __init__(self, logger):
        self.logger = logger

    async def backup_from_collection(
        self,
        collection: Any,
        memory_sql_manager: MemorySqlManager,
        source: str,
        provider_id: str = "",
    ) -> Dict[str, int]:
        return await asyncio.to_thread(
            self._backup_from_collection_sync,
            collection,
            memory_sql_manager,
            source,
            provider_id,
        )

    def _backup_from_collection_sync(
        self,
        collection: Any,
        memory_sql_manager: MemorySqlManager,
        source: str,
        provider_id: str = "",
    ) -> Dict[str, int]:
        start_time = time.time()
        self.logger.info(
            f"[simple_backup] start source={source} provider={provider_id or 'unknown'}"
        )
        self.logger.info("[simple_backup] phase=fetch_metadatas")

        try:
            result = collection.get(include=["metadatas"])
            metadatas: List[Dict] = list((result or {}).get("metadatas") or [])
        except Exception as e:
            self.logger.error(
                f"[simple_backup] failed source={source} error=读取向量记忆失败: {e}",
                exc_info=True,
            )
            return {"scanned": 0, "deduped": 0, "upserted": 0, "failed": 1}
        self.logger.info(f"[simple_backup] phase=normalize scanned_raw={len(metadatas)}")

        normalized_memories: List[Dict] = []
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            judgment = str(meta.get("judgment") or "").strip()
            if not judgment:
                continue
            normalized_memories.append(
                {
                    "memory_type": meta.get("memory_type", "知识记忆"),
                    "judgment": judgment,
                    "reasoning": str(meta.get("reasoning") or "").strip(),
                    "tags": meta.get("tags", ""),
                    "strength": meta.get("strength", 1),
                    "is_active": meta.get("is_active", False),
                    "memory_scope": meta.get("memory_scope", "public"),
                    "created_at": meta.get("created_at", 0),
                }
            )

        self.logger.info(
            f"[simple_backup] phase=upsert input={len(normalized_memories)}"
        )
        stats = asyncio.run(
            memory_sql_manager.upsert_memories_by_judgment(normalized_memories)
        )
        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[simple_backup] done "
            f"scanned={stats.get('scanned', 0)} "
            f"deduped={stats.get('deduped', 0)} "
            f"upserted={stats.get('upserted', 0)} "
            f"failed={stats.get('failed', 0)} "
            f"cost_ms={cost_ms}"
        )
        return stats
