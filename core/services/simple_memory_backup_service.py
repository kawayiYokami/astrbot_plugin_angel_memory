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
        start_time = time.time()
        self.logger.info(
            f"[simple_backup] 开始备份 source={source} provider={provider_id or 'unknown'}"
        )
        self.logger.info("[simple_backup] 阶段=读取向量元数据")

        try:
            result = await asyncio.to_thread(
                collection.get,
                include=["metadatas", "documents"],
            )
            ids: List[str] = list((result or {}).get("ids") or [])
            metadatas: List[Dict] = list((result or {}).get("metadatas") or [])
            documents: List[str] = list((result or {}).get("documents") or [])
        except Exception as e:
            self.logger.error(
                f"[simple_backup] 备份失败 source={source} error=读取向量记忆失败: {e}",
                exc_info=True,
            )
            return {"scanned": 0, "deduped": 0, "upserted": 0, "failed": 1}
        scanned = max(len(ids), len(metadatas), len(documents))
        self.logger.info(f"[simple_backup] 阶段=数据整理 scanned_raw={scanned}")

        normalized_memories: List[Dict] = []
        skipped_no_judgment = 0
        for idx in range(scanned):
            meta: Dict = (
                metadatas[idx]
                if idx < len(metadatas) and isinstance(metadatas[idx], dict)
                else {}
            )
            doc_text = str(documents[idx] or "").strip() if idx < len(documents) else ""
            judgment = str(meta.get("judgment") or "").strip() or doc_text
            if not judgment:
                skipped_no_judgment += 1
                continue

            reasoning = str(meta.get("reasoning") or "").strip()
            if not reasoning and doc_text and doc_text != judgment:
                reasoning = doc_text

            normalized_memories.append(
                {
                    "memory_type": meta.get("memory_type", "知识记忆"),
                    "judgment": judgment,
                    "reasoning": reasoning,
                    "tags": meta.get("tags", ""),
                    "strength": meta.get("strength", 1),
                    "is_active": meta.get("is_active", False),
                    "memory_scope": meta.get("memory_scope", "public"),
                    "created_at": meta.get("created_at") or int(time.time()),
                }
            )

        self.logger.info(
            f"[simple_backup] 阶段=写入Simple库 input={len(normalized_memories)} skipped_no_judgment={skipped_no_judgment}"
        )
        try:
            stats = await memory_sql_manager.upsert_memories_by_judgment(normalized_memories)
        except Exception as e:
            self.logger.error(
                f"[simple_backup] 备份失败 source={source} error=写入Simple库失败: {e}",
                exc_info=True,
            )
            return {
                "scanned": len(metadatas),
                "deduped": 0,
                "upserted": 0,
                "failed": len(normalized_memories) or 1,
            }
        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[simple_backup] 备份完成 "
            f"scanned={stats.get('scanned', 0)} "
            f"deduped={stats.get('deduped', 0)} "
            f"upserted={stats.get('upserted', 0)} "
            f"failed={stats.get('failed', 0)} "
            f"skipped_no_judgment={skipped_no_judgment} "
            f"cost_ms={cost_ms}"
        )
        return stats
