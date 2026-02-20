from __future__ import annotations

import asyncio
import time
from typing import Dict, Set

from ...llm_memory.components.memory_sql_manager import MemorySqlManager
from ...llm_memory.service.cognitive_service import CognitiveService


class MemoryVectorSyncService:
    """记忆向量库同步服务：将中央记忆索引缺失项同步到向量库。"""

    def __init__(self, logger):
        self.logger = logger

    async def sync_memory_vector_index(
        self,
        cognitive_service: CognitiveService,
        memory_sql_manager: MemorySqlManager,
        provider_id: str = "",
    ) -> Dict[str, int]:
        start_time = time.time()
        self.logger.info(
            f"[记忆向量库同步] 开始，供应商={provider_id or 'unknown'}"
        )

        index_collection = cognitive_service.vector_store.get_or_create_collection_with_dimension_check(
            "memory_index"
        )

        try:
            vector_result = await asyncio.to_thread(index_collection.get)
            vector_ids_raw = list((vector_result or {}).get("ids") or [])
            vector_ids: Set[str] = {str(mid) for mid in vector_ids_raw if mid}
        except Exception as e:
            self.logger.error(
                f"[记忆向量库同步] 失败：读取向量库异常: {e}",
                exc_info=True,
            )
            return {"sql_total": 0, "vector_total": 0, "missing": 0, "migrated": 0, "failed": 1}

        try:
            sql_index_rows = await memory_sql_manager.list_memory_index_rows()
        except Exception as e:
            self.logger.error(
                f"[记忆向量库同步] 读取中央记忆索引失败: {e}",
                exc_info=True,
            )
            sql_index_rows = []

        missing = [
            item
            for item in sql_index_rows
            if str(item.get("id") or "").strip() and str(item.get("id")) not in vector_ids
        ]

        migrated = 0
        failed = 0
        try:
            await cognitive_service.vector_store.upsert_memory_index_rows(
                collection=index_collection,
                rows=missing,
            )
            migrated = len(missing)
        except Exception as e:
            failed = len(missing) if missing else 1
            self.logger.warning(
                f"[记忆向量库同步] 写入向量索引失败: {e}"
            )

        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[记忆向量库同步] 完成 "
            f"中央总数={len(sql_index_rows)} "
            f"向量总数={len(vector_ids)} "
            f"缺失数={len(missing)} "
            f"同步成功={migrated} "
            f"同步失败={failed} "
            f"耗时毫秒={cost_ms}"
        )
        return {
            "sql_total": len(sql_index_rows),
            "vector_total": len(vector_ids),
            "missing": len(missing),
            "migrated": migrated,
            "failed": failed,
        }
