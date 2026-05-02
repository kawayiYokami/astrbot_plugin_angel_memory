from __future__ import annotations

import time
from typing import Dict

from ...llm_memory.components.memory_sql_manager import MemorySqlManager
from ...llm_memory.service.cognitive_service import CognitiveService


class MemoryVectorSyncService:
    """记忆 FAISS 索引同步服务：同步中央记忆索引并清理孤儿向量。"""

    def __init__(self, logger):
        self.logger = logger

    async def sync_memory_vector_index(
        self,
        cognitive_service: CognitiveService,
        memory_sql_manager: MemorySqlManager,
        provider_id: str = "",
    ) -> Dict[str, int]:
        start_time = time.time()
        self.logger.info(f"[FAISS向量索引] 开始 任务名=记忆索引同步 provider_id={provider_id or 'unknown'}")

        try:
            sql_index_rows = await memory_sql_manager.list_memory_index_rows()
        except Exception as e:
            self.logger.error(
                f"[FAISS向量索引] 失败 任务名=记忆索引同步 阶段=读取中央记忆索引 异常={e}",
                exc_info=True,
            )
            return {"sql_total": 0, "vector_total": 0, "missing": 0, "orphan": 0, "migrated": 0, "deleted": 0, "failed": 1}

        index_collection = cognitive_service.memory_index_collection
        try:
            result = await index_collection.sync_rows(sql_index_rows)
        except Exception as e:
            self.logger.error(
                f"[FAISS向量索引] 失败 任务名=记忆索引同步 阶段=同步FAISS 异常={e}",
                exc_info=True,
            )
            return {
                "sql_total": len(sql_index_rows),
                "vector_total": 0,
                "missing": 0,
                "orphan": 0,
                "migrated": 0,
                "deleted": 0,
                "failed": 1,
            }

        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[FAISS向量索引] 完成 任务名=记忆索引同步 "
            f"中央总数={result.get('sql_total', len(sql_index_rows))} "
            f"向量总数={result.get('vector_total', 0)} "
            f"缺失数={result.get('missing', 0)} "
            f"孤儿数={result.get('orphan', 0)} "
            f"同步成功={result.get('migrated', 0)} "
            f"删除成功={result.get('deleted', 0)} "
            f"同步失败={result.get('failed', 0)} "
            f"耗时毫秒={cost_ms}"
        )
        return result
