from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Set

from ...llm_memory.components.memory_sql_manager import MemorySqlManager
from ...llm_memory.service.cognitive_service import CognitiveService


class MemoryVectorSyncService:
    """记忆向量库同步服务：将中央记忆索引缺失项同步到向量库，并清理孤儿向量。"""

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    async def _delete_collection_ids(collection, ids: List[str], batch_size: int = 5000) -> int:
        """分批删除向量库中的记录，返回实际删除的记录数"""
        if not ids:
            return 0

        deleted_count = 0
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            try:
                await asyncio.to_thread(collection.delete, ids=chunk)
                deleted_count += len(chunk)
            except Exception as e:
                raise RuntimeError(f"删除批次失败（已删除 {deleted_count} 条）: {e}") from e

        return deleted_count

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
            return {"sql_total": 0, "vector_total": 0, "missing": 0, "orphan": 0, "migrated": 0, "deleted": 0, "failed": 1}

        try:
            sql_index_rows = await memory_sql_manager.list_memory_index_rows()
        except Exception as e:
            self.logger.error(
                f"[记忆向量库同步] 读取中央记忆索引失败: {e}",
                exc_info=True,
            )
            sql_index_rows = []

        sql_ids: Set[str] = {
            str(item.get("id") or "").strip()
            for item in sql_index_rows
            if str(item.get("id") or "").strip()
        }

        # 缺失：SQL 有，向量没有
        missing = [
            item
            for item in sql_index_rows
            if str(item.get("id") or "").strip() and str(item.get("id")) not in vector_ids
        ]

        # 孤儿：向量有，SQL 没有
        orphan_ids = list(vector_ids - sql_ids)

        migrated = 0
        failed = 0
        deleted = 0

        # 补缺失
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

        # 删孤儿
        if orphan_ids:
            try:
                deleted = await self._delete_collection_ids(index_collection, orphan_ids)
                self.logger.debug(f"[记忆向量库同步] 成功删除 {deleted} 个孤儿向量")
            except Exception as e:
                # 尝试从异常消息中提取已删除的计数
                import re
                match = re.search(r"已删除\s+(\d+)\s*条", str(e))
                deleted = int(match.group(1)) if match else 0
                if deleted > 0:
                    self.logger.warning(
                        f"[记忆向量库同步] 删除孤儿向量部分成功: 已删除 {deleted} 个，失败原因: {e}"
                    )
                else:
                    self.logger.warning(
                        f"[记忆向量库同步] 删除孤儿向量失败: {e}"
                    )

        cost_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "[记忆向量库同步] 完成 "
            f"中央总数={len(sql_index_rows)} "
            f"向量总数={len(vector_ids)} "
            f"缺失数={len(missing)} "
            f"孤儿数={len(orphan_ids)} "
            f"同步成功={migrated} "
            f"删除成功={deleted} "
            f"同步失败={failed} "
            f"耗时毫秒={cost_ms}"
        )
        return {
            "sql_total": len(sql_index_rows),
            "vector_total": len(vector_ids),
            "missing": len(missing),
            "orphan": len(orphan_ids),
            "migrated": migrated,
            "deleted": deleted,
            "failed": failed,
        }
