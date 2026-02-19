"""
记忆 scope 字段迁移。

将历史记录中缺失 memory_scope 的 metadata 补齐为 public。
"""

import asyncio


class MemoryScopeMigration:
    """缺失 memory_scope 的增量迁移器。"""

    def __init__(self, logger):
        self.logger = logger

    async def migrate_missing_memory_scope(
        self,
        collection,
        batch_size: int = 300,
        sleep_seconds: float = 0.05,
    ) -> None:
        if collection is None:
            self.logger.warning("memory_scope 迁移跳过：collection 不可用。")
            return

        scanned = 0
        patched = 0
        offset = 0
        failed_update_batches = []

        while True:
            results = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            ids = results.get("ids", []) if results else []
            metadatas = results.get("metadatas", []) if results else []

            if not ids:
                break

            scanned += len(ids)
            to_update_ids = []
            to_update_metas = []

            for idx, mem_id in enumerate(ids):
                try:
                    meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                    scope = str(meta.get("memory_scope", "")).strip() if isinstance(meta, dict) else ""
                    if scope:
                        continue

                    if not isinstance(meta, dict):
                        meta = {}
                    new_meta = dict(meta)
                    new_meta["memory_scope"] = "public"
                    to_update_ids.append(mem_id)
                    to_update_metas.append(new_meta)
                except Exception as e:
                    self.logger.warning(f"memory_scope 迁移跳过异常记录 id={mem_id}: {e}")

            if to_update_ids:
                try:
                    collection.update(ids=to_update_ids, metadatas=to_update_metas)
                    patched += len(to_update_ids)
                except Exception as e:
                    self.logger.error(
                        f"memory_scope 迁移批次更新失败: {e}; ids={to_update_ids}"
                    )
                    failed_update_batches.append((to_update_ids, to_update_metas))

            self.logger.info(
                f"[memory_scope迁移] 已扫描={scanned} 已补齐={patched} 当前批次={len(ids)}"
            )

            offset += len(ids)
            await asyncio.sleep(sleep_seconds)

        failed_update_ids = []
        for ids_batch, metas_batch in failed_update_batches:
            try:
                collection.update(ids=ids_batch, metadatas=metas_batch)
                patched += len(ids_batch)
                self.logger.info(
                    f"[memory_scope迁移] 重试成功，补齐={len(ids_batch)}"
                )
            except Exception as retry_error:
                failed_update_ids.extend(ids_batch)
                self.logger.error(
                    f"memory_scope 迁移重试失败: {retry_error}; ids={ids_batch}"
                )

        self.logger.info(
            f"[memory_scope迁移] 完成，总扫描={scanned}，总补齐={patched}"
        )
        if failed_update_ids:
            self.logger.error(
                f"[memory_scope迁移] 仍有未补齐记录，失败ID数量={len(failed_update_ids)}，ids={failed_update_ids}"
            )
