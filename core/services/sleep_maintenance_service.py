from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from ..migrations.memory_scope_migration import MemoryScopeMigration
from .simple_memory_backup_service import SimpleMemoryBackupService
from .memory_vector_sync_service import MemoryVectorSyncService


class SleepMaintenanceService:
    """睡眠维护管线：任务按固定顺序执行，任务内部自行判定。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind
        plugin_context = getattr(deepmind, "plugin_context", None)
        base_dir = plugin_context.get_memory_center_dir() if plugin_context else Path(".")
        self._state_file = Path(base_dir) / "maintenance_state.json"

    def _load_state(self) -> Dict[str, Any]:
        default_state: Dict[str, Any] = {
            "last_updated_at": 0.0,
            "memory_scope_migration_done": False,
            "deprecated_field_cleanup_done": False,
            "vector_to_center_migration_last_provider": "",
            "memory_vector_sync_last_provider": "",
            "notes_vector_sync_last_provider": "",
            "daily_json_backup_last_day": "",
            "cleanup_last_sleep_at": 0.0,
        }
        if not self._state_file.exists():
            return default_state
        try:
            with self._state_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return default_state
            merged = dict(default_state)
            merged.update(loaded)
            return merged
        except Exception:
            self.deepmind.logger.warning("读取 maintenance_state.json 失败，已回退默认状态。")
            return default_state

    def _save_state(self, state: Dict[str, Any]) -> None:
        state["last_updated_at"] = time.time()
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with self._state_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def run_pre_consolidate(self) -> str:
        """清理前维护：回灌前置，确保本轮清理可覆盖回灌数据。"""
        deepmind = self.deepmind
        plugin_context = getattr(deepmind, "plugin_context", None)
        if plugin_context is None:
            return "skipped"

        state = self._load_state()
        try:
            migration_status = await self._task_vector_to_center_migration(state)
            sync_status = await self._task_memory_vector_sync(state)
            self._save_state(state)
            if "failed" in (migration_status, sync_status):
                return "failed"
            if migration_status == "success" or sync_status == "success":
                return "success"
            return "skipped"
        except Exception as e:
            deepmind.logger.error(
                f"[睡眠维护] 前置维护执行失败: {e}",
                exc_info=True,
            )
            self._save_state(state)
            return "failed"

    async def _task_vector_to_center_migration(self, state: Dict[str, Any]) -> str:
        """向量主集合 -> 中央SQL（按 provider 仅执行一次，provider 变化时重跑）。"""
        deepmind = self.deepmind
        plugin_context = deepmind.plugin_context
        enable_simple = getattr(deepmind.config, "enable_simple_memory", False)
        if enable_simple:
            deepmind.logger.info(
                "[睡眠维护] 向量到中央迁移：跳过（simple 模式）"
            )
            return "skipped"

        provider = plugin_context.get_current_provider()
        current_provider = str(provider) if provider is not None else ""
        last_provider = str(
            state.get("vector_to_center_migration_last_provider", "") or ""
        )
        if current_provider and last_provider == current_provider:
            deepmind.logger.info(
                "[睡眠维护] 向量到中央迁移：跳过（当前供应商已迁移）"
            )
            return "skipped"

        cognitive_service = plugin_context.get_component("cognitive_service")
        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        if (
            cognitive_service is None
            or memory_sql_manager is None
            or not hasattr(cognitive_service, "main_collection")
        ):
            deepmind.logger.warning(
                "[睡眠维护] 向量到中央迁移：失败（组件不可用）"
            )
            return "failed"

        backup_service = SimpleMemoryBackupService(deepmind.logger)
        result = await backup_service.backup_from_collection(
            collection=cognitive_service.main_collection,
            memory_sql_manager=memory_sql_manager,
            source="main_collection",
            provider_id=current_provider,
        )
        if int(result.get("failed", 0)) > 0:
            deepmind.logger.warning(
                "[睡眠维护] 向量到中央迁移：失败（存在失败项）"
            )
            return "failed"

        state["vector_to_center_migration_last_provider"] = current_provider
        deepmind.logger.info(
            "[睡眠维护] 向量到中央迁移：完成 "
            f"写入条数={result.get('upserted', 0)}"
        )
        return "success"

    async def run_post_consolidate(self, cleanup_completed_at: float) -> None:
        """清理后维护：迁移/清理/备份。"""
        deepmind = self.deepmind
        plugin_context = getattr(deepmind, "plugin_context", None)
        if plugin_context is None:
            return

        state = self._load_state()
        state["cleanup_last_sleep_at"] = cleanup_completed_at

        start = time.time()
        success = 0
        failed = 0
        skipped = 0

        deepmind.logger.info("[睡眠维护] 后置维护开始")
        for runner in (
            self._task_memory_scope_migration,
            self._task_deprecated_field_cleanup,
            self._task_notes_vector_sync,
            self._task_daily_json_backup,
        ):
            try:
                status = await runner(state)
                if status == "success":
                    success += 1
                elif status == "failed":
                    failed += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                deepmind.logger.error(f"[睡眠维护] 任务异常 {runner.__name__}: {e}", exc_info=True)

        self._save_state(state)
        cost_ms = int((time.time() - start) * 1000)
        deepmind.logger.info(
            f"[睡眠维护] 后置维护完成 成功={success} 失败={failed} 跳过={skipped} 耗时毫秒={cost_ms}"
        )

    async def _task_memory_vector_sync(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        plugin_context = deepmind.plugin_context
        enable_simple = getattr(deepmind.config, "enable_simple_memory", False)
        provider = plugin_context.get_current_provider()
        current_provider = str(provider) if provider is not None else ""
        target_provider = "simple" if enable_simple else current_provider

        if enable_simple:
            if state.get("memory_vector_sync_last_provider", "") != "simple":
                state["memory_vector_sync_last_provider"] = "simple"
            deepmind.logger.info("[睡眠维护] 记忆向量库同步：跳过（当前为 simple 模式）")
            return "skipped"

        last_provider = str(state.get("memory_vector_sync_last_provider", "") or "")
        if last_provider == target_provider:
            deepmind.logger.info("[睡眠维护] 记忆向量库同步：跳过（供应商未变化）")
            return "skipped"

        cognitive_service = plugin_context.get_component("cognitive_service")
        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        if cognitive_service is None or memory_sql_manager is None:
            deepmind.logger.warning("[睡眠维护] 记忆向量库同步：失败（组件不可用）")
            return "failed"

        sync_service = MemoryVectorSyncService(deepmind.logger)
        result = await sync_service.sync_memory_vector_index(
            cognitive_service=cognitive_service,
            memory_sql_manager=memory_sql_manager,
            provider_id=current_provider,
        )
        if int(result.get("failed", 0)) > 0:
            deepmind.logger.warning("[睡眠维护] 记忆向量库同步：失败（存在失败项）")
            return "failed"

        state["memory_vector_sync_last_provider"] = target_provider
        deepmind.logger.info("[睡眠维护] 记忆向量库同步：完成")
        return "success"

    async def _task_notes_vector_sync(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        plugin_context = deepmind.plugin_context
        if getattr(deepmind.config, "enable_simple_memory", False):
            deepmind.logger.info("[睡眠维护] 笔记向量库同步：跳过（simple 模式）")
            return "skipped"

        provider = plugin_context.get_current_provider()
        current_provider = str(provider) if provider is not None else ""
        last_provider = str(state.get("notes_vector_sync_last_provider", "") or "")
        # 与记忆迁移/同步保持一致：首次无专用记录时，复用已有 provider 状态避免误判全量重建
        if not last_provider:
            last_provider = str(state.get("vector_to_center_migration_last_provider", "") or "")
        if not last_provider:
            last_provider = str(state.get("memory_vector_sync_last_provider", "") or "")

        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        vector_store = plugin_context.get_component("vector_store")
        if memory_sql_manager is None or vector_store is None:
            deepmind.logger.warning("[睡眠维护] 笔记向量库同步：失败（组件不可用）")
            return "failed"

        try:
            notes_index = vector_store.get_or_create_collection_with_dimension_check("notes_index")
            note_service = plugin_context.get_component("note_service")
            if note_service is not None:
                note_service.notes_index_collection = notes_index
            rows = await memory_sql_manager.list_note_index_vector_rows()

            # 供应商变化：全量重建（embedding 语义空间变化）
            if current_provider and last_provider != current_provider:
                deepmind.logger.info(
                    "[睡眠维护] 笔记向量库同步：检测到供应商变化，开始全量重建 "
                    f"旧供应商={last_provider or '空'} 新供应商={current_provider} "
                    f"待写入条数={len(rows)}"
                )
                vector_store.clear_collection(notes_index)
                notes_index = vector_store.get_or_create_collection_with_dimension_check("notes_index")
                if note_service is not None:
                    note_service.notes_index_collection = notes_index
                await vector_store.upsert_note_index_rows(notes_index, rows)
                state["notes_vector_sync_last_provider"] = current_provider
                deepmind.logger.info(
                    "[睡眠维护] 笔记向量库同步：完成（供应商变化全量重建） "
                    f"写入条数={len(rows)}"
                )
                return "success"

            # 供应商未变化：增量同步（只补新增、删失效）
            sql_ids = {str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()}
            vector_ids = await self._list_collection_ids(notes_index)
            vector_id_set = set(vector_ids)

            to_add = sql_ids - vector_id_set
            to_delete = vector_id_set - sql_ids

            if to_delete:
                await self._delete_collection_ids(notes_index, list(to_delete))

            if to_add:
                add_rows = [row for row in rows if str(row.get("id") or "").strip() in to_add]
                await vector_store.upsert_note_index_rows(notes_index, add_rows)

            if not to_add and not to_delete:
                if current_provider:
                    state["notes_vector_sync_last_provider"] = current_provider
                deepmind.logger.info("[睡眠维护] 笔记向量库同步：跳过（无增量变更）")
                return "skipped"

            deepmind.logger.info(
                "[睡眠维护] 笔记向量库同步：完成（增量同步） "
                f"新增={len(to_add)} 删除={len(to_delete)}"
            )
            if current_provider:
                state["notes_vector_sync_last_provider"] = current_provider
            return "success"
        except Exception as e:
            deepmind.logger.warning(
                f"[睡眠维护] 笔记向量库同步：失败，异常={e}",
                exc_info=True,
            )
            return "failed"

    @staticmethod
    async def _list_collection_ids(collection, batch_size: int = 5000) -> list[str]:
        ids: list[str] = []
        offset = 0
        while True:
            result = collection.get(limit=batch_size, offset=offset, include=[])
            batch = result.get("ids", []) if isinstance(result, dict) else []
            if not batch:
                break
            ids.extend([str(x) for x in batch if str(x)])
            if len(batch) < batch_size:
                break
            offset += len(batch)
        return ids

    @staticmethod
    async def _delete_collection_ids(collection, ids: list[str], batch_size: int = 5000) -> None:
        if not ids:
            return
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            collection.delete(ids=chunk)

    async def _task_memory_scope_migration(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        if bool(state.get("memory_scope_migration_done", False)):
            deepmind.logger.info("[睡眠维护] memory_scope 补齐：跳过（已完成）")
            return "skipped"

        # 中央库真相源模式下，不再依赖旧向量 metadata 的 memory_scope 迁移。
        if deepmind.plugin_context.get_component("memory_sql_manager") is not None:
            state["memory_scope_migration_done"] = True
            deepmind.logger.info("[睡眠维护] memory_scope 补齐：跳过（中央库模式无需执行）")
            return "skipped"

        if getattr(deepmind.config, "enable_simple_memory", False):
            deepmind.logger.info("[睡眠维护] memory_scope 补齐：跳过（simple 模式）")
            return "skipped"

        cognitive_service = deepmind.plugin_context.get_component("cognitive_service")
        if cognitive_service is None or not hasattr(cognitive_service, "main_collection"):
            deepmind.logger.warning("[睡眠维护] memory_scope 补齐：失败（集合不可用）")
            return "failed"

        migrator = MemoryScopeMigration(deepmind.logger)
        await migrator.migrate_missing_memory_scope(collection=cognitive_service.main_collection)
        state["memory_scope_migration_done"] = True
        deepmind.logger.info("[睡眠维护] memory_scope 补齐：完成")
        return "success"

    async def _task_deprecated_field_cleanup(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        if bool(state.get("deprecated_field_cleanup_done", False)):
            deepmind.logger.info("[睡眠维护] 废弃字段清理：跳过（已完成）")
            return "skipped"

        # 中央库真相源模式下，不再依赖旧向量 metadata 的 is_consolidated 清理。
        if deepmind.plugin_context.get_component("memory_sql_manager") is not None:
            state["deprecated_field_cleanup_done"] = True
            deepmind.logger.info("[睡眠维护] 废弃字段清理：跳过（中央库模式无需执行）")
            return "skipped"

        if getattr(deepmind.config, "enable_simple_memory", False):
            deepmind.logger.info("[睡眠维护] 废弃字段清理：跳过（simple 模式）")
            return "skipped"

        cognitive_service = deepmind.plugin_context.get_component("cognitive_service")
        if cognitive_service is None or not hasattr(cognitive_service, "main_collection"):
            deepmind.logger.warning("[睡眠维护] 废弃字段清理：失败（集合不可用）")
            return "failed"

        collection = cognitive_service.main_collection
        offset = 0
        batch_size = 300
        cleaned = 0
        while True:
            results = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            ids = results.get("ids", []) if results else []
            metas = results.get("metadatas", []) if results else []
            if not ids:
                break

            update_ids = []
            update_metas = []
            for idx, mem_id in enumerate(ids):
                meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                if "is_consolidated" in meta:
                    new_meta = dict(meta)
                    new_meta.pop("is_consolidated", None)
                    update_ids.append(mem_id)
                    update_metas.append(new_meta)

            if update_ids:
                collection.update(ids=update_ids, metadatas=update_metas)
                cleaned += len(update_ids)

            offset += len(ids)

        state["deprecated_field_cleanup_done"] = True
        deepmind.logger.info(f"[睡眠维护] 废弃字段清理：完成 清理条数={cleaned}")
        return "success"

    async def _task_daily_json_backup(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        plugin_context = deepmind.plugin_context
        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        if memory_sql_manager is None:
            deepmind.logger.warning("[睡眠维护] 每日 JSON 备份：失败（memory_sql_manager 不可用）")
            return "failed"

        today = time.strftime("%Y%m%d", time.localtime())
        if str(state.get("daily_json_backup_last_day", "")) == today:
            deepmind.logger.info("[睡眠维护] 每日 JSON 备份：跳过（今日已备份）")
            return "skipped"

        center_dir = plugin_context.get_memory_center_dir()
        backup_dir = Path(center_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / f"memory_backup_{today}.json"

        snapshot = await memory_sql_manager.export_backup_snapshot()
        with backup_file.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        backup_files = sorted(backup_dir.glob("memory_backup_*.json"))
        while len(backup_files) > 3:
            stale = backup_files.pop(0)
            try:
                stale.unlink(missing_ok=True)
            except Exception as e:
                deepmind.logger.warning(f"[睡眠维护] 删除旧备份失败 文件={stale.name} 异常={e}")

        state["daily_json_backup_last_day"] = today
        deepmind.logger.info(
            f"[睡眠维护] 每日 JSON 备份：完成 文件={backup_file.name}"
        )
        return "success"
