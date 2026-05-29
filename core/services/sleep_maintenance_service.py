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
        if plugin_context.get_component("cognitive_service") is None:
            deepmind.logger.info("[睡眠维护] 向量到中央迁移：跳过（当前为 BM25-only）")
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
        if cognitive_service is None or memory_sql_manager is None:
            deepmind.logger.warning("[睡眠维护] 向量到中央迁移：失败（组件不可用）")
            return "failed"

        if getattr(cognitive_service, "main_collection", None) is None:
            state["vector_to_center_migration_last_provider"] = current_provider
            deepmind.logger.info(
                "[睡眠维护] 向量到中央迁移：跳过（当前已使用中央库+FAISS轻量索引）"
            )
            return "skipped"

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
        is_bm25_only = plugin_context.get_component("cognitive_service") is None
        provider = plugin_context.get_current_provider()
        current_provider = str(provider) if provider is not None else ""
        target_provider = "bm25_only" if is_bm25_only else current_provider

        if is_bm25_only:
            if state.get("memory_vector_sync_last_provider", "") != "bm25_only":
                state["memory_vector_sync_last_provider"] = "bm25_only"
            deepmind.logger.info("[睡眠维护] 记忆向量库同步：跳过（当前为 BM25-only）")
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

        if deepmind.plugin_context.get_component("vector_store") is None:
            deepmind.logger.info("[睡眠维护] memory_scope 补齐：跳过（当前为 BM25-only）")
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

        if deepmind.plugin_context.get_component("vector_store") is None:
            deepmind.logger.info("[睡眠维护] 废弃字段清理：跳过（当前为 BM25-only）")
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
