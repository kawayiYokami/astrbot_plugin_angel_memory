import json
import time
from pathlib import Path
from typing import Any, Dict


class SleepMaintenanceService:
    """睡眠维护管线：p-memory 模式仅保留每日 JSON 备份。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind
        plugin_context = getattr(deepmind, "plugin_context", None)
        base_dir = plugin_context.get_memory_center_dir() if plugin_context else Path(".")
        self._state_file = Path(base_dir) / "maintenance_state.json"

    def _load_state(self) -> Dict[str, Any]:
        default_state: Dict[str, Any] = {
            "last_updated_at": 0.0,
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
            for key in default_state:
                if key in loaded:
                    merged[key] = loaded[key]
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
        """p-memory 模式无前置任务，直接跳过。"""
        return "skipped"

    async def run_post_consolidate(self, cleanup_completed_at: float) -> None:
        """清理后维护：仅每日 JSON 备份。"""
        deepmind = self.deepmind
        plugin_context = getattr(deepmind, "plugin_context", None)
        if plugin_context is None:
            return

        state = self._load_state()
        state["cleanup_last_sleep_at"] = cleanup_completed_at

        start = time.time()
        try:
            status = await self._task_daily_json_backup(state)
        except Exception as e:
            status = "failed"
            deepmind.logger.error(f"[睡眠维护] 任务异常 _task_daily_json_backup: {e}", exc_info=True)

        self._save_state(state)
        cost_ms = int((time.time() - start) * 1000)
        if status == "success":
            deepmind.logger.info(f"[睡眠维护] 后置维护完成 成功=1 失败=0 跳过=0 耗时毫秒={cost_ms}")
        elif status == "failed":
            deepmind.logger.info(f"[睡眠维护] 后置维护完成 成功=0 失败=1 跳过=0 耗时毫秒={cost_ms}")
        else:
            deepmind.logger.info(f"[睡眠维护] 后置维护完成 成功=0 失败=0 跳过=1 耗时毫秒={cost_ms}")

    async def _task_daily_json_backup(self, state: Dict[str, Any]) -> str:
        deepmind = self.deepmind
        plugin_context = deepmind.plugin_context
        memory_store = plugin_context.get_component("memory_store")
        if memory_store is None:
            deepmind.logger.warning("[睡眠维护] 每日 JSON 备份：失败（记忆组件不可用）")
            return "failed"

        today = time.strftime("%Y%m%d", time.localtime())
        if str(state.get("daily_json_backup_last_day", "")) == today:
            deepmind.logger.info("[睡眠维护] 每日 JSON 备份：跳过（今日已备份）")
            return "skipped"

        center_dir = plugin_context.get_memory_center_dir()
        backup_dir = Path(center_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / f"memory_backup_{today}.json"

        import asyncio

        snapshot = await asyncio.to_thread(memory_store.export_snapshot)
        source = "p-memory"
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
            f"[睡眠维护] 每日 JSON 备份：完成 来源={source} "
            f"文件={backup_file.name} 记录数={len(snapshot.get('records') or [])}"
        )
        return "success"
