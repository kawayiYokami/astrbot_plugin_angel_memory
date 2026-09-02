from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT.parent))

from astrbot_plugin_angel_memory.core.services.sleep_maintenance_service import (
    SleepMaintenanceService,
)


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _MemoryManager:
    async def export_backup_snapshot(self):
        return {"records": [{"id": "memory-1"}], "global_tags": []}


class _PluginContext:
    def __init__(self, root):
        self.root = root
        self.manager = _MemoryManager()

    def get_memory_center_dir(self):
        return self.root

    def get_component(self, name):
        return self.manager if name == "memory_sql_manager" else None


class _DeepMind:
    def __init__(self, root):
        self.plugin_context = _PluginContext(root)
        self.logger = _Logger()


def test_daily_json_backup_is_atomic_and_private(tmp_path):
    service = SleepMaintenanceService(_DeepMind(tmp_path))
    state = {}

    status = asyncio.run(service._task_daily_json_backup(state))

    today = time.strftime("%Y%m%d", time.localtime())
    backup_file = tmp_path / "backups" / f"memory_backup_{today}.json"
    assert status == "success"
    assert state["daily_json_backup_last_day"] == today
    assert json.loads(backup_file.read_text(encoding="utf-8"))["records"] == [
        {"id": "memory-1"}
    ]
    assert list((tmp_path / "backups").glob(".*.tmp")) == []
    if os.name == "posix":
        assert stat.S_IMODE(backup_file.stat().st_mode) == 0o600
