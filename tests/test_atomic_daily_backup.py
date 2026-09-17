from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT.parent))

from astrbot_plugin_angel_memory.core.services.sleep_maintenance_service import (
    SleepMaintenanceService,
)


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))


class _MemoryManager:
    def __init__(self):
        self.export_calls = 0

    async def export_backup_snapshot(self):
        self.export_calls += 1
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
    backup_dir = tmp_path / "backups"
    if os.name == "posix":
        backup_dir.mkdir(mode=0o777)
        backup_dir.chmod(0o777)

    service = SleepMaintenanceService(_DeepMind(tmp_path))
    state = {}

    status = asyncio.run(service._task_daily_json_backup(state))

    today = time.strftime("%Y%m%d", time.localtime())
    backup_file = backup_dir / f"memory_backup_{today}.json"
    assert status == "success"
    assert state["daily_json_backup_last_day"] == today
    assert json.loads(backup_file.read_text(encoding="utf-8"))["records"] == [
        {"id": "memory-1"}
    ]
    assert list(backup_dir.glob(".*.tmp")) == []
    if os.name == "posix":
        assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(backup_file.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_backup_stops_when_private_directory_cannot_be_secured(monkeypatch, tmp_path):
    deepmind = _DeepMind(tmp_path)
    service = SleepMaintenanceService(deepmind)
    original_chmod = Path.chmod

    def guarded_chmod(path, mode, *args, **kwargs):
        if path.name == "backups":
            raise PermissionError("simulated chmod failure")
        return original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", guarded_chmod)

    with pytest.raises(PermissionError, match="simulated chmod failure"):
        asyncio.run(service._task_daily_json_backup({}))

    assert deepmind.plugin_context.manager.export_calls == 0
    assert list((tmp_path / "backups").glob("memory_backup_*.json")) == []


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-specific")
def test_unsupported_directory_fsync_still_marks_backup_complete(monkeypatch, tmp_path):
    deepmind = _DeepMind(tmp_path)
    service = SleepMaintenanceService(deepmind)
    state = {}
    real_fsync = os.fsync

    def fsync_with_unsupported_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "directory fsync unsupported")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_with_unsupported_directory)

    assert asyncio.run(service._task_daily_json_backup(state)) == "success"
    assert asyncio.run(service._task_daily_json_backup(state)) == "skipped"
    assert deepmind.plugin_context.manager.export_calls == 1
    assert deepmind.logger.warnings


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-specific")
def test_real_directory_fsync_error_is_not_swallowed(monkeypatch, tmp_path):
    service = SleepMaintenanceService(_DeepMind(tmp_path))
    state = {}
    real_fsync = os.fsync

    def fsync_with_io_error(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "simulated I/O failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_with_io_error)

    with pytest.raises(OSError) as exc_info:
        asyncio.run(service._task_daily_json_backup(state))

    assert exc_info.value.errno == errno.EIO
    assert "daily_json_backup_last_day" not in state
