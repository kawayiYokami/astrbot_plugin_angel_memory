"""睡眠调度时刻测试：每天 04:00 触发一次，同日不重复，不做启动补偿。"""

from __future__ import annotations

import datetime
import sys
import time
import types
from pathlib import Path

import pytest

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

CORE_PACKAGE = f"{PACKAGE_NAME}.core"
if CORE_PACKAGE not in sys.modules:
    package = types.ModuleType(CORE_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "core")]
    sys.modules[CORE_PACKAGE] = package

from astrbot_plugin_angel_memory.core.sleep_schedule import (
    seconds_until_next_sleep,
    should_run_sleep,
    today_key,
)


def test_next_sleep_is_same_day_before_four():
    now = datetime.datetime(2026, 9, 10, 3, 0, 0)

    assert seconds_until_next_sleep(now.timestamp()) == pytest.approx(3600, abs=1)


def test_next_sleep_rolls_to_tomorrow_after_four():
    now = datetime.datetime(2026, 9, 10, 5, 30, 0)

    assert seconds_until_next_sleep(now.timestamp()) == pytest.approx(
        22.5 * 3600, abs=1
    )


def test_next_sleep_at_exact_four_rolls_to_tomorrow():
    now = datetime.datetime(2026, 9, 10, 4, 0, 0)

    assert seconds_until_next_sleep(now.timestamp()) == pytest.approx(86400, abs=1)


def test_next_sleep_never_returns_zero():
    now = datetime.datetime(2026, 9, 10, 4, 0, 0)

    assert seconds_until_next_sleep(now.timestamp()) > 0


def test_should_run_sleep_blocks_same_day():
    today = today_key()

    assert should_run_sleep("") is True
    assert should_run_sleep(today) is False
    assert should_run_sleep("2000-01-01") is True


def test_today_key_is_date_only():
    assert today_key() == time.strftime("%Y-%m-%d", time.localtime())
