"""睡眠调度时刻计算。

睡眠固定每天凌晨 4 点执行一次，按运行环境本地时间计算，写死不可配置。
进程在 04:00 未运行时当天不补偿，启动时只重算下一次触发时刻。
"""

from __future__ import annotations

import datetime
import time

SLEEP_HOUR = 4
SLEEP_MINUTE = 0


def seconds_until_next_sleep(now: float | None = None) -> float:
    """距下一个 04:00 的秒数。"""
    current = datetime.datetime.fromtimestamp(
        time.time() if now is None else float(now)
    )
    target = current.replace(
        hour=SLEEP_HOUR, minute=SLEEP_MINUTE, second=0, microsecond=0
    )
    if target <= current:
        target += datetime.timedelta(days=1)
    return max(1.0, (target - current).total_seconds())


def today_key(now: float | None = None) -> str:
    """当前日期键，用于「同一天只跑一次」判定。"""
    return time.strftime(
        "%Y-%m-%d", time.localtime(time.time() if now is None else float(now))
    )


def should_run_sleep(last_sleep_day: str, now: float | None = None) -> bool:
    """今天是否还没跑过睡眠。"""
    return str(last_sleep_day or "") != today_key(now)
