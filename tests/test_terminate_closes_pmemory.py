"""terminate() 的 p-memory 兜底：任何路径都要把知识库关掉，释放目录锁。

pmemory 用替身，不真的开库。替身按真实语义模拟：已打开时再 _open 会抛
"data directory is already open"，close 之后才允许重新打开。
"""

import logging

import pytest

import astrbot_plugin_angel_memory.main as plugin_main


class FakePmemory:
    """pmemory.close() 的替身：幂等，并把「锁」状态复位。"""

    def __init__(self, opened: bool = True):
        self.close_calls = 0
        self._kb = object() if opened else None

    def close(self) -> None:
        """真实 close() 不返回结果：它一定会关干净。

        替身跟着返回 None，保持契约一致——上一版替身按已经不存在的 `if not
        closed` 分支写了注释，属于过期描述。
        """
        self.close_calls += 1
        self._kb = None

    def _open(self):
        if self._kb is not None:
            raise RuntimeError("data directory is already open")
        self._kb = object()
        return self._kb


class OkManager:
    def __init__(self):
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def get_status(self):
        return {"state": "stopped"}


class BoomManager:
    """shutdown 抛出非 RuntimeError 的异常，验证 except 已放宽到 Exception。"""

    async def shutdown(self) -> None:
        raise ValueError("关闭链炸了")

    def get_status(self):
        return {"state": "unknown"}


def _plugin(plugin_manager=None) -> plugin_main.AngelMemoryPlugin:
    plugin = object.__new__(plugin_main.AngelMemoryPlugin)
    plugin.logger = logging.getLogger("test-terminate")
    plugin._is_terminating = False
    plugin._background_tasks = set()
    plugin.plugin_manager = plugin_manager
    return plugin


@pytest.mark.asyncio
async def test_terminate_closes_pmemory_without_plugin_manager(monkeypatch):
    fake = FakePmemory()
    monkeypatch.setattr(plugin_main, "pmemory", fake)

    await _plugin(plugin_manager=None).terminate()

    assert fake.close_calls == 1


@pytest.mark.asyncio
async def test_terminate_closes_pmemory_when_shutdown_raises(monkeypatch):
    fake = FakePmemory()
    monkeypatch.setattr(plugin_main, "pmemory", fake)

    await _plugin(plugin_manager=BoomManager()).terminate()

    assert fake.close_calls == 1


@pytest.mark.asyncio
async def test_terminate_closes_pmemory_on_normal_path(monkeypatch):
    fake = FakePmemory()
    monkeypatch.setattr(plugin_main, "pmemory", fake)
    manager = OkManager()

    # 模拟 component_factory.shutdown() 里已经关过一次，terminate 再关一次必须无害
    fake.close()
    await _plugin(plugin_manager=manager).terminate()

    assert manager.shutdown_calls == 1
    assert fake.close_calls == 2


@pytest.mark.asyncio
async def test_reopen_succeeds_after_terminate(monkeypatch):
    fake = FakePmemory()
    monkeypatch.setattr(plugin_main, "pmemory", fake)
    plugin = _plugin(plugin_manager=None)

    # 未关闭前再打开必须失败，这才是「锁被占着」的真实形态
    with pytest.raises(RuntimeError):
        fake._open()

    await plugin.terminate()

    # 关闭之后可以重新打开：锁确实释放了
    assert fake._open() is not None


@pytest.mark.asyncio
async def test_close_failure_does_not_break_terminate(monkeypatch):
    class BrokenPmemory(FakePmemory):
        def close(self) -> None:
            raise OSError("native 句柄异常")

    monkeypatch.setattr(plugin_main, "pmemory", BrokenPmemory())

    # 兜底自身失败也不能把 terminate 再炸一次
    await _plugin(plugin_manager=None).terminate()
