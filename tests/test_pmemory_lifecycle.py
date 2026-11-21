"""p-memory 生命周期回归测试。

覆盖三件事，都是踩过的坑：
1. 打开只发生在 configure()，访问方不再触发开库
2. close() 之后访问抛 ClosedError，且能被 except PMemoryError 接住
3. 同一数据目录不会被两个实例持有
4. 迁移的 shutdown() 真的等到写库线程退出，而不只是取消协程
5. 迁移线程在等待期间抛错，也不能让 executor 逃过关闭
6. close() 等同步线程退出，关干净后同一目录能重新打开（reload 依赖这点）
"""

import asyncio
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from p_memory.errors import ClosedError, PMemoryError

from astrbot_plugin_angel_memory.pmemory.kb import PMemory


@pytest.mark.asyncio
async def test_configure_opens_library_immediately():
    """configure() 之后库必须已经打开，不能等到首次访问。"""
    pm = PMemory()
    await pm.configure(tempfile.mkdtemp())
    try:
        assert pm._kb is not None
    finally:
        pm.close()


@pytest.mark.asyncio
async def test_kb_before_configure_raises_closed_error():
    """未配置就访问，抛 ClosedError 而不是顺手打开。"""
    pm = PMemory()
    with pytest.raises(ClosedError):
        _ = pm.kb


@pytest.mark.asyncio
async def test_kb_after_close_raises_closed_error():
    """关闭后访问抛 ClosedError，且必须能被 except PMemoryError 接住。

    notes.py 十处都是 except PMemoryError；抛裸 RuntimeError 会穿透到工具层。
    """
    pm = PMemory()
    await pm.configure(tempfile.mkdtemp())
    pm.close()  # 一定会关干净，没有「跳过」这个结果

    with pytest.raises(ClosedError) as excinfo:
        _ = pm.kb
    assert isinstance(excinfo.value, PMemoryError)


@pytest.mark.asyncio
async def test_same_path_shares_handle_across_instances():
    """同一数据目录在进程内共享句柄，第二个实例不再撞锁。

    reload 之后新实例会再构造一次；如果每次都真去开目录，就会撞上 native 的
    进程级锁——那正是「重载后记忆功能整轮失效」的根因。
    """
    root = tempfile.mkdtemp()
    first = PMemory()
    await first.configure(root)

    second = PMemory()
    await second.configure(root)
    assert second._kb is not None, "同路径第二个实例没打开"
    # 共享的是底层 native 句柄，Python 侧的 KnowledgeBase 对象仍是两个
    assert second._kb._native is first._kb._native, "没有共享同一个 native 句柄"

    # 关掉一个，另一个仍然可用（引用计数还没归零）
    first.close()
    assert second.kb is not None
    second.close()

    # 归零之后锁释放，新实例能重新打开
    third = PMemory()
    await third.configure(root)
    assert third._kb is not None
    third.close()


@pytest.mark.asyncio
async def test_close_waits_for_sync_thread():
    """close() 等同步线程退出，不设超时、不跳过。

    sync 按批调嵌入回调，关闭标志一置上，下一批就抛异常中断（实测 0.07s 退出），
    所以线程一定会结束。这里用一个 0.3 秒才退出的线程验证：close 必须等到它，
    而不是等一会儿就放弃。
    """
    pm = PMemory()
    root = tempfile.mkdtemp()
    await pm.configure(root)

    finished = threading.Event()

    def worker():
        time.sleep(0.3)
        finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    pm._sync_thread = thread
    thread.start()

    started = time.time()
    pm.close()
    elapsed = time.time() - started

    assert finished.is_set(), "close 没等线程跑完就返回了"
    assert elapsed >= 0.25, f"close 只用了 {elapsed:.2f}s，没等到线程结束"
    assert pm._kb is None, "库没关掉"
    assert pm._sync_thread is None, "线程引用没清"

    # 关干净之后锁已释放：同一目录能重新打开，reload 依赖这一点
    again = PMemory()
    await again.configure(root)
    again.close()


def test_migration_shutdown_waits_for_worker_thread():
    """shutdown() 必须等到写库线程退出，而不只是取消协程。

    asyncio.to_thread 的 Task.cancel() 只取消外层 Future，底层线程继续跑
    （实测 gather 0.2s 返回、线程 1.5s 才结束）。所以要自管 executor。
    """
    from astrbot_plugin_angel_memory.core.services.legacy_memory_migration import (
        LegacyMemoryMigration,
    )

    obj = object.__new__(LegacyMemoryMigration)
    import logging

    obj.logger = logging.getLogger("test-migration")
    obj._task = None
    obj._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legacy-migration")
    obj._migrate_future = None

    started = time.time()

    def slow_write():
        time.sleep(1.0)

    obj._migrate_future = obj._executor.submit(slow_write)

    asyncio.run(obj.shutdown())
    elapsed = time.time() - started

    assert elapsed >= 0.9, f"shutdown 只用了 {elapsed:.2f}s，没等到线程结束"
    assert obj._migrate_future is None


def test_migration_shutdown_closes_executor_when_worker_raises():
    """写库线程在等待期间抛错，也不能让 executor 逃过关闭。

    done() 检查与 result(30) 之间有 TOCTOU：进 try 时线程还活着，30 秒内它
    抛错结束，result() 会把异常原样重抛。这不是超时，若只 except TimeoutError
    就会逃出 shutdown()，后面的 executor.shutdown(wait=True) 整条被跳过，
    worker 线程留存到解释器退出。实测过：ESCAPED ValueError + 线程还在。
    """
    from astrbot_plugin_angel_memory.core.services.legacy_memory_migration import (
        LegacyMemoryMigration,
    )

    obj = object.__new__(LegacyMemoryMigration)
    import logging

    obj.logger = logging.getLogger("test-migration")
    obj._task = None
    obj._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legacy-raiser")
    obj._migrate_future = None

    def exploding_write():
        time.sleep(0.3)
        raise ValueError("db write failed")

    obj._migrate_future = obj._executor.submit(exploding_write)

    # 不许抛：异常已在 shutdown 内部记日志消化
    asyncio.run(obj.shutdown())

    # executor 确实关了 —— 关过的 executor 拒绝再提交
    with pytest.raises(RuntimeError):
        obj._executor.submit(lambda: None)


@pytest.mark.asyncio
async def test_close_interrupts_real_vector_sync():
    """关闭标志能让真实的向量同步在下一批嵌入回调处停下。

    只验证「join 存在」不够：中断只发生在 `embed()` 入口，如果同步线程还没走到
    第一次嵌入回调，join 就是无界的。这条用真实库 + 慢嵌入回调跑一遍。
    """

    class SlowProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_documents(self, texts):
            self.calls += 1
            await asyncio.sleep(0.15)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        def get_model_info(self):
            return {"model_name": "slow"}

        def is_available(self) -> bool:
            return True

    from astrbot_plugin_angel_memory.pmemory.config import memory_namespace
    from astrbot_plugin_angel_memory.pmemory.memory import MemoryStore

    pm = PMemory()
    provider = SlowProvider()
    await pm.configure(tempfile.mkdtemp(), None, provider, None, None)

    store = MemoryStore(pm)
    for index in range(10):
        store.remember(f"内容 {index}", "", ["t"], "知识", "public")

    kb = pm._kb
    space_id = pm._space_id
    sync_thread = threading.Thread(
        target=lambda: kb.embeddings.sync(
            space_id, batch=1, namespace=memory_namespace()
        ),
        daemon=True,
        name="pmemory-sync-test",
    )
    pm._sync_thread = sync_thread
    sync_thread.start()
    await asyncio.sleep(0.3)  # 让它进入第一批嵌入

    started = time.time()
    # 和 configure 一样放工作线程：close 要 join 同步线程，而同步线程的嵌入回调
    # 要回到这条循环，直接在循环线程上等就是两边互等。
    await asyncio.to_thread(pm.close)
    elapsed = time.time() - started

    assert not sync_thread.is_alive(), "close 返回时同步线程还活着"
    assert elapsed < 5, f"close 用了 {elapsed:.1f}s，中断没生效"
    # 中断生效的话不会把 10 条全部嵌入完；跑满了说明关闭标志没被看到
    assert provider.calls < 10, f"同步跑完了全部 {provider.calls} 批，中断没生效"
    assert pm._kb is None


def test_requires_pmemory_version():
    """版本不够时启动就带着修法报错，而不是等重载撞锁才发现。

    0.4.0 之前同一路径第二次打开会撞 native 目录锁，症状是重载后记忆功能整轮
    静默失效。这条检查让它在启动时就停下来，并说清该装哪个版本。
    """
    import p_memory

    from astrbot_plugin_angel_memory.pmemory.kb import require_pmemory_version

    original = p_memory.__version__
    try:
        p_memory.__version__ = "0.3.8"
        with pytest.raises(RuntimeError, match="版本过低"):
            require_pmemory_version()
        with pytest.raises(RuntimeError, match="pip install -U"):
            require_pmemory_version()

        p_memory.__version__ = "0.4.0"
        require_pmemory_version()

        # 按数值比较而不是字符串：0.10.0 必须大于 0.4.0
        p_memory.__version__ = "0.10.0"
        require_pmemory_version()
    finally:
        p_memory.__version__ = original
