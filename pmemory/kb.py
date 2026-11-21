"""p-memory 知识库单例。

一个数据目录承载全部记忆、笔记与向量。嵌入与重排由本项目注册回调提供，
回调是运行时状态、不落盘，所以每次打开数据目录都要重新注册。

打开时机：ComponentFactory 创建完提供商后调用 `configure()`，在那里完成开库。
此后任何线程访问 `kb` 时库一定已经打开；没配置或已关闭一律抛 `ClosedError`，
不存在「谁先访问谁打开」。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
from pathlib import Path
from typing import Any

from astrbot.api import logger
from p_memory import KnowledgeBase
from p_memory.errors import ClosedError

from .config import (
    DOMAIN_DIRS,
    SCOPE_PUBLIC,
    iter_domains,
    knowledge_namespace,
    memory_namespace,
)
from .providers import EmbeddingBridge, RerankBridge, running_loop

SPACE_ENCODING = "sq8"
SCOPES_FILE = "scopes.json"

# 插件依赖 p-memory 的最低版本。
# 0.4.0 起同一路径在进程内共享句柄并按引用计数关闭；更早的版本第二次打开会
# 直接撞 native 目录锁，插件重载后记忆功能整轮失效，且只在日志里留一行。
REQUIRED_PMEMORY_VERSION = "0.4.0"


def _version_tuple(text: str) -> tuple:
    """把版本串变成可比较的元组，最后一位标记是否正式版。

    优先用 packaging（完整 PEP 440 语义：`0.4.0.post1 > 0.4.0 > 0.4.0rc1`，
    `v0.4.0` 与 `0.4.0` 等价）。拿不到就退回只比数字核——至少不会把 post
    当成预发布而误杀。
    """
    raw = str(text or "").strip()
    try:
        from packaging.version import Version

        parsed = Version(raw)
        return (
            parsed.major,
            parsed.minor,
            parsed.micro,
            0 if parsed.is_prerelease else 1,
        )
    except Exception:
        numbers = re.findall(r"\d+", raw)[:3]
        parts = [int(piece) for piece in numbers] + [0] * (3 - len(numbers))
        return (parts[0], parts[1], parts[2], 1)


def require_pmemory_version() -> None:
    """启动时校验 p-memory 版本，不满足就带着修法报错。

    与其等到重载时抛一句无法归因的 LockedError，不如启动就说清版本不对、
    该装哪个版本。这条检查在组件创建最前面调用，不通过就整条初始化停住。
    """
    import p_memory

    current = str(getattr(p_memory, "__version__", "0.0.0"))
    if _version_tuple(current) >= _version_tuple(REQUIRED_PMEMORY_VERSION):
        return
    raise RuntimeError(
        f"p-memory 版本过低：当前 {current}，需要 >= {REQUIRED_PMEMORY_VERSION}。"
        "旧版不支持同一数据目录在进程内共享句柄，插件重载后会撞上目录锁，"
        "记忆功能整轮失效。"
        f'请执行 pip install -U "p-memory>={REQUIRED_PMEMORY_VERSION}"，然后重启 AstrBot。'
    )


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return text or "unknown"


def _log_event(event: dict) -> None:
    """库在检索线程上同步调用，这里只写一行日志，不做任何 IO 或上报。

    检索太频繁，正常路径不落日志；只有降级才报出来。
    """
    kind = str(event.get("kind") or "")
    if kind == "search":
        degraded = event.get("degraded")
        if degraded:
            logger.warning(
                "[PMEMORY] 检索降级 ms=%s 原因=%s 候选=%s 命中=%s",
                event.get("ms"),
                degraded,
                event.get("candidates"),
                event.get("hits"),
            )
        return
    if kind == "index_rebuild":
        logger.info(
            "[PMEMORY] 索引重建事件 ms=%s 文档=%s 格式=%s",
            event.get("ms"),
            event.get("documents"),
            event.get("format"),
        )
    else:
        logger.debug("[PMEMORY] 未知事件 kind=%s", kind)


class PMemory:
    """知识库单例：数据目录、回调注册、笔记域根目录都在这里收口。"""

    # 同一数据目录的句柄由 p_memory 自己共享并按引用计数关闭：同进程第二次打开
    # 拿到的是同一个句柄，归零才真正关。这里不再自己维护路径占用表。

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._root: Path | None = None
        self._raw_dir: Path | None = None
        self._kb: KnowledgeBase | None = None
        self._embedding_provider: Any = None
        self._rerank_provider: Any = None
        self._rerank_options: dict[str, int] = {}
        self._embedding: EmbeddingBridge | None = None
        self._reranker: RerankBridge | None = None
        self._space_id: str | None = None
        self._sync_started = False
        self._sync_thread: threading.Thread | None = None
        # 关闭后拒绝再打开：close() 之后任何访问都必须失败，否则后台线程会把库
        # 重新打开、把目录锁又拿回去，下次重载必撞锁
        self._closed = False
        # 提供商对象所属的事件循环：回调必须回到这条循环，否则跨 loop 使用会报错
        self._owner_loop: Any | None = None
        # 已知 scope 名单：库不支持「跨全部 scope」的读，生命周期操作只能显式列全
        self._scopes: set[str] = {SCOPE_PUBLIC}
        self._scopes_loaded = False

    # === 配置 ===

    async def configure(
        self,
        root: str | Path,
        raw_dir: str | Path | None = None,
        embedding_provider: Any = None,
        rerank_provider: Any = None,
        rerank_options: dict[str, int] | None = None,
    ) -> None:
        """设置数据目录与提供商，并在这里完成打开。

        打开动作只允许发生在这一处：配置完成后立刻开库，此后任何线程访问 `kb`
        时库一定已经打开。重复调用会先关旧库再按新配置重开。
        """
        # close 必须在锁外调用：它内部要 join 向量同步线程，持锁 join 会死锁。
        # 但也不能直接同步调：configure 跑在事件循环线程上，而同步线程里的嵌入
        # 回调要投递回这条循环——在这儿 join 就是两边互等，循环冻到嵌入超时为止。
        # 放到工作线程去等，循环保持空闲，回调才跑得回来。
        await asyncio.to_thread(self.close)

        with self._lock:
            # 重新配置视为重新启用：close() 置上的关闭标志在这里清掉
            self._closed = False
            self._root = Path(root)
            self._raw_dir = Path(raw_dir) if raw_dir else None
            self._embedding_provider = embedding_provider
            self._rerank_provider = rerank_provider
            self._rerank_options = dict(rerank_options or {})
            self._owner_loop = running_loop() or self._owner_loop
            self._scopes = {SCOPE_PUBLIC}
            self._scopes_loaded = False

        # 开库放工作线程，不能留在主循环线程上：
        # 库的嵌入回调是同步的、在调用线程上直接跑，而 _CallbackRunner 判定
        # 「当前线程就是归属循环线程」时会退回自带后台循环，跨 loop 使用提供商
        # 对象会抛 RuntimeError。放到工作线程，回调才能正常投递回主循环。
        try:
            await asyncio.to_thread(self.ensure_open)
        except Exception:
            logger.exception("[PMEMORY] 知识库打开失败，本次运行记忆功能不可用")

    def ensure_open(self) -> None:
        """打开知识库。

        只应由 configure() 调用。其他线程一律通过 `kb` 只读访问——打开时机
        集中在这里，才不会出现「谁先访问谁打开」。
        """
        with self._lock:
            if self._closed:
                raise ClosedError("p-memory 知识库已关闭，不再接受访问")
            if self._kb is None:
                self._open()

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def kb(self) -> KnowledgeBase:
        """只读访问。库没打开或已关闭都抛 ClosedError。

        抛 ClosedError 而不是 RuntimeError：它是 PMemoryError 子类，调用方
        现有的 `except PMemoryError` 都能接住并降级，不会让异常穿透到工具层。
        """
        with self._lock:
            if self._closed:
                raise ClosedError("p-memory 知识库已关闭，不再接受访问")
            if self._kb is None:
                raise ClosedError(
                    "p-memory 知识库尚未打开：打开只发生在 configure()，"
                    "调用方不应依赖首次访问触发"
                )
            return self._kb

    @property
    def space_id(self) -> str | None:
        return self._space_id

    # === scope 名单 ===

    @property
    def scopes(self) -> list[str]:
        with self._lock:
            if self._root is not None and not self._scopes_loaded:
                self._load_scopes()
            return sorted(self._scopes)

    def register_scopes(self, names) -> None:
        """登记已知 scope。库的读过滤必须显式列出 scope，跨会话操作靠这份名单。"""
        with self._lock:
            if self._root is not None:
                self._load_scopes()
            new_names = {
                str(name).strip()
                for name in (names or [])
                if str(name or "").strip()
            }
            if not new_names - self._scopes:
                return
            self._scopes |= new_names
            self._save_scopes()

    def _load_scopes(self) -> None:
        self._scopes_loaded = True
        if self._root is None:
            return
        path = self._root / SCOPES_FILE
        if not path.is_file():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                self._scopes |= {str(item).strip() for item in loaded if str(item).strip()}
        except Exception:
            logger.warning("[PMEMORY] 读取 scope 名单失败，按已登记项继续")

    def _save_scopes(self) -> None:
        if self._root is None:
            return
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with (self._root / SCOPES_FILE).open("w", encoding="utf-8") as f:
                json.dump(sorted(self._scopes), f, ensure_ascii=False)
        except Exception:
            logger.warning("[PMEMORY] 写入 scope 名单失败")

    def close(self) -> None:
        """关闭知识库。"""
        # 1) 先置关闭标志并取出同步线程引用。标志必须在锁内先落，
        #    这样正在跑的向量同步下一次访问 kb 就会失败退出，不会把库重新打开。
        with self._lock:
            self._closed = True
            # 只读，不清：清掉的话下一次 close()（卸载链上会调两次）看不到线程，
            # 会直接进关库流程，把这层保护作废。
            sync_thread = self._sync_thread

        # 2) 等同步线程收尾。必须在锁外 join：它内部要拿同一把锁，锁内等会死锁。
        #    正常情况下一批嵌入回调就会中断（实测 0.07 秒），这里 30 秒是兜底：
        #    中断机制自己坏掉时，宁可留一个没关的句柄，也不能让 terminate 永远挂住。
        #    不关库是安全的——句柄还留在共享表里，重载后新实例照样能拿到它。
        if sync_thread is not None and sync_thread.is_alive():
            sync_thread.join(timeout=30)
            if sync_thread.is_alive():
                logger.error(
                    "[PMEMORY] 向量同步线程 30 秒未退出，本次不关库；"
                    "句柄留在共享表里，重载后仍可打开，进程退出时释放"
                )
                return

        # 3) 真正关库与回调线程
        with self._lock:
            # 线程确认结束，此时才清引用
            self._sync_thread = None
            if self._kb is not None:
                try:
                    self._kb.close()
                except Exception:
                    logger.exception("[PMEMORY] 关闭知识库失败")
                self._kb = None
            for bridge in (self._embedding, self._reranker):
                if bridge is not None:
                    with contextlib.suppress(Exception):
                        bridge.close()
            self._embedding = None
            self._reranker = None
            self._space_id = None
            self._sync_started = False

    # === 打开 ===

    def _open(self) -> None:
        if self._root is None:
            raise RuntimeError("p-memory 数据目录未配置")
        self._root.mkdir(parents=True, exist_ok=True)
        kb = KnowledgeBase(str(self._root))
        self._kb = kb
        try:
            kb.register_event_sink(_log_event)
        except Exception:
            logger.exception("[PMEMORY] 注册事件回调失败")

        try:
            self._register_note_roots(kb)
        except Exception:
            logger.exception("[PMEMORY] 登记笔记根目录失败")

        try:
            self._register_callbacks(kb)
        except Exception:
            logger.exception("[PMEMORY] 提供商回调注册失败，检索降级为纯全文")

        try:
            self._configure_vectorization(kb)
        except Exception:
            logger.exception("[PMEMORY] 向量开关配置失败")

        try:
            counts = kb.counts()
        except Exception:
            counts = {}
        logger.info(
            "[PMEMORY] 知识库已打开 目录=%s 空间=%s 计数=%s",
            self._root,
            self._space_id or "无（纯全文）",
            counts,
        )

        self._start_vector_sync()

    def _register_note_roots(self, kb: KnowledgeBase) -> None:
        if self._raw_dir is None:
            logger.debug("[PMEMORY] 未提供笔记根目录，跳过 set_root")
            return
        for domain in iter_domains():
            relative = DOMAIN_DIRS[domain]
            directory = self._raw_dir / relative
            if not directory.is_dir():
                logger.debug("[PMEMORY] 笔记目录不存在，跳过 域=%s 目录=%s", domain, directory)
                continue
            try:
                kb.notes.set_root(knowledge_namespace(domain), str(directory))
                logger.debug("[PMEMORY] 笔记根目录已登记 域=%s 目录=%s", domain, directory)
            except Exception:
                logger.exception("[PMEMORY] 笔记根目录登记失败 域=%s 目录=%s", domain, directory)

    def _register_callbacks(self, kb: KnowledgeBase) -> None:
        # 上游提供商对象内部锁绑定在创建它的事件循环上，回调必须投递回同一条循环
        owner_loop = running_loop() or self._owner_loop
        if self._rerank_provider is not None:
            try:
                bridge = RerankBridge(self._rerank_provider, loop=owner_loop)
                options = {
                    "max_tokens_total": int(self._rerank_options.get("max_tokens_total", 8192)),
                    "max_candidates": int(self._rerank_options.get("max_candidates", 64)),
                    "max_tokens_per_doc": int(self._rerank_options.get("max_tokens_per_doc", 1024)),
                }
                kb.register_reranker(bridge.rerank, **options)
                self._reranker = bridge
                logger.info("[PMEMORY] 已注册重排回调 %s", options)
            except Exception:
                logger.exception("[PMEMORY] 注册重排回调失败，检索不重排")

        if self._embedding_provider is None:
            logger.info("[PMEMORY] 未提供嵌入提供商，记忆检索走纯全文")
            return
        try:
            if not self._embedding_provider.is_available():
                logger.warning("[PMEMORY] 嵌入提供商当前不可用，跳过向量回调注册")
                return

            bridge = EmbeddingBridge(
                self._embedding_provider,
                loop=owner_loop,
                should_stop=lambda: self._closed,
            )
            dimension = bridge.probe_dimension()
            if dimension <= 0:
                logger.warning("[PMEMORY] 嵌入维度探测失败，跳过向量空间注册")
                bridge.close()
                return
            space_id = f"angel-{_slug(bridge.model_name())}-{dimension}"
            kb.embeddings.register_space(
                {
                    "id": space_id,
                    "model": bridge.model_name(),
                    "dimension": dimension,
                    "encoding": SPACE_ENCODING,
                }
            )
            kb.embeddings.register_embedder(space_id, bridge.embed)
            self._embedding = bridge
            self._space_id = space_id
            logger.info("[PMEMORY] 已注册嵌入回调 空间=%s 维度=%s", space_id, dimension)
        except Exception:
            logger.exception("[PMEMORY] 注册嵌入回调失败，记忆检索走纯全文")

    def _configure_vectorization(self, kb: KnowledgeBase) -> None:
        # 记忆开向量；笔记全部关闭。
        kb.embeddings.set_vectorization(memory_namespace(), "memory", True)
        for domain in iter_domains():
            namespace = knowledge_namespace(domain)
            kb.embeddings.set_vectorization(namespace, "notes", False)
        logger.info("[PMEMORY] 向量开关：记忆开、笔记关")

    def _start_vector_sync(self) -> None:
        """后台补齐记忆向量；笔记不向量化，不参与。"""
        if self._space_id is None or self._sync_started:
            return
        self._sync_started = True
        space_id = self._space_id

        def _run() -> None:
            if self._closed:
                return
            try:
                report = self.kb.embeddings.sync(space_id, namespace=memory_namespace())
                value = report.get("value") if isinstance(report, dict) else None
                interrupted = (value or {}).get("interrupted")
                if interrupted:
                    # 关闭时由嵌入回调抛异常中断，库把它记在 interrupted 里，
                    # 剩下的记录留到下次启动继续补，不算失败。
                    # 用 info：_closed 若被误置，这里就是唯一的痕迹，debug 看不见。
                    logger.info(
                        "[PMEMORY] 记忆向量同步被中断（%s），剩余记录下次启动补齐",
                        interrupted,
                    )
                else:
                    logger.info(
                        "[PMEMORY] 记忆向量同步完成 空间=%s 结果=%s", space_id, report
                    )
            except ClosedError:
                # close() 期间访问 kb 会抛这个，属正常收尾，不记异常
                logger.debug("[PMEMORY] 记忆向量同步在关闭过程中中断")
            except Exception:
                logger.exception("[PMEMORY] 记忆向量同步失败，检索在补齐前走全文")

        thread = threading.Thread(target=_run, daemon=True, name="pmemory-sync")
        self._sync_thread = thread
        thread.start()

    # === 运维信息 ===

    def health(self) -> dict[str, Any]:
        try:
            kb = self.kb
            return {
                "root": str(self._root or ""),
                "space": self._space_id,
                "counts": kb.counts(),
                "integrity": kb.integrity_check(),
            }
        except Exception as exc:
            return {"root": str(self._root or ""), "space": self._space_id, "error": str(exc)}


pmemory = PMemory()
