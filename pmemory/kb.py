"""p-memory 知识库单例。

一个数据目录承载全部记忆、笔记与向量。嵌入与重排由本项目注册回调提供，
回调是运行时状态、不落盘，所以每次打开数据目录都要重新注册。

打开时机：ComponentFactory 创建完提供商后调用 `configure()`，首次访问 `kb`
时才真正开库，避免启动期就把锁和索引初始化压在关键路径上。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from p_memory import KnowledgeBase

from .config import (
    DOMAIN_DIRS,
    SCOPE_PUBLIC,
    iter_domains,
    knowledge_namespace,
    memory_namespace,
)
from .providers import EmbeddingBridge, RerankBridge, running_loop

logger = logging.getLogger(__name__)

SPACE_ENCODING = "sq8"
SCOPES_FILE = "scopes.json"


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

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._root: Optional[Path] = None
        self._raw_dir: Optional[Path] = None
        self._kb: Optional[KnowledgeBase] = None
        self._embedding_provider: Any = None
        self._rerank_provider: Any = None
        self._rerank_options: Dict[str, int] = {}
        self._embedding: Optional[EmbeddingBridge] = None
        self._reranker: Optional[RerankBridge] = None
        self._space_id: Optional[str] = None
        self._sync_started = False
        # 提供商对象所属的事件循环：回调必须回到这条循环，否则跨 loop 使用会报错
        self._owner_loop: Optional[Any] = None
        # 已知 scope 名单：库不支持「跨全部 scope」的读，生命周期操作只能显式列全
        self._scopes: set[str] = {SCOPE_PUBLIC}
        self._scopes_loaded = False

    # === 配置 ===

    def configure(
        self,
        root: str | Path,
        raw_dir: str | Path | None = None,
        embedding_provider: Any = None,
        rerank_provider: Any = None,
        rerank_options: Optional[Dict[str, int]] = None,
    ) -> None:
        """设置数据目录与提供商。重复调用会关掉旧库，下次访问按新配置重开。"""
        with self._lock:
            self.close()
            self._root = Path(root)
            self._raw_dir = Path(raw_dir) if raw_dir else None
            self._embedding_provider = embedding_provider
            self._rerank_provider = rerank_provider
            self._rerank_options = dict(rerank_options or {})
            self._owner_loop = running_loop() or self._owner_loop
            self._scopes = {SCOPE_PUBLIC}
            self._scopes_loaded = False

    @property
    def root(self) -> Optional[Path]:
        return self._root

    @property
    def kb(self) -> KnowledgeBase:
        with self._lock:
            if self._kb is None:
                self._open()
            assert self._kb is not None
            return self._kb

    @property
    def space_id(self) -> Optional[str]:
        self.kb  # 触发打开与回调注册
        return self._space_id

    # === scope 名单 ===

    @property
    def scopes(self) -> List[str]:
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
        with self._lock:
            if self._kb is not None:
                try:
                    self._kb.close()
                except Exception:
                    logger.exception("[PMEMORY] 关闭知识库失败")
                self._kb = None
            for bridge in (self._embedding, self._reranker):
                if bridge is not None:
                    try:
                        bridge.close()
                    except Exception:
                        pass
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
                provider_type = ""
                try:
                    provider_type = str(self._embedding_provider.get_provider_type() or "")
                except Exception:
                    provider_type = ""
                if provider_type == "local":
                    logger.info("[PMEMORY] 本地嵌入提供商懒加载，首次向量化时载入")
                else:
                    logger.warning("[PMEMORY] 嵌入提供商当前不可用，跳过向量回调注册")
                    return

            bridge = EmbeddingBridge(self._embedding_provider, loop=owner_loop)
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
            try:
                report = self.kb.embeddings.sync(space_id, namespace=memory_namespace())
                logger.info("[PMEMORY] 记忆向量同步完成 空间=%s 结果=%s", space_id, report)
            except Exception:
                logger.exception("[PMEMORY] 记忆向量同步失败，检索在补齐前走全文")

        threading.Thread(target=_run, daemon=True, name="pmemory-sync").start()

    # === 运维信息 ===

    def health(self) -> Dict[str, Any]:
        kb = self.kb
        try:
            return {
                "root": str(self._root or ""),
                "space": self._space_id,
                "counts": kb.counts(),
                "integrity": kb.integrity_check(),
            }
        except Exception as exc:
            return {"root": str(self._root or ""), "space": self._space_id, "error": str(exc)}


pmemory = PMemory()
