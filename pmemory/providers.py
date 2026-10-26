"""p-memory 的提供商桥。

库在**调用线程上持 GIL 同步执行**嵌入与重排回调，所以这里不能直接 await。
做法与 story 一致：为每个回调开一条独立后台事件循环，用
`run_coroutine_threadsafe` 把协程丢过去取结果。

回调抛错时按 `EmbedCallbackError.kind` 告诉库该怎么处理：
- `too_large`：批太大，库会减半重试
- `rate_limited`：被限流，库会退避重试
- `other`：不重试，直接降级
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from p_memory.errors import EmbedCallbackError

logger = logging.getLogger(__name__)

_TOO_LARGE_HINTS = ("too large", "too long", "context length", "max tokens", "payload")
_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "429", "too many requests")


def _classify(message: str) -> EmbedCallbackError:
    text = str(message or "").lower()
    if any(hint in text for hint in _TOO_LARGE_HINTS):
        return EmbedCallbackError.too_large(message)
    if any(hint in text for hint in _RATE_LIMIT_HINTS):
        return EmbedCallbackError.rate_limited(message)
    return EmbedCallbackError.other(message)


class _BackgroundLoop:
    """一条常驻后台事件循环，供同步回调使用。"""

    def __init__(self, name: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name=name)
        self._thread.start()
        self._closed = False

    def run(self, coro, timeout: Optional[float]):
        if self._closed:
            raise RuntimeError("回调事件循环已关闭")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        self._thread.join(timeout=2)


# 循环 id -> 跑这条循环的线程 id：回调若已在该线程上，不能再投递回去等结果
_LOOP_THREADS: Dict[int, int] = {}


def running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """取当前正在运行的循环；没有则为 None。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    _LOOP_THREADS[id(loop)] = threading.get_ident()
    return loop


def loop_thread_ident(loop: Optional[asyncio.AbstractEventLoop]) -> Optional[int]:
    """查这条循环跑在哪个线程上；没记录过就返回 None。"""
    if loop is None:
        return None
    return _LOOP_THREADS.get(id(loop))


class _CallbackRunner:
    """把回调协程送回提供商所属的事件循环执行。

    上游提供商对象（openai AsyncClient 及其内部锁、Event）绑定在创建它的事件循环上，
    换一条循环去调用会抛 `is bound to a different event loop`。所以优先投递回原循环；
    只有拿不到归属循环、原循环已停、或回调本身就阻塞在原循环线程上时，
    才退回自带后台循环（此时可能仍然失败，但不会死锁）。
    """

    def __init__(self, name: str, owner: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._name = name
        self._owner = owner
        self._owner_thread = loop_thread_ident(owner)
        self._fallback: Optional[_BackgroundLoop] = None
        self._fallback_lock = threading.Lock()

    def run(self, coro, timeout: Optional[float]):
        owner = self._owner
        if (
            owner is not None
            and owner.is_running()
            and threading.get_ident() != self._owner_thread
        ):
            return asyncio.run_coroutine_threadsafe(coro, owner).result(timeout=timeout)
        return self._fallback_loop().run(coro, timeout)

    def _fallback_loop(self) -> _BackgroundLoop:
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = _BackgroundLoop(f"{self._name}-fallback")
            return self._fallback

    @property
    def same_loop(self) -> bool:
        """是否确实绑在提供商自己的循环上（用于日志与自检）。"""
        return self._owner is not None

    def close(self) -> None:
        with self._fallback_lock:
            if self._fallback is not None:
                self._fallback.close()
                self._fallback = None


class EmbeddingBridge:
    """把异步 `embed_documents(texts) -> list[list[float]]` 包成同步回调。"""

    def __init__(
        self,
        provider: Any,
        timeout: float = 120.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._provider = provider
        self._timeout = float(timeout)
        self._runner = _CallbackRunner("pmemory-embed", loop)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            result = self._runner.run(
                self._provider.embed_documents(list(texts)), self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise EmbedCallbackError.other(f"嵌入超时 timeout={self._timeout}s") from exc
        except Exception as exc:  # 提供商异常按类别上报，库据此重试或降级
            raise _classify(str(exc)) from exc
        if not isinstance(result, list):
            raise EmbedCallbackError.other("嵌入返回格式非法")
        return result

    def probe_dimension(self) -> int:
        probe = self.embed(["维度探针"])
        if not probe or not probe[0]:
            return 0
        return len(probe[0])

    def model_name(self) -> str:
        try:
            info = self._provider.get_model_info() or {}
        except Exception:
            info = {}
        return str(info.get("model_name") or info.get("provider_id") or "unknown").strip() or "unknown"

    @property
    def same_loop(self) -> bool:
        return self._runner.same_loop

    def close(self) -> None:
        self._runner.close()


class RerankBridge:
    """把异步 `rerank(query, documents)` 包成同步回调，输出与文档等长的分数表。"""

    def __init__(
        self,
        provider: Any,
        timeout: float = 60.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._provider = provider
        self._timeout = float(timeout)
        self._runner = _CallbackRunner("pmemory-rerank", loop)

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        docs = [str(doc or "") for doc in (documents or [])]
        if not docs:
            return []
        try:
            response = self._runner.run(
                self._provider.rerank(query=query, documents=docs), self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise EmbedCallbackError.other(f"重排超时 timeout={self._timeout}s") from exc
        except Exception as exc:
            raise EmbedCallbackError.other(str(exc)) from exc
        return _normalize_scores(response, len(docs))

    @property
    def same_loop(self) -> bool:
        return self._runner.same_loop

    def close(self) -> None:
        self._runner.close()


def _normalize_scores(response: Any, count: int) -> List[float]:
    """把各种重排返回形状归一到「与输入文档等长的分数表」。

    支持：`{results: [...]}` / `{data: [...]}` / 字典列表（index + relevance_score|score）
    / 纯数字列表。没给到的位置记 0 分。
    """
    items = response
    if isinstance(response, dict):
        items = response.get("results")
        if items is None:
            items = response.get("data")
        if isinstance(items, dict):
            items = items.get("results") or items.get("items")
    scores = [0.0] * count
    if not isinstance(items, list):
        return scores
    if items and all(isinstance(item, (int, float)) for item in items):
        for index, value in enumerate(items[:count]):
            scores[index] = float(value)
        return scores
    for position, item in enumerate(items):
        if isinstance(item, dict):
            index = item.get("index", position)
            score = item.get("relevance_score", item.get("score", 0.0))
        else:
            index = getattr(item, "index", position)
            score = getattr(item, "relevance_score", getattr(item, "score", 0.0))
        if isinstance(index, int) and 0 <= index < count:
            try:
                scores[index] = float(score or 0.0)
            except (TypeError, ValueError):
                scores[index] = 0.0
    return scores
