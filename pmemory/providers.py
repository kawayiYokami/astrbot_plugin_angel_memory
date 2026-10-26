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
from typing import Any, List, Optional, Sequence

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


class EmbeddingBridge:
    """把异步 `embed_documents(texts) -> list[list[float]]` 包成同步回调。"""

    def __init__(self, provider: Any, timeout: float = 120.0) -> None:
        self._provider = provider
        self._timeout = float(timeout)
        self._loop = _BackgroundLoop("pmemory-embed")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            result = self._loop.run(self._provider.embed_documents(list(texts)), self._timeout)
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

    def close(self) -> None:
        self._loop.close()


class RerankBridge:
    """把异步 `rerank(query, documents)` 包成同步回调，输出与文档等长的分数表。"""

    def __init__(self, provider: Any, timeout: float = 60.0) -> None:
        self._provider = provider
        self._timeout = float(timeout)
        self._loop = _BackgroundLoop("pmemory-rerank")

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        docs = [str(doc or "") for doc in (documents or [])]
        if not docs:
            return []
        try:
            response = self._loop.run(self._provider.rerank(query=query, documents=docs), self._timeout)
        except asyncio.TimeoutError as exc:
            raise EmbedCallbackError.other(f"重排超时 timeout={self._timeout}s") from exc
        except Exception as exc:
            raise EmbedCallbackError.other(str(exc)) from exc
        return _normalize_scores(response, len(docs))

    def close(self) -> None:
        self._loop.close()


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
