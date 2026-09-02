"""Rerank provider adapters and failover orchestration for AngelMemory."""

from __future__ import annotations

import inspect
import time
from typing import Any

import aiohttp

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def _limit_fallback_documents(
    documents: list[str],
    *,
    max_documents: int,
    max_document_chars: int,
) -> list[str]:
    """Bound a local fallback workload while keeping useful context at both ends."""
    limited = documents[:max_documents] if max_documents > 0 else documents
    if max_document_chars <= 0:
        return limited

    marker = "\n[…]\n"
    bounded: list[str] = []
    for document in limited:
        text = str(document or "")
        if len(text) <= max_document_chars:
            bounded.append(text)
            continue

        content_chars = max(1, max_document_chars - len(marker))
        head_chars = max(1, content_chars * 3 // 4)
        tail_chars = max(0, content_chars - head_chars)
        tail = text[-tail_chars:] if tail_chars else ""
        bounded.append(f"{text[:head_chars]}{marker}{tail}")
    return bounded


async def _call_provider(
    provider: Any,
    *,
    query: str,
    documents: list[str],
    top_n: int | None,
) -> list[dict[str, Any]]:
    rerank_method = provider.rerank
    call_kwargs: dict[str, Any] = {
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        try:
            parameters = inspect.signature(rerank_method).parameters.values()
            accepts_top_n = any(
                parameter.name == "top_n"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_top_n = True
        if accepts_top_n:
            call_kwargs["top_n"] = int(top_n)

    result = rerank_method(**call_kwargs)
    if inspect.isawaitable(result):
        result = await result
    normalized = _normalize_results(result)
    if not normalized:
        raise RuntimeError("Rerank Provider 返回空结果")
    if top_n is not None:
        return normalized[: max(0, int(top_n))]
    return normalized


def _should_retry_with_shorter_documents(exc: Exception) -> bool:
    """Retry local server errors that may be llama.cpp physical-batch overflow."""
    status = getattr(exc, "status", None)
    message = str(exc).lower()
    return status == 500 or "too large to process" in message or "physical batch" in message


def _normalize_results(response: Any) -> list[dict[str, Any]]:
    """Normalize AstrBot objects and OpenAI-compatible JSON to one result shape."""
    if isinstance(response, dict):
        items = response.get("results")
        if items is None:
            data = response.get("data")
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("results") or data.get("items")
    else:
        items = response
    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            try:
                index = int(item.get("index", -1))
                score = float(
                    item.get("relevance_score", item.get("score", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                continue
        else:
            try:
                index = int(getattr(item, "index"))
                score = float(
                    getattr(item, "relevance_score", getattr(item, "score", 0.0))
                    or 0.0
                )
            except (AttributeError, TypeError, ValueError):
                continue
        if index < 0:
            continue
        normalized.append(
            {
                "index": index,
                "score": score,
                "relevance_score": score,
            }
        )
    return normalized


class CredentialBackedOpenAIRerankProvider:
    """Use an existing OpenAI chat provider's URL and rotating credentials for /rerank."""

    def __init__(self, provider_id: str, credential_provider: Any, timeout: float = 8.0):
        self.provider_id = provider_id
        self.credential_provider = credential_provider
        self.timeout = max(2.0, min(60.0, float(timeout)))

    def _base_url(self) -> str:
        client = getattr(self.credential_provider, "client", None)
        base_url = str(getattr(client, "base_url", "") or "").rstrip("/")
        if not base_url:
            raise RuntimeError(f"Rerank Provider {self.provider_id} 缺少 API Base URL")
        return base_url

    def _api_key(self) -> str:
        getter = getattr(self.credential_provider, "get_current_key", None)
        key = getter() if callable(getter) else ""
        if not key:
            raise RuntimeError(f"Rerank Provider {self.provider_id} 缺少 API Key")
        return str(key)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        model_getter = getattr(self.credential_provider, "get_model", None)
        model = model_getter() if callable(model_getter) else ""
        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)

        provider_config = getattr(self.credential_provider, "provider_config", {}) or {}
        proxy = str(provider_config.get("proxy") or "").strip() or None
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url()}/rerank",
                json=payload,
                headers=headers,
                proxy=proxy,
            ) as response:
                response.raise_for_status()
                return _normalize_results(await response.json())


class FailoverRerankProvider:
    """Try providers in order and temporarily circuit-break repeatedly failing primaries."""

    def __init__(
        self,
        providers: list[tuple[str, Any]],
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        fallback_max_documents: int = 0,
        fallback_max_document_chars: int = 0,
        fallback_retry_document_chars: int = 0,
    ):
        self.providers = providers
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(5.0, float(cooldown_seconds))
        self.fallback_max_documents = max(0, int(fallback_max_documents))
        self.fallback_max_document_chars = max(0, int(fallback_max_document_chars))
        self.fallback_retry_document_chars = max(
            0, int(fallback_retry_document_chars)
        )
        self._failure_counts: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        now = time.monotonic()
        for index, (provider_id, provider) in enumerate(self.providers):
            is_last = index == len(self.providers) - 1
            if not is_last and self._cooldown_until.get(provider_id, 0.0) > now:
                continue
            try:
                provider_documents = documents
                provider_top_n = top_n
                if index > 0:
                    provider_documents = _limit_fallback_documents(
                        documents,
                        max_documents=self.fallback_max_documents,
                        max_document_chars=self.fallback_max_document_chars,
                    )
                    if provider_top_n is not None:
                        provider_top_n = min(
                            int(provider_top_n), len(provider_documents)
                        )
                try:
                    normalized = await _call_provider(
                        provider,
                        query=query,
                        documents=provider_documents,
                        top_n=provider_top_n,
                    )
                except Exception as first_exc:
                    retry_chars = self.fallback_retry_document_chars
                    can_retry = (
                        index > 0
                        and retry_chars > 0
                        and any(len(str(doc or "")) > retry_chars for doc in provider_documents)
                        and _should_retry_with_shorter_documents(first_exc)
                    )
                    if not can_retry:
                        raise
                    retry_documents = _limit_fallback_documents(
                        provider_documents,
                        max_documents=0,
                        max_document_chars=retry_chars,
                    )
                    logger.warning(
                        "Rerank Provider %s 输入超过本地处理能力，单篇缩短至 %s 字后重试",
                        provider_id,
                        retry_chars,
                    )
                    normalized = await _call_provider(
                        provider,
                        query=query,
                        documents=retry_documents,
                        top_n=provider_top_n,
                    )
                self._failure_counts[provider_id] = 0
                self._cooldown_until.pop(provider_id, None)
                return normalized
            except Exception as exc:
                last_error = exc
                failures = self._failure_counts.get(provider_id, 0) + 1
                self._failure_counts[provider_id] = failures
                if not is_last and failures >= self.failure_threshold:
                    self._cooldown_until[provider_id] = (
                        time.monotonic() + self.cooldown_seconds
                    )
                    self._failure_counts[provider_id] = 0
                next_provider = (
                    self.providers[index + 1][0] if not is_last else "BM25/向量排序"
                )
                logger.warning(
                    "Rerank Provider %s 失败，回退到 %s: %s",
                    provider_id,
                    next_provider,
                    exc,
                )
        if last_error:
            logger.warning("全部 Rerank Provider 均不可用: %s", last_error)
        return []
