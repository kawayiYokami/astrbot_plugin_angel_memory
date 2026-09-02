from __future__ import annotations

import asyncio

from llm_memory.components.embedding_provider import APIEmbeddingProvider


class ShortEmbeddingProvider:
    """返回固定长度向量的嵌入提供商。"""

    async def get_embeddings(self, texts):
        """返回三组固定向量。"""
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    async def get_embedding(self, text):
        """返回固定向量。"""
        return [1.0, 0.0, 0.0]


class AggregatingEmbeddingProvider:
    """返回聚合单条向量的嵌入提供商。"""

    async def get_embeddings(self, texts):
        """始终只返回一个聚合向量。"""
        return [[1.0, 2.0, 3.0]]

    async def get_embedding(self, text):
        """返回固定聚合向量。"""
        return [1.0, 2.0, 3.0]


class MetaEmbeddingProvider:
    """携带 meta 信息的嵌入提供商。"""

    def __init__(self, meta):
        """保存元数据。"""
        self._meta = meta

    def meta(self):
        """返回元数据。"""
        return self._meta

    async def get_embeddings(self, texts):
        """返回固定向量列表。"""
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        """返回固定向量。"""
        return [1.0, 0.0, 0.0]


class ClosedEmbeddingProvider:
    """模拟已关闭的嵌入提供商。"""

    async def get_embeddings(self, texts):
        """始终抛出 provider 已关闭错误。"""
        raise RuntimeError("old provider is closed")

    async def get_embedding(self, text):
        """始终抛出 provider 已关闭错误。"""
        raise RuntimeError("old provider is closed")


class RecordingEmbeddingProvider:
    """记录每次调用文本列表的嵌入提供商。"""

    def __init__(self):
        """初始化调用记录列表。"""
        self.calls = []

    async def get_embeddings(self, texts):
        """记录调用并返回固定向量。"""
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        """返回固定向量。"""
        return [1.0, 0.0, 0.0]


class RateLimitedEmbeddingProvider:
    """模拟首次请求触发 429、重试后成功的嵌入提供商。"""

    def __init__(self):
        """初始化调用计数。"""
        self.calls = 0

    async def get_embeddings(self, texts):
        """首次调用返回 429，后续返回固定向量。"""
        self.calls += 1
        if self.calls == 1:
            raise Exception("429 Throttling.RateQuota - rate limit exceeded")
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        """返回与 get_embeddings 相同的固定向量。"""
        return [1.0, 0.0, 0.0]


class ReplacingContext:
    """返回固定替换 provider 的上下文。"""

    def __init__(self, replacement):
        """保存替换 provider。"""
        self.replacement = replacement

    def get_provider_by_id(self, provider_id):
        """返回替换 provider。"""
        assert provider_id == "api_provider"
        return self.replacement


class RotatingContext:
    """按调用次数轮换替换 provider 的上下文。"""

    def __init__(self, replacements):
        """保存替换 provider 列表。"""
        self.replacements = list(replacements)
        self.calls = 0

    def get_provider_by_id(self, provider_id):
        """按顺序返回替换 provider。"""
        assert provider_id == "api_provider"
        index = min(self.calls, len(self.replacements) - 1)
        self.calls += 1
        return self.replacements[index]


class FakeOpenAIClient:
    """模拟 OpenAI client 的关闭状态。"""

    def __init__(self, closed=False):
        """初始化关闭状态。"""
        self.closed = closed

    def is_closed(self):
        """返回当前 client 是否已关闭。"""
        return self.closed


class OpenAIStyleEmbeddingProvider:
    """模拟 OpenAI 风格 embedding provider。"""

    def __init__(self):
        """初始化配置、client 和调用计数。"""
        self.provider_config = {
            "id": "api_provider",
            "type": "openai_embedding",
            "embedding_api_key": "test-key",
            "embedding_api_base": "https://example.com",
            "timeout": 20,
            "embedding_model": "text-embedding-3-small",
        }
        self.provider_settings = {}
        self.client = FakeOpenAIClient(closed=True)
        self.model = self.provider_config["embedding_model"]
        self.calls = 0

    async def get_embeddings(self, texts):
        """client 已关闭时抛出异常，否则返回固定向量。"""
        self.calls += 1
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        """client 已关闭时抛出异常，否则返回固定向量。"""
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [1.0, 0.0, 0.0]


class FlakyClosedClientEmbeddingProvider(OpenAIStyleEmbeddingProvider):
    """首次请求抛出 client 已关闭错误，后续成功的 embedding provider。"""

    def __init__(self):
        """初始化 provider 并保持 client 未关闭。"""
        super().__init__()
        self.client = FakeOpenAIClient(closed=False)

    async def get_embeddings(self, texts):
        """首次调用触发关闭错误，后续返回固定向量。"""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_api_embedding_provider_rejects_short_batch_results():
    """嵌入提供商返回数量不足时，应抛出数量不匹配异常。"""
    async def run():
        """运行异步测试逻辑。"""
        provider = APIEmbeddingProvider(ShortEmbeddingProvider(), "short_provider")
        provider._available = True
        provider._cache_enabled = False

        try:
            await provider.embed_documents(["alpha", "beta", "gamma"])
        except RuntimeError as exc:
            assert "返回数量不匹配" in str(exc)
        else:
            raise AssertionError("expected short embedding results to fail")

    asyncio.run(run())


def test_api_embedding_provider_refreshes_provider_reference_before_batch_request():
    """批量请求前应刷新 provider 引用。"""
    async def run():
        """运行异步测试逻辑。"""
        replacement = MetaEmbeddingProvider({"model": "BAAI/bge-m3"})
        provider = APIEmbeddingProvider(
            ClosedEmbeddingProvider(),
            "api_provider",
            context=ReplacingContext(replacement),
        )
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert provider.provider is replacement

    asyncio.run(run())


def test_api_embedding_provider_uses_one_provider_reference_per_batch_attempt():
    """每个批次尝试应使用同一个 provider 引用。"""
    async def run():
        """运行异步测试逻辑。"""
        first = RecordingEmbeddingProvider()
        second = RecordingEmbeddingProvider()
        context = RotatingContext([first, second])
        provider = APIEmbeddingProvider(
            ClosedEmbeddingProvider(),
            "api_provider",
            context=context,
        )
        provider._available = True
        provider._cache_enabled = False
        provider.batch_size = 1

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert context.calls == 1
        assert first.calls == [["alpha"], ["beta"]]
        assert second.calls == []
        assert provider.provider is first

    asyncio.run(run())


def test_api_embedding_provider_rebuilds_closed_openai_client_before_request():
    """上游 client 已关闭时，应在请求前重建。"""
    async def run():
        """运行异步测试逻辑。"""
        upstream = OpenAIStyleEmbeddingProvider()
        provider = APIEmbeddingProvider(upstream, "api_provider")
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert upstream.calls == 1
        assert upstream.client.is_closed() is False

    asyncio.run(run())


def test_api_embedding_provider_rebuilds_and_retries_after_closed_client_error():
    """client 关闭错误后应重建并重试成功。"""
    async def run():
        """运行异步测试逻辑。"""
        upstream = FlakyClosedClientEmbeddingProvider()
        provider = APIEmbeddingProvider(upstream, "api_provider")
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert upstream.calls == 2
        assert upstream.client.is_closed() is False

    asyncio.run(run())


def test_api_embedding_provider_detects_closed_client_from_exception_cause():
    """应能从异常 cause 链中识别 client 已关闭错误。"""
    provider = APIEmbeddingProvider(MetaEmbeddingProvider({"model": "BAAI/bge-m3"}), "api_provider")
    outer = RuntimeError("Connection error.")
    outer.__cause__ = RuntimeError("Cannot send a request, as the client has been closed.")

    assert provider._is_closed_client_error(outer) is True


def test_api_embedding_provider_refreshes_model_info_cache_after_provider_change():
    """provider 引用变化后应刷新模型信息缓存。"""
    first = MetaEmbeddingProvider({"model": "Old/Embedding"})
    second = MetaEmbeddingProvider({"model": "New/Embedding"})
    context = ReplacingContext(first)
    provider = APIEmbeddingProvider(
        first,
        "api_provider",
        context=context,
    )
    provider._available = True

    assert provider.get_model_info()["model_name"] == "Old_Embedding"

    context.replacement = second

    assert provider.get_model_info()["model_name"] == "New_Embedding"
    assert provider.provider is second


def test_api_embedding_provider_exposes_clean_model_name_from_meta():
    """应从 meta 中提取并清洗模型名。"""
    provider = APIEmbeddingProvider(
        MetaEmbeddingProvider({"model": "BAAI/bge-m3"}),
        "api_provider",
    )
    provider._available = True

    info = provider.get_model_info()

    assert info["model_name"] == "BAAI_bge-m3"


def test_api_embedding_provider_exposes_clean_model_name_from_nested_meta():
    """应从嵌套 meta 中提取并清洗模型名。"""
    provider = APIEmbeddingProvider(
        MetaEmbeddingProvider({"model_config": {"model": "Qwen/Qwen3-Embedding-8B"}}),
        "api_provider",
    )
    provider._available = True

    info = provider.get_model_info()

    assert info["model_name"] == "Qwen_Qwen3-Embedding-8B"


def test_api_embedding_provider_expands_aggregated_embedding():
    """聚合嵌入结果应扩展为批次内所有文本的向量。"""
    async def run():
        """运行异步测试逻辑。"""
        provider = APIEmbeddingProvider(AggregatingEmbeddingProvider(), "aggregating_provider")
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

    asyncio.run(run())


def test_is_batch_too_large_error_detects_dashscope_batch_limit():
    """应能识别 DashScope 返回的批量大小超限错误。"""
    error = Exception(
        "DashScope Embedding API request failed (HTTP 400): InvalidParameter - "
        "InternalError.Algo.InvalidParameter: Value error, batch size is invalid, "
        "it should not be larger than 10."
    )

    assert APIEmbeddingProvider._is_batch_too_large_error(error) is True


def test_is_rate_limit_error_detects_http_429_and_throttling():
    """应能识别 HTTP 429 与限流文本错误。"""
    class _RateLimitError(Exception):
        status_code = 429

    assert APIEmbeddingProvider._is_rate_limit_error(_RateLimitError("rate limited")) is True
    assert (
        APIEmbeddingProvider._is_rate_limit_error(
            Exception("Throttling.RateQuota - Requests rate limit exceeded")
        )
        is True
    )
    assert (
        APIEmbeddingProvider._is_rate_limit_error(
            Exception("some other invalid parameter")
        )
        is False
    )


def test_api_embedding_provider_retries_single_batch_on_rate_limit():
    """单批请求遇到 429 后应自动重试并成功返回。"""

    async def run():
        """运行异步测试逻辑。"""
        provider = APIEmbeddingProvider(
            RateLimitedEmbeddingProvider(),
            "rate_limited_provider",
        )
        provider._available = True
        provider._cache_enabled = False
        provider.batch_size = 100

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    asyncio.run(run())
