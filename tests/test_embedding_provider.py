from __future__ import annotations

import asyncio

from llm_memory.components.embedding_provider import APIEmbeddingProvider


class ShortEmbeddingProvider:
    async def get_embeddings(self, texts):
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    async def get_embedding(self, text):
        return [1.0, 0.0, 0.0]


class AggregatingEmbeddingProvider:
    async def get_embeddings(self, texts):
        return [[1.0, 2.0, 3.0]]

    async def get_embedding(self, text):
        return [1.0, 2.0, 3.0]


class MetaEmbeddingProvider:
    def __init__(self, meta):
        self._meta = meta

    def meta(self):
        return self._meta

    async def get_embeddings(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        return [1.0, 0.0, 0.0]


class ClosedEmbeddingProvider:
    async def get_embeddings(self, texts):
        raise RuntimeError("old provider is closed")

    async def get_embedding(self, text):
        raise RuntimeError("old provider is closed")


class RecordingEmbeddingProvider:
    def __init__(self):
        self.calls = []

    async def get_embeddings(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        return [1.0, 0.0, 0.0]


class ReplacingContext:
    def __init__(self, replacement):
        self.replacement = replacement

    def get_provider_by_id(self, provider_id):
        assert provider_id == "api_provider"
        return self.replacement


class RotatingContext:
    def __init__(self, replacements):
        self.replacements = list(replacements)
        self.calls = 0

    def get_provider_by_id(self, provider_id):
        assert provider_id == "api_provider"
        index = min(self.calls, len(self.replacements) - 1)
        self.calls += 1
        return self.replacements[index]


class FakeOpenAIClient:
    def __init__(self, closed=False):
        self.closed = closed

    def is_closed(self):
        return self.closed


class OpenAIStyleEmbeddingProvider:
    def __init__(self):
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
        self.calls += 1
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embedding(self, text):
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [1.0, 0.0, 0.0]


class FlakyClosedClientEmbeddingProvider(OpenAIStyleEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.client = FakeOpenAIClient(closed=False)

    async def get_embeddings(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        if self.client.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_api_embedding_provider_rejects_short_batch_results():
    async def run():
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
    async def run():
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
    async def run():
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
    async def run():
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
    async def run():
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
    provider = APIEmbeddingProvider(MetaEmbeddingProvider({"model": "BAAI/bge-m3"}), "api_provider")
    outer = RuntimeError("Connection error.")
    outer.__cause__ = RuntimeError("Cannot send a request, as the client has been closed.")

    assert provider._is_closed_client_error(outer) is True


def test_api_embedding_provider_refreshes_model_info_cache_after_provider_change():
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
    provider = APIEmbeddingProvider(
        MetaEmbeddingProvider({"model": "BAAI/bge-m3"}),
        "api_provider",
    )
    provider._available = True

    info = provider.get_model_info()

    assert info["model_name"] == "BAAI_bge-m3"


def test_api_embedding_provider_exposes_clean_model_name_from_nested_meta():
    provider = APIEmbeddingProvider(
        MetaEmbeddingProvider({"model_config": {"model": "Qwen/Qwen3-Embedding-8B"}}),
        "api_provider",
    )
    provider._available = True

    info = provider.get_model_info()

    assert info["model_name"] == "Qwen_Qwen3-Embedding-8B"


def test_api_embedding_provider_expands_aggregated_embedding():
    async def run():
        provider = APIEmbeddingProvider(AggregatingEmbeddingProvider(), "aggregating_provider")
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

    asyncio.run(run())
