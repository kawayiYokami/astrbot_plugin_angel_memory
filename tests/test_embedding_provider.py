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
