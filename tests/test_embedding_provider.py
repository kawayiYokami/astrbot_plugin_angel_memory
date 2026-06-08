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


def test_api_embedding_provider_expands_aggregated_embedding():
    async def run():
        provider = APIEmbeddingProvider(AggregatingEmbeddingProvider(), "aggregating_provider")
        provider._available = True
        provider._cache_enabled = False

        result = await provider.embed_documents(["alpha", "beta"])

        assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

    asyncio.run(run())
