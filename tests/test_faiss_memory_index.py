from __future__ import annotations

import asyncio

from llm_memory.components.faiss_memory_index import FaissVectorStore


class FakeEmbeddingProvider:
    def __init__(self, model_name: str = "fake-v1", dimension: int = 3):
        self.model_name = model_name
        self.dimension = dimension

    async def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def _embed(self, text: str):
        lowered = str(text or "").lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0][: self.dimension]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0][: self.dimension]
        if "gamma" in lowered:
            return [0.0, 0.0, 1.0][: self.dimension]
        return [1.0, 1.0, 0.0][: self.dimension]

    def get_model_info(self):
        return {
            "provider_id": "fake-provider",
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    def get_provider_type(self):
        return "api"

    def is_available(self):
        return True


def test_faiss_upsert_search_delete_and_reload(tmp_path):
    async def run():
        store = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(),
            index_dir=tmp_path / "memory_provider_a" / "index" / "faiss",
            provider_id="provider_a",
        )
        index = store.get_or_create_collection_with_dimension_check("memory_index")
        await store.upsert_memory_index_rows(
            index,
            [
                {"id": "m_alpha", "vector_text": "alpha memory"},
                {"id": "m_beta", "vector_text": "beta memory"},
            ],
        )

        hits = await store.recall_memory_ids(index, query="alpha", limit=2, similarity_threshold=0.0)
        assert hits[0][0] == "m_alpha"
        assert hits[0][1] > hits[1][1]

        deleted = index.delete(ids=["m_alpha"])
        assert deleted == 1
        hits_after_delete = await store.recall_memory_ids(
            index,
            query="alpha",
            limit=2,
            similarity_threshold=0.0,
        )
        assert "m_alpha" not in [item_id for item_id, _ in hits_after_delete]

        reloaded_store = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(),
            index_dir=tmp_path / "memory_provider_a" / "index" / "faiss",
            provider_id="provider_a",
        )
        reloaded_index = reloaded_store.get_or_create_collection_with_dimension_check("memory_index")
        reloaded_hits = await reloaded_store.recall_memory_ids(
            reloaded_index,
            query="beta",
            limit=1,
            similarity_threshold=0.0,
        )
        assert reloaded_hits == [("m_beta", 1.0)]

    asyncio.run(run())


def test_faiss_provider_dirs_are_isolated(tmp_path):
    async def run():
        provider_a = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(),
            index_dir=tmp_path / "memory_provider_a" / "index" / "faiss",
            provider_id="provider_a",
        )
        provider_b = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(),
            index_dir=tmp_path / "memory_provider_b" / "index" / "faiss",
            provider_id="provider_b",
        )
        index_a = provider_a.get_or_create_collection_with_dimension_check("memory_index")
        index_b = provider_b.get_or_create_collection_with_dimension_check("memory_index")

        await provider_a.upsert_memory_index_rows(index_a, [{"id": "m_alpha", "vector_text": "alpha"}])
        await provider_b.upsert_memory_index_rows(index_b, [{"id": "m_beta", "vector_text": "beta"}])

        assert index_a.list_ids() == ["m_alpha"]
        assert index_b.list_ids() == ["m_beta"]

    asyncio.run(run())


def test_faiss_model_change_clears_stale_vector_space(tmp_path):
    async def run():
        index_dir = tmp_path / "memory_provider_a" / "index" / "faiss"
        old_store = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(model_name="fake-v1"),
            index_dir=index_dir,
            provider_id="provider_a",
        )
        old_index = old_store.get_or_create_collection_with_dimension_check("memory_index")
        await old_store.upsert_memory_index_rows(old_index, [{"id": "old_alpha", "vector_text": "alpha"}])
        assert old_index.list_ids() == ["old_alpha"]

        new_store = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(model_name="fake-v2"),
            index_dir=index_dir,
            provider_id="provider_a",
        )
        new_index = new_store.get_or_create_collection_with_dimension_check("memory_index")
        stale_hits = await new_store.recall_memory_ids(
            new_index,
            query="alpha",
            limit=1,
            similarity_threshold=0.0,
        )
        assert stale_hits == []

        await new_store.upsert_memory_index_rows(new_index, [{"id": "new_beta", "vector_text": "beta"}])
        assert new_index.list_ids() == ["new_beta"]

    asyncio.run(run())


def test_faiss_sync_rows_checks_truth_layer_consistency(tmp_path):
    async def run():
        store = FaissVectorStore(
            embedding_provider=FakeEmbeddingProvider(),
            index_dir=tmp_path / "memory_provider_a" / "index" / "faiss",
            provider_id="provider_a",
        )
        index = store.get_or_create_collection_with_dimension_check("memory_index")
        await index.sync_rows(
            [
                {"id": "m_alpha", "vector_text": "alpha"},
                {"id": "m_beta", "vector_text": "beta"},
            ]
        )

        result = await index.sync_rows(
            [
                {"id": "m_alpha", "vector_text": "alpha changed"},
                {"id": "m_gamma", "vector_text": "gamma"},
            ]
        )

        assert result["missing"] == 1
        assert result["orphan"] == 1
        assert result["changed"] == 1
        assert result["migrated"] == 2
        assert result["deleted"] == 1
        assert set(index.list_ids()) == {"m_alpha", "m_gamma"}

    asyncio.run(run())
