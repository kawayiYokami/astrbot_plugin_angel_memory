from __future__ import annotations


class VectorStore:
    """Backward-compatible alias for the FAISS vector store.

    ChromaDB has been removed from the runtime path. Existing imports that still
    refer to VectorStore now receive the lightweight FAISS implementation.
    """

    def __new__(cls, *args, **kwargs):
        from .faiss_memory_index import FaissVectorStore

        return FaissVectorStore(*args, **kwargs)
