from __future__ import annotations

from .faiss_memory_index import FaissVectorStore


class VectorStore(FaissVectorStore):
    """Backward-compatible alias for the FAISS vector store.

    ChromaDB has been removed from the runtime path. Existing imports that still
    refer to VectorStore now receive the lightweight FAISS implementation.
    """

    pass
