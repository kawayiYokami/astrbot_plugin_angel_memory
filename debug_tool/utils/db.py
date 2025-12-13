import chromadb
import logging
import sqlite3
import os
import ast
from typing import List, Dict, Any
from .embedding import create_embedding_function

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, db_path: str, provider_config: dict):
        self.db_path = db_path
        self.provider_config = provider_config
        self.provider_id = provider_config.get('id', 'default')
        self.embedding_fn = create_embedding_function(provider_config)
        self.client = None
        self.tag_conn = None
        self._connect()
        self._connect_tag_db()
        self.ranker = Ranker() if FLASHRANK_AVAILABLE else None

    def _connect_tag_db(self):
        """Connect to SQLite Tag Database."""
        try:
            base_dir = os.path.dirname(self.db_path)
            tag_db_path = os.path.join(base_dir, "index", f"tag_index_{self.provider_id}.db")

            if os.path.exists(tag_db_path):
                self.tag_conn = sqlite3.connect(tag_db_path, check_same_thread=False)
                logger.info(f"Connected to Tag DB at {tag_db_path}")
            else:
                logger.warning(f"Tag DB not found at {tag_db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to Tag DB: {e}")

    def resolve_tag_ids(self, tag_ids_str: str) -> List[str]:
        """Convert '[1, 2, 3]' string to list of tag names."""
        if not self.tag_conn or not tag_ids_str:
            return []

        try:
            tag_ids = ast.literal_eval(tag_ids_str)
            if not isinstance(tag_ids, list) or not tag_ids:
                return []

            placeholders = ",".join("?" * len(tag_ids))
            query = f"SELECT name FROM tag_{self.provider_id} WHERE id IN ({placeholders})"
            cursor = self.tag_conn.cursor()
            cursor.execute(query, tag_ids)
            names = [row[0] for row in cursor.fetchall()]
            logger.info(f"Resolved tag IDs {tag_ids} to names: {names}")
            return names

        except Exception as e:
            logger.error(f"Error resolving tags: {e}")
            return []

    def _connect(self):
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            logger.info(f"Connected to ChromaDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            raise e

    def get_collections(self) -> List[str]:
        """List all collection names."""
        if not self.client:
            return []
        colls = self.client.list_collections()
        return [c.name for c in colls]

    def query_collections(self, query_text: str, collection_names: List[str], n_results: int = 5, use_flashrank: bool = False, flashrank_ratio: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for name in collection_names:
            try:
                collection = self.client.get_collection(name=name, embedding_function=self.embedding_fn)

                candidate_limit = n_results * 5  # Get more candidates for re-ranking
                initial_results = collection.query(
                    query_texts=[query_text],
                    n_results=candidate_limit,
                    include=["documents", "metadatas", "distances"]
                )

                if not initial_results['ids'] or not initial_results['ids'][0]:
                    results[name] = []
                    continue

                ids = initial_results['ids'][0]
                docs = initial_results['documents'][0]
                metas = initial_results['metadatas'][0]
                dists = initial_results['distances'][0]

                # Prepare for re-ranking
                passages = [{"id": str(_id), "text": doc} for _id, doc in zip(ids, docs)]

                reranked_passages = []
                if use_flashrank and FLASHRANK_AVAILABLE and self.ranker:
                    try:
                        rerank_request = RerankRequest(query=query_text, passages=passages)
                        reranked_passages = self.ranker.rerank(rerank_request)
                    except Exception as e:
                        logger.error(f"FlashRank failed for collection {name}: {e}. Falling back to vector search.")
                        reranked_passages = [] # Reset to fallback

                # Merge results
                formatted_res = []
                if reranked_passages:
                    # Build a map from id to original data
                    original_data_map = {
                        str(_id): {"meta": meta, "dist": dist, "base_score": max(0.0, 1.0 - (dist / 2.0))}
                        for _id, meta, dist in zip(ids, metas, dists)
                    }

                    for p in reranked_passages[:n_results]:
                        orig = original_data_map.get(p['id'])
                        if not orig:
                            continue

                        base_score = orig['base_score']
                        rerank_score = p['score']
                        final_score = (base_score * (1 - flashrank_ratio)) + (rerank_score * flashrank_ratio)

                        formatted_res.append({
                            "id": p['id'],
                            "document": p['text'],
                            "metadata": orig['meta'],
                            "distance": orig['dist'],
                            "score": final_score,
                            "base_score": base_score,
                            "rerank_score": rerank_score
                        })

                else: # Fallback to standard vector search
                    for i, _id in enumerate(ids[:n_results]):
                        base_score = max(0.0, 1.0 - (dists[i] / 2.0))
                        formatted_res.append({
                            "id": _id,
                            "document": docs[i] if docs else "",
                            "metadata": metas[i] if (metas and metas[i]) else {},
                            "distance": dists[i] if dists else 0.0,
                            "score": base_score,
                            "base_score": base_score,
                            "rerank_score": 0.0
                        })

                results[name] = formatted_res

            except Exception as e:
                logger.error(f"Error querying collection {name}: {e}")
                results[name] = [{"error": str(e)}]

        return results

    def get_collection_stats(self, collection_name: str) -> dict:
        try:
            collection = self.client.get_collection(name=collection_name)
            return {"count": collection.count()}
        except Exception:
            return {"count": 0}

    def browse_collection(self, collection_name: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Browse items in a collection without query."""
        try:
            collection = self.client.get_collection(name=collection_name)
            res = collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            formatted = []
            if res['ids']:
                for i, _id in enumerate(res['ids']):
                    meta = res['metadatas'][i] if (res['metadatas'] and res['metadatas'][i]) else {}
                    doc = res['documents'][i] if (res['documents'] and res['documents'][i]) else ""
                    formatted.append({
                        "id": _id,
                        "document": doc,
                        "metadata": meta
                    })
            return formatted
        except Exception as e:
            logger.error(f"Error browsing collection {collection_name}: {e}")
            return []