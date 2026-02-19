import chromadb
import logging
import sqlite3
import os
import ast
from typing import List, Dict, Any
from .embedding import create_embedding_function

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

    def query_collections(
        self,
        query_text: str,
        collection_names: List[str],
        n_results: int = 5,
        where_filter: Dict[str, Any] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for name in collection_names:
            try:
                collection = self.client.get_collection(name=name, embedding_function=self.embedding_fn)

                query_params = {
                    "query_texts": [query_text],
                    "n_results": n_results,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_filter:
                    query_params["where"] = where_filter

                initial_results = collection.query(
                    **query_params
                )

                if not initial_results['ids'] or not initial_results['ids'][0]:
                    results[name] = []
                    continue

                ids = initial_results['ids'][0]
                docs = initial_results['documents'][0]
                metas = initial_results['metadatas'][0]
                dists = initial_results['distances'][0]

                formatted_res = []
                for i, _id in enumerate(ids[:n_results]):
                    base_score = max(0.0, 1.0 - (dists[i] / 2.0))
                    formatted_res.append({
                        "id": _id,
                        "document": docs[i] if docs else "",
                        "metadata": metas[i] if (metas and metas[i]) else {},
                        "distance": dists[i] if dists else 0.0,
                        "score": base_score,
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

    def browse_collection(
        self,
        collection_name: str,
        limit: int = 10,
        offset: int = 0,
        where_filter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Browse items in a collection without query."""
        try:
            collection = self.client.get_collection(name=collection_name)
            get_params = {
                "limit": limit,
                "offset": offset,
                "include": ["documents", "metadatas"],
            }
            if where_filter:
                get_params["where"] = where_filter

            res = collection.get(
                **get_params
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
