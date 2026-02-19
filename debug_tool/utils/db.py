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
        self.simple_conn = None
        self._connect()
        self._connect_tag_db()
        self._connect_simple_db()

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

    def _connect_simple_db(self):
        """Connect to simple_memory.db."""
        try:
            base_dir = os.path.dirname(self.db_path)
            simple_db_path = os.path.join(base_dir, "index", "simple_memory.db")
            if os.path.exists(simple_db_path):
                self.simple_conn = sqlite3.connect(simple_db_path, check_same_thread=False)
                self.simple_conn.row_factory = sqlite3.Row
                logger.info(f"Connected to Simple Memory DB at {simple_db_path}")
            else:
                logger.warning(f"Simple Memory DB not found at {simple_db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to Simple Memory DB: {e}")

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

    def has_simple_memory_db(self) -> bool:
        return self.simple_conn is not None

    def get_simple_memory_stats(self) -> Dict[str, Any]:
        if not self.simple_conn:
            return {"count": 0, "scopes": []}
        try:
            cursor = self.simple_conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM memory_records")
            total = int(cursor.fetchone()["cnt"])
            cursor.execute(
                "SELECT DISTINCT memory_scope FROM memory_records ORDER BY memory_scope ASC"
            )
            scopes = [row["memory_scope"] for row in cursor.fetchall() if row["memory_scope"]]
            return {"count": total, "scopes": scopes}
        except Exception as e:
            logger.error(f"Error reading simple memory stats: {e}")
            return {"count": 0, "scopes": []}

    def browse_simple_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        scope: str = "",
        keyword: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.simple_conn:
            return []
        try:
            where_clauses = []
            params: List[Any] = []

            scope_text = (scope or "").strip()
            if scope_text:
                where_clauses.append("mr.memory_scope = ?")
                params.append(scope_text)

            keyword_text = (keyword or "").strip()
            if keyword_text:
                where_clauses.append(
                    "(mr.judgment LIKE ? OR mr.reasoning LIKE ? OR IFNULL(tags.tags_text, '') LIKE ?)"
                )
                like_val = f"%{keyword_text}%"
                params.extend([like_val, like_val, like_val])

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sql = f"""
                SELECT
                    mr.id,
                    mr.memory_type,
                    mr.judgment,
                    mr.reasoning,
                    mr.strength,
                    mr.is_active,
                    mr.memory_scope,
                    mr.created_at,
                    mr.updated_at,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                LEFT JOIN (
                    SELECT
                        mtr.memory_id AS memory_id,
                        GROUP_CONCAT(mt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr
                    JOIN memory_tags mt ON mt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                {where_sql}
                ORDER BY mr.created_at DESC, mr.strength DESC
                LIMIT ? OFFSET ?
            """
            params.extend([int(limit), int(offset)])
            cursor = self.simple_conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            formatted: List[Dict[str, Any]] = []
            for row in rows:
                metadata = {
                    "memory_type": row["memory_type"],
                    "judgment": row["judgment"],
                    "reasoning": row["reasoning"],
                    "strength": row["strength"],
                    "is_active": bool(row["is_active"]),
                    "memory_scope": row["memory_scope"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "tags": row["tags"],
                }
                formatted.append(
                    {
                        "id": row["id"],
                        "document": row["judgment"],
                        "metadata": metadata,
                    }
                )
            return formatted
        except Exception as e:
            logger.error(f"Error browsing simple memories: {e}")
            return []
