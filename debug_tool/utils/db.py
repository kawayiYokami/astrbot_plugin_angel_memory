import ast
import datetime
import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from .config_loader import ConfigLoader
from .embedding import create_embedding_function

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self, loader: ConfigLoader, provider_config: Optional[dict]):
        self.loader = loader
        self.provider_config = provider_config or {}
        self.provider_id = self.provider_config.get("id", "")
        self.embedding_fn = None
        self.client = None
        self.central_conn = None

        self.chromadb_path = ""
        self.simple_db_path = self.loader.get_simple_memory_db_path()
        self.maintenance_state_path = self.loader.get_maintenance_state_path()
        self.backup_dir = self.loader.get_backup_dir()

        self._connect_central_db()
        self._connect_vector_db()

    # ===== connect =====

    def _connect_central_db(self):
        try:
            if os.path.exists(self.simple_db_path):
                self.central_conn = sqlite3.connect(
                    self.simple_db_path, check_same_thread=False
                )
                self.central_conn.row_factory = sqlite3.Row
                logger.info("Connected to central memory DB: %s", self.simple_db_path)
            else:
                logger.warning("Central memory DB not found: %s", self.simple_db_path)
        except Exception as e:
            logger.error("Failed to connect central memory DB: %s", e)

    def _connect_vector_db(self):
        if not self.provider_id:
            return
        try:
            self.chromadb_path = self.loader.get_data_dir(self.provider_id)
            if not os.path.exists(self.chromadb_path):
                logger.warning("ChromaDB path not found: %s", self.chromadb_path)
                return
            self.client = chromadb.PersistentClient(path=self.chromadb_path)
            try:
                self.embedding_fn = create_embedding_function(self.provider_config)
            except Exception as e:
                logger.warning("Embedding function init failed, query mode disabled: %s", e)
            logger.info("Connected to ChromaDB: %s", self.chromadb_path)
        except Exception as e:
            logger.error("Failed to connect ChromaDB: %s", e)

    # ===== status =====

    def has_vector_db(self) -> bool:
        return self.client is not None

    def has_central_db(self) -> bool:
        return self.central_conn is not None

    def get_overview(self) -> Dict[str, Any]:
        out = {
            "provider_id": self.provider_id or "(未检测到可用 embedding provider)",
            "chromadb_path": self.chromadb_path or "(未连接)",
            "simple_db_path": self.simple_db_path,
            "maintenance_state_path": self.maintenance_state_path,
            "backup_dir": self.backup_dir,
        }
        if self.has_vector_db():
            cols = self.get_collections()
            out["vector_collections"] = cols
            out["memory_index_count"] = self.get_collection_stats("memory_index").get("count", 0)
        else:
            out["vector_collections"] = []
            out["memory_index_count"] = 0
        if self.has_central_db():
            out.update(self.get_central_stats())
        else:
            out.update({"memory_count": 0, "global_tag_count": 0, "scopes": []})
        return out

    # ===== vector =====

    def get_collections(self) -> List[str]:
        if not self.client:
            return []
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception as e:
            logger.error("list_collections failed: %s", e)
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        if not self.client:
            return {"count": 0}
        try:
            collection = self.client.get_collection(collection_name)
            return {"count": int(collection.count())}
        except Exception:
            return {"count": 0}

    def browse_collection(self, collection_name: str, limit: int = 20, offset: int = 0):
        if not self.client:
            return []
        try:
            collection = self.client.get_collection(collection_name)
            res = collection.get(
                limit=int(limit),
                offset=int(offset),
                include=["documents", "metadatas"],
            )
            out = []
            for i, _id in enumerate(res.get("ids", []) or []):
                out.append(
                    {
                        "id": _id,
                        "document": (res.get("documents", []) or [""])[i]
                        if i < len(res.get("documents", []) or [])
                        else "",
                        "metadata": (res.get("metadatas", []) or [{}])[i]
                        if i < len(res.get("metadatas", []) or [])
                        else {},
                    }
                )
            return out
        except Exception as e:
            logger.error("browse_collection failed: %s", e)
            return []

    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        if not self.embedding_fn:
            return [{"error": "当前 provider 无法初始化 embedding，无法执行向量查询"}]
        try:
            collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_fn
            )
            query_params: Dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": int(n_results),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_params["where"] = where_filter
            res = collection.query(**query_params)
            out = []
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for i, _id in enumerate(ids):
                distance = float(dists[i]) if i < len(dists) else 2.0
                out.append(
                    {
                        "id": _id,
                        "document": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": distance,
                        "score": max(0.0, 1.0 - distance / 2.0),
                    }
                )
            return out
        except Exception as e:
            logger.error("query_collection failed: %s", e)
            return [{"error": str(e)}]

    # ===== central memory =====

    def get_central_stats(self) -> Dict[str, Any]:
        if not self.central_conn:
            return {"memory_count": 0, "global_tag_count": 0, "scopes": []}
        try:
            c = self.central_conn.cursor()
            c.execute("SELECT COUNT(*) AS cnt FROM memory_records")
            memory_count = int(c.fetchone()["cnt"])
            c.execute("SELECT COUNT(*) AS cnt FROM global_tags")
            tag_count = int(c.fetchone()["cnt"])
            c.execute(
                "SELECT DISTINCT memory_scope FROM memory_records ORDER BY memory_scope ASC"
            )
            scopes = [row["memory_scope"] for row in c.fetchall() if row["memory_scope"]]
            return {
                "memory_count": memory_count,
                "global_tag_count": tag_count,
                "scopes": scopes,
            }
        except Exception as e:
            logger.error("get_central_stats failed: %s", e)
            return {"memory_count": 0, "global_tag_count": 0, "scopes": []}

    def browse_central_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        scope: str = "",
        keyword: str = "",
        return_total: bool = False,
    ):
        if not self.central_conn:
            return ([], 0) if return_total else []
        try:
            where = []
            params: List[Any] = []

            if (scope or "").strip():
                where.append("mr.memory_scope = ?")
                params.append(scope.strip())

            if (keyword or "").strip():
                kw = (
                    keyword.strip()
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                where.append(
                    "(mr.judgment LIKE ? ESCAPE '\\' OR mr.reasoning LIKE ? ESCAPE '\\' OR IFNULL(tags.tags_text, '') LIKE ? ESCAPE '\\')"
                )
                like = f"%{kw}%"
                params.extend([like, like, like])

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""

            total = 0
            if return_total:
                sql_count = f"""
                    SELECT COUNT(*) AS cnt
                    FROM memory_records mr
                    LEFT JOIN (
                        SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                        FROM memory_tag_rel mtr
                        JOIN global_tags gt ON gt.id = mtr.tag_id
                        GROUP BY mtr.memory_id
                    ) tags ON tags.memory_id = mr.id
                    {where_sql}
                """
                cur = self.central_conn.cursor()
                cur.execute(sql_count, list(params))
                total = int((cur.fetchone() or {"cnt": 0})["cnt"])

            sql = f"""
                SELECT
                    mr.id, mr.memory_type, mr.judgment, mr.reasoning,
                    mr.strength, mr.is_active, mr.memory_scope,
                    mr.created_at, mr.updated_at,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                LEFT JOIN (
                    SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr
                    JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                {where_sql}
                ORDER BY mr.created_at DESC, mr.strength DESC
                LIMIT ? OFFSET ?
            """
            query_params = list(params) + [int(limit), int(offset)]
            cur = self.central_conn.cursor()
            cur.execute(sql, query_params)
            rows = cur.fetchall()
            out = []
            for row in rows:
                out.append(
                    {
                        "id": row["id"],
                        "document": row["judgment"],
                        "metadata": {
                            "memory_type": row["memory_type"],
                            "judgment": row["judgment"],
                            "reasoning": row["reasoning"],
                            "strength": row["strength"],
                            "is_active": bool(row["is_active"]),
                            "memory_scope": row["memory_scope"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "tags": row["tags"],
                        },
                    }
                )
            return (out, total) if return_total else out
        except Exception as e:
            logger.error("browse_central_memories failed: %s", e)
            return ([], 0) if return_total else []

    def get_global_tags(self, limit: int = 200, offset: int = 0, keyword: str = ""):
        if not self.central_conn:
            return []
        try:
            where_sql = ""
            params: List[Any] = []
            if (keyword or "").strip():
                where_sql = "WHERE gt.name LIKE ? ESCAPE '\\'"
                kw = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{kw}%")

            sql = f"""
                SELECT
                    gt.id,
                    gt.name,
                    (
                        SELECT COUNT(*) FROM memory_tag_rel mtr WHERE mtr.tag_id = gt.id
                    ) AS memory_refs,
                    (
                        SELECT COUNT(*) FROM note_tag_rel ntr WHERE ntr.tag_id = gt.id
                    ) AS note_refs
                FROM global_tags gt
                {where_sql}
                ORDER BY (memory_refs + note_refs) DESC, gt.id ASC
                LIMIT ? OFFSET ?
            """
            params.extend([int(limit), int(offset)])
            cur = self.central_conn.cursor()
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("get_global_tags failed: %s", e)
            return []

    def unified_tag_hit_search(
        self, query_text: str, limit: int = 50, scope: str = ""
    ) -> Dict[str, Any]:
        if not self.central_conn:
            return {"matched_tags": [], "matched_tag_ids": [], "memory_hits": []}
        query = str(query_text or "")
        if not query.strip():
            return {"matched_tags": [], "matched_tag_ids": [], "memory_hits": []}

        try:
            cur = self.central_conn.cursor()
            cur.execute("SELECT id, name FROM global_tags ORDER BY id ASC")
            all_tags = cur.fetchall()
            matched = [row for row in all_tags if row["name"] and row["name"] in query]
            if not matched:
                return {"matched_tags": [], "matched_tag_ids": [], "memory_hits": []}

            tag_ids = [int(row["id"]) for row in matched]
            tag_names = [str(row["name"]) for row in matched]

            placeholders = ",".join(["?" for _ in tag_ids])
            scope_clause = ""
            scope_params: List[Any] = []
            if (scope or "").strip():
                s = scope.strip()
                if s == "public":
                    scope_clause = "AND mr.memory_scope = ?"
                    scope_params = ["public"]
                else:
                    scope_clause = "AND (mr.memory_scope = ? OR mr.memory_scope = 'public')"
                    scope_params = [s]

            sql = f"""
                SELECT
                    mr.id, mr.memory_type, mr.judgment, mr.reasoning,
                    mr.strength, mr.is_active, mr.memory_scope,
                    mr.created_at, mr.updated_at,
                    COUNT(DISTINCT gt.id) AS hit_count,
                    CAST(mr.created_at / 86400 AS INTEGER) AS day_bucket,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                JOIN memory_tag_rel mtr ON mtr.memory_id = mr.id
                JOIN global_tags gt ON gt.id = mtr.tag_id
                LEFT JOIN (
                    SELECT mtr2.memory_id, GROUP_CONCAT(gt2.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr2
                    JOIN global_tags gt2 ON gt2.id = mtr2.tag_id
                    GROUP BY mtr2.memory_id
                ) tags ON tags.memory_id = mr.id
                WHERE gt.id IN ({placeholders})
                {scope_clause}
                GROUP BY mr.id
                ORDER BY hit_count DESC, day_bucket DESC, mr.strength DESC, mr.created_at DESC
                LIMIT ?
            """
            params: List[Any] = [*tag_ids, *scope_params, int(limit)]
            cur.execute(sql, params)
            rows = cur.fetchall()
            memory_hits = []
            for row in rows:
                memory_hits.append(
                    {
                        "id": row["id"],
                        "hit_count": int(row["hit_count"] or 0),
                        "document": row["judgment"],
                        "metadata": {
                            "memory_type": row["memory_type"],
                            "judgment": row["judgment"],
                            "reasoning": row["reasoning"],
                            "strength": row["strength"],
                            "is_active": bool(row["is_active"]),
                            "memory_scope": row["memory_scope"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "tags": row["tags"],
                        },
                    }
                )
            return {
                "matched_tags": tag_names,
                "matched_tag_ids": tag_ids,
                "memory_hits": memory_hits,
            }
        except Exception as e:
            logger.error("unified_tag_hit_search failed: %s", e)
            return {"matched_tags": [], "matched_tag_ids": [], "memory_hits": [], "error": str(e)}

    # ===== import / export =====

    @staticmethod
    def _parse_tags(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            return [x.strip() for x in stripped.split(",") if x.strip()]
        return []

    def _replace_tags(self, conn: sqlite3.Connection, memory_id: str, tags: List[str]):
        conn.execute("DELETE FROM memory_tag_rel WHERE memory_id = ?", (memory_id,))
        if not tags:
            return
        conn.executemany("INSERT OR IGNORE INTO global_tags(name) VALUES (?)", [(t,) for t in tags])
        placeholders = ",".join(["?" for _ in tags])
        rows = conn.execute(
            f"SELECT id, name FROM global_tags WHERE name IN ({placeholders})", tuple(tags)
        ).fetchall()
        id_map = {str(r["name"]): int(r["id"]) for r in rows}
        rel = [(memory_id, id_map[t]) for t in tags if t in id_map]
        if rel:
            conn.executemany(
                "INSERT OR IGNORE INTO memory_tag_rel(memory_id, tag_id) VALUES (?, ?)", rel
            )

    def export_central_snapshot(self) -> Dict[str, Any]:
        if not self.central_conn:
            return {"schema_version": 1, "exported_at": int(time.time()), "records": [], "global_tags": [], "memory_tag_rel": []}
        cur = self.central_conn.cursor()
        records = [
            dict(row)
            for row in cur.execute(
                """
                SELECT id, memory_type, judgment, reasoning, strength, is_active,
                       memory_scope, created_at, updated_at
                FROM memory_records
                ORDER BY created_at DESC
                """
            ).fetchall()
        ]
        tags = [dict(row) for row in cur.execute("SELECT id, name FROM global_tags ORDER BY id ASC").fetchall()]
        rel = [dict(row) for row in cur.execute("SELECT memory_id, tag_id FROM memory_tag_rel").fetchall()]
        return {
            "schema_version": 1,
            "exported_at": int(time.time()),
            "records": records,
            "global_tags": tags,
            "memory_tag_rel": rel,
        }

    def _upsert_by_judgment(self, conn: sqlite3.Connection, item: Dict[str, Any]) -> str:
        judgment = str(item.get("judgment") or "").strip()
        if not judgment:
            return "skip"
        now = time.time()
        created_at = float(item.get("created_at") or now)
        rows = conn.execute(
            "SELECT id, created_at FROM memory_records WHERE judgment = ? ORDER BY created_at DESC, updated_at DESC",
            (judgment,),
        ).fetchall()
        tags = self._parse_tags(item.get("tags", []))
        memory_type = str(item.get("memory_type") or "知识记忆").strip() or "知识记忆"
        reasoning = str(item.get("reasoning") or "").strip()
        strength = int(item.get("strength", 1) or 1)
        is_active = 1 if bool(item.get("is_active", False)) else 0
        memory_scope = str(item.get("memory_scope") or "public").strip() or "public"

        if rows:
            keep_id = str(rows[0]["id"])
            old_created = float(rows[0]["created_at"] or 0)
            if created_at >= old_created:
                conn.execute(
                    """
                    UPDATE memory_records
                    SET memory_type=?, reasoning=?, strength=?, is_active=?,
                        memory_scope=?, created_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        memory_type,
                        reasoning,
                        strength,
                        is_active,
                        memory_scope,
                        created_at,
                        now,
                        keep_id,
                    ),
                )
                self._replace_tags(conn, keep_id, tags)
            dup_ids = [str(x["id"]) for x in rows[1:]]
            if dup_ids:
                placeholders = ",".join(["?" for _ in dup_ids])
                conn.execute(f"DELETE FROM memory_records WHERE id IN ({placeholders})", tuple(dup_ids))
                conn.execute(f"DELETE FROM memory_tag_rel WHERE memory_id IN ({placeholders})", tuple(dup_ids))
            return "upsert"

        memory_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO memory_records(
                id, memory_type, judgment, reasoning, strength, is_active,
                memory_scope, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, memory_type, judgment, reasoning, strength, is_active, memory_scope, created_at, now),
        )
        self._replace_tags(conn, memory_id, tags)
        return "insert"

    def import_central_payload(self, payload: Any) -> Dict[str, int]:
        if not self.central_conn:
            return {"inserted": 0, "upserted": 0, "skipped": 0, "failed": 1}

        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
            rel_map: Dict[str, List[str]] = {}
            tags_map: Dict[int, str] = {}
            for row in payload.get("global_tags", []) or []:
                try:
                    tags_map[int(row.get("id"))] = str(row.get("name"))
                except Exception:
                    continue
            for row in payload.get("memory_tag_rel", []) or []:
                mid = str(row.get("memory_id") or "").strip()
                tid = row.get("tag_id")
                if not mid:
                    continue
                if isinstance(tid, int) and tid in tags_map:
                    rel_map.setdefault(mid, []).append(tags_map[tid])
            memories = []
            for r in records:
                item = dict(r)
                item["tags"] = rel_map.get(str(item.get("id") or ""), [])
                memories.append(item)
        elif isinstance(payload, dict) and isinstance(payload.get("memories"), list):
            memories = payload["memories"]
        elif isinstance(payload, list):
            memories = payload
        else:
            raise ValueError("JSON 格式错误：支持 records/memories/数组 三种格式。")

        inserted = 0
        upserted = 0
        skipped = 0
        failed = 0
        with self.central_conn:
            for raw in memories:
                try:
                    status = self._upsert_by_judgment(self.central_conn, raw if isinstance(raw, dict) else {})
                    if status == "insert":
                        inserted += 1
                    elif status == "upsert":
                        upserted += 1
                    else:
                        skipped += 1
                except Exception:
                    failed += 1
            self.central_conn.execute(
                "DELETE FROM memory_tag_rel WHERE memory_id NOT IN (SELECT id FROM memory_records)"
            )
        return {"inserted": inserted, "upserted": upserted, "skipped": skipped, "failed": failed}

    # ===== maintenance =====

    def get_maintenance_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.maintenance_state_path):
            return {}
        try:
            with open(self.maintenance_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}

    def list_backups(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.backup_dir):
            return []
        rows = []
        for name in sorted(os.listdir(self.backup_dir), reverse=True):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.backup_dir, name)
            try:
                stat = os.stat(path)
                rows.append(
                    {
                        "name": name,
                        "path": path,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )
            except Exception:
                continue
        return rows

    def load_backup_preview(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "schema_version": data.get("schema_version"),
                "exported_at": data.get("exported_at"),
                "records": len(data.get("records") or []),
                "global_tags": len(data.get("global_tags") or []),
                "memory_tag_rel": len(data.get("memory_tag_rel") or []),
            }
        except Exception as e:
            return {"error": str(e)}

