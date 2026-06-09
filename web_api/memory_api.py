"""
记忆相关 API
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_central_conn(self) -> Optional[sqlite3.Connection]:
        """获取中央记忆数据库连接"""
        path_manager = self.plugin_context.get_path_manager()
        db_path = str(path_manager.get_simple_memory_db_path())
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_faiss_path(self) -> str:
        """获取 FAISS 索引目录路径"""
        try:
            path_manager = self.plugin_context.get_path_manager()
            if not path_manager.is_provider_set():
                return ""
            faiss_dir = str(path_manager.get_faiss_index_dir())
            return faiss_dir if os.path.isdir(faiss_dir) else ""
        except (ValueError, AttributeError):
            return ""

    def _get_sqlite_vector_path(self) -> str:
        """获取 SQLite 向量索引目录路径"""
        try:
            path_manager = self.plugin_context.get_path_manager()
            if not path_manager.is_provider_set():
                return ""
            sqlite_dir = str(path_manager.get_sqlite_vector_index_dir())
            return sqlite_dir if os.path.isdir(sqlite_dir) else ""
        except (ValueError, AttributeError):
            return ""

    def _get_vector_store(self):
        try:
            return self.plugin_context.get_component("vector_store")
        except Exception:
            return None

    def _get_vector_backend_name(self) -> str:
        vector_store = self._get_vector_store()
        return str(getattr(vector_store, "backend_name", "") or "").strip()

    def _get_vector_storage_path(self) -> str:
        backend = self._get_vector_backend_name()
        if backend == "sqlite":
            return self._get_sqlite_vector_path()
        if backend == "faiss":
            return self._get_faiss_path()
        return self._get_sqlite_vector_path() or self._get_faiss_path()

    @staticmethod
    def _count_vector_rows(db_path: str, backend: str) -> int:
        table = "vector_rows" if backend == "sqlite" else "index_rows"
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(f"SELECT COUNT(1) FROM {table}").fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    async def get_overview(self):
        """总览统计"""
        conn = self._get_central_conn()
        result: Dict[str, Any] = {
            "provider_id": self.plugin_context.get_embedding_provider_id() or "(未配置)",
            "llm_provider_id": self.plugin_context.get_llm_provider_id() or "(未配置)",
            "has_providers": self.plugin_context.has_providers(),
            "memory_count": 0,
            "global_tag_count": 0,
            "note_index_count": 0,
            "scopes": [],
            "index_dir": str(self.plugin_context.get_index_dir()),
        }

        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) AS cnt FROM memory_records")
                result["memory_count"] = int(cur.fetchone()["cnt"])
                cur.execute("SELECT COUNT(*) AS cnt FROM global_tags")
                result["global_tag_count"] = int(cur.fetchone()["cnt"])
                cur.execute("SELECT COUNT(*) AS cnt FROM note_index_records")
                result["note_index_count"] = int(cur.fetchone()["cnt"])
                cur.execute("SELECT DISTINCT memory_scope FROM memory_records ORDER BY memory_scope ASC")
                result["scopes"] = [row["memory_scope"] for row in cur.fetchall() if row["memory_scope"]]
            except Exception as e:
                result["error"] = str(e)
            finally:
                conn.close()

        # 向量索引状态
        vector_backend = self._get_vector_backend_name()
        vector_path = self._get_vector_storage_path()
        result["vector_backend"] = vector_backend or "(未初始化)"
        result["has_vector_db"] = bool(vector_path)
        if vector_path:
            collections = []
            try:
                for name in os.listdir(vector_path):
                    if name.endswith(".sqlite"):
                        col_name = name[:-len(".sqlite")]
                        if vector_backend == "faiss":
                            idx_path = os.path.join(vector_path, f"{col_name}.index")
                            if not os.path.exists(idx_path):
                                continue
                        collections.append(col_name)
            except Exception:
                pass
            result["vector_collections"] = sorted(collections)

        return jsonify(result)

    async def browse_memories(self):
        """分页浏览记忆"""
        scope = request.args.get("scope", "")
        keyword = request.args.get("keyword", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})

        try:
            where = []
            params: List[Any] = []

            if scope.strip():
                where.append("mr.memory_scope = ?")
                params.append(scope.strip())

            if keyword.strip():
                kw = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where.append(
                    "(mr.judgment LIKE ? ESCAPE '\\' OR mr.reasoning LIKE ? ESCAPE '\\' "
                    "OR IFNULL(tags.tags_text, '') LIKE ? ESCAPE '\\')"
                )
                like = f"%{kw}%"
                params.extend([like, like, like])

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""

            # 总数
            sql_count = f"""
                SELECT COUNT(*) AS cnt
                FROM memory_records mr
                LEFT JOIN (
                    SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                {where_sql}
            """
            cur = conn.cursor()
            cur.execute(sql_count, list(params))
            total = int(cur.fetchone()["cnt"])

            # 数据
            sql = f"""
                SELECT
                    mr.id, mr.memory_type, mr.judgment, mr.reasoning,
                    mr.strength, mr.is_active, mr.memory_scope,
                    mr.created_at, mr.updated_at,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                LEFT JOIN (
                    SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                {where_sql}
                ORDER BY mr.created_at DESC, mr.strength DESC
                LIMIT ? OFFSET ?
            """
            cur.execute(sql, list(params) + [page_size, offset])
            items = []
            for row in cur.fetchall():
                items.append({
                    "id": row["id"],
                    "memory_type": row["memory_type"],
                    "judgment": row["judgment"],
                    "reasoning": row["reasoning"],
                    "strength": row["strength"],
                    "is_active": bool(row["is_active"]),
                    "memory_scope": row["memory_scope"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "tags": row["tags"],
                })

            return jsonify({
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(total / page_size) if total > 0 else 1,
            })
        except Exception as e:
            return jsonify({"error": str(e), "items": [], "total": 0}), 500
        finally:
            conn.close()

    async def delete_memory(self):
        """删除记忆"""
        data = await request.get_json()
        memory_id = str(data.get("id", "")).strip()
        if not memory_id:
            return jsonify({"success": False, "error": "缺少 id 参数"}), 400

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"success": False, "error": "数据库不可用"}), 500

        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM memory_tag_rel WHERE memory_id = ?", (memory_id,))
            cur.execute("DELETE FROM memory_records WHERE id = ?", (memory_id,))
            conn.commit()
            deleted = cur.rowcount > 0
            return jsonify({"success": deleted, "id": memory_id})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
        finally:
            conn.close()

    async def vector_collections(self):
        """获取向量集合列表"""
        backend = self._get_vector_backend_name()
        vector_path = self._get_vector_storage_path()
        if not vector_path:
            return jsonify({"collections": []})

        collections = []
        try:
            for name in os.listdir(vector_path):
                if name.endswith(".sqlite"):
                    col_name = name[:-len(".sqlite")]
                    if backend == "faiss":
                        idx_path = os.path.join(vector_path, f"{col_name}.index")
                        if not os.path.exists(idx_path):
                            continue
                    sidecar = os.path.join(vector_path, name)
                    collections.append({
                        "name": col_name,
                        "count": self._count_vector_rows(sidecar, backend),
                        "backend": backend or "unknown",
                    })
        except Exception as e:
            return jsonify({"collections": [], "error": str(e)})

        return jsonify({"collections": sorted(collections, key=lambda x: x["name"])})

    async def vector_search(self):
        """向量检索"""
        collection = request.args.get("collection", "memory_index")
        query_text = request.args.get("text", "").strip()
        top_k = min(50, max(1, int(request.args.get("top_k", 10))))

        if not query_text:
            return jsonify({"results": [], "error": "查询文本不能为空"}), 400

        try:
            vector_store = self._get_vector_store()
            if vector_store is None:
                return jsonify({"results": [], "error": "向量存储组件不可用，系统可能仍在启动中"})

            backend = str(getattr(vector_store, "backend_name", "") or "").strip()
            vector_path = self._get_vector_storage_path()
            if not vector_path:
                return jsonify({"results": [], "error": "向量索引不可用"})

            sidecar_path = os.path.join(vector_path, f"{collection}.sqlite")
            if not os.path.exists(sidecar_path):
                return jsonify({"results": [], "error": f"集合 {collection} 不存在"})
            if backend == "faiss":
                index_path = os.path.join(vector_path, f"{collection}.index")
                if not os.path.exists(index_path):
                    return jsonify({"results": [], "error": f"集合 {collection} 不存在"})

            index_collection = vector_store.get_or_create_collection_with_dimension_check(collection)
            hits = await vector_store.recall_memory_ids(
                index_collection,
                query=query_text,
                limit=top_k,
                similarity_threshold=-1.0,
            )
            if not hits:
                return jsonify({"results": []})

            hit_ids = [item_id for item_id, _ in hits]
            table = "vector_rows" if backend == "sqlite" else "index_rows"
            sc = sqlite3.connect(sidecar_path)
            sc.row_factory = sqlite3.Row
            placeholders = ",".join(["?" for _ in hit_ids])
            rows = sc.execute(
                f"SELECT item_id, vector_text FROM {table} WHERE item_id IN ({placeholders})",
                tuple(hit_ids),
            ).fetchall()
            sc.close()

            row_map = {str(r["item_id"]): dict(r) for r in rows}
            results = []
            for item_id, score in hits:
                row = row_map.get(str(item_id))
                if not row:
                    continue
                results.append({
                    "id": row["item_id"],
                    "document": row["vector_text"],
                    "score": float(score),
                })

            return jsonify({"results": results, "backend": backend or "unknown"})

        except Exception as e:
            logger.error(f"[WebUI] 向量检索失败: {e}", exc_info=True)
            return jsonify({"results": [], "error": str(e)})

    async def vector_browse(self):
        """浏览向量索引内容"""
        collection = request.args.get("collection", "memory_index")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        backend = self._get_vector_backend_name()
        vector_path = self._get_vector_storage_path()
        if not vector_path:
            return jsonify({"items": [], "total": 0})

        sidecar_path = os.path.join(vector_path, f"{collection}.sqlite")
        if not os.path.exists(sidecar_path):
            return jsonify({"items": [], "total": 0})

        try:
            sc = sqlite3.connect(sidecar_path)
            sc.row_factory = sqlite3.Row
            table = "vector_rows" if backend == "sqlite" else "index_rows"
            total_row = sc.execute(f"SELECT COUNT(1) AS cnt FROM {table}").fetchone()
            total = int(total_row["cnt"]) if total_row else 0
            if backend == "sqlite":
                rows = sc.execute(
                    "SELECT item_id, vector_text, dimension, updated_at "
                    "FROM vector_rows ORDER BY updated_at ASC, item_id ASC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
            else:
                rows = sc.execute(
                    "SELECT item_id, vector_id, vector_text, dimension, updated_at "
                    "FROM index_rows ORDER BY vector_id ASC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
            sc.close()

            items = []
            for index, row in enumerate(rows):
                items.append({
                    "id": row["item_id"],
                    "vector_id": int(row["vector_id"]) if "vector_id" in row.keys() else offset + index + 1,
                    "document": row["vector_text"],
                    "dimension": int(row["dimension"]),
                    "updated_at": float(row["updated_at"] or 0),
                    "backend": backend or "unknown",
                })

            return jsonify({
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(total / page_size) if total > 0 else 1,
            })
        except Exception as e:
            return jsonify({"items": [], "total": 0, "error": str(e)}), 500
