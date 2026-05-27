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
        faiss_path = self._get_faiss_path()
        result["has_vector_db"] = bool(faiss_path)
        if faiss_path:
            collections = []
            try:
                for name in os.listdir(faiss_path):
                    if name.endswith(".sqlite"):
                        col_name = name[:-len(".sqlite")]
                        idx_path = os.path.join(faiss_path, f"{col_name}.index")
                        if os.path.exists(idx_path):
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
        faiss_path = self._get_faiss_path()
        if not faiss_path:
            return jsonify({"collections": []})

        collections = []
        try:
            for name in os.listdir(faiss_path):
                if name.endswith(".sqlite"):
                    col_name = name[:-len(".sqlite")]
                    idx_path = os.path.join(faiss_path, f"{col_name}.index")
                    if os.path.exists(idx_path):
                        # 获取记录数
                        sidecar = os.path.join(faiss_path, name)
                        count = 0
                        try:
                            sc = sqlite3.connect(sidecar)
                            row = sc.execute("SELECT COUNT(1) FROM index_rows").fetchone()
                            count = int(row[0]) if row else 0
                            sc.close()
                        except Exception:
                            pass
                        collections.append({"name": col_name, "count": count})
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

        faiss_path = self._get_faiss_path()
        if not faiss_path:
            return jsonify({"results": [], "error": "向量索引不可用"})

        sidecar_path = os.path.join(faiss_path, f"{collection}.sqlite")
        index_path = os.path.join(faiss_path, f"{collection}.index")
        if not os.path.exists(sidecar_path) or not os.path.exists(index_path):
            return jsonify({"results": [], "error": f"集合 {collection} 不存在"})

        # 尝试获取 embedding provider 进行向量化
        try:
            vector_store = self.plugin_context.get_component("vector_store")
            if vector_store is None:
                return jsonify({"results": [], "error": "向量存储组件不可用，系统可能仍在启动中"})

            # 使用 vector_store 的 embedding provider 进行查询
            import faiss as faiss_lib
            import numpy as np

            embedding_provider = getattr(vector_store, "embedding_provider", None)
            if embedding_provider is None:
                return jsonify({"results": [], "error": "嵌入提供商不可用"})

            # 向量化查询
            vectors = await embedding_provider.embed_documents([query_text])
            if not vectors or not vectors[0]:
                return jsonify({"results": [], "error": "查询向量化失败"})
            query_vector = vectors[0]

            # FAISS 检索
            index = faiss_lib.read_index(index_path)
            arr = np.asarray(query_vector, dtype="float32").reshape(1, -1)
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            arr = arr / norm

            actual_k = min(top_k, int(index.ntotal))
            if actual_k <= 0:
                return jsonify({"results": []})

            scores, ids = index.search(arr, actual_k)

            # 从 sidecar 获取元数据
            vector_ids = [int(x) for x in ids[0].tolist() if int(x) >= 0]
            if not vector_ids:
                return jsonify({"results": []})

            sc = sqlite3.connect(sidecar_path)
            sc.row_factory = sqlite3.Row
            placeholders = ",".join(["?" for _ in vector_ids])
            rows = sc.execute(
                f"SELECT item_id, vector_id, vector_text FROM index_rows WHERE vector_id IN ({placeholders})",
                tuple(vector_ids),
            ).fetchall()
            sc.close()

            row_map = {int(r["vector_id"]): dict(r) for r in rows}
            results = []
            for pos, vid in enumerate(ids[0].tolist()):
                vid = int(vid)
                if vid < 0:
                    continue
                row = row_map.get(vid)
                if not row:
                    continue
                results.append({
                    "id": row["item_id"],
                    "document": row["vector_text"],
                    "score": float(scores[0][pos]),
                })

            return jsonify({"results": results})

        except Exception as e:
            logger.error(f"[WebUI] 向量检索失败: {e}", exc_info=True)
            return jsonify({"results": [], "error": str(e)})

    async def vector_browse(self):
        """浏览向量索引内容"""
        collection = request.args.get("collection", "memory_index")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        faiss_path = self._get_faiss_path()
        if not faiss_path:
            return jsonify({"items": [], "total": 0})

        sidecar_path = os.path.join(faiss_path, f"{collection}.sqlite")
        if not os.path.exists(sidecar_path):
            return jsonify({"items": [], "total": 0})

        try:
            sc = sqlite3.connect(sidecar_path)
            sc.row_factory = sqlite3.Row
            total_row = sc.execute("SELECT COUNT(1) AS cnt FROM index_rows").fetchone()
            total = int(total_row["cnt"]) if total_row else 0

            rows = sc.execute(
                "SELECT item_id, vector_id, vector_text, dimension, updated_at "
                "FROM index_rows ORDER BY vector_id ASC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
            sc.close()

            items = [{
                "id": row["item_id"],
                "vector_id": int(row["vector_id"]),
                "document": row["vector_text"],
                "dimension": int(row["dimension"]),
                "updated_at": float(row["updated_at"] or 0),
            } for row in rows]

            return jsonify({
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(total / page_size) if total > 0 else 1,
            })
        except Exception as e:
            return jsonify({"items": [], "total": 0, "error": str(e)}), 500
