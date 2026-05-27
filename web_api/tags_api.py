"""
标签相关 API
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TagsAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_central_conn(self) -> Optional[sqlite3.Connection]:
        path_manager = self.plugin_context.get_path_manager()
        db_path = str(path_manager.get_simple_memory_db_path())
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    async def get_tags(self):
        """全局标签列表"""
        keyword = request.args.get("keyword", "")
        limit = min(500, max(1, int(request.args.get("limit", 200))))
        offset = max(0, int(request.args.get("offset", 0)))

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"tags": []})

        try:
            where_sql = ""
            params: List[Any] = []
            if keyword.strip():
                where_sql = "WHERE gt.name LIKE ? ESCAPE '\\'"
                kw = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{kw}%")

            sql = f"""
                SELECT
                    gt.id,
                    gt.name,
                    (SELECT COUNT(*) FROM memory_tag_rel mtr WHERE mtr.tag_id = gt.id) AS memory_refs,
                    (SELECT COUNT(*) FROM note_tag_rel ntr WHERE ntr.tag_id = gt.id) AS note_refs
                FROM global_tags gt
                {where_sql}
                ORDER BY (memory_refs + note_refs) DESC, gt.id ASC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            cur = conn.cursor()
            cur.execute(sql, params)
            tags = [dict(row) for row in cur.fetchall()]
            return jsonify({"tags": tags})
        except Exception as e:
            return jsonify({"tags": [], "error": str(e)}), 500
        finally:
            conn.close()

    async def hit_search(self):
        """标签命中搜索"""
        data = await request.get_json()
        query_text = str(data.get("query", "")).strip()
        scope = str(data.get("scope", "")).strip()
        limit = min(200, max(1, int(data.get("limit", 50))))

        if not query_text:
            return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})

        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM global_tags ORDER BY id ASC")
            all_tags = cur.fetchall()
            matched = [row for row in all_tags if row["name"] and row["name"] in query_text]
            if not matched:
                return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": []})

            tag_ids = [int(row["id"]) for row in matched]
            tag_names = [str(row["name"]) for row in matched]

            placeholders = ",".join(["?" for _ in tag_ids])
            scope_clause = ""
            scope_params: List[Any] = []
            if scope:
                if scope == "public":
                    scope_clause = "AND mr.memory_scope = ?"
                    scope_params = ["public"]
                else:
                    scope_clause = "AND (mr.memory_scope = ? OR mr.memory_scope = 'public')"
                    scope_params = [scope]

            sql = f"""
                SELECT
                    mr.id, mr.memory_type, mr.judgment, mr.reasoning,
                    mr.strength, mr.is_active, mr.memory_scope,
                    mr.created_at, mr.updated_at,
                    COUNT(DISTINCT gt.id) AS hit_count,
                    IFNULL(tags.tags_text, '') AS tags
                FROM memory_records mr
                JOIN memory_tag_rel mtr ON mtr.memory_id = mr.id
                JOIN global_tags gt ON gt.id = mtr.tag_id
                LEFT JOIN (
                    SELECT mtr2.memory_id, GROUP_CONCAT(gt2.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr2 JOIN global_tags gt2 ON gt2.id = mtr2.tag_id
                    GROUP BY mtr2.memory_id
                ) tags ON tags.memory_id = mr.id
                WHERE gt.id IN ({placeholders})
                {scope_clause}
                GROUP BY mr.id
                ORDER BY hit_count DESC, mr.strength DESC, mr.created_at DESC
                LIMIT ?
            """
            params: List[Any] = [*tag_ids, *scope_params, limit]
            cur.execute(sql, params)
            rows = cur.fetchall()
            memory_hits = []
            for row in rows:
                memory_hits.append({
                    "id": row["id"],
                    "hit_count": int(row["hit_count"] or 0),
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
                "matched_tags": tag_names,
                "matched_tag_ids": tag_ids,
                "memory_hits": memory_hits,
            })
        except Exception as e:
            return jsonify({"matched_tags": [], "matched_tag_ids": [], "memory_hits": [], "error": str(e)}), 500
        finally:
            conn.close()
