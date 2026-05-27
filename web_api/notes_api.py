"""
笔记相关 API
"""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class NotesAPI:
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

    def _get_raw_dir(self) -> Path:
        path_manager = self.plugin_context.get_path_manager()
        return path_manager.get_raw_dir()

    async def browse_notes(self):
        """分页浏览笔记索引"""
        keyword = request.args.get("keyword", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})

        try:
            where_sql = ""
            params: List[Any] = []
            if keyword.strip():
                kw = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where_sql = (
                    "WHERE (nir.source_file_path LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h1,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h2,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h3,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h4,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h5,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(nir.heading_h6,'') LIKE ? ESCAPE '\\' "
                    "OR IFNULL(tags.tags_text, '') LIKE ? ESCAPE '\\')"
                )
                like = f"%{kw}%"
                params.extend([like] * 8)

            # 总数
            sql_count = f"""
                SELECT COUNT(*) AS cnt
                FROM note_index_records nir
                LEFT JOIN (
                    SELECT ntr.source_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM note_tag_rel ntr JOIN global_tags gt ON gt.id = ntr.tag_id
                    GROUP BY ntr.source_id
                ) tags ON tags.source_id = nir.source_id
                {where_sql}
            """
            cur = conn.cursor()
            cur.execute(sql_count, params)
            total = int(cur.fetchone()["cnt"])

            # 数据
            sql = f"""
                SELECT
                    nir.source_id, IFNULL(nir.note_short_id, -1) AS note_short_id,
                    nir.file_id, nir.source_file_path,
                    IFNULL(nir.heading_h1, '') AS heading_h1,
                    IFNULL(nir.heading_h2, '') AS heading_h2,
                    IFNULL(nir.heading_h3, '') AS heading_h3,
                    IFNULL(nir.heading_h4, '') AS heading_h4,
                    IFNULL(nir.heading_h5, '') AS heading_h5,
                    IFNULL(nir.heading_h6, '') AS heading_h6,
                    IFNULL(nir.total_lines, 0) AS total_lines,
                    nir.updated_at,
                    IFNULL(tags.tags_text, '') AS tags_text
                FROM note_index_records nir
                LEFT JOIN (
                    SELECT ntr.source_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM note_tag_rel ntr JOIN global_tags gt ON gt.id = ntr.tag_id
                    GROUP BY ntr.source_id
                ) tags ON tags.source_id = nir.source_id
                {where_sql}
                ORDER BY nir.updated_at DESC, nir.source_file_path ASC
                LIMIT ? OFFSET ?
            """
            cur.execute(sql, params + [page_size, offset])
            items = [dict(row) for row in cur.fetchall()]

            return jsonify({
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(total / page_size) if total > 0 else 1,
            })
        except Exception as e:
            return jsonify({"items": [], "total": 0, "error": str(e)}), 500
        finally:
            conn.close()

    async def recall_note(self):
        """模拟 note_recall"""
        data = await request.get_json()
        note_short_id = int(data.get("note_short_id", -1))
        start_line = int(data.get("start_line", 1))
        end_line = int(data.get("end_line", 200))

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"error": "数据库不可用"}), 500

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT source_file_path FROM note_index_records WHERE note_short_id = ? LIMIT 1",
                (note_short_id,),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": f"未找到 note_short_id={note_short_id}"}), 404

            rel_path = str(row["source_file_path"] or "").replace("\\", "/").strip().lstrip("/")
            raw_dir = self._get_raw_dir()
            target = (raw_dir / rel_path).resolve()
            raw_root = raw_dir.resolve()

            if raw_root not in target.parents and target != raw_root:
                return jsonify({"error": "路径越界"}), 400

            if not target.exists():
                return jsonify({"error": f"文件不存在: {rel_path}"}), 404

            text = target.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            total_lines = len(lines)

            if start_line > end_line:
                return jsonify({"error": "start_line 不能大于 end_line"}), 400

            actual_start = min(max(1, start_line), total_lines) if total_lines > 0 else 0
            actual_end = min(max(1, end_line), total_lines) if total_lines > 0 else 0
            content = "\n".join(lines[actual_start - 1:actual_end]) if total_lines > 0 else ""

            return jsonify({
                "content": content,
                "note_short_id": note_short_id,
                "source_file_path": rel_path,
                "total_lines": total_lines,
                "actual_start_line": actual_start,
                "actual_end_line": actual_end,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    async def list_note_files(self):
        """列出笔记文件"""
        raw_dir = self._get_raw_dir()
        if not raw_dir.exists():
            return jsonify({"files": [], "error": f"目录不存在: {raw_dir}"})

        files = []
        for root, _, filenames in os.walk(str(raw_dir)):
            for f in filenames:
                if f.endswith((".md", ".txt")):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, str(raw_dir)).replace("\\", "/")
                    files.append(rel_path)
        files.sort()
        return jsonify({"files": files})

    async def get_file_content(self):
        """获取笔记文件内容"""
        rel_path = request.args.get("path", "").strip()
        if not rel_path:
            return jsonify({"error": "缺少 path 参数"}), 400

        raw_dir = self._get_raw_dir()
        target = (raw_dir / rel_path.replace("/", os.sep)).resolve()
        raw_root = raw_dir.resolve()

        if raw_root not in target.parents and target != raw_root:
            return jsonify({"error": "路径越界"}), 400

        if not target.exists():
            return jsonify({"error": f"文件不存在: {rel_path}"}), 404

        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
            return jsonify({"content": content, "path": rel_path})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
