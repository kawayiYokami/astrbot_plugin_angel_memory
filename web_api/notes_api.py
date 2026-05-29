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
    KEYWORD_SEARCH_MAX_CANDIDATES = 500

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

    def _load_note_registry_map(self, paths: List[str], file_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """按路径和 file_id 读取笔记注册表，用于补齐切片搜索结果。"""
        normalized_paths = sorted({
            str(path or "").replace("\\", "/").strip().lstrip("/")
            for path in paths
            if str(path or "").strip()
        })
        normalized_file_ids = sorted({
            str(file_id or "").strip()
            for file_id in file_ids
            if str(file_id or "").strip()
        })
        if not normalized_paths and not normalized_file_ids:
            return {}

        conn = self._get_central_conn()
        if not conn:
            return {}

        try:
            clauses: List[str] = []
            params: List[Any] = []
            if normalized_paths:
                placeholders = ",".join(["?" for _ in normalized_paths])
                clauses.append(f"source_file_path IN ({placeholders})")
                params.extend(normalized_paths)
            if normalized_file_ids:
                placeholders = ",".join(["?" for _ in normalized_file_ids])
                clauses.append(f"file_id IN ({placeholders})")
                params.extend(normalized_file_ids)
            where_sql = " OR ".join(clauses)
            rows = conn.execute(
                f"""
                SELECT
                    source_id, IFNULL(note_short_id, -1) AS note_short_id,
                    file_id, source_file_path,
                    IFNULL(heading_h1, '') AS heading_h1,
                    IFNULL(heading_h2, '') AS heading_h2,
                    IFNULL(heading_h3, '') AS heading_h3,
                    IFNULL(heading_h4, '') AS heading_h4,
                    IFNULL(heading_h5, '') AS heading_h5,
                    IFNULL(heading_h6, '') AS heading_h6,
                    IFNULL(total_lines, 0) AS total_lines,
                    updated_at,
                    '' AS tags_text
                FROM note_index_records
                WHERE {where_sql}
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            path = str(item.get("source_file_path") or "").replace("\\", "/").strip().lstrip("/")
            file_id = str(item.get("file_id") or "").strip()
            if path:
                result[f"path:{path}"] = item
            if file_id:
                result[f"file:{file_id}"] = item
        return result

    async def browse_notes(self):
        """分页浏览笔记索引（使用切片搜索）"""
        keyword = request.args.get("keyword", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        # 如果有关键词，优先使用切片搜索
        if keyword.strip():
            search_engine = self.plugin_context.get_component("note_chunk_search")
            if search_engine is not None:
                try:
                    requested_end = offset + page_size
                    search_limit = min(
                        self.KEYWORD_SEARCH_MAX_CANDIDATES,
                        max(page_size * 3, requested_end + 1),
                    )
                    results = await search_engine.search_async(
                        query=keyword.strip(),
                        limit=search_limit,
                        max_chunks_per_file=1,
                    )
                    has_more = len(results) > requested_end
                    paged = results[offset:offset + page_size]
                    registry_map = self._load_note_registry_map(
                        [str(r.get("source_file_path") or "") for r in paged],
                        [str(r.get("file_id") or "") for r in paged],
                    )
                    items = []
                    for r in paged:
                        source_file_path = str(r.get("source_file_path") or "").replace("\\", "/").strip().lstrip("/")
                        file_id = str(r.get("file_id") or "").strip()
                        registry = (
                            registry_map.get(f"path:{source_file_path}")
                            or registry_map.get(f"file:{file_id}")
                            or {}
                        )
                        items.append({
                            "source_id": registry.get("source_id", ""),
                            "note_short_id": int(registry.get("note_short_id", -1) or -1),
                            "source_file_path": source_file_path,
                            "file_id": file_id,
                            "heading_h1": registry.get("heading_h1", ""),
                            "heading_h2": registry.get("heading_h2", ""),
                            "heading_h3": registry.get("heading_h3", ""),
                            "heading_h4": registry.get("heading_h4", ""),
                            "heading_h5": registry.get("heading_h5", ""),
                            "heading_h6": registry.get("heading_h6", ""),
                            "total_lines": int(registry.get("total_lines", 0) or 0),
                            "updated_at": registry.get("updated_at"),
                            "tags_text": registry.get("tags_text", ""),
                            "chunk_index": r.get("chunk_index", 0),
                            "line_start": r.get("line_start", 0),
                            "line_end": r.get("line_end", 0),
                            "content": str(r.get("content") or "")[:200],
                            "score": float(r.get("score", 0.0)),
                        })
                    total = offset + len(items) + (1 if has_more else 0)
                    return jsonify({
                        "items": items,
                        "total": total,
                        "page": page,
                        "page_size": page_size,
                        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
                        "search_mode": "chunk",
                        "has_more": has_more,
                        "total_is_estimated": has_more or search_limit >= self.KEYWORD_SEARCH_MAX_CANDIDATES,
                    })
                except Exception as e:
                    logger.warning(f"切片搜索失败，降级到 SQL 浏览: {e}")

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})

        try:
            where_sql = ""
            params: List[Any] = []
            if keyword.strip():
                kw = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where_sql = (
                    "WHERE nir.source_file_path LIKE ? ESCAPE '\\'"
                )
                like = f"%{kw}%"
                params.append(like)

            # 总数
            sql_count = f"""
                SELECT COUNT(*) AS cnt
                FROM note_index_records nir
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
                    '' AS tags_text
                FROM note_index_records nir
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
        """模拟 angel_note_read"""
        data = await request.get_json()
        note_short_id = int(data.get("note_short_id", -1))
        offset = int(data.get("offset", data.get("start_line", 1)))
        if "limit" in data:
            limit = int(data.get("limit", 200))
        else:
            legacy_end = int(data.get("end_line", offset + 199))
            limit = max(1, legacy_end - offset + 1)

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

            actual_start = min(max(1, offset), total_lines) if total_lines > 0 else 0
            actual_end = min(actual_start + max(1, limit) - 1, total_lines) if total_lines > 0 else 0
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

    async def search_chunks(self):
        """切片搜索 API"""
        query = request.args.get("query", "").strip()
        limit = min(100, max(1, int(request.args.get("limit", 20))))

        if not query:
            return jsonify({"error": "缺少 query 参数"}), 400

        search_engine = self.plugin_context.get_component("note_chunk_search")
        if search_engine is None:
            return jsonify({"error": "切片搜索引擎不可用"}), 503

        try:
            results = await search_engine.search_async(query=query, limit=limit)
            items = []
            for r in results:
                items.append({
                    "source_file_path": r.get("source_file_path", ""),
                    "file_id": r.get("file_id", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "line_start": r.get("line_start", 0),
                    "line_end": r.get("line_end", 0),
                    "content": str(r.get("content") or ""),
                    "score": float(r.get("score", 0.0)),
                })
            return jsonify({"items": items, "total": len(items), "query": query})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    async def chunk_stats(self):
        """切片索引统计"""
        search_engine = self.plugin_context.get_component("note_chunk_search")
        chunk_store = self.plugin_context.get_component("note_chunk_store")

        stats = {
            "search_engine_available": search_engine is not None,
            "chunk_store_available": chunk_store is not None,
            "chunk_count": 0,
            "file_count": 0,
        }

        if chunk_store is not None:
            try:
                store_stats = chunk_store.get_stats()
                stats["chunk_count"] = store_stats.get("total_chunks", 0)
                stats["file_count"] = store_stats.get("total_files", 0)
            except Exception:
                pass

        return jsonify(stats)
