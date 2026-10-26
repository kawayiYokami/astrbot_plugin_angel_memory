"""
笔记相关 API（p-memory 版）

聊天链路已切到 `note_index`（NotesStore），本文件三个读接口同样改到它：
- 浏览 / 关键词：走 `search_notes_async` 或文件漫步，形状兼容旧前端
  （`note_short_id` 即 `note_id`，标题/行数现场读文件）。
- 单篇读取：`note_id` 定位，兼容旧 `note_short_id` 参数名。
- 切片搜索 / 统计：走 `note_index`，旧 `note_chunk_*` 组件已不存在。
"""

from __future__ import annotations

import asyncio
import math
import os
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

    def _get_raw_dir(self) -> Path:
        path_manager = self.plugin_context.get_path_manager()
        return path_manager.get_raw_dir()

    def _note_index(self):
        try:
            return self.plugin_context.get_component("note_index")
        except Exception:
            return None

    # === 文件元信息（现场读，标题取首个 # 行） ===

    @staticmethod
    def _file_meta(abs_path: Path) -> Dict[str, Any]:
        try:
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {"total_lines": 0, "heading_h1": "", "updated_at": 0}
        lines = text.splitlines()
        heading = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if heading:
                    break
        if not heading:
            for line in lines:
                if line.strip():
                    heading = line.strip()[:80]
                    break
        try:
            mtime = int(abs_path.stat().st_mtime)
        except OSError:
            mtime = 0
        return {"total_lines": len(lines), "heading_h1": heading, "updated_at": mtime}

    def _resolve_note_id(self, note_index, rel_path: str, abs_path: Path) -> Optional[int]:
        """尽力解析 note_id，失败返回 None（文件尚未索引）。"""
        try:
            domain = note_index.domain_for_path(abs_path, self._get_raw_dir())
        except Exception:
            return None
        if not domain:
            return None
        try:
            return note_index._notes.resolve_note_id(domain, abs_path)
        except Exception:
            return None

    def _item_for_file(
        self,
        source_file_path: str,
        note_id: int,
        preview: str = "",
        score: float = 0.0,
        domain: str = "",
    ) -> Dict[str, Any]:
        raw_dir = self._get_raw_dir()
        abs_path = (raw_dir / source_file_path.replace("/", os.sep))
        meta = self._file_meta(abs_path) if abs_path.exists() else {
            "total_lines": 0, "heading_h1": "", "updated_at": 0,
        }
        return {
            "source_id": "",
            "note_id": int(note_id),
            "note_short_id": int(note_id),
            "source_file_path": source_file_path,
            "file_id": "",
            "heading_h1": meta["heading_h1"],
            "heading_h2": "",
            "heading_h3": "",
            "heading_h4": "",
            "heading_h5": "",
            "heading_h6": "",
            "total_lines": int(meta["total_lines"]),
            "updated_at": meta["updated_at"],
            "tags_text": "",
            "domain": domain,
            "content": str(preview or "")[:200],
            "score": float(score or 0.0),
        }

    async def browse_notes(self):
        """分页浏览笔记：有关键词走索引搜，无关键词漫步文件。"""
        keyword = request.args.get("keyword", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        offset = (page - 1) * page_size

        note_index = self._note_index()
        if note_index is None:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size,
                            "error": "笔记索引服务不可用"}), 503

        # 有关键词：走 p-memory 索引，按文件聚合后分页
        if keyword.strip():
            try:
                requested_end = offset + page_size
                search_limit = min(
                    self.KEYWORD_SEARCH_MAX_CANDIDATES,
                    max(page_size * 3, requested_end + 1),
                )
                results = await note_index.search_notes_async(
                    query=keyword.strip(),
                    max_results=search_limit,
                    domains=None,
                )
                grouped: Dict[str, Dict[str, Any]] = {}
                for row in results:
                    metadata = row.get("metadata", {}) or {}
                    path = str(metadata.get("source_file_path") or "")
                    if not path or path in grouped:
                        continue
                    grouped[path] = {
                        "path": path,
                        "note_id": int(metadata.get("note_id", 0) or 0),
                        "score": float(row.get("similarity", 0.0) or 0.0),
                        "preview": str(row.get("content") or ""),
                        "domain": str(metadata.get("domain") or ""),
                    }
                files = list(grouped.values())
                total = len(files)
                paged = files[offset:offset + page_size]
                items = [
                    self._item_for_file(
                        f["path"], f["note_id"], f["preview"], f["score"], f["domain"]
                    )
                    for f in paged
                ]
                return jsonify({
                    "items": items,
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": math.ceil(total / page_size) if total > 0 else 1,
                    "search_mode": "note_index",
                    "has_more": False,
                    "total_is_estimated": False,
                })
            except Exception as e:
                logger.warning(f"笔记索引搜索失败: {e}")
                return jsonify({"items": [], "total": 0, "page": page,
                                "page_size": page_size, "error": str(e)}), 500

        # 无关键词：漫步 raw/ 文件，尽力补 note_id
        raw_dir = self._get_raw_dir()
        if not raw_dir.exists():
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})
        all_files: List[str] = []
        for root, _, filenames in os.walk(str(raw_dir)):
            for name in filenames:
                if name.endswith((".md", ".txt")):
                    abs_path = os.path.join(root, name)
                    rel_path = os.path.relpath(abs_path, str(raw_dir)).replace("\\", "/")
                    all_files.append(rel_path)
        all_files.sort()
        total = len(all_files)
        paged_paths = all_files[offset:offset + page_size]
        items = []
        for idx, rel_path in enumerate(paged_paths):
            abs_path = raw_dir / rel_path.replace("/", os.sep)
            note_id = await asyncio.to_thread(
                self._resolve_note_id, note_index, rel_path, abs_path
            )
            # 尚未索引的文件给唯一的负数占位，保证表格 row-key 不重复；
            # 这类 id 调 recall 会 404，索引追平后自动变成正数实 id。
            if note_id is None:
                note_id = -(offset + idx + 1)
            items.append(self._item_for_file(rel_path, int(note_id)))
        return jsonify({
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": math.ceil(total / page_size) if total > 0 else 1,
        })

    async def recall_note(self):
        """模拟 angel_note_read：兼容 note_id / note_short_id 两种参数名。"""
        data = await request.get_json(silent=True) or {}
        raw_id = data.get("note_id", data.get("note_short_id", -1))
        try:
            note_id = int(raw_id)
        except (TypeError, ValueError):
            return jsonify({"error": f"无效的 note_id={raw_id}"}), 400
        offset = int(data.get("offset", data.get("start_line", 1)))
        if "limit" in data:
            limit = int(data.get("limit", 200))
        else:
            legacy_end = int(data.get("end_line", offset + 199))
            limit = max(1, legacy_end - offset + 1)

        note_index = self._note_index()
        if note_index is None:
            return jsonify({"error": "笔记索引服务不可用"}), 503
        if note_id <= 0:
            return jsonify({"error": f"未找到 note_id={note_id}"}), 404

        target = note_index.note_file(note_id)
        if target is None:
            return jsonify({"error": f"未找到 note_id={note_id}"}), 404
        rel_path = note_index.relative_path(note_id) or target.name

        raw_dir = self._get_raw_dir()
        try:
            resolved = target.resolve()
        except OSError:
            return jsonify({"error": f"文件不存在: {rel_path}"}), 404
        raw_root = raw_dir.resolve()
        if raw_root not in resolved.parents and resolved != raw_root:
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
            "note_id": note_id,
            "note_short_id": note_id,
            "source_file_path": rel_path,
            "total_lines": total_lines,
            "actual_start_line": actual_start,
            "actual_end_line": actual_end,
        })

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
        """切片搜索 API（走 note_index，不再依赖旧切片引擎）。"""
        query = request.args.get("query", "").strip()
        limit = min(100, max(1, int(request.args.get("limit", 20))))

        if not query:
            return jsonify({"error": "缺少 query 参数"}), 400

        note_index = self._note_index()
        if note_index is None:
            return jsonify({"error": "笔记索引服务不可用"}), 503

        try:
            results = await note_index.search_notes_async(
                query=query, max_results=limit, domains=None
            )
            items = []
            for r in results:
                metadata = r.get("metadata", {}) or {}
                items.append({
                    "source_file_path": metadata.get("source_file_path", ""),
                    "file_id": "",
                    "note_id": int(metadata.get("note_id", 0) or 0),
                    "chunk_index": 0,
                    "line_start": int(metadata.get("line_start", 0) or 0),
                    "line_end": int(metadata.get("line_end", 0) or 0),
                    "content": str(r.get("content") or ""),
                    "score": float(r.get("similarity", 0.0) or 0.0),
                    "domain": str(metadata.get("domain") or ""),
                })
            return jsonify({"items": items, "total": len(items), "query": query})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    async def chunk_stats(self):
        """索引统计（走 note_index，旧切片库已下线）。"""
        note_index = self._note_index()
        if note_index is None:
            return jsonify({
                "search_engine_available": False,
                "chunk_store_available": False,
                "chunk_count": 0,
                "file_count": 0,
            })
        try:
            status = note_index.get_status()
        except Exception:
            status = {}
        return jsonify({
            "search_engine_available": True,
            "chunk_store_available": False,
            "chunk_count": 0,
            "file_count": int(status.get("tracked_files", 0) or 0),
            "domains": status.get("domains", []),
            "raw_dir": str(status.get("raw_dir", "")),
        })
