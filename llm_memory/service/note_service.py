"""
笔记服务（重构版）

仅保留中央 SQL 索引 + notes_index 轻量向量索引链路。
不再支持“正文全量向量化”旧链路，也不提供回退。
"""

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .id_service import IDService

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class NoteServiceError(Exception):
    pass


class NoteNotFoundError(NoteServiceError):
    pass


class NoteOperationError(NoteServiceError):
    pass


class NoteService:
    def __init__(self, plugin_context=None, vector_store=None):
        self.logger = logger
        self.plugin_context = plugin_context
        self.vector_store = vector_store
        self.notes_index_collection = None
        self.collections_initialized = False

        if plugin_context:
            self.id_service = IDService.from_plugin_context(plugin_context)
            self.logger.info("笔记服务初始化完成（PluginContext模式）")
        elif vector_store:
            self.id_service = IDService()
            self._initialize_collections()
            self.logger.info("笔记服务初始化完成（兼容模式）")
        else:
            raise ValueError("必须提供 plugin_context 或 vector_store 参数")

    def _initialize_collections(self):
        if not self.vector_store:
            return
        self.notes_index_collection = (
            self.vector_store.get_or_create_collection_with_dimension_check("notes_index")
        )
        self.collections_initialized = True

    def set_vector_store(self, vector_store):
        self.vector_store = vector_store
        self._initialize_collections()
        self.logger.info("VectorStore已设置，notes_index集合初始化完成")

    def _get_memory_sql_manager(self):
        if self.plugin_context is None:
            raise RuntimeError("NoteService 缺少 plugin_context")
        memory_sql_manager = self.plugin_context.get_component("memory_sql_manager")
        if memory_sql_manager is None:
            raise RuntimeError("memory_sql_manager 不可用，笔记新链路无法执行")
        return memory_sql_manager

    def ensure_ready(self):
        self._get_memory_sql_manager()
        return True

    async def search_notes(
        self,
        query: str,
        max_results: int = 10,
        tag_filter: List[str] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        del tag_filter, threshold
        rows = await self._search_notes_v2(query=query, recall_count=max_results)
        return rows[:max_results]

    async def search_notes_by_top_k(
        self,
        query: str,
        recall_count: int = 100,
        top_k: int = 20,
        tag_filter: List[str] = None,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        del tag_filter
        candidates = await self._search_notes_v2(query=query, recall_count=recall_count, vector=vector)
        return candidates[: max(0, int(top_k))]

    async def _search_notes_v2(
        self,
        query: str,
        recall_count: int = 100,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        memory_sql_manager = self._get_memory_sql_manager()

        rows: List[Dict] = []
        if self.vector_store is not None and self.notes_index_collection is not None:
            try:
                recalled = await self.vector_store.recall_note_source_ids(
                    collection=self.notes_index_collection,
                    query=query,
                    limit=recall_count,
                    vector=vector,
                    similarity_threshold=0.5,
                )
            except Exception as e:
                # notes_index 在睡眠维护中可能被重建，旧句柄会失效；此处自动刷新并重试一次。
                msg = str(e)
                if "does not exist" in msg or "Error getting collection" in msg:
                    self.logger.warning("notes_index 集合句柄已失效，正在自动刷新并重试。")
                    self.notes_index_collection = (
                        self.vector_store.get_or_create_collection_with_dimension_check("notes_index")
                    )
                    recalled = await self.vector_store.recall_note_source_ids(
                        collection=self.notes_index_collection,
                        query=query,
                        limit=recall_count,
                        vector=vector,
                        similarity_threshold=0.5,
                    )
                else:
                    raise
            source_ids = [sid for sid, _ in recalled]
            score_map = {sid: score for sid, score in recalled}
            if source_ids:
                by_ids = await memory_sql_manager.get_note_index_by_source_ids(source_ids)
                row_map = {str(row.get("source_id")): row for row in by_ids}
                rows = [row_map[sid] for sid in source_ids if sid in row_map]
                for row in rows:
                    row["similarity"] = float(score_map.get(str(row.get("source_id")), 0.0))
        else:
            rows = await memory_sql_manager.search_note_index_by_tags(query=query, limit=recall_count)
            for row in rows:
                row["similarity"] = float(row.get("hit_count", 0))

        return [self._format_note_index_row(row) for row in rows]

    @staticmethod
    def _format_note_index_row(row: Dict) -> Dict:
        tags_text = str(row.get("tags_text") or "")
        tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        h_values = [str(row.get(f"heading_h{i}") or "").strip() for i in range(1, 7)]
        heading_text = " / ".join([h for h in h_values if h]) or "(无标题)"
        source_file_path = str(row.get("source_file_path") or "")
        preview = f"{source_file_path} | {heading_text}"
        return {
            "id": str(row.get("source_id") or ""),
            "content": preview,
            "metadata": {
                "source_id": str(row.get("source_id") or ""),
                "note_short_id": int(row.get("note_short_id") or -1),
                "source_file_path": source_file_path,
                "file_id": str(row.get("file_id") or ""),
                "heading_h1": h_values[0],
                "heading_h2": h_values[1],
                "heading_h3": h_values[2],
                "heading_h4": h_values[3],
                "heading_h5": h_values[4],
                "heading_h6": h_values[5],
                "total_lines": int(row.get("total_lines") or 0),
                "tags": tags_text,
                "updated_at": float(row.get("updated_at") or 0),
            },
            "tags": tags,
            "similarity": float(row.get("similarity", 0.0)),
        }

    def get_note(self, note_id: str) -> Dict:
        raise NoteOperationError(
            f"已废弃按 note_id 读取正文接口（note_id={note_id}）。"
            "请改用 note_recall(note_short_id + 行范围)。"
        )

    def parse_and_store_file_sync(self, file_path: str, relative_path: str = None) -> tuple:
        import time

        timings: Dict[str, float] = {}
        file_obj = Path(file_path)
        if not file_obj.exists():
            self.logger.warning(f"文件不存在: {file_path}")
            return 0, timings

        file_timestamp = int(file_obj.stat().st_mtime)
        if relative_path is None:
            relative_path = file_obj.name

        t0 = time.time()
        file_id = self.id_service.file_to_id(relative_path, file_timestamp)
        timings["id_lookup"] = (time.time() - t0) * 1000

        memory_sql_manager = self._get_memory_sql_manager()

        t0 = time.time()
        asyncio.run(memory_sql_manager.delete_note_index_by_file_id(str(file_id)))
        timings["delete_old_index"] = (time.time() - t0) * 1000

        t0 = time.time()
        entries = self._build_note_index_entries(file_path, relative_path, str(file_id), file_timestamp)
        timings["parse"] = (time.time() - t0) * 1000

        t0 = time.time()
        asyncio.run(memory_sql_manager.upsert_note_index_entries(entries))
        timings["store_total"] = (time.time() - t0) * 1000

        return len(entries), timings

    @staticmethod
    def _entry_to_vector_row(entry: Dict) -> Optional[Dict[str, str]]:
        source_id = str(entry.get("source_id") or "").strip()
        tags = [str(x).strip() for x in (entry.get("tags") or []) if str(x).strip()]
        if not source_id or not tags:
            return None
        return {"id": source_id, "vector_text": " ".join(dict.fromkeys(tags))}

    @staticmethod
    def _extract_path_tags(relative_path: str) -> List[str]:
        normalized = str(relative_path or "").replace("\\", "/").strip("/")
        if not normalized:
            return []
        return [part.strip() for part in normalized.split("/") if part.strip()]

    @staticmethod
    def _safe_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _build_source_id(self, relative_path: str, h_values: List[str], section_index: int) -> str:
        raw = f"{relative_path}|{'|'.join(h_values)}|{section_index}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"note_{digest[:32]}"

    def _build_markdown_entries(
        self,
        content: str,
        relative_path: str,
        file_id: str,
        updated_at: int,
        total_lines: int,
    ) -> List[Dict]:
        lines = content.split("\n")
        header_re = re.compile(r"^(#{1,6})\s+(.+)$")
        headings: List[Tuple[int, str, int]] = []
        for idx, line in enumerate(lines):
            match = header_re.match(line)
            if not match:
                continue
            level = len(match.group(1))
            text = self._safe_text(match.group(2))
            headings.append((level, text, idx))

        path_tags = self._extract_path_tags(relative_path)
        if not headings:
            return [
                {
                    "source_id": self._build_source_id(relative_path, ["", "", "", "", "", ""], 0),
                    "file_id": file_id,
                    "source_file_path": relative_path,
                    "h1": "",
                    "h2": "",
                    "h3": "",
                    "h4": "",
                    "h5": "",
                    "h6": "",
                    "total_lines": int(total_lines),
                    "tags": list(dict.fromkeys(path_tags)),
                    "updated_at": updated_at,
                }
            ]

        # 叶子节点索引：仅为“没有子标题”的标题节点建索引
        entries: List[Dict] = []
        current_h = ["", "", "", "", "", ""]
        leaf_index = 0
        for idx, (level, text, _) in enumerate(headings):
            current_h[level - 1] = text
            for j in range(level, 6):
                current_h[j] = ""

            next_level = headings[idx + 1][0] if idx + 1 < len(headings) else 0
            is_leaf = (idx + 1 >= len(headings)) or (next_level <= level)
            if not is_leaf:
                continue

            leaf_index += 1
            tags = list(dict.fromkeys(path_tags + [h for h in current_h if h]))
            entries.append(
                {
                    "source_id": self._build_source_id(relative_path, current_h, leaf_index),
                    "file_id": file_id,
                    "source_file_path": relative_path,
                    "h1": current_h[0],
                    "h2": current_h[1],
                    "h3": current_h[2],
                    "h4": current_h[3],
                    "h5": current_h[4],
                    "h6": current_h[5],
                    "total_lines": int(total_lines),
                    "tags": tags,
                    "updated_at": updated_at,
                }
            )

        return entries

    @staticmethod
    def _count_total_lines(file_path: str) -> int:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore", newline=None) as f:
                text = f.read()
        except Exception:
            return 0
        if not text:
            return 0
        return len(text.splitlines())

    def _build_note_index_entries(
        self,
        file_path: str,
        relative_path: str,
        file_id: str,
        updated_at: int,
    ) -> List[Dict]:
        total_lines = self._count_total_lines(file_path)
        extension = Path(file_path).suffix.lower()
        if extension in [".md", ".txt"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                content = ""
            if extension == ".md":
                return self._build_markdown_entries(
                    content, relative_path, file_id, updated_at, total_lines
                )

        path_tags = self._extract_path_tags(relative_path)
        return [
            {
                "source_id": self._build_source_id(relative_path, ["", "", "", "", "", ""], 0),
                "file_id": file_id,
                "source_file_path": relative_path,
                "h1": "",
                "h2": "",
                "h3": "",
                "h4": "",
                "h5": "",
                "h6": "",
                "total_lines": int(total_lines),
                "tags": path_tags,
                "updated_at": updated_at,
            }
        ]

    def remove_file_data(self, file_path: str) -> bool:
        try:
            relative_path = Path(file_path).name
            file_id = self.id_service.id_to_file(relative_path)
            if not file_id:
                return True
            return self.remove_file_data_by_file_id(file_id)
        except Exception as e:
            self.logger.error(f"删除文件相关数据失败: {file_path}, 错误: {e}")
            return False

    def remove_file_data_by_file_id(self, file_id: int) -> bool:
        try:
            memory_sql_manager = self._get_memory_sql_manager()
            source_ids = asyncio.run(memory_sql_manager.delete_note_index_by_file_id(str(file_id)))
            if self.vector_store is not None and self.notes_index_collection is not None and source_ids:
                try:
                    self.notes_index_collection.delete(ids=source_ids)
                except Exception as e:
                    self.logger.warning(f"删除 notes_index 缓存失败（不影响主流程）: {e}")
            return True
        except Exception as e:
            self.logger.error(f"根据file_id删除文件数据失败: {file_id}, 错误: {e}")
            return False

    def close(self):
        try:
            if hasattr(self, "id_service"):
                self.id_service.close()
            self.logger.debug("笔记服务已关闭")
        except Exception as e:
            self.logger.error(f"关闭笔记服务失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def from_plugin_context(cls, plugin_context):
        return cls(plugin_context=plugin_context)

    def get_status(self):
        return {
            "ready": self.collections_initialized,
            "has_vector_store": self.vector_store is not None,
            "has_plugin_context": self.plugin_context is not None,
            "provider_id": self.id_service.provider_id if hasattr(self, "id_service") else None,
        }
