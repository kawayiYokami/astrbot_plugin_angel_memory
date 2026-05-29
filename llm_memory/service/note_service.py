"""笔记服务。

当前笔记检索只走切片 SQLite + Tantivy 搜索；中央 SQL 仅保留文件级
note_short_id 注册表，用于 angel_note_read 定位原始文件。
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from .id_service import IDService
from ..parser.note_chunker import chunk_file
from ..components.note_chunk_search import NoteChunkSearchEngine

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
    def __init__(self, plugin_context=None):
        self.logger = logger
        self.plugin_context = plugin_context

        if plugin_context:
            self.id_service = IDService.from_plugin_context(plugin_context)
            self.logger.info("笔记服务初始化完成（PluginContext模式）")
        else:
            raise ValueError("必须提供 plugin_context 参数")

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
        search_engine = self._get_chunk_search_engine()
        if search_engine is None:
            return []
        raw_results = await search_engine.search_async(query=query, limit=max_results)
        results = self._format_chunk_results(raw_results)
        return self._enrich_with_short_ids(results)

    async def search_notes_by_top_k(
        self,
        query: str,
        recall_count: int = 100,
        top_k: int = 20,
        tag_filter: List[str] = None,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        del tag_filter, vector
        search_engine = self._get_chunk_search_engine()
        if search_engine is None:
            return []
        raw_results = await search_engine.search_async(query=query, limit=max(0, int(top_k)))
        results = self._format_chunk_results(raw_results)
        return self._enrich_with_short_ids(results)

    def _get_chunk_search_engine(self) -> Optional[NoteChunkSearchEngine]:
        """获取切片搜索引擎实例"""
        if self.plugin_context is None:
            return None
        engine = self.plugin_context.get_component("note_chunk_search")
        return engine

    @staticmethod
    def _format_chunk_results(raw_results: List[Dict]) -> List[Dict]:
        """将切片搜索结果格式化为统一的笔记结果格式"""
        formatted: List[Dict] = []
        for r in raw_results:
            source_file_path = str(r.get("source_file_path") or "")
            content = str(r.get("content") or "").strip()
            line_start = int(r.get("line_start", 0))
            line_end = int(r.get("line_end", 0))
            formatted.append({
                "id": f"{r.get('file_id', '')}#{r.get('chunk_index', 0)}",
                "content": content,
                "metadata": {
                    "source_file_path": source_file_path,
                    "file_id": str(r.get("file_id") or ""),
                    "line_start": line_start,
                    "line_end": line_end,
                    "chunk_index": int(r.get("chunk_index", 0)),
                    "note_short_id": -1,
                },
                "tags": [],
                "similarity": float(r.get("score", 0.0)),
            })
        return formatted

    def _enrich_with_short_ids(self, results: List[Dict]) -> List[Dict]:
        """为结果补充 note_short_id"""
        if not results:
            return results
        memory_sql_manager = None
        try:
            memory_sql_manager = self._get_memory_sql_manager()
        except Exception:
            return results
        paths = list({r.get("metadata", {}).get("source_file_path", "") for r in results if r.get("metadata", {}).get("source_file_path")})
        if not paths:
            return results
        path_to_short_id = memory_sql_manager.get_short_ids_by_paths_sync(paths)
        for r in results:
            path = r.get("metadata", {}).get("source_file_path", "")
            if path in path_to_short_id:
                r["metadata"]["note_short_id"] = path_to_short_id[path]
        return results

    async def _search_notes_v2(
        self,
        query: str,
        recall_count: int = 100,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        del recall_count, vector
        return await self.search_notes(query=query)

    def get_note(self, note_id: str) -> Dict:
        raise NoteOperationError(
            f"已废弃按 note_id 读取正文接口（note_id={note_id}）。"
            "请改用 angel_note_read(note_short_id + offset/limit)。"
        )

    def parse_and_store_file_sync(
        self,
        file_path: str,
        relative_path: str = None,
        *,
        update_search_index: bool = True,
    ) -> tuple:
        import time

        timings: Dict[str, float] = {}
        file_obj = Path(file_path)
        if not file_obj.exists():
            self.logger.warning(f"文件不存在: {file_path}")
            return 0, timings

        file_timestamp = int(file_obj.stat().st_mtime)
        if relative_path is None:
            relative_path = file_obj.name

        if file_obj.suffix.lower() not in {".md", ".txt"}:
            self.logger.info(f"跳过不支持的笔记文件类型: {relative_path}")
            return 0, timings

        t0 = time.time()
        file_id = self.id_service.file_to_id(relative_path, file_timestamp)
        timings["id_lookup"] = (time.time() - t0) * 1000

        memory_sql_manager = self._get_memory_sql_manager()

        t0 = time.time()
        total_lines = self._count_total_lines(file_path)
        timings["parse"] = (time.time() - t0) * 1000

        t0 = time.time()
        asyncio.run(
            memory_sql_manager.upsert_note_file_entry(
                file_id=str(file_id),
                source_file_path=relative_path,
                total_lines=total_lines,
                updated_at=file_timestamp,
            )
        )
        timings["store_total"] = (time.time() - t0) * 1000

        # 切片生成与持久化；批量重建时搜索索引由调用方统一重建。
        t0 = time.time()
        chunk_count = self._build_and_store_chunks(
            file_path,
            relative_path,
            str(file_id),
            update_search_index=update_search_index,
            timings=timings,
        )
        timings["chunk_total"] = (time.time() - t0) * 1000
        timings["chunk_count"] = chunk_count

        return chunk_count, timings

    def _build_and_store_chunks(
        self,
        file_path: str,
        relative_path: str,
        file_id: str,
        *,
        update_search_index: bool = True,
        timings: Optional[Dict[str, float]] = None,
    ) -> int:
        """对文件执行切片，写入切片库，并可选更新搜索索引。"""
        if Path(file_path).suffix.lower() not in {".md", ".txt"}:
            return 0
        try:
            import time

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():
                return 0
            t0 = time.time()
            chunks = chunk_file(content, relative_path)
            if timings is not None:
                timings["chunk_generate"] = (time.time() - t0) * 1000
            if not chunks:
                return 0

            count = len(chunks)
            if self.plugin_context is not None:
                chunk_store = self.plugin_context.get_component("note_chunk_store")
                if chunk_store is not None:
                    try:
                        t0 = time.time()
                        count = chunk_store.upsert_chunks(
                            file_id=file_id,
                            source_file_path=relative_path,
                            chunks=chunks,
                        )
                        if timings is not None:
                            timings["chunk_store"] = (time.time() - t0) * 1000
                    except Exception as e:
                        self.logger.warning(f"切片存储写入失败（不影响主流程）: {e}")

            if update_search_index:
                search_engine = self._get_chunk_search_engine()
                if search_engine is not None:
                    t0 = time.time()
                    count = search_engine.index_chunks(
                        file_id=file_id,
                        source_file_path=relative_path,
                        chunks=chunks,
                    )
                    if timings is not None:
                        timings["chunk_search_index"] = (time.time() - t0) * 1000
            return count
        except Exception as e:
            self.logger.warning(f"切片处理失败（不影响主流程）: {e}")
            return 0

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
            asyncio.run(memory_sql_manager.delete_note_index_by_file_id(str(file_id)))

            if self.plugin_context is not None:
                chunk_store = self.plugin_context.get_component("note_chunk_store")
                if chunk_store is not None:
                    try:
                        chunk_store.delete_by_file_id(str(file_id))
                    except Exception as e:
                        self.logger.warning(f"删除切片存储失败（不影响主流程）: {e}")

            search_engine = self._get_chunk_search_engine()
            if search_engine is not None:
                try:
                    search_engine.delete_by_file_id(str(file_id))
                except Exception as e:
                    self.logger.warning(f"删除切片索引失败（不影响主流程）: {e}")
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
            "ready": self.plugin_context is not None,
            "has_vector_store": False,
            "has_plugin_context": self.plugin_context is not None,
            "provider_id": self.id_service.provider_id if hasattr(self, "id_service") else None,
        }
