"""
angel_recall - 统一检索工具

一次调用同时检索记忆和笔记切片，合并返回。
"""

from typing import List
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

from ..llm_memory.models.data_models import BaseMemory
from ..core.utils.memory_formatter import MemoryFormatter

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CoreMemoryRecallTool(FunctionTool):
    name: str = "angel_recall"
    description: str = "搜索天使记忆和笔记中的内容。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键字，多个词用空格或|分隔",
                    "minLength": 1,
                },
            },
            "required": ["query"],
        }
    )

    def __post_init__(self):
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        query: str,
    ) -> str:
        if query is None or not str(query).strip():
            return "参数错误：query 为必填且不能为空。"

        query = str(query).strip()
        self.logger.debug(f"{self.name} - LLM 调用: query='{query}'")

        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            return "错误：无法获取插件上下文。"

        plugin_context = event.plugin_context

        try:
            memory_runtime = plugin_context.get_component("memory_runtime")
            if not memory_runtime:
                raise ValueError("memory_runtime 未注册。")
            memory_scope = await plugin_context.resolve_memory_scope_from_event(event)
            if not isinstance(memory_scope, str):
                memory_scope = str(memory_scope)
        except Exception as e:
            self.logger.error(f"{self.name}: 获取上下文失败: {e}")
            return "错误：无法确定当前会话，检索已拒绝。"

        parts: List[str] = []

        # === 记忆检索 ===
        try:
            all_memories: List[BaseMemory] = await memory_runtime.comprehensive_recall(
                query=query,
                limit=50,
                event=event,
                memory_scope=memory_scope,
            )
            all_active = [mem for mem in all_memories if mem.is_active]

            if all_active:
                memory_sql_manager = plugin_context.get_component("memory_sql_manager")
                memory_text = MemoryFormatter.format_session_memories(
                    all_active, short_id_registry=memory_sql_manager
                )
                parts.append(f"[记忆检索结果 ({len(all_active)}条)]\n{memory_text}")
            else:
                parts.append("[记忆检索结果]\n无相关记忆。")
        except Exception as e:
            self.logger.error(f"{self.name}: 记忆检索失败: {e}", exc_info=True)
            parts.append(f"[记忆检索结果]\n检索失败：{e}")

        # === 笔记切片检索（50条，精准过滤后按文件聚合） ===
        try:
            note_service = plugin_context.get_component("note_service")
            if note_service:
                note_results = await note_service.search_notes(query=query, max_results=50)
                # 精准过滤：content 必须包含原始查询词
                if note_results:
                    note_results = self._precise_filter(note_results, query)
                if note_results:
                    # 按文件聚合
                    aggregated = self._aggregate_by_file(note_results)
                    note_lines = []
                    for file_info in aggregated:
                        path = file_info["path"]
                        short_id = file_info["short_id"]
                        chunks = file_info["chunks"]
                        id_hint = f"ID:{short_id}" if short_id >= 0 else ""
                        header = f"[{path}] ({id_hint}, 命中{len(chunks)}处)"
                        chunk_previews = []
                        for c in chunks:
                            preview = c["content"][:80]
                            chunk_previews.append(f"  L:{c['line_start']}-{c['line_end']} | {preview}")
                        note_lines.append(header + "\n" + "\n".join(chunk_previews))
                    parts.append(
                        f"[笔记检索结果 ({len(aggregated)}个文件)]\n"
                        "[展开正文: angel_note_read(note_short_id, offset, limit)]\n\n"
                        + "\n\n".join(note_lines)
                    )
                else:
                    parts.append("[笔记检索结果]\n无相关笔记。")
            else:
                parts.append("[笔记检索结果]\n笔记服务不可用。")
        except Exception as e:
            self.logger.error(f"{self.name}: 笔记检索失败: {e}", exc_info=True)
            parts.append(f"[笔记检索结果]\n检索失败：{e}")

        return "\n\n".join(parts)

    @staticmethod
    def _aggregate_by_file(note_results: List[dict]) -> List[dict]:
        """将切片搜索结果按文件聚合"""
        from collections import OrderedDict
        file_map: OrderedDict = OrderedDict()
        for note in note_results:
            metadata = note.get("metadata", {})
            path = str(metadata.get("source_file_path") or "")
            if not path:
                continue
            if path not in file_map:
                file_map[path] = {
                    "path": path,
                    "short_id": int(metadata.get("note_short_id", -1)),
                    "chunks": [],
                }
            file_map[path]["chunks"].append({
                "line_start": int(metadata.get("line_start", 0)),
                "line_end": int(metadata.get("line_end", 0)),
                "content": str(note.get("content") or "").replace("\n", " "),
            })
        return list(file_map.values())

    @staticmethod
    def _precise_filter(note_results: List[dict], query: str) -> List[dict]:
        """精准过滤：content 必须包含原始查询词（按空格/|拆词，至少命中一个完整词）"""
        import re
        normalized = query.strip().lower()
        terms = [t.strip() for t in re.split(r"\s*\|\s*|\s+", normalized) if t.strip()]
        if not terms:
            return note_results
        filtered = []
        for note in note_results:
            content = str(note.get("content") or "").lower()
            if any(term in content for term in terms):
                filtered.append(note)
        return filtered
