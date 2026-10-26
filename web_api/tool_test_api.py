"""工具测试 API

面板上手动触发两条检索链路，只读观测，不做任何写入：

- 自动 RAG：`DeepMindRetrievalService.retrieve_memories_and_notes`（被动注入前的召回）
- 主动搜索：`angel_recall` 工具本体（LLM 主动调用时看到的那段文本）

真实 `AstrMessageEvent` 在面板里拿不到，这里构造一个只带必要字段的替身；
检索链路对事件只做 `hasattr` 与 getter 取值，不依赖平台适配器。
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

_OK = lambda data=None, message="": json_response(
    {"status": "ok", "message": message, "data": data}
)
_ERR = lambda message, status_code=400: error_response(message, status_code=status_code)

# 面板不传会话时用的示例会话ID，只用于解析 scope
DEFAULT_UMO = "aiocqhttp:GroupMessage:0"

_MEMORY_COUNT_RE = re.compile(r"\[记忆检索结果 \((\d+)条\)\]")
_NOTE_COUNT_RE = re.compile(r"\[笔记检索结果 \((\d+)个文件\)\]")


class PreviewEvent:
    """替身事件：只提供检索链路会读的字段。"""

    def __init__(self, umo: str, group_id: str, plugin_context: "PluginContext"):
        self.unified_msg_origin = umo
        self._group_id = group_id
        self.plugin_context = plugin_context
        # 面板里没有天使之心上下文：不设这两个属性，检索侧按 `hasattr` 判断会自然跳过
        self.role = "user"
        self.message_str = ""

    def get_group_id(self) -> str:
        return self._group_id

    def get_platform_name(self) -> str:
        return ""

    def get_sender_id(self) -> str:
        return ""


class ToolTestAPI:
    """两条检索链路的只读探针。"""

    def __init__(self, plugin_context: "PluginContext"):
        self._ctx = plugin_context

    # === 入参解析 ===

    @staticmethod
    async def _payload() -> Dict[str, Any]:
        data = await request.json(default=None)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _normalize(payload: Dict[str, Any]) -> Dict[str, str]:
        return {
            "query": str(payload.get("query") or "").strip(),
            "umo": str(payload.get("umo") or "").strip() or DEFAULT_UMO,
            "group_id": str(payload.get("group_id") or "").strip(),
        }

    async def _scope_info(self, event: PreviewEvent) -> Dict[str, Any]:
        """解析记忆域并带上命中来源，供页面判断 scope 是否退化了。"""
        persona_name = await self._ctx.get_event_persona_name(event)
        conversation_id = self._ctx.get_event_conversation_id(event)
        scope, matched_by, matched_key = self._ctx.resolve_memory_scope_with_source(
            conversation_id, persona_name=persona_name
        )
        return {
            "memory_scope": scope,
            "matched_by": matched_by,
            "matched_key": matched_key,
            "persona_name": persona_name,
            "conversation_id": conversation_id,
        }

    # === 自动 RAG ===

    async def auto_rag(self):
        payload = self._normalize(await self._payload())
        query = payload["query"]
        if not query:
            return _ERR("query 不能为空")

        deepmind = self._ctx.get_component("deepmind")
        if deepmind is None:
            return _ERR("潜意识组件不可用，插件可能仍在初始化")

        event = PreviewEvent(payload["umo"], payload["group_id"], self._ctx)

        try:
            scope_info = await self._scope_info(event)
        except Exception as e:
            logger.warning(f"[工具测试] 自动RAG 解析记忆域失败: {e}")
            return _ERR(f"解析记忆域失败：{e}")

        started = time.perf_counter()
        try:
            retrieval = await deepmind.retrieval_service.retrieve_memories_and_notes(
                event, query, readonly=True
            )
        except Exception as e:
            logger.error(f"[工具测试] 自动RAG 召回失败: {e}", exc_info=True)
            return _OK(
                {
                    **scope_info,
                    "query": query,
                    "processed_query": "",
                    "auto_domains": [],
                    "memories": [],
                    "notes": [],
                    "injection_preview": "",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(e).__name__}: {e}",
                },
                message="召回失败",
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        memories = self._memory_items(retrieval.get("long_term_memories") or [])
        notes = self._note_items(retrieval.get("candidate_notes") or [])

        auto_domains = self._ctx.auto_domains(payload["group_id"])
        processed_query = ""
        try:
            processed_query = deepmind.query_processor.process_query_for_memory(query, event)
        except Exception as e:
            logger.warning(f"[工具测试] 自动RAG 查询词预处理失败: {e}")

        return _OK(
            {
                **scope_info,
                "query": query,
                "processed_query": processed_query,
                "auto_domains": auto_domains,
                "note_recall_enabled": bool(getattr(deepmind, "note_recall_enabled", True)),
                "note_inject_top_k": int(getattr(deepmind, "note_inject_top_k", 0)),
                "note_candidate_top_k": int(getattr(deepmind, "note_candidate_top_k", 0)),
                "memories": memories,
                "notes": notes,
                "injection_preview": self._injection_preview(
                    deepmind, retrieval
                ),
                "elapsed_ms": elapsed_ms,
                "error": None,
            }
        )

    # === 主动搜索 ===

    async def active_recall(self):
        payload = self._normalize(await self._payload())
        query = payload["query"]
        if not query:
            return _ERR("query 不能为空")

        from ..tools.angel_recall import CoreMemoryRecallTool

        event = PreviewEvent(payload["umo"], payload["group_id"], self._ctx)

        try:
            scope_info = await self._scope_info(event)
        except Exception as e:
            logger.warning(f"[工具测试] 主动搜索 解析记忆域失败: {e}")
            return _ERR(f"解析记忆域失败：{e}")

        visible = self._ctx.visible_domains(payload["group_id"])

        started = time.perf_counter()
        try:
            raw = await CoreMemoryRecallTool().run(event, query)
        except Exception as e:
            logger.error(f"[工具测试] 主动搜索 工具执行失败: {e}", exc_info=True)
            return _OK(
                {
                    **scope_info,
                    "query": query,
                    "visible_domains": visible,
                    "raw": "",
                    "memory_count": None,
                    "note_file_count": None,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(e).__name__}: {e}",
                },
                message="工具执行失败",
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        return _OK(
            {
                **scope_info,
                "query": query,
                "visible_domains": visible,
                "raw": raw,
                "chars": len(raw),
                "memory_count": self._count(_MEMORY_COUNT_RE, raw),
                "note_file_count": self._count(_NOTE_COUNT_RE, raw),
                "elapsed_ms": elapsed_ms,
                "error": None,
            }
        )

    # === 形状转换 ===

    @staticmethod
    def _count(pattern: re.Pattern, text: str):
        match = pattern.search(text or "")
        return int(match.group(1)) if match else None

    @staticmethod
    def _memory_items(memories: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for memory in memories:
            memory_type = getattr(memory, "memory_type", "")
            items.append(
                {
                    "id": str(getattr(memory, "id", "") or ""),
                    "judgment": str(getattr(memory, "judgment", "") or ""),
                    "reasoning": str(getattr(memory, "reasoning", "") or ""),
                    "memory_type": getattr(memory_type, "value", memory_type) or "",
                    "memory_scope": str(getattr(memory, "memory_scope", "") or ""),
                    "strength": int(getattr(memory, "strength", 0) or 0),
                    "is_active": bool(getattr(memory, "is_active", False)),
                    "tags": [str(t) for t in (getattr(memory, "tags", []) or [])],
                    "similarity": float(getattr(memory, "similarity", 0.0) or 0.0),
                }
            )
        return items

    @staticmethod
    def _note_items(notes: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for note in notes or []:
            if not isinstance(note, dict):
                continue
            metadata = note.get("metadata") or {}
            content = str(note.get("content") or "")
            items.append(
                {
                    "path": str(metadata.get("source_file_path") or ""),
                    "note_id": int(metadata.get("note_id") or -1),
                    "domain": str(metadata.get("domain") or ""),
                    "line_start": int(metadata.get("line_start") or 0),
                    "line_end": int(metadata.get("line_end") or 0),
                    "similarity": float(note.get("similarity") or 0.0),
                    "content": content[:200],
                }
            )
        return items

    @staticmethod
    def _injection_preview(deepmind, retrieval: Dict[str, Any]) -> str:
        """组装预览：笔记上下文 + 记忆块。不是完整提示词。"""
        parts: List[str] = []
        try:
            from ..core.utils.note_context_builder import NoteContextBuilder

            selected = (retrieval.get("candidate_notes") or [])[
                : max(0, int(getattr(deepmind, "note_inject_top_k", 0)))
            ]
            note_context = NoteContextBuilder.build_candidate_list_for_prompt(selected)
            if note_context:
                parts.append(note_context)
        except Exception as e:
            logger.warning(f"[工具测试] 组装笔记上下文失败: {e}")

        try:
            from ..core.utils.memory_formatter import MemoryFormatter

            memories = retrieval.get("long_term_memories") or []
            if memories:
                memory_text = MemoryFormatter.format_session_memories(
                    memories, short_id_registry=None
                )
                if memory_text:
                    parts.append(memory_text)
        except Exception as e:
            logger.warning(f"[工具测试] 组装记忆块失败: {e}")

        return "\n\n".join(parts)


__all__ = ["ToolTestAPI", "PreviewEvent"]
