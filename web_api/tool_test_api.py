"""工具测试 API

一次调用同时跑两条检索链路，返回纯文本，只读观测，不做任何写入：

- 自动 RAG：`DeepMindRetrievalService.retrieve_memories_and_notes`（被动注入前的召回）
- 主动搜索：`angel_recall` 工具本体（LLM 主动调用时看到的那段文本）

真实 `AstrMessageEvent` 在面板里拿不到，这里构造一个只带必要字段的替身；
检索链路对事件只做 `hasattr` 与 getter 取值，不依赖平台适配器。

任何环节出错都写进 `errors` 并原样落在文本里，不吞异常、不伪装成「无结果」。
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List

from astrbot.api import logger
from astrbot.api.web import json_response, request

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

# 面板不传会话时用的示例会话ID，只用于解析 scope
DEFAULT_UMO = "aiocqhttp:GroupMessage:0"


def _ok(data: Dict[str, Any]):
    return json_response(data)


def _fail(message: str):
    """错误也用 200 返回：桥接只认响应体的 data，非 200 会把消息吞成通用错误。"""
    return json_response({"text": "", "errors": [message]})


class PreviewEvent:
    """替身事件：只提供检索链路会读的字段。"""

    def __init__(self, umo: str, group_id: str, plugin_context: "PluginContext"):
        self.unified_msg_origin = umo
        self._group_id = group_id
        self.plugin_context = plugin_context
        self.role = "user"
        self.message_str = ""

    def get_group_id(self) -> str:
        return self._group_id

    def get_platform_name(self) -> str:
        return ""

    def get_sender_id(self) -> str:
        return ""


def _format_error(exc: BaseException, with_trace: bool = False) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if with_trace:
        tail = traceback.format_exception_only(type(exc), exc)[-1].strip()
        frames = traceback.format_exc().strip().splitlines()
        last = [line.strip() for line in frames if line.strip().startswith("File ")]
        if last:
            text = f"{text}（{last[-1]}）"
        else:
            text = tail or text
    return text


class ToolTestAPI:
    """两条检索链路的只读探针。"""

    def __init__(self, plugin_context: "PluginContext"):
        self._ctx = plugin_context

    # === 入参 ===

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

    # === 入口 ===

    async def run(self):
        payload = self._normalize(await self._payload())
        query = payload["query"]
        if not query:
            return _fail("查询词不能为空")

        errors: List[str] = []
        lines: List[str] = []
        event = PreviewEvent(payload["umo"], payload["group_id"], self._ctx)

        for section in (
            self._auto_rag_section,
            self._active_recall_section,
        ):
            try:
                text, section_errors = await section(event, query, payload["group_id"])
            except Exception as e:
                logger.error(f"[工具测试] 段落执行失败: {e}", exc_info=True)
                name = "自动 RAG" if section is self._auto_rag_section else "主动搜索"
                text = f"【{name}】\n[错误] 未捕获异常：{_format_error(e)}"
                section_errors = [f"{name}：{_format_error(e)}"]
            lines.append(text)
            errors.extend(section_errors)

        return _ok({"text": "\n\n".join(lines), "errors": errors})

    # === 自动 RAG ===

    async def _auto_rag_section(self, event: PreviewEvent, query: str, group_id: str):
        lines = ["【自动 RAG】被动注入前的召回"]
        errors: List[str] = []

        deepmind = self._ctx.get_component("deepmind")
        if deepmind is None:
            msg = "潜意识组件不可用，插件可能仍在初始化"
            lines.append(f"[错误] {msg}")
            return "\n".join(lines), [f"自动 RAG：{msg}"]

        scope, scope_error = await self._scope_line(event, lines)
        if scope_error:
            errors.append(f"自动 RAG：{scope_error}")

        started = time.perf_counter()
        retrieval: Dict[str, Any] | None = None
        try:
            retrieval = await deepmind.retrieval_service.retrieve_memories_and_notes(
                event, query, readonly=True
            )
        except Exception as e:
            logger.error(f"[工具测试] 自动RAG 召回失败: {e}", exc_info=True)
            detail = _format_error(e, with_trace=True)
            lines.append(f"[错误] 召回失败：{detail}")
            errors.append(f"自动 RAG 召回失败：{detail}")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        auto_domains = self._ctx.auto_domains(group_id)
        try:
            processed = deepmind.query_processor.process_query_for_memory(query, event)
        except Exception as e:
            processed = query
            detail = _format_error(e)
            lines.append(f"[错误] 查询词预处理失败：{detail}")
            errors.append(f"自动 RAG 查询词预处理失败：{detail}")

        lines.append(
            f"群={group_id or '（未指定，按私聊）'}  耗时={elapsed_ms}ms  "
            f"自动域={'、'.join(auto_domains) if auto_domains else '（无）'}"
        )
        lines.append(f"预处理后查询词：{processed}")

        if not auto_domains:
            lines.append("笔记侧不会召回：这个群没有标记为「自动」的知识域。")

        if retrieval is not None:
            for inner in retrieval.get("errors") or []:
                lines.append(f"[错误] {inner}")
                errors.append(f"自动 RAG {inner}")

            memories = self._memory_lines(retrieval.get("long_term_memories") or [])
            notes = self._note_lines(retrieval.get("candidate_notes") or [])
            lines.append("")
            lines.extend(memories)
            lines.append("")
            lines.extend(notes)

            try:
                preview = self._injection_preview(deepmind, retrieval)
            except Exception as e:
                preview = ""
                detail = _format_error(e)
                lines.append(f"[错误] 组装注入预览失败：{detail}")
                errors.append(f"自动 RAG 注入预览失败：{detail}")
            if preview:
                lines.append("")
                lines.append("--- 注入预览 ---")
                lines.append(preview)

        return "\n".join(lines), errors

    # === 主动搜索 ===

    async def _active_recall_section(self, event: PreviewEvent, query: str, group_id: str):
        lines = ["【主动搜索】angel_recall 工具原文"]
        errors: List[str] = []

        from ..tools.angel_recall import CoreMemoryRecallTool

        scope, scope_error = await self._scope_line(event, lines)
        if scope_error:
            errors.append(f"主动搜索：{scope_error}")

        visible = self._ctx.visible_domains(group_id)
        lines.append(
            f"群={group_id or '（未指定，按私聊）'}  "
            f"可见域={'、'.join(visible) if visible else '（无）'}"
        )

        started = time.perf_counter()
        try:
            raw = await CoreMemoryRecallTool().run(event, query)
        except Exception as e:
            logger.error(f"[工具测试] 主动搜索 工具执行失败: {e}", exc_info=True)
            detail = _format_error(e, with_trace=True)
            lines.append(f"[错误] 工具执行失败：{detail}")
            errors.append(f"主动搜索执行失败：{detail}")
            return "\n".join(lines), errors
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        lines.append(f"耗时={elapsed_ms}ms  字符数={len(raw)}")
        lines.append("")
        lines.append(raw)

        # 工具内部对两段各自兜底，失败只写进文本，不抛异常：这里把它捞成显式错误
        for marker in ("检索失败：", "笔记服务不可用", "检索已拒绝", "错误："):
            if marker in raw:
                errors.append(f"主动搜索：工具返回里出现「{marker}」，详见下方原文")
                break

        return "\n".join(lines), errors

    # === 公共片段 ===

    async def _scope_line(self, event: PreviewEvent, lines: List[str]):
        try:
            persona_name = await self._ctx.get_event_persona_name(event)
            conversation_id = self._ctx.get_event_conversation_id(event)
            scope, matched_by, matched_key = self._ctx.resolve_memory_scope_with_source(
                conversation_id, persona_name=persona_name
            )
        except Exception as e:
            detail = _format_error(e)
            lines.append(f"[错误] 解析记忆域失败：{detail}")
            return "", f"解析记忆域失败：{detail}"

        lines.append(
            f"记忆域={scope}（命中来源 {matched_by}/{matched_key}）  "
            f"会话={conversation_id}"
        )
        return scope, ""

    @staticmethod
    def _memory_lines(memories: List[Any]) -> List[str]:
        lines = [f"记忆 {len(memories)} 条"]
        if not memories:
            lines.append("（无相关记忆）")
        for index, memory in enumerate(memories, 1):
            memory_type = getattr(memory, "memory_type", "")
            type_text = getattr(memory_type, "value", memory_type) or ""
            active = " 主动" if bool(getattr(memory, "is_active", False)) else ""
            lines.append(
                f"{index}. [{type_text}|{getattr(memory, 'memory_scope', '')}|"
                f"强度{getattr(memory, 'strength', 0)}{active}] "
                f"{getattr(memory, 'judgment', '')}"
            )
            reasoning = str(getattr(memory, "reasoning", "") or "")
            if reasoning:
                lines.append(f"   {reasoning}")
            tags = [str(t) for t in (getattr(memory, "tags", []) or [])]
            if tags:
                lines.append(f"   标签：{'、'.join(tags)}")
        return lines

    @staticmethod
    def _note_lines(notes: List[Any]) -> List[str]:
        lines = [f"笔记 {len(notes)} 条"]
        if not notes:
            lines.append("（无相关笔记）")
        for index, note in enumerate(notes or [], 1):
            if not isinstance(note, dict):
                continue
            metadata = note.get("metadata") or {}
            content = str(note.get("content") or "").replace("\n", " ")[:200]
            lines.append(
                f"{index}. [{metadata.get('domain') or '未知域'}] "
                f"ID={metadata.get('note_id')} "
                f"相似度={float(note.get('similarity') or 0):.3f} "
                f"L{metadata.get('line_start')}-{metadata.get('line_end')}"
            )
            lines.append(f"   {metadata.get('source_file_path')}")
            if content:
                lines.append(f"   {content}")
        return lines

    @staticmethod
    def _injection_preview(deepmind, retrieval: Dict[str, Any]) -> str:
        """组装预览：笔记上下文 + 记忆块。不是完整提示词。"""
        parts: List[str] = []
        from ..core.utils.note_context_builder import NoteContextBuilder

        selected = (retrieval.get("candidate_notes") or [])[
            : max(0, int(getattr(deepmind, "note_inject_top_k", 0)))
        ]
        note_context = NoteContextBuilder.build_candidate_list_for_prompt(selected)
        if note_context:
            parts.append(note_context)

        from ..core.utils.memory_formatter import MemoryFormatter

        memories = retrieval.get("long_term_memories") or []
        if memories:
            memory_text = MemoryFormatter.format_session_memories(
                memories, short_id_registry=None
            )
            if memory_text:
                parts.append(memory_text)

        return "\n\n".join(parts)


__all__ = ["ToolTestAPI", "PreviewEvent", "DEFAULT_UMO"]
