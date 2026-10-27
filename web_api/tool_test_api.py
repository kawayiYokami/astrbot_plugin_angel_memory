"""工具测试 API

两条检索链路各自独立触发，返回的都是原文，不做任何包装：

- 自动 RAG：跑真实的注入流程，把最终会交给模型的上下文原样返回
- 主动搜索：`angel_recall` 工具 `run()` 的返回值原文

真实 `AstrMessageEvent` 在面板里拿不到，这里构造一个只带必要字段的替身。
注入过程写到独立预览会话，不碰任何真实会话的工作缓存。
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, List

from astrbot.api import logger
from astrbot.api.web import json_response, request

if TYPE_CHECKING:
    from ..core.plugin_context import PluginContext

# 面板不传会话时用的示例会话ID，只用于解析 scope
DEFAULT_UMO = "aiocqhttp:GroupMessage:0"
# 注入预览专用会话键：与任何真实会话隔离
PREVIEW_SESSION_ID = "__tool_test_preview__"


def _ok(text: str, errors: List[str] | None = None):
    return json_response({"text": text, "errors": list(errors or [])})


def _format_error(exc: BaseException) -> str:
    """异常带上出错位置，排障时能直接定位。"""
    text = f"{type(exc).__name__}: {exc}"
    lines = [
        line.strip()
        for line in traceback.format_exc().splitlines()
        if line.strip().startswith("File ")
    ]
    return f"{text}（{lines[-1]}）" if lines else text


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

    # === 自动 RAG ===

    async def auto_rag(self):
        payload = self._normalize(await self._payload())
        if not payload["query"]:
            return _ok("", ["查询词不能为空"])

        errors: List[str] = []
        event = PreviewEvent(payload["umo"], payload["group_id"], self._ctx)

        deepmind = self._ctx.get_component("deepmind")
        if deepmind is None:
            return _ok("", ["潜意识组件不可用，插件可能仍在初始化"])

        try:
            retrieval = await deepmind.retrieval_service.retrieve_memories_and_notes(
                event, payload["query"], readonly=True
            )
        except Exception as e:
            logger.error(f"[工具测试] 自动RAG 召回失败: {e}", exc_info=True)
            return _ok("", [f"召回失败：{_format_error(e)}"])

        for inner in retrieval.get("errors") or []:
            errors.append(inner)

        try:
            text = await self._inject_preview(deepmind, event, retrieval)
        except Exception as e:
            logger.error(f"[工具测试] 自动RAG 注入预览失败: {e}", exc_info=True)
            errors.append(f"注入预览失败：{_format_error(e)}")
            text = ""

        return _ok(text, errors)

    async def _inject_preview(self, deepmind, event: PreviewEvent, retrieval: Dict[str, Any]) -> str:
        """跑真实注入流程，返回最终会交给模型的那段上下文。"""
        from astrbot.api.provider import ProviderRequest

        from ..core.utils.note_context_builder import NoteContextBuilder

        memories = retrieval.get("long_term_memories") or []
        candidate_notes = retrieval.get("candidate_notes") or []

        note_context = ""
        selected = candidate_notes[: max(0, int(getattr(deepmind, "note_inject_top_k", 0)))]
        if selected:
            note_context = NoteContextBuilder.build_candidate_list_for_prompt(selected)

        soul_values = None
        if getattr(deepmind, "soul", None):
            try:
                soul_values = deepmind.soul.get_scores()
            except Exception as e:
                logger.warning(f"[工具测试] 取灵魂状态失败: {e}")

        memory_scope = await self._ctx.resolve_memory_scope_from_event(event)
        has_secretary_decision = bool(retrieval.get("secretary_decision"))

        # 注入读的是会话工作缓存，真实链路也是先把召回结果填进去再注入
        deepmind.session_memory_manager.add_memories_to_session(PREVIEW_SESSION_ID, memories)
        try:
            provider_request = ProviderRequest()
            provider_request.system_prompt = ""
            await deepmind.injection_service.inject_memories_to_request(
                provider_request,
                PREVIEW_SESSION_ID,
                note_context,
                soul_values,
                has_secretary_decision,
                memory_scope=memory_scope,
            )
            parts = [
                str(block.get("content") or "")
                for block in (provider_request.contexts or [])
                if isinstance(block, dict)
            ]
            return "\n\n".join(part for part in parts if part)
        finally:
            deepmind.session_memory_manager.clear_session(PREVIEW_SESSION_ID)

    # === 主动搜索 ===

    async def active_recall(self):
        payload = self._normalize(await self._payload())
        if not payload["query"]:
            return _ok("", ["查询词不能为空"])

        from ..tools.angel_recall import CoreMemoryRecallTool

        event = PreviewEvent(payload["umo"], payload["group_id"], self._ctx)

        try:
            raw = await CoreMemoryRecallTool().run(event, payload["query"])
        except Exception as e:
            logger.error(f"[工具测试] 主动搜索 工具执行失败: {e}", exc_info=True)
            return _ok("", [f"工具执行失败：{_format_error(e)}"])

        # 工具内部对记忆段和笔记段各自兜底，失败只写进文本不抛异常
        errors: List[str] = []
        for marker in ("检索失败：", "笔记服务不可用", "检索已拒绝", "错误："):
            if marker in raw:
                errors.append(f"工具返回里出现「{marker}」")
                break

        return _ok(raw, errors)


__all__ = ["ToolTestAPI", "PreviewEvent", "DEFAULT_UMO", "PREVIEW_SESSION_ID"]
