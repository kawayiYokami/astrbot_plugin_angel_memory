import json
from typing import Any, Dict, List

from astrbot.api.event import AstrMessageEvent


class DeepMindRetrievalService:
    """DeepMind 的检索相关职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    def parse_memory_context(
        self, event: AstrMessageEvent
    ) -> Dict[str, Any] | None:
        if not hasattr(event, "angelmemory_context"):
            return None

        if event.angelmemory_context is None:
            return None

        try:
            context_data = json.loads(event.angelmemory_context)
            return {
                "session_id": context_data["session_id"],
                "query": context_data.get("recall_query", ""),
                "user_list": context_data.get("user_list", []),
                "raw_chat_records": context_data.get("raw_chat_records", []),
                "raw_memories": context_data.get("raw_memories", []),
                "raw_notes": context_data.get("raw_notes", []),
                "core_topic": context_data.get("core_topic", ""),
                "memory_id_mapping": context_data.get("memory_id_mapping", {}),
                "note_id_mapping": context_data.get("note_id_mapping", {}),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.deepmind.logger.warning(f"解析记忆上下文失败: {e}")
            return None

    async def retrieve_memories_and_notes(
        self, event: AstrMessageEvent, query: str, readonly: bool = False
    ) -> Dict[str, Any]:
        """检索长期记忆与候选笔记。

        `readonly=True` 时跳过灵魂共鸣等会改状态的分支，供面板探针复用同一条链路。
        """
        deepmind = self.deepmind

        errors: List[str] = []

        memory_query = deepmind.query_processor.process_query_for_memory(query, event)

        long_term_memories = []
        if deepmind.memory_system:
            try:
                memory_scope = await deepmind.plugin_context.resolve_memory_scope_from_event(event)
                rag_fields = deepmind.query_processor.extract_rag_fields(event)
                entities = rag_fields.get("entities", [])

                dynamic_limit = deepmind.CHAINED_RECALL_PER_TYPE_LIMIT

                long_term_memories = await deepmind.memory_system.chained_recall(
                    query=memory_query,
                    entities=entities,
                    per_type_limit=int(dynamic_limit),
                    final_limit=int(dynamic_limit * 1.5),
                    event=event,
                    vector=None,
                    memory_scope=memory_scope,
                )

                if deepmind.soul and not readonly:
                    entries = []
                    for mem in long_term_memories:
                        scores = getattr(mem, "soul_state", None)
                        if not isinstance(scores, dict) or not scores:
                            continue
                        entries.append(
                            {"scores": scores, "strength": getattr(mem, "strength", 1)}
                        )
                    if entries:
                        deepmind.soul.resonate(entries)

            except Exception as e:
                deepmind.logger.error(f"链式召回失败，跳过记忆检索: {e}", exc_info=True)
                errors.append(f"链式召回失败：{type(e).__name__}: {e}")
                long_term_memories = []

        secretary_decision = {}
        try:
            if hasattr(event, "angelheart_context") and event.angelheart_context is not None:
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.deepmind.logger.warning(f"无法获取 secretary_decision 信息: {e}")

        note_recall_enabled = getattr(deepmind, "note_recall_enabled", True)

        candidate_notes = []
        if note_recall_enabled and deepmind.note_service:
            auto_domains = self._auto_domains(event)
            if auto_domains:
                note_query = deepmind.query_processor.process_query_for_notes(query, event)
                merged = []
                for domain in auto_domains:
                    merged.extend(
                        await deepmind.note_service.search_notes_by_top_k(
                            query=note_query,
                            recall_count=deepmind.note_candidate_top_k,
                            top_k=deepmind.note_candidate_top_k,
                            domains=[domain],
                        )
                    )
                merged.sort(key=lambda row: row.get("similarity") or 0.0, reverse=True)
                candidate_notes = merged[: deepmind.note_candidate_top_k]
            else:
                deepmind.logger.debug("当前会话没有标记为自动的知识库，跳过知识自动召回")

        return {
            "long_term_memories": long_term_memories,
            "candidate_notes": candidate_notes,
            "note_id_mapping": {},
            "memory_id_mapping": {},
            "secretary_decision": secretary_decision,
            "core_topic": secretary_decision.get("topic", ""),
            # 被吞掉的内部异常，供探针上报；主链路不读这个字段
            "errors": errors,
        }

    @staticmethod
    def _group_id(event: AstrMessageEvent) -> str:
        if hasattr(event, "get_group_id"):
            try:
                return str(event.get_group_id() or "").strip()
            except Exception:
                return ""
        return ""

    def _auto_domains(self, event: AstrMessageEvent) -> list:
        """当前会话里标记为「自动」的知识域。私聊与未配置群一律为空。"""
        group_id = self._group_id(event)
        if not group_id:
            return []
        return self.deepmind.plugin_context.auto_domains(group_id)
