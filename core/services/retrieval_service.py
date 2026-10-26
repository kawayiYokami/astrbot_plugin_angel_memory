import json
from typing import Any, Dict

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
        self, event: AstrMessageEvent, query: str
    ) -> Dict[str, Any]:
        deepmind = self.deepmind

        memory_query = deepmind.query_processor.process_query_for_memory(query, event)

        long_term_memories = []
        if deepmind.memory_system:
            try:
                memory_scope = await deepmind.plugin_context.resolve_memory_scope_from_event(event)
                rag_fields = deepmind.query_processor.extract_rag_fields(event)
                entities = rag_fields.get("entities", [])

                dynamic_limit = deepmind.CHAINED_RECALL_PER_TYPE_LIMIT
                if deepmind.soul:
                    try:
                        dynamic_limit = deepmind.soul.get_value("RecallDepth")
                        deepmind.logger.info(
                            f"🧠 灵魂回忆深度: {dynamic_limit} (E={deepmind.soul.energy['RecallDepth']:.1f})"
                        )
                    except Exception as e:
                        deepmind.logger.warning(f"获取灵魂参数失败，使用默认值: {e}")

                long_term_memories = await deepmind.memory_system.chained_recall(
                    query=memory_query,
                    entities=entities,
                    per_type_limit=int(dynamic_limit),
                    final_limit=int(dynamic_limit * 1.5),
                    event=event,
                    vector=None,
                    memory_scope=memory_scope,
                )

                if deepmind.soul:
                    snapshots = [
                        mem.state_snapshot
                        for mem in long_term_memories
                        if hasattr(mem, "state_snapshot") and mem.state_snapshot
                    ]
                    if snapshots:
                        deepmind.soul.resonate(snapshots)

            except Exception as e:
                deepmind.logger.error(f"链式召回失败，跳过记忆检索: {e}")
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
                deepmind.logger.info("当前会话没有标记为自动的知识库，跳过知识自动召回")

        return {
            "long_term_memories": long_term_memories,
            "candidate_notes": candidate_notes,
            "note_id_mapping": {},
            "memory_id_mapping": {},
            "secretary_decision": secretary_decision,
            "core_topic": secretary_decision.get("topic", ""),
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
