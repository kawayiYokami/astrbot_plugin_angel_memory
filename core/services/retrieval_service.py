import json
from typing import Any, Dict, Optional

from astrbot.api.event import AstrMessageEvent

from ..utils.memory_id_resolver import MemoryIDResolver


class DeepMindRetrievalService:
    """DeepMind 的检索相关职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    def parse_memory_context(
        self, event: AstrMessageEvent
    ) -> Optional[Dict[str, Any]]:
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
        self, event: AstrMessageEvent, query: str, precompute_vectors: bool = False
    ) -> Dict[str, Any]:
        deepmind = self.deepmind

        if precompute_vectors:
            memory_query, memory_vector = (
                await deepmind.query_processor.process_query_for_memory_with_vector(
                    query, event
                )
            )
        else:
            memory_query = deepmind.query_processor.process_query_for_memory(query, event)
            memory_vector = None

        long_term_memories = []
        if deepmind.memory_system:
            try:
                memory_scope = deepmind.plugin_context.resolve_memory_scope_from_event(event)
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
                    vector=memory_vector,
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

        if precompute_vectors:
            note_query, note_vector = (
                await deepmind.query_processor.process_query_for_notes_with_vector(
                    query, event
                )
            )
        else:
            note_query = deepmind.query_processor.process_query_for_notes(query, event)
            note_vector = None

        candidate_notes = []
        if deepmind.note_service:
            candidate_notes = await deepmind.note_service.search_notes_by_token_limit(
                query=note_query,
                max_tokens=deepmind.small_model_note_budget,
                recall_count=deepmind.NOTE_CANDIDATE_COUNT,
                vector=note_vector,
            )
        elif getattr(deepmind.config, "enable_simple_memory", False):
            deepmind.logger.debug("当前为简化记忆模式，已跳过笔记检索。")

        note_id_mapping = {}
        for note in candidate_notes:
            note_id = note.get("id")
            if note_id:
                short_id = MemoryIDResolver.generate_short_id(note_id)
                note_id_mapping[short_id] = note_id

        memory_id_mapping = {}
        if long_term_memories:
            memory_id_mapping = MemoryIDResolver.generate_id_mapping(
                [memory.to_dict() for memory in long_term_memories], "id"
            )

        return {
            "long_term_memories": long_term_memories,
            "candidate_notes": candidate_notes,
            "note_id_mapping": note_id_mapping,
            "memory_id_mapping": memory_id_mapping,
            "secretary_decision": secretary_decision,
            "core_topic": secretary_decision.get("topic", ""),
        }
