from typing import Any, Dict, List

from ..utils.feedback_queue import get_feedback_queue
from ..utils.memory_id_resolver import MemoryIDResolver


class DeepMindFeedbackService:
    """DeepMind 的反馈与异步分析职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    async def update_memory_system(
        self,
        feedback_data: Dict[str, Any],
        long_term_memories: List,
        session_id: str,
        persona_name: str = "",
    ) -> None:
        deepmind = self.deepmind
        memory_scope = deepmind.plugin_context.resolve_memory_scope(
            session_id, persona_name=persona_name
        )

        useful_memory_ids = feedback_data.get("useful_memory_ids", [])
        recalled_memory_ids = [memory.id for memory in (long_term_memories or []) if getattr(memory, "id", None)]
        new_memories_raw = feedback_data.get("new_memories", {})
        merge_groups_raw = feedback_data.get("merge_groups", [])

        useful_long_term_memories = []
        if useful_memory_ids:
            memory_map = {memory.id: memory for memory in long_term_memories}
            useful_long_term_memories = [
                memory_map[memory_id]
                for memory_id in useful_memory_ids
                if memory_id in memory_map
            ]

        new_memories_normalized = MemoryIDResolver.normalize_new_memories_format(
            new_memories_raw, deepmind.logger
        )
        new_memory_objects = []
        if new_memories_normalized:
            from ...llm_memory.models.data_models import BaseMemory, MemoryType

            for mem_dict in new_memories_normalized:
                try:
                    init_data = mem_dict.copy()
                    if "type" in init_data:
                        init_data["memory_type"] = MemoryType(init_data.pop("type"))
                    new_memory_objects.append(BaseMemory(**init_data))
                except Exception as e:
                    deepmind.logger.warning(f"为新记忆创建BaseMemory对象失败: {e}")

        useful_memory_ids = [memory.id for memory in useful_long_term_memories]
        deepmind.session_memory_manager.update_session_memories(
            session_id, new_memory_objects, useful_memory_ids
        )

        merge_groups = MemoryIDResolver.normalize_merge_groups_format(merge_groups_raw)
        if useful_memory_ids or new_memories_normalized or merge_groups:
            task_payload = {
                "feedback_fn": self.execute_feedback_task,
                "session_id": session_id,
                "payload": {
                    "useful_memory_ids": list(useful_memory_ids),
                    "recalled_memory_ids": recalled_memory_ids,
                    "new_memories": new_memories_normalized,
                    "merge_groups": merge_groups,
                    "session_id": session_id,
                    "memory_scope": memory_scope,
                },
            }
            await get_feedback_queue().submit(task_payload)

    async def execute_feedback_task(
        self,
        useful_memory_ids: List[str] | None = None,
        recalled_memory_ids: List[str] | None = None,
        new_memories: List[Dict[str, Any]] | None = None,
        merge_groups: List[List[str]] | None = None,
        session_id: str = "",
        memory_scope: str = "public",
    ) -> None:
        deepmind = self.deepmind
        if deepmind.memory_system is not None:
            await deepmind.memory_system.feedback(
                useful_memory_ids=useful_memory_ids or [],
                recalled_memory_ids=recalled_memory_ids or [],
                new_memories=new_memories or [],
                merge_groups=merge_groups or [],
                memory_scope=memory_scope,
            )
        else:
            deepmind.logger.error("记忆系统不可用，跳过反馈")

    async def submit_async_analysis_task(self, reflection_input):
        deepmind = self.deepmind
        session_id = getattr(reflection_input, "session_id", "unknown")
        task_payload = {
            "feedback_fn": deepmind._execute_async_analysis_task,
            "session_id": session_id,
            "payload": {
                "reflection_input": reflection_input,
            },
        }
        await get_feedback_queue().submit(task_payload)
