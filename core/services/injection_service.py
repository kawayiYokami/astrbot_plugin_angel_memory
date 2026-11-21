from typing import Any

from astrbot.api.provider import ProviderRequest

from ..session_memory import is_source_alive


class DeepMindInjectionService:
    """DeepMind 的请求注入职责。"""

    BEHAVIOR_RULES = (
        "【关于你记得的事】\n"
        "下面附上的记忆、笔记和你此刻的状态，都是你自己记得的、感受到的，不是临时查来的资料。\n"
        "- 直接用它们说话，就像你本来就知道。不交代来源，不说「根据记忆」「资料显示」「我记得你以前提过」。\n"
        "- 不要为了显得记性好而主动翻旧账；对话真的需要时才自然带出。\n"
        "- 状态只影响你说话的语气，不必向对方解释。\n"
        "- 当前对话永远优先，记忆是底气，不是话题。"
    )

    # 实验性注入路径，暂不对外开放，保持 False。
    # False（正式路径）：短期记忆合成一块，只追加到 request.contexts 末尾一条。
    # True（实验路径）：调 attach_history_blocks，把每条记忆挂回它当初召回时那条
    #       历史消息后面，末尾块只放当前这条消息带来的记忆。
    # 实验路径的定位方式：拿 _message_order 给的账本原文，跟 contexts 里的用户消息
    # 从尾部逐条一对一配对，配不上就丢掉。账本存正文、上下文带发言人前缀与时间标签，
    # 两者不同源时一条都配不上；群聊重复短句也会配错位置。开启前必须先确认能对上。
    # 未经真机验证，不要接进配置项，也不要对外说明。
    INJECT_HISTORY_BLOCKS = False

    def __init__(self, deepmind):
        self.deepmind = deepmind

    def create_tendency_bar(self, normalized_value: float) -> str:
        bar_length = 10
        filled_length = round(normalized_value * bar_length)
        return "█" * filled_length + " " * (bar_length - filled_length)

    REMINDER_OPEN = "<system_reminder>"

    @classmethod
    def _has_reminder(cls, content: Any) -> bool:
        """这段内容是不是已经是一个提醒块。"""
        return isinstance(content, str) and cls.REMINDER_OPEN in content

    @classmethod
    def _already_injected(cls, request: ProviderRequest) -> bool:
        """本次请求已经挂过提醒块，就不再挂第二块。"""
        if getattr(request, "_angel_memory_injected", False):
            return True
        for part in request.extra_user_content_parts or []:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")
            if cls._has_reminder(text):
                return True
        return False

    @staticmethod
    def _plain_text(content: Any) -> str:
        """把消息内容归一成可比较的纯文本，账本与 contexts 两侧用同一套口径。"""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                elif getattr(item, "type", "") == "text":
                    parts.append(str(getattr(item, "text", "") or ""))
            return " ".join(parts).strip()
        return ""

    @classmethod
    def _render_block(cls, parts: list) -> str:
        return cls.REMINDER_OPEN + "\n" + "\n\n".join(parts) + "\n</system_reminder>"

    def _attach_current_block(self, request: ProviderRequest, parts: list) -> bool:
        """本轮注入内容合成一条消息，追加到上下文末尾，位置在本轮用户发言之前。"""
        if not parts:
            return False
        if self._already_injected(request):
            return False
        contexts = request.contexts
        if not isinstance(contexts, list):
            contexts = []
            request.contexts = contexts
        contexts.append({"role": "user", "content": self._render_block(parts), "_no_save": True})
        request._angel_memory_injected = True
        return True

    def attach_history_blocks(
        self,
        request: ProviderRequest,
        message_order,
        memories_by_message,
        current_message_id: str = "",
    ) -> list:
        """实验性路径：按消息顺序给历史消息挂记忆块，插在各自那条消息之后。

        `message_order` 是 (消息ID, 消息纯文本) 的顺序表，`memories_by_message` 是
        召回来源消息 ID 到记忆列表的映射。账本与上下文从尾部对齐后一对一配对，账本多
        出来的那几条视为已被截断，直接丢弃；配对后内容对不上的也丢弃，不猜位置。
        返回真正挂上去的那批记忆。

        前提是 `message_order` 里的文本与 `request.contexts` 里的用户消息同源，
        否则一条都配不上，等于记忆全部不注入。暂不对外开放，只由
        INJECT_HISTORY_BLOCKS 控制，未经真机验证不要启用。
        """
        snapshot_messages = [
            (message_id, self._plain_text(text))
            for message_id, text in (message_order or [])
            if message_id and message_id != current_message_id
        ]
        if not snapshot_messages:
            return []

        contexts = request.contexts or []
        context_users = [
            (index, self._plain_text(block.get("content")))
            for index, block in enumerate(contexts)
            if isinstance(block, dict) and block.get("role") == "user"
        ]
        if not context_users:
            return []

        # 从尾部往前一对一配对，截断发生在头部，尾部总是对得上
        pairs = []
        snapshot_index = len(snapshot_messages) - 1
        context_index = len(context_users) - 1
        while snapshot_index >= 0 and context_index >= 0:
            pairs.append((snapshot_messages[snapshot_index], context_users[context_index]))
            snapshot_index -= 1
            context_index -= 1
        pairs.reverse()

        pending = []
        attached = []
        for (message_id, snapshot_plain), (anchor, context_plain) in pairs:
            if not snapshot_plain or not is_source_alive(snapshot_plain, [context_plain]):
                continue
            memories = memories_by_message.get(message_id) or []
            if not memories:
                continue
            # 锚点后面已经是提醒块，说明这条消息挂过了
            neighbour = contexts[anchor + 1] if anchor + 1 < len(contexts) else None
            if isinstance(neighbour, dict) and self._has_reminder(neighbour.get("content")):
                continue
            memory_context = self.deepmind.memory_injector.format_session_memories_for_prompt(
                memories, short_id_registry=None
            )
            if not memory_context:
                continue
            pending.append(
                (
                    anchor,
                    self._render_block([f"<memories>\n{memory_context}\n</memories>"]),
                )
            )
            attached.extend(memories)

        for anchor, block in sorted(pending, reverse=True):
            request.contexts.insert(
                anchor + 1,
                {"role": "user", "content": block, "_no_save": True},
            )
        return attached

    def group_session_memories_by_message(self, session_id: str) -> list[tuple]:
        """按召回来源消息分组短期记忆里的记忆条目，直接取短期记忆里那份内容，不再回查长期库。"""
        items = self.deepmind.session_memory_manager.get_session_memories(session_id)
        groups: dict = {}
        for item in items:
            memory = getattr(item, "memory", None)
            if memory is None:
                continue
            groups.setdefault(item.recalled_message_id, []).append(memory)
        return list(groups.items())

    def session_memories(self, session_id: str) -> list:
        """本轮要注入的记忆：短期记忆表里的全部条目，直接取表里那份内容，不回查长期库。"""
        items = self.deepmind.session_memory_manager.get_session_memories(session_id)
        return [item.memory for item in items if getattr(item, "memory", None) is not None]

    async def inject_memories_to_request(
        self,
        request: ProviderRequest,
        session_id: str,
        note_context: str,
        soul_state_values: dict[str, Any] | None = None,
        message_id: str = "",
        message_order=None,
    ) -> list:
        deepmind = self.deepmind
        body_parts = []
        soul_block = ""

        if soul_state_values and deepmind.config.enable_soul_system:
            norm_joy = float(soul_state_values.get("joy", 0.5) or 0.5)
            norm_surprise = float(soul_state_values.get("surprise", 0.5) or 0.5)
            norm_anger = float(soul_state_values.get("anger", 0.5) or 0.5)
            norm_disgust = float(soul_state_values.get("disgust", 0.5) or 0.5)

            bar_joy = self.create_tendency_bar(norm_joy)
            bar_surprise = self.create_tendency_bar(norm_surprise)
            bar_anger = self.create_tendency_bar(norm_anger)
            bar_disgust = self.create_tendency_bar(norm_disgust)

            soul_block = (
                f"<soul_state>\n"
                f"• 社交倾向: 内向 {bar_joy} 外向 [{norm_joy:.2f}]\n"
                f"• 认知倾向: 好奇 {bar_surprise} 观望 [{norm_surprise:.2f}]\n"
                f"• 表达倾向: 简洁 {bar_anger} 详尽 [{norm_anger:.2f}]\n"
                f"• 情绪倾向: 活泼 {bar_disgust} 严肃 [{norm_disgust:.2f}]\n"
                f"</soul_state>"
            )

        if self.INJECT_HISTORY_BLOCKS:
            groups = self.group_session_memories_by_message(session_id)
            memories_by_message = {bound: memories for bound, memories in groups}
            attached = []
            if message_order:
                attached.extend(
                    self.attach_history_blocks(
                        request,
                        message_order,
                        memories_by_message,
                        current_message_id=message_id,
                    )
                )
            if message_id:
                current_memories = list(memories_by_message.get(message_id) or [])
            else:
                current_memories = [memory for memories in memories_by_message.values() for memory in memories]
        else:
            current_memories = self.session_memories(session_id)
            attached = []

        user_profile_context = ""
        if hasattr(deepmind, "user_profile_service") and deepmind.user_profile_service:
            user_profile_context = deepmind.user_profile_service.format_session_profiles(
                session_id,
                short_id_registry=None,
            )
            current_memories = deepmind.user_profile_service.filter_regular_memories(session_id, current_memories)

        memory_context = deepmind.memory_injector.format_session_memories_for_prompt(
            current_memories,
            short_id_registry=None,
        )
        if memory_context:
            body_parts.append(f"<memories>\n{memory_context}\n</memories>")
            attached.extend(current_memories)

        if user_profile_context:
            body_parts.append(f"<user_profiles>\n{user_profile_context}\n</user_profiles>")

        if note_context:
            body_parts.append(f"<notes>\n{note_context}\n</notes>")

        # 状态排在最后：它只调语气，不是要给模型读的内容
        if soul_block:
            body_parts.append(soul_block)

        if body_parts:
            self._attach_current_block(request, body_parts)

        memory_guide = (
            "【记忆系统】\n"
            "你拥有长期记忆，应主动使用：\n"
            "- 对话牵涉过去的事、拿不准的细节，先用 angel_recall 检索再回答\n"
            "- 出现值得长期记住的信息（群友稳定特征、重要约定、关键事实）时，用 angel_remember 记下："
            "新信息 create，发现错误或过时及时 update，重复 merge\n"
            "- 宁可多记一条后续修正，不可漏记\n"
            "- 想起的内容自然融入当前回答，对话焦点始终跟随用户"
        )
        addition = f"{self.BEHAVIOR_RULES}\n\n{memory_guide}"
        request.system_prompt = f"{request.system_prompt}\n\n{addition}" if request.system_prompt else addition
        return attached
