"""
小模型提示词构建器

负责构建用于小模型的记忆整理提示词。
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from ..session_memory import MemoryItem
from ..config import MemoryConstants
from .memory_formatter import MemoryFormatter


class SmallModelPromptBuilder:
    """小模型提示词构建器"""

    @staticmethod
    def format_relative_time(timestamp: float) -> str:
        """
        格式化相对时间

        Args:
            timestamp: Unix时间戳

        Returns:
            相对时间字符串，如"刚刚"、"2分钟前"等
        """
        if not timestamp:
            return ""

        now = datetime.now().timestamp()
        diff = now - timestamp

        if diff < MemoryConstants.TIME_MINUTE:
            return "刚刚"
        if diff < MemoryConstants.TIME_HOUR:
            return f"{int(diff / MemoryConstants.TIME_MINUTE)}分钟前"
        if diff < MemoryConstants.TIME_DAY:
            return f"{int(diff / MemoryConstants.TIME_HOUR)}小时前"
        return f"{int(diff / MemoryConstants.TIME_DAY)}天前"

    @staticmethod
    def extract_text_from_content(content) -> str:
        """
        从content中提取文本

        Args:
            content: 消息内容，可能是字符串、列表或字典

        Returns:
            提取的文本内容
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
        return ""

    # 剧本式排版的断档阈值：相邻两条消息间隔达到这个秒数就补一行时间
    CHAT_TIME_GAP_SECONDS = 600
    CHAT_TIMEZONE = timezone(timedelta(hours=8))

    @staticmethod
    def _format_chat_clock(timestamp) -> str:
        """把发送时间转成 UTC+8 的 HH:MM；取不到有效时间时返回空串。"""
        try:
            value = float(timestamp or 0)
        except (TypeError, ValueError):
            return ""
        if value <= 0:
            return ""
        moment = datetime.fromtimestamp(value, tz=SmallModelPromptBuilder.CHAT_TIMEZONE)
        return moment.strftime("%H:%M")

    @staticmethod
    def _split_quote_fragments(text: str) -> Tuple[List[str], str]:
        """抽出 [引用消息(某人: 被引内容)] 片段，返回 (引用列表, 剩余正文)。

        括号里可能嵌套 [At:...] 或圆括号，所以按深度配对吃到最外层右括号，
        不能在第一个 ')' 处断开。括号没闭合时原样留在正文里。
        """
        marker = "[引用消息("
        quotes: List[str] = []
        rest: List[str] = []
        cursor = 0
        while True:
            start = text.find(marker, cursor)
            if start < 0:
                rest.append(text[cursor:])
                break
            rest.append(text[cursor:start])

            depth = 1
            index = start + len(marker)
            while index < len(text) and depth > 0:
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                index += 1

            if depth > 0:
                rest.append(text[start:])
                break

            inner = text[start + len(marker) : index - 1].strip()
            if index < len(text) and text[index] == "]":
                index += 1
            if inner:
                quotes.append(inner)
            cursor = index

        return quotes, "".join(rest)

    @staticmethod
    def _render_quote(inner: str) -> str:
        """把「某人: 被引内容」渲染成「  > 某人：被引内容」。"""
        text = (inner or "").strip()
        for separator in (": ", "：", ":"):
            if separator in text:
                name, _, content = text.partition(separator)
                name = name.strip()
                content = content.strip()
                if name:
                    return f"  > {name}：{content}"
                break
        return f"  > {text}"

    @staticmethod
    def _render_chat_body(text: str) -> List[str]:
        """正文逐行缩进两空格；空行不产出，避免留下只有空白的行。"""
        return [f"  {line}" for line in text.splitlines() if line.strip()]

    @staticmethod
    def format_chat_records(chat_records: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        格式化对话历史为剧本式纯文本，并生成对话参与者清单。

        排版规则：
        - 严格按发送时间升序，时间戳相同保持传入顺序
        - 相邻两条间隔达到 600 秒就插一行 HH:MM（UTC+8），第一条之前也插
        - 换人才起新段，段首写「名字（号）」不缩进，该人后续消息逐行缩进两空格
        - 引用片段单列成「> 名字：内容」，放在该条消息正文之前
        - [At:...]、[图片] 这类占位符留在正文原位，不加任何 markdown 标记

        Args:
            chat_records: 对话记录列表

        Returns:
            一个元组，包含：
            - 格式化后的对话文本字符串
            - 对话参与者清单（去重后）
        """
        usable: List[Dict] = []
        for msg in chat_records or []:
            if not isinstance(msg, dict):
                continue

            # 过滤掉toolcall相关消息
            if msg.get("is_structured_toolcall"):
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                toolcall_keywords = ["调用", "tool_call", "function", "工具调用结果："]
                if any(keyword in content for keyword in toolcall_keywords):
                    continue

            usable.append(msg)

        # 严格按发送时间升序：sorted 是稳定排序，时间戳相同就保持传入顺序
        usable.sort(key=lambda item: float(item.get("timestamp", 0) or 0))

        lines: List[str] = []
        participants: Dict[str, Dict] = {}
        last_timestamp: Optional[float] = None
        last_speaker_key: Optional[Tuple] = None

        for msg in usable:
            role = str(msg.get("role", "") or "").strip()
            raw_text = SmallModelPromptBuilder.extract_text_from_content(
                msg.get("content", "")
            )
            quotes, leftover = SmallModelPromptBuilder._split_quote_fragments(
                raw_text or ""
            )
            body = leftover.strip()

            # 内容（含引用）全空的记录直接跳过，不产出空段
            if not body and not quotes:
                continue

            try:
                timestamp = float(msg.get("timestamp", 0) or 0)
            except (TypeError, ValueError):
                timestamp = 0.0

            clock = SmallModelPromptBuilder._format_chat_clock(timestamp)
            if clock and (
                last_timestamp is None
                or timestamp - last_timestamp >= SmallModelPromptBuilder.CHAT_TIME_GAP_SECONDS
            ):
                lines.append(clock)

            if role == "assistant":
                speaker_key: Tuple = ("assistant",)
                speaker_label = "我"
            else:
                sender_id = str(msg.get("sender_id", "") or "").strip()
                sender_name = str(msg.get("sender_name", "") or "").strip()
                speaker_key = ("user", sender_id or sender_name or "群友")

                if sender_name:
                    speaker_label = (
                        f"{sender_name}（{sender_id}）" if sender_id else sender_name
                    )
                elif sender_id:
                    speaker_label = f"群友（{sender_id}）"
                else:
                    speaker_label = "群友"

                if (
                    sender_id
                    and sender_id.lower() != "unknown"
                    and sender_id not in participants
                ):
                    participants[sender_id] = {
                        "id": sender_id,
                        "name": sender_name or "群友",
                    }

            # 换人才起新段；同一个人连续发言不重复署名
            if speaker_key != last_speaker_key:
                lines.append(speaker_label)
                last_speaker_key = speaker_key

            for quote in quotes:
                lines.append(SmallModelPromptBuilder._render_quote(quote))
            lines.extend(SmallModelPromptBuilder._render_chat_body(body))

            last_timestamp = timestamp

        return "\n".join(lines), list(participants.values())

    @staticmethod
    def build_memory_prompt(
        formatted_query: str,
        memories: List[MemoryItem],
        user_list: List[Dict],
        candidate_notes: List[Dict] = None,
        secretary_decision: Dict = None,
        core_topic: str = None,
        system_prompt: str = None,
        memory_config=None,
    ) -> str:
        """
        构建用于小模型的记忆整理提示词

        Args:
            formatted_query: 已格式化的对话历史字符串
            memories: 记忆列表
            user_list: 对话参与者清单
            candidate_notes: 候选笔记列表（可选）
            secretary_decision: 秘书决策信息，包含AI人格和别名（可选）
            core_topic: 当前对话的核心话题（可选）
            system_prompt: 系统提示词（可选，如果不提供则需要在方法内获取）

        Returns:
            完整的提示词字符串
        """
        # 获取系统提示词
        if system_prompt is None:
            # p-memory 模式：直接取提示词装配器，不再经过已下线的 CognitiveService
            from ..prompts.prompt_assembler import PromptAssembler

            system_prompt = PromptAssembler.build_memory_system_guide()

        # 1. 构建参与者信息（使用列表推导式）
        participants_section = "# 对话参与者\n" + (
            "\n".join(
                f"- {p.get('name', '未知')}: {p.get('id', '未知')}" for p in user_list
            )
            if user_list
            else "暂无详细信息\n"
        )

        # 2. 构建新的规则
        rules_section = (
            '\n重要规则： 在生成任何判断或记忆时，你必须使用"对话参与者"部分提供的具体用户昵称来指代人物。'
            '绝对禁止使用"用户"、"提问者"等模糊代词。'
        )

        # 3. 构建AI身份信息
        ai_identity_section = ""
        if secretary_decision:
            persona_name = secretary_decision.get("persona_name", "")
            alias = secretary_decision.get("alias", "")
            if persona_name or alias:
                ai_info_parts = []
                if persona_name:
                    ai_info_parts.append(f"助理的名字是：{persona_name}")
                if alias:
                    ai_info_parts.append(f"昵称有：{alias}")
                ai_identity_section = (
                    "\n# AI身份信息\n" + "\n".join(ai_info_parts) + "\n"
                )

        # 4. 构建核心话题信息
        topic_section = ""
        if core_topic and core_topic.strip():
            topic_section = f"\n# 当前对话核心话题\n{core_topic.strip()}\n"

        # 5. 组装新的 system_prompt (已移除废弃的笔记清单)
        final_system_prompt = f"{system_prompt}\n\n{participants_section}{rules_section}{ai_identity_section}{topic_section}"

        # 7. 构建完整提示词（简化记忆处理）
        parts = [final_system_prompt]
        if formatted_query:
            parts.append(f"\n\n对话历史：\n{formatted_query}")
        if memories:
            parts.append(
                f"\n\n你回忆起了：\n{MemoryFormatter.format_memories_for_display(memories)}"
            )
        parts.append("\n\n请按照任务进行处理")

        return "".join(parts)

    @staticmethod
    def build_post_hoc_analysis_prompts(
        historical_query: str,
        main_llm_response: str,
        raw_memories: List,
        core_topic: str = "",
        memory_id_mapping: Dict[str, str] = None,
        config=None,
    ) -> Tuple[str, str]:
        """
        构建用于事后反思的提示词，返回 (system_prompt, user_prompt)

        system_prompt 固定为记忆系统指南，便于命中提示词缓存；
        user_prompt 只放本次对话数据与输出要求。

        Args:
            historical_query: 历史对话内容
            main_llm_response: 主LLM的回答内容
            raw_memories: 原始记忆列表
            core_topic: 核心话题（可选）
            memory_id_mapping: 记忆ID映射表（可选）
            config: 配置对象（可选）

        Returns:
            (system_prompt, user_prompt)
        """
        from ...llm_memory.prompts.prompt_assembler import PromptAssembler

        system_prompt = PromptAssembler.build_memory_system_guide()

        # 构建数据上下文（仅包含实际数据，不包含指令）
        sections = ["# 本次对话数据", "", "## 历史对话内容", historical_query]
        if str(main_llm_response or "").strip():
            sections += ["", "## 你的最新回复", str(main_llm_response)]
        sections += ["", "## 召回的相关记忆"]
        data_context = "\n".join(sections) + "\n"

        # 添加记忆信息
        if raw_memories:
            for memory in raw_memories:
                short_id = (
                    memory_id_mapping.get(memory.id, memory.id)
                    if memory_id_mapping
                    else memory.id
                )
                data_context += f"id:{short_id} - {memory.judgment}\n"
        else:
            data_context += "\n无相关记忆\n"

        user_prompt = f"""{data_context}

---

现在请开始分析并输出JSON结果。
"""

        return system_prompt, user_prompt
