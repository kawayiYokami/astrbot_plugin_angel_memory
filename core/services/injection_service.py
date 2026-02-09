from typing import Any, Dict, Optional

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart


class DeepMindInjectionService:
    """DeepMind 的请求注入职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    def normalize_soul_value(self, dimension: str, value: float) -> float:
        deepmind = self.deepmind
        if not deepmind.soul:
            return 0.5

        cfg = deepmind.soul.config.get(dimension, {})
        min_val = cfg.get("min", 0)
        max_val = cfg.get("max", 1)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def create_tendency_bar(self, normalized_value: float) -> str:
        bar_length = 10
        filled_length = int(round(normalized_value * bar_length))
        return "█" * filled_length + " " * (bar_length - filled_length)

    def inject_memories_to_request(
        self,
        request: ProviderRequest,
        session_id: str,
        note_context: str,
        soul_state_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        deepmind = self.deepmind
        system_context_parts = []

        instruction = (
            "<instruction>\n"
            "以下是系统自动检索的背景上下文（包含记忆、笔记及当前状态）。\n"
            "请利用这些信息辅助回答，但必须遵守以下规则：\n"
            "1. **对话优先**：始终优先响应用户的当前对话内容，记忆和笔记仅作为补充参考。\n"
            "2. **绝对隐形**：严禁在回复中提及\"soul_state\"、\"系统提示\"、\"XML标签\"等来源信息。\n"
            "3. **状态保密**：soul_state 仅用于调整你的回复风格，绝不可在回复中泄露或讨论。\n"
            "</instruction>"
        )
        system_context_parts.append(instruction)

        if soul_state_values and deepmind.config.enable_soul_system:
            norm_recall = self.normalize_soul_value(
                "RecallDepth", soul_state_values.get("RecallDepth", 0.5)
            )
            norm_impression = self.normalize_soul_value(
                "ImpressionDepth", soul_state_values.get("ImpressionDepth", 0.5)
            )
            norm_expression = self.normalize_soul_value(
                "ExpressionDesire", soul_state_values.get("ExpressionDesire", 0.5)
            )
            norm_creativity = self.normalize_soul_value(
                "Creativity", soul_state_values.get("Creativity", 0.5)
            )

            bar_recall = self.create_tendency_bar(norm_recall)
            bar_impression = self.create_tendency_bar(norm_impression)
            bar_expression = self.create_tendency_bar(norm_expression)
            bar_creativity = self.create_tendency_bar(norm_creativity)

            soul_state_content = (
                f"<soul_state>\n"
                f"• 社交倾向: 内向 {bar_recall} 外向 [{norm_recall:.2f}]\n"
                f"• 认知倾向: 指导 {bar_impression} 好奇 [{norm_impression:.2f}]\n"
                f"• 表达倾向: 简洁 {bar_expression} 详尽 [{norm_expression:.2f}]\n"
                f"• 情绪倾向: 严肃 {bar_creativity} 活泼 [{norm_creativity:.2f}]\n"
                f"</soul_state>"
            )
            system_context_parts.append(soul_state_content)

        short_term_memories = deepmind.session_memory_manager.get_session_memories(session_id)
        memory_context = deepmind.memory_injector.format_session_memories_for_prompt(
            short_term_memories
        )
        if memory_context:
            system_context_parts.append(f"<memories>\n{memory_context}\n</memories>")

        if note_context:
            system_context_parts.append(f"<notes>\n{note_context}\n</notes>")

        if system_context_parts:
            full_system_context = (
                "<system_context>\n"
                + "\n\n".join(system_context_parts)
                + "\n</system_context>"
            )
            text_part = TextPart(text=full_system_context)
            request.extra_user_content_parts.append(text_part)
