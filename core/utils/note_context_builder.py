"""
笔记上下文构建器

仅负责将候选笔记条目格式化为提示词文本。
旧的 note_id 正文扩展链路已废弃。
"""

from typing import Dict, List


class NoteContextBuilder:
    _MIN_DISPLAY_CONFIDENCE = 0.50
    _MAX_DISPLAY_CONFIDENCE = 0.70

    @staticmethod
    def _normalize_raw_confidence(score: float) -> float:
        """将不同检索分值统一到 0~1 的原始置信度。"""
        if score <= 0:
            return 0.0
        if score <= 1.0:
            return float(score)
        # simple 模式下 score 可能是命中次数，做平滑归一化
        return float(score / (score + 2.0))

    @staticmethod
    def build_candidate_list_for_prompt(notes: List[Dict]) -> str:
        if not notes:
            return ""

        raw_confidences: List[float] = [
            NoteContextBuilder._normalize_raw_confidence(
                float((note.get("similarity", 0.0) or 0.0))
            )
            for note in notes
        ]
        top_conf = max(raw_confidences) if raw_confidences else 0.0
        cap = min(
            NoteContextBuilder._MAX_DISPLAY_CONFIDENCE,
            max(NoteContextBuilder._MIN_DISPLAY_CONFIDENCE, top_conf * 0.8),
        )

        note_parts = []
        for idx, note in enumerate(notes):
            metadata = note.get("metadata", {}) or {}
            source_file_path = metadata.get("source_file_path", "")
            note_short_id = int(metadata.get("note_short_id") or -1)
            heading_values = [
                metadata.get("heading_h1", ""),
                metadata.get("heading_h2", ""),
                metadata.get("heading_h3", ""),
                metadata.get("heading_h4", ""),
                metadata.get("heading_h5", ""),
                metadata.get("heading_h6", ""),
            ]
            heading_text = " / ".join(
                [str(x).strip() for x in heading_values if str(x).strip()]
            ) or "(无标题)"
            total_lines = int(metadata.get("total_lines") or 0)
            tags = note.get("tags", [])
            tags_str = ", ".join(tags) if tags else "无"
            raw_conf = raw_confidences[idx] if idx < len(raw_confidences) else 0.0

            # 统一展示到 [0.5, cap]，确保最低不低于 0.5（哪怕只有 1 条）
            if top_conf > 0 and cap > NoteContextBuilder._MIN_DISPLAY_CONFIDENCE:
                relative = max(0.0, min(1.0, raw_conf / top_conf))
                confidence = NoteContextBuilder._MIN_DISPLAY_CONFIDENCE + relative * (
                    cap - NoteContextBuilder._MIN_DISPLAY_CONFIDENCE
                )
            else:
                confidence = NoteContextBuilder._MIN_DISPLAY_CONFIDENCE

            note_text = (
                f"[笔记候选]\n"
                f"短ID: {note_short_id}\n"
                f"路径: {source_file_path}\n"
                f"标题: {heading_text}\n"
                f"置信度: {confidence:.2f}\n"
                f"总行数: {total_lines}\n"
                f"标签: {tags_str}\n"
                f"提示: 如需正文，请调用 note_recall(note_short_id, start_line, end_line)。"
            )
            note_parts.append(note_text)

        notes_str = "\n\n---\n\n".join(note_parts)
        time_warning = "[注意：以下笔记内容可能不具备时效性，请勿作为最新消息看待]\n\n"
        return time_warning + notes_str
