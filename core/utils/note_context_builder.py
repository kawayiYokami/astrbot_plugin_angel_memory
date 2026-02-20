"""
笔记上下文构建器

仅负责将候选笔记条目格式化为提示词文本。
旧的 note_id 正文扩展链路已废弃。
"""

from typing import Dict, List


class NoteContextBuilder:
    @staticmethod
    def build_candidate_list_for_prompt(notes: List[Dict]) -> str:
        if not notes:
            return ""

        note_parts = []
        for note in notes:
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

            note_text = (
                f"[笔记候选]\n"
                f"短ID: {note_short_id}\n"
                f"路径: {source_file_path}\n"
                f"标题: {heading_text}\n"
                f"总行数: {total_lines}\n"
                f"标签: {tags_str}\n"
                f"提示: 如需正文，请调用 note_recall(note_short_id, start_line, end_line)。"
            )
            note_parts.append(note_text)

        notes_str = "\n\n---\n\n".join(note_parts)
        time_warning = "[注意：以下笔记内容可能不具备时效性，请勿作为最新消息看待]\n\n"
        return time_warning + notes_str
