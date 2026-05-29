"""
笔记上下文构建器

将切片搜索结果格式化为提示词文本，包含内容摘要和位置信息。
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
            content = str(note.get("content") or "").strip()
            line_start = int(metadata.get("line_start") or 0)
            line_end = int(metadata.get("line_end") or 0)

            # 构建位置信息
            location = f"L{line_start}-{line_end}" if line_start and line_end else ""

            # 内容截断（避免过长）
            max_content_len = 300
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."

            parts = [f"[笔记切片]"]
            if note_short_id >= 0:
                parts.append(f"短ID: {note_short_id}")
            parts.append(f"路径: {source_file_path}")
            if location:
                parts.append(f"位置: {location}")
            if content:
                parts.append(f"内容: {content}")

            note_parts.append("\n".join(parts))

        notes_str = "\n\n---\n\n".join(note_parts)
        header = (
            "[注意：以下笔记内容可能不具备时效性，请勿作为最新消息看待]\n"
            "[如需完整正文，请调用 angel_note_read(note_short_id, offset, limit)]\n\n"
        )
        return header + notes_str
