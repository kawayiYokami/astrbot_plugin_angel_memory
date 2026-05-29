from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class NoteRecallTool(FunctionTool):
    name: str = "angel_note_read"
    description: str = "读取笔记正文。使用 note_short_id 定位文件，支持 offset+limit 分页读取。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "note_short_id": {
                    "type": "integer",
                    "description": "笔记短ID（数字ID）",
                    "minimum": 0,
                },
                "offset": {
                    "type": "integer",
                    "description": "起始行偏移（从第几行开始读，默认1）",
                    "minimum": 1,
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "读取行数上限（默认30000字符内）",
                    "minimum": 1,
                },
            },
            "required": ["note_short_id"],
        }
    )

    # 默认最大字符数
    MAX_CHARS = 30000

    def __post_init__(self):
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        note_short_id: int,
        offset: int = 1,
        limit: Optional[int] = None,
    ) -> str:
        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            return "错误：无法获取插件上下文。"
        plugin_context = event.plugin_context

        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        if memory_sql_manager is None:
            return "错误：memory_sql_manager 不可用。"
        row = await memory_sql_manager.get_note_index_by_short_id(int(note_short_id))
        if not row:
            return f"错误：未找到 note_short_id={int(note_short_id)} 对应的笔记。"
        rel_path = str(row.get("source_file_path") or "").replace("\\", "/").strip().lstrip("/")

        raw_dir = plugin_context.get_path_manager().get_raw_dir()
        target_path = (Path(raw_dir) / rel_path).resolve()
        raw_root = Path(raw_dir).resolve()
        if raw_root not in target_path.parents and target_path != raw_root:
            return "错误：路径越界。"
        if not target_path.exists():
            return f"错误：文件不存在：{rel_path}"

        try:
            text = target_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"错误：读取文件失败：{e}"

        all_lines = text.splitlines()
        total_lines = len(all_lines)

        # offset: 从第几行开始（1-based）
        start = max(1, int(offset))
        if start > total_lines:
            return (
                f"[angel_note_read]\n"
                f"note_short_id: {note_short_id}\n"
                f"source_file_path: {rel_path}\n"
                f"total_lines: {total_lines}\n"
                f"offset: {start}\n"
                f"内容已到末尾，无更多行。"
            )

        # 按 limit 或字符上限截取
        selected_lines = []
        char_count = 0
        line_limit = int(limit) if limit else total_lines
        idx = start - 1
        lines_read = 0

        while idx < total_lines and lines_read < line_limit:
            line = all_lines[idx]
            if char_count + len(line) + 1 > self.MAX_CHARS and selected_lines:
                break
            selected_lines.append(line)
            char_count += len(line) + 1
            idx += 1
            lines_read += 1

        actual_end = start + len(selected_lines) - 1
        has_more = actual_end < total_lines
        content = "\n".join(selected_lines)

        result = (
            f"[angel_note_read]\n"
            f"note_short_id: {note_short_id}\n"
            f"source_file_path: {rel_path}\n"
            f"total_lines: {total_lines}\n"
            f"returned: L{start}-{actual_end}\n"
        )
        if has_more:
            result += f"has_more: true (继续读取请用 offset={actual_end + 1})\n"
        else:
            result += "has_more: false\n"
        result += f"\n{content}"

        return result
