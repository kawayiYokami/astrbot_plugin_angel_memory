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
    description: str = "读取笔记正文。使用 note_id 定位文件，支持 offset+limit 分页读取。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "笔记的整数 ID（检索结果里 ID: 后面那个数字）",
                    "minimum": 1,
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
            "required": ["note_id"],
        }
    )

    # 默认最大字符数
    MAX_CHARS = 30000

    def __post_init__(self):
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        note_id: int,
        offset: int = 1,
        limit: Optional[int] = None,
    ) -> str:
        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            return "错误：无法获取插件上下文。"
        plugin_context = event.plugin_context

        note_index = plugin_context.get_component("note_index")
        if note_index is None:
            return "错误：笔记索引服务不可用。"

        # 可见性校验：直读必须和主动搜索走同一套白名单。
        # 只在白名单域里找一次：hidden 域碰都不碰，与不存在统一报“未找到”。
        group_id = self._group_id(event)
        try:
            visible = plugin_context.visible_domains(group_id)
        except Exception:
            visible = []
        domain, note = note_index.find_note(int(note_id), domains=list(visible or []))
        if domain is None or note is None:
            return f"错误：未找到 note_id={int(note_id)} 对应的笔记。"

        target_path = note_index.file_for_note(note)
        if target_path is None:
            return f"错误：未找到 note_id={int(note_id)} 对应的笔记。"
        rel_path = note_index.path_for_note(domain, note) or target_path.name

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
                f"note_id: {note_id}\n"
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
            f"note_id: {note_id}\n"
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

    @staticmethod
    def _group_id(event: AstrMessageEvent) -> str:
        """群号；私聊返回空串，私聊按全部域可见处理。"""
        if hasattr(event, "get_group_id"):
            try:
                return str(event.get_group_id() or "").strip()
            except Exception:
                return ""
        return ""
