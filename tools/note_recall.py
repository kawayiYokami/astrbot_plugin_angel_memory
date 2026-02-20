from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

from ..llm_memory.utils.token_utils import truncate_by_tokens

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class NoteRecallTool(FunctionTool):
    name: str = "note_recall"
    description: str = "读取笔记内容。请使用 note_short_id，支持按行范围读取。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "note_short_id": {
                    "type": "integer",
                    "description": "笔记短ID（数字ID，从0开始）",
                    "minimum": 0,
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（1-based，闭区间，可选）",
                    "minimum": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（1-based，闭区间，可选）",
                    "minimum": 1,
                },
                "token_budget": {
                    "type": "integer",
                    "description": "返回内容 token 限制（默认 2000）",
                    "default": 2000,
                    "minimum": 100,
                    "maximum": 12000,
                },
            },
            "required": ["note_short_id"],
        }
    )

    def __post_init__(self):
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        note_short_id: int,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        token_budget: int = 2000,
    ) -> str:
        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            return "错误：无法获取插件上下文。"
        plugin_context = event.plugin_context

        memory_sql_manager = plugin_context.get_component("memory_sql_manager")
        if memory_sql_manager is None:
            return "错误：memory_sql_manager 不可用，无法按 note_short_id 查询。"
        row = await memory_sql_manager.get_note_index_by_short_id(int(note_short_id))
        if not row:
            return f"错误：未找到 note_short_id={int(note_short_id)} 对应的笔记。"
        rel_path = str(row.get("source_file_path") or "").replace("\\", "/").strip().lstrip("/")
        resolved_short_id = int(row.get("note_short_id") or note_short_id)

        raw_dir = plugin_context.get_path_manager().get_raw_dir()
        target_path = (Path(raw_dir) / rel_path).resolve()
        raw_root = Path(raw_dir).resolve()
        if raw_root not in target_path.parents and target_path != raw_root:
            return "错误：source_file_path 非法（越界路径）。"
        if not target_path.exists():
            return f"错误：文件不存在：{rel_path}"

        try:
            text = target_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"错误：读取文件失败：{e}"

        all_lines = text.splitlines()
        total_lines = len(all_lines)

        req_start = int(start_line) if start_line is not None else 1
        req_end = int(end_line) if end_line is not None else total_lines
        if req_start > req_end:
            return "错误：start_line 不能大于 end_line。"

        if total_lines <= 0:
            actual_start_line = 0
            actual_end_line = 0
            output_raw = ""
        else:
            actual_start_line = min(max(1, req_start), total_lines)
            actual_end_line = min(max(1, req_end), total_lines)
            if actual_start_line > actual_end_line:
                actual_start_line = actual_end_line
            selected_lines = all_lines[actual_start_line - 1: actual_end_line]
            output_raw = "\n".join(selected_lines)

        output = truncate_by_tokens(output_raw, int(token_budget))
        truncated_by_token_budget = output != output_raw
        if truncated_by_token_budget and actual_start_line > 0:
            visible_lines = output.splitlines()
            if visible_lines:
                actual_end_line = min(
                    total_lines, actual_start_line + len(visible_lines) - 1
                )
            else:
                actual_end_line = actual_start_line
        return (
            f"[note_recall]\n"
            f"note_short_id: {resolved_short_id}\n"
            f"source_file_path: {rel_path}\n"
            f"total_lines: {total_lines}\n"
            f"actual_start_line: {actual_start_line}\n"
            f"actual_end_line: {actual_end_line}\n\n"
            f"truncated_by_token_budget: {str(truncated_by_token_budget).lower()}\n\n"
            f"{output}"
        )
