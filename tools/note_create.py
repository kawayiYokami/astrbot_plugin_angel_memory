import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class NoteCreateTool(FunctionTool):
    name: str = "note_create"
    description: str = (
        "把成体系的知识整理成一篇 Markdown 笔记并永久存档（自动纳入知识库，之后可检索）。"
        "两种时机会想写笔记：一是用户明确要求你记笔记、整理资料、写总结时（必须写）；"
        "二是当你在学习或交流中积累了不少心得、值得沉淀下来时，可以主动写一篇。"
        "用多级标题（#/##）分节组织，方便日后按段检索与展开。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "笔记标题，简短概括主题。例：'Python异步编程'",
                },
                "content": {
                    "type": "string",
                    "description": "笔记正文，Markdown 格式，建议用多级标题（#/##）组织结构。",
                },
            },
            "required": ["title", "content"],
        }
    )

    def __post_init__(self):
        self.logger = logger

    @staticmethod
    def _sanitize_title(title: str) -> str:
        """清洗标题中的非法文件名字符"""
        safe = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", str(title or "").strip())
        safe = safe.strip(". ")
        return safe[:80] if safe else "untitled"

    async def run(
        self,
        event: AstrMessageEvent,
        title: str,
        content: str,
    ) -> str:
        self.logger.debug(f"{self.name} - LLM 调用: title='{title}'")

        if not str(content or "").strip():
            return "错误：笔记内容不能为空。"

        if not hasattr(event, "plugin_context") or event.plugin_context is None:
            return "错误：无法获取插件上下文。"
        plugin_context = event.plugin_context

        note_service = plugin_context.get_component("note_service")
        if note_service is None:
            return "错误：笔记服务不可用，系统可能仍在初始化中。"

        try:
            raw_dir = plugin_context.get_path_manager().get_raw_dir()
        except Exception as e:
            self.logger.error(f"{self.name}: 获取 raw 目录失败: {e}")
            return "错误：无法获取笔记目录。"

        # 文件名：安全标题 + 当地时间戳（精确到秒）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._sanitize_title(title)
        filename = f"{safe_title}_{timestamp}.md"

        # AI 自主生成的笔记统一存放在 raw/.angel/note/，与用户手动放入的文件区分开
        rel_dir = ".angel/note"
        notes_dir = Path(raw_dir) / ".angel" / "note"
        notes_dir.mkdir(parents=True, exist_ok=True)
        target_path = notes_dir / filename
        relative_path = f"{rel_dir}/{filename}"

        # 正文首行补上标题（若 LLM 未自带 H1）
        body = str(content).strip()
        if not body.lstrip().startswith("#"):
            body = f"# {title}\n\n{body}"

        try:
            target_path.write_text(body, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"{self.name}: 写入笔记失败: {e}", exc_info=True)
            return f"错误：写入笔记失败：{e}"

        # 主动触发索引，确保立即可检索（相对路径需与文件监控扫描结果一致）
        try:
            import asyncio
            await asyncio.to_thread(
                note_service.parse_and_store_file_sync, str(target_path), relative_path
            )
        except Exception as e:
            self.logger.warning(f"{self.name}: 笔记已写入但索引同步失败（稍后扫描会补上）: {e}", exc_info=True)
            return f"笔记《{title}》已保存为 {filename}，但索引同步稍有延迟，稍后即可检索。"

        self.logger.info(f"{self.name}: 完成 笔记={filename} 标题={title}")
        return f"学习笔记《{title}》已保存并归档为 {filename}，已纳入知识库索引，之后可通过检索或 note_recall 查看。"
