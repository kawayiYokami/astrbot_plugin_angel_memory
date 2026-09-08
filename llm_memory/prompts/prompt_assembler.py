from __future__ import annotations

from ..utils.path_manager import PathManager


class PromptAssembler:
    """组装反思提示词。指南为单一文件，便于整体审阅约束之间的冲突。"""

    @classmethod
    def get_guide_path(cls) -> str:
        """获取记忆系统指南文件路径。"""
        return str(PathManager.get_prompt_guide_path())

    @classmethod
    def build_memory_system_guide(cls) -> str:
        guide_path = PathManager.get_prompt_guide_path()
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read().strip() + "\n"
