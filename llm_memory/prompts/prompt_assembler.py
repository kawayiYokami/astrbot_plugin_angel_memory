from __future__ import annotations

from pathlib import Path
from typing import List

from ..utils.path_manager import PathManager


class PromptAssembler:
    """组装反思提示词 sections，避免手改单一巨型文件。"""

    SECTION_FILES = [
        "00_intro.md",
        "10_output_schema.md",
        "20_actions.md",
        "30_memory_fields.md",
        "40_user_profiles.md",
        "50_generation_rules.md",
        "60_examples.md",
        "70_checklist.md",
    ]

    @classmethod
    def get_section_paths(cls) -> List[Path]:
        sections_dir = PathManager.get_prompt_sections_dir()
        return [sections_dir / filename for filename in cls.SECTION_FILES]

    @classmethod
    def build_memory_system_guide(cls) -> str:
        parts: List[str] = []
        for path in cls.get_section_paths():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts).strip() + "\n"
