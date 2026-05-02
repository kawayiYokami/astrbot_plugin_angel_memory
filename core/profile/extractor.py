"""
画像提取器 — 从对话消息中提取用户事实标签
Pipeline: 规则粗筛 → LLM 精提 → 同类型内去重 → 更新覆盖
"""
import re
import json
import asyncio
import logging
from typing import Any

from .config import ProfileExtractionConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 提取 Prompt
# ------------------------------------------------------------

EXTRACTION_PROMPT = """你是一个用户画像提取器。从以下对话消息中提取用户的事实性标签。

规则：
1. 只提取用户**主动透露**的、不敏感的公开信息
2. 标签类型仅限：{green_tags}
3. 输出格式：{{"tags": [{{"type": "标签类型", "value": "事实内容", "temporal": "present/past"}}]}}
4. 值不超过30字，是摘要不是原文
5. 没有可提取内容时返回 {{"tags": []}}
6. 严禁提取：密码、号码、地址、亲密话题
7. 区分当前状态(present)和历史陈述(past)——"我以前当过老师"是past

消息内容：
{message}

请只输出JSON，不要输出其他内容。"""

# ------------------------------------------------------------
# 画像条目
# ------------------------------------------------------------

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
TAG_TYPE_RE = re.compile(r"^[a-z_]+$")


def _validate_tag(tag: dict, allowed_types: set | None = None) -> bool:
    """校验单条标签是否合法"""
    if not isinstance(tag, dict):
        return False
    tag_type = tag.get("type", "")
    value = tag.get("value", "")
    if not isinstance(tag_type, str) or not isinstance(value, str):
        return False
    if not TAG_TYPE_RE.match(tag_type):
        return False
    if allowed_types and tag_type not in allowed_types:
        return False
    if not value or len(value) > 30:
        return False
    return True


class ProfileExtractor:
    """画像提取器"""

    def __init__(self, config: ProfileExtractionConfig | None = None):
        self.config = config or ProfileExtractionConfig()
        # 频率控制：{user_id: 上次提取时间戳}
        self._last_extraction: dict[str, float] = {}
        self._extract_cooldown: float = 300.0  # 5 分钟冷却

    # ---- 规则粗筛 ----

    def should_skip(self, text: str) -> bool:
        """是否跳过此消息（不过提取）"""
        stripped = text.strip()
        if len(stripped) < self.config.min_message_length:
            return True
        if CODE_BLOCK_RE.search(stripped):
            return True
        if self.config.contains_red(stripped):
            return True
        return False

    # ---- LLM 提取 ----

    def build_prompt(self, message: str) -> str:
        """构建 LLM 提取提示词，注入当前绿灯+已启用的黄灯标签"""
        allowed = [*self.config.green_tags, *self.config.yellow_tags]
        return EXTRACTION_PROMPT.format(green_tags=", ".join(allowed), message=message)

    async def extract(
        self,
        message: str,
        context: Any = None,
        provider_id: str = "",
    ) -> list[dict]:
        """
        从单条消息中提取画像标签。

        Returns:
            [{"type": "preferred_name", "value": "小貔貅", "temporal": "present"}, ...]
        """
        if self.should_skip(message):
            return []

        if not context or not provider_id:
            return []  # 无 LLM 可用时静默跳过

        try:
            provider = context.get_provider_by_id(provider_id)
            if not provider:
                return []

            prompt = self.build_prompt(message)
            llm_response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt),
                timeout=30.0,
            )

            if not llm_response or not getattr(llm_response, "completion_text", ""):
                return []

            response_text = llm_response.completion_text

            # 解析 JSON
            data = self._parse_json(response_text)
            if not data:
                return []

            tags = data.get("tags", [])
            if not isinstance(tags, list):
                return []

            allowed_types = set(self.config.green_tags) | set(self.config.yellow_tags)
            return [t for t in tags if _validate_tag(t, allowed_types=allowed_types)]

        except Exception:
            logger.error("[画像提取] LLM提取失败", exc_info=True)
            return []

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """从 LLM 响应中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试从markdown代码块中提取
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None
