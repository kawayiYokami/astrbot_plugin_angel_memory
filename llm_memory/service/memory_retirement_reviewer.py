"""记忆淘汰审查器。

睡眠回收阶段把强度归零的候选交给一次独立的 LLM 裁决：明确 delete 的删除，
明确 keep 的升级为永久记忆，未出现在裁决表里的保持原状。任何失败（找不到
provider、调用超时、返回无法解析）都抛出 RetirementReviewError，由调用方整批
跳过，一条都不删。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..models.data_models import BaseMemory
from ..prompts.prompt_assembler import PromptAssembler


class RetirementReviewError(Exception):
    """淘汰审查失败，携带已发出的提示词与已收到的返回，供 trace 记录。"""

    def __init__(
        self,
        message: str,
        *,
        prompt: str = "",
        system_prompt: str = "",
        response: str = "",
    ):
        super().__init__(message)
        self.prompt = str(prompt or "")
        self.system_prompt = str(system_prompt or "")
        self.response = str(response or "")


@dataclass
class RetirementReviewOutcome:
    """一次淘汰审查的结果。"""

    decisions: Dict[str, str]
    prompt: str
    system_prompt: str
    response: str
    candidate_count: int


class MemoryRetirementReviewer:
    """把候选记忆清单交给 LLM，回收「记忆编号 -> 裁决」表。"""

    REVIEW_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        context: Any,
        provider_id: str,
        logger: Any = None,
        id_token: str = "",
    ):
        self.context = context
        self.provider_id = str(provider_id or "")
        self.logger = logger
        # 清单编号带一次性后缀：提示词示例里的 m0/m1 与本次候选不同域，
        # 模型复述示例也不会映射到真实记忆
        self._id_token = str(id_token or uuid.uuid4().hex[:4])

    async def review(self, candidates: Sequence[BaseMemory]) -> RetirementReviewOutcome:
        """审查候选列表；失败抛 RetirementReviewError，成功返回裁决表。"""
        candidate_list = [mem for mem in (candidates or []) if getattr(mem, "id", None)]
        short_ids = self._build_short_ids(len(candidate_list))
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(candidate_list, short_ids)

        provider = (
            self.context.get_provider_by_id(self.provider_id) if self.context else None
        )
        if provider is None:
            raise RetirementReviewError(
                f"找不到提供者: {self.provider_id}",
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

        try:
            llm_response = await asyncio.wait_for(
                provider.text_chat(prompt=user_prompt, system_prompt=system_prompt),
                timeout=self.REVIEW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise RetirementReviewError(
                f"审查调用超时（{int(self.REVIEW_TIMEOUT_SECONDS)} 秒）",
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            raise RetirementReviewError(
                f"审查调用失败: {e}",
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

        response_text = str(getattr(llm_response, "completion_text", "") or "")
        if not response_text.strip():
            raise RetirementReviewError(
                "审查调用返回空响应",
                prompt=user_prompt,
                system_prompt=system_prompt,
                response=response_text,
            )

        try:
            decisions = self._parse_decisions(response_text, candidate_list, short_ids)
        except ValueError as e:
            raise RetirementReviewError(
                str(e),
                prompt=user_prompt,
                system_prompt=system_prompt,
                response=response_text,
            )

        return RetirementReviewOutcome(
            decisions=decisions,
            prompt=user_prompt,
            system_prompt=system_prompt,
            response=response_text,
            candidate_count=len(candidate_list),
        )

    def _build_short_ids(self, count: int) -> List[str]:
        return [f"m{index}-{self._id_token}" for index in range(count)]

    def build_system_prompt(self) -> str:
        """审查规则，内容固定，便于命中提示词缓存。"""
        return PromptAssembler.build_memory_retirement_review()

    def build_user_prompt(
        self,
        candidates: Sequence[BaseMemory],
        short_ids: Sequence[str],
    ) -> str:
        """本次候选清单与输出要求。"""
        listing = self._build_listing(candidates, short_ids)
        return (
            "# 记忆清单\n\n"
            f"{listing}\n\n"
            "请按以上规则逐条裁决，仅输出 JSON。\n"
        )

    def _build_listing(
        self,
        candidates: Sequence[BaseMemory],
        short_ids: Sequence[str],
    ) -> str:
        now = time.time()
        blocks: List[str] = []
        for index, mem in enumerate(candidates):
            blocks.append(f"- id: `{short_ids[index]}`")
            blocks.append(f"  type: `{self._type_label(mem)}`")
            blocks.append(f"  judgment: `{getattr(mem, 'judgment', '') or ''}`")
            blocks.append(f"  reasoning: `{getattr(mem, 'reasoning', '') or ''}`")
            blocks.append(f"  tags: {self._tags_text(mem)}")
            blocks.append(
                f"  创建于: `{self._age_text(getattr(mem, 'created_at', 0.0), now)}`"
                f"　最后召回: `{self._recalled_text(getattr(mem, 'last_recalled_at', 0.0), now)}`"
                f"　被判定有用: `{int(getattr(mem, 'useful_count', 0) or 0)} 次`"
            )
            blocks.append("")
        return "\n".join(blocks).rstrip()

    @staticmethod
    def _type_label(mem: BaseMemory) -> str:
        memory_type = getattr(mem, "memory_type", "")
        return str(getattr(memory_type, "value", memory_type) or "")

    @staticmethod
    def _tags_text(mem: BaseMemory) -> str:
        tags = [str(tag) for tag in (getattr(mem, "tags", None) or []) if str(tag).strip()]
        if not tags:
            return "（无）"
        return "、".join(f"`{tag}`" for tag in tags)

    @staticmethod
    def _age_text(created_at: float, now: float) -> str:
        try:
            days = int(max(0.0, now - float(created_at or 0.0)) // 86400)
        except (TypeError, ValueError):
            return "未知"
        return "今天" if days == 0 else f"{days} 天前"

    @staticmethod
    def _recalled_text(last_recalled_at: float, now: float) -> str:
        try:
            value = float(last_recalled_at or 0.0)
        except (TypeError, ValueError):
            return "从未"
        if value <= 0:
            return "从未"
        days = int(max(0.0, now - value) // 86400)
        return "今天" if days == 0 else f"{days} 天前"

    def _parse_decisions(
        self,
        text: str,
        candidates: Sequence[BaseMemory],
        short_ids: Sequence[str],
    ) -> Dict[str, str]:
        """把模型返回解析为「真实记忆 ID -> 裁决」表；无法解析时抛 ValueError。"""
        data = self._extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("审查返回无法解析为 JSON 对象")

        raw_decisions = data.get("decisions")
        if not isinstance(raw_decisions, list):
            raise ValueError("审查返回缺少 decisions 列表")

        short_id_map = {
            str(short_ids[index]): str(mem.id) for index, mem in enumerate(candidates)
        }
        decisions: Dict[str, str] = {}
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            short_id = str(item.get("id", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if short_id not in short_id_map or verdict not in {"keep", "delete"}:
                continue
            decisions[short_id_map[short_id]] = verdict
        return decisions

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
                return parsed
            except (json.JSONDecodeError, ValueError):
                continue
        return None
