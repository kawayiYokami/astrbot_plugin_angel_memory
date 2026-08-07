from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


PROFILE_ATTRIBUTE_TAGS = {
    "用户别名",
    "事实属性",
    "技能树",
    "关系图谱",
    "活跃项目",
}


def normalize_tags(tags: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for tag in tags or []:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def is_user_id_tag(tag: str, known_user_ids: Optional[Sequence[str]] = None) -> bool:
    """判断 tag 是否为用户 ID。

    两级判定：
    1. known_user_ids（用户账本）非空且 tag 命中 → 直接判定为 ID，不猜形态。
    2. 账本未命中或为空 → 回退形态排除法兜底（冷启动、新用户、账本外 ID）。
    """
    text = str(tag or "").strip()
    if not text:
        return False
    if known_user_ids:
        known = {str(uid).strip() for uid in known_user_ids if str(uid or "").strip()}
        if text in known:
            return True
    # 形态兜底：长度 > 5，且不含中文、不含空白、不是纯英文词形
    if len(text) <= 5:
        return False
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return False
    if any(c.isspace() for c in text):
        return False
    if text.replace("-", "").replace("_", "").isalpha():
        return False
    return True


def extract_user_id_from_tags(
    tags: Iterable[str],
    known_user_ids: Optional[Sequence[str]] = None,
) -> str:
    """从 tags 中提取用户 ID。

    账本命中优先：返回 tags 中第一个命中 known_user_ids 的 tag。
    兜底：退回形态判定，仅当恰好一个疑似 ID 时返回。
    """
    normalized = normalize_tags(tags)
    if known_user_ids:
        known = {str(uid).strip() for uid in known_user_ids if str(uid or "").strip()}
        for tag in normalized:
            if tag in known:
                return tag
    user_ids = [tag for tag in normalized if is_user_id_tag(tag)]
    return user_ids[0] if len(user_ids) == 1 else ""


def extract_profile_attribute_from_tags(tags: Iterable[str]) -> str:
    for tag in normalize_tags(tags):
        if tag in PROFILE_ATTRIBUTE_TAGS:
            return tag
    return ""


def extract_user_nickname_from_tags(
    tags: Iterable[str],
    known_user_ids: Optional[Sequence[str]] = None,
) -> str:
    for tag in normalize_tags(tags):
        if is_user_id_tag(tag, known_user_ids=known_user_ids) or tag in PROFILE_ATTRIBUTE_TAGS:
            continue
        return tag
    return ""


def is_user_profile_tags(
    tags: Iterable[str],
    known_user_ids: Optional[Sequence[str]] = None,
) -> bool:
    normalized = normalize_tags(tags)
    return bool(extract_user_id_from_tags(normalized, known_user_ids=known_user_ids)) and bool(
        extract_profile_attribute_from_tags(normalized)
    )


def normalize_judgment(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())
