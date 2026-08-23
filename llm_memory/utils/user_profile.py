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


def extract_user_ids_from_tags(
    tags: Iterable[str],
    known_user_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    """从 tags 中提取全部用户 ID。

    一个记忆可关联多个用户（如「小明和小红是好友」的关系图谱），
    因此返回所有命中 ID 而非单个：
    1. 账本命中优先：返回所有命中 known_user_ids 的 tag。
    2. 无账本命中 → 退回形态判定，返回所有疑似 ID。
    """
    normalized = normalize_tags(tags)
    if known_user_ids:
        known = {str(uid).strip() for uid in known_user_ids if str(uid or "").strip()}
        ledger_hits = [tag for tag in normalized if tag in known]
        if ledger_hits:
            return ledger_hits
    return [tag for tag in normalized if is_user_id_tag(tag)]


def extract_user_id_from_tags(
    tags: Iterable[str],
    known_user_ids: Optional[Sequence[str]] = None,
) -> str:
    """从 tags 中提取用户 ID（单值兼容版）。

    恰好一个 ID 时返回它；多个 ID（记忆关联多个用户）时返回第一个，
    需要全部 ID 请使用 extract_user_ids_from_tags。
    """
    user_ids = extract_user_ids_from_tags(tags, known_user_ids=known_user_ids)
    return user_ids[0] if user_ids else ""


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
