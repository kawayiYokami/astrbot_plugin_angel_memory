from __future__ import annotations

from typing import Iterable, List


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


def is_user_id_tag(tag: str) -> bool:
    """判断 tag 是否为用户 ID。

    排除法规则：长度 > 5，且不含中文、不含空白、不是纯英文词形。
    纯数字、数字+字母混合、含下划线/连字符且含数字的平台标识符等算用户 ID。
    描述性标签（含中文、含空格、纯英文词形）一律不算。
    """
    text = str(tag or "").strip()
    if len(text) <= 5:
        return False
    # 含中文字符 → 描述性标签，不是平台 ID（如「1998年10月19日」）
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return False
    # 含空白 → 描述性短语，不是平台 ID（如「docker restart」）
    if any(c.isspace() for c in text):
        return False
    # 去掉连字符/下划线后是纯英文字母 → 词形用户名，不是 ID（如 Test-Bot、weixin_oc_adapter）
    if text.replace("-", "").replace("_", "").isalpha():
        return False
    return True


def extract_user_id_from_tags(tags: Iterable[str]) -> str:
    user_ids = [tag for tag in normalize_tags(tags) if is_user_id_tag(tag)]
    return user_ids[0] if len(user_ids) == 1 else ""


def extract_profile_attribute_from_tags(tags: Iterable[str]) -> str:
    for tag in normalize_tags(tags):
        if tag in PROFILE_ATTRIBUTE_TAGS:
            return tag
    return ""


def extract_user_nickname_from_tags(tags: Iterable[str]) -> str:
    for tag in normalize_tags(tags):
        if is_user_id_tag(tag) or tag in PROFILE_ATTRIBUTE_TAGS:
            continue
        return tag
    return ""


def is_user_profile_tags(tags: Iterable[str]) -> bool:
    normalized = normalize_tags(tags)
    return bool(extract_user_id_from_tags(normalized)) and bool(
        extract_profile_attribute_from_tags(normalized)
    )


def normalize_judgment(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())
