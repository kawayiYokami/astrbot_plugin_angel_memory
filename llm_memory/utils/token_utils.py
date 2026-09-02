"""
Token 工具模块

提供基于字符规则的轻量级 Token 估算功能。
不再依赖 tiktoken，以提升性能和减少依赖。

规则（v1.6.6 对齐实测）：
1 个英文字符 ≈ 0.25 个 token  (≈4 字符/token，对齐 deepseek-harness CHARS_PER_TOKEN=4)
1 个中文字符 ≈ 0.48 个 token  (实测 README 8838字符=2946token 反推，0.25/0.48 误差 3.6% 偏保守)
校准样本：README.md 5166 ASCII + 3672 非ASCII → 3054 token ≈ 实测 2946
"""

import math

# 对齐实测的固定密度（保守略高，避免超 8192 被 400）
_ASCII_TOKENS = 0.25
_NON_ASCII_TOKENS = 0.48

def count_tokens(text: str) -> int:
    """
    计算文本的 token 数量（基于字符规则估算）

    规则：
    - ASCII 字符 (英文、数字、符号等): 0.25 token
    - 非 ASCII 字符 (中文等): 0.48 token

    Args:
        text: 要计算的文本

    Returns:
        估算的 token 数量
    """
    if not text:
        return 0

    token_count = 0.0
    for char in text:
        if ord(char) < 128:
            token_count += _ASCII_TOKENS
        else:
            token_count += _NON_ASCII_TOKENS

    return math.ceil(token_count)


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """
    按 token 数量截断文本（基于字符规则估算）

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数量

    Returns:
        截断后的文本
    """
    if not text:
        return ""

    current_tokens = 0.0
    truncated_text = []

    for char in text:
        char_tokens = _ASCII_TOKENS if ord(char) < 128 else _NON_ASCII_TOKENS

        if current_tokens + char_tokens > max_tokens:
            break

        current_tokens += char_tokens
        truncated_text.append(char)

    return "".join(truncated_text)


def truncate_by_tokens_from_end(text: str, max_tokens: int) -> str:
    """
    按 token 数量从后往前截断文本（保留末尾部分）

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数量（必须 >= 0）

    Returns:
        截断后的文本（保留末尾）
    """
    if max_tokens < 0:
        raise ValueError("max_tokens 必须大于或等于 0")

    if not text:
        return ""

    current_tokens = 0.0
    truncated_text = []

    # 反向遍历
    for char in reversed(text):
        char_tokens = _ASCII_TOKENS if ord(char) < 128 else _NON_ASCII_TOKENS

        if current_tokens + char_tokens > max_tokens:
            break

        current_tokens += char_tokens
        truncated_text.append(char)

    return "".join(reversed(truncated_text))
