"""
Token 工具模块

提供基于字符规则的轻量级 Token 估算功能。
不再依赖 tiktoken，以提升性能和减少依赖。

规则：
1 个英文字符 ≈ 0.3 个 token
1 个中文字符 ≈ 0.6 个 token
"""

import math

def count_tokens(text: str) -> int:
    """
    计算文本的 token 数量（基于字符规则估算）

    规则：
    - ASCII 字符 (英文、数字、符号等): 0.3 token
    - 非 ASCII 字符 (中文等): 0.6 token

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
            token_count += 0.3
        else:
            token_count += 0.6

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
        char_tokens = 0.3 if ord(char) < 128 else 0.6

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
        char_tokens = 0.3 if ord(char) < 128 else 0.6

        if current_tokens + char_tokens > max_tokens:
            break

        current_tokens += char_tokens
        truncated_text.append(char)

    return "".join(reversed(truncated_text))
