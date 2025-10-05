"""
Token 工具模块

提供基于 tiktoken 的 token 计算功能。
"""

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None

# 默认使用的编码器
_DEFAULT_ENCODING = "cl100k_base"  # 适用于 GPT-3.5 和 GPT-4

# 编码器实例缓存
_encoder_cache = {}


def get_tokenizer(encoding_name: str = _DEFAULT_ENCODING):
    """
    获取 tokenizer 实例
    
    Args:
        encoding_name: 编码器名称，默认为 cl100k_base
        
    Returns:
        tiktoken 编码器实例
    """
    global _encoder_cache
    
    if not _TIKTOKEN_AVAILABLE:
        raise ImportError(
            "tiktoken library is not installed. "
            "Please install it with: pip install tiktoken"
        )
    
    if encoding_name not in _encoder_cache:
        _encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    
    return _encoder_cache[encoding_name]


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """
    计算文本的 token 数量

    Args:
        text: 要计算的文本
        encoding_name: 编码器名称，默认为 cl100k_base

    Returns:
        token 数量
    """
    if not text:
        return 0

    try:
        tokenizer = get_tokenizer(encoding_name)
        return len(tokenizer.encode(text))
    except (ValueError, RuntimeError):
        # 编码器相关错误：不支持的编码或运行时错误
        # 回退到简单的字符计数（除以4近似）
        return len(text) // 4
    except Exception as e:
        # 其他预期外的错误，重新抛出以保持追踪
        raise RuntimeError(f"Token counting failed: {e}") from e


def truncate_by_tokens(text: str, max_tokens: int, encoding_name: str = _DEFAULT_ENCODING) -> str:
    """
    按 token 数量截断文本

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数量
        encoding_name: 编码器名称，默认为 cl100k_base

    Returns:
        截断后的文本
    """
    if not text:
        return ""

    try:
        tokenizer = get_tokenizer(encoding_name)
        tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    except (ValueError, RuntimeError):
        # 编码器相关错误：不支持的编码或运行时错误
        # 回退到简单的字符截断
        return text[:max_tokens * 4]  # 粗略估计
    except Exception as e:
        # 其他预期外的错误，重新抛出以保持追踪
        raise RuntimeError(f"Token truncation failed: {e}") from e