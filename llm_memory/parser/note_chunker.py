"""
笔记切片器

将 Markdown/TXT 文件切分为段落级切片，每个切片带行号定位。
设计参考 story 项目的 _chunk_markdown 模型。

核心设计：
- 第 0 行为虚拟路径行（相对路径），参与后续检索
- 真实内容从第 1 行开始
- 按段落（空行）为自然边界切分
- 代码块和表格保持完整不拆分
- 超长段落按字符数强制截断
"""

import re
from typing import List, Dict


# 切片最大字符数（清洗后纯文本计）
MAX_CHUNK_CHARS = 220


def chunk_file(
    content: str,
    relative_path: str,
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> List[Dict]:
    """
    将文件内容切分为切片列表。

    第 0 行是虚拟路径行（relative_path），真实内容从第 1 行开始。
    每个切片包含原始文本（含 Markdown 标记），行号用于后续定位。

    Args:
        content: 文件原始文本内容
        relative_path: 文件相对路径（作为第 0 行）
        max_chunk_chars: 单个切片最大字符数

    Returns:
        切片列表，每项:
            - chunk_index: int 切片序号（从 1 开始）
            - line_start: int 起始行（0 = 路径虚拟行）
            - line_end: int 结束行
            - content: str 切片原始文本
    """
    if not content or not content.strip():
        # 空文件只产出路径行切片
        return [
            {
                "chunk_index": 1,
                "line_start": 0,
                "line_end": 0,
                "content": relative_path,
            }
        ]

    lines = content.split("\n")

    # 先识别特殊块（代码块、表格），标记行号范围
    special_ranges = _identify_special_ranges(lines)

    # 按段落切分（空行为边界），特殊块保持完整
    raw_chunks = _split_by_paragraphs(lines, special_ranges, max_chunk_chars)

    # 组装最终结果：第一个切片包含路径行
    chunks: List[Dict] = []
    chunk_index = 1

    for raw_chunk in raw_chunks:
        line_start = raw_chunk["line_start"]
        line_end = raw_chunk["line_end"]
        text = raw_chunk["text"]

        if not text.strip():
            continue

        # 第一个切片前置路径行
        if chunk_index == 1:
            chunk_content = relative_path + "\n" + text
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "line_start": 0,
                    "line_end": line_end,
                    "content": chunk_content,
                }
            )
        else:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "line_start": line_start,
                    "line_end": line_end,
                    "content": text,
                }
            )
        chunk_index += 1

    # 兜底：如果内容全是空行，至少产出路径行
    if not chunks:
        chunks.append(
            {
                "chunk_index": 1,
                "line_start": 0,
                "line_end": 0,
                "content": relative_path,
            }
        )

    return chunks


def _identify_special_ranges(lines: List[str]) -> List[Dict]:
    """
    识别代码块和表格的行号范围（这些区域不拆分）。

    Returns:
        [{"start": int, "end": int, "type": "code"|"table"}, ...]
        行号从 1 开始（对应文件真实行号）
    """
    ranges = []
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]

        # 代码块：``` 开始到 ``` 结束
        if re.match(r"^```", line):
            start = i + 1  # 转为 1-based
            i += 1
            while i < total and not re.match(r"^```\s*$", lines[i]):
                i += 1
            end = i + 1 if i < total else total
            ranges.append({"start": start, "end": end, "type": "code"})
            i += 1
            continue

        # 表格：连续的 | 开头行
        if re.match(r"^\|", line):
            start = i + 1
            while i < total and re.match(r"^\|", lines[i]):
                i += 1
            end = i  # 最后一个表格行
            ranges.append({"start": start, "end": end, "type": "table"})
            continue

        i += 1

    return ranges


def _is_in_special_range(line_1based: int, special_ranges: List[Dict]) -> bool:
    """检查某行是否在特殊块范围内"""
    for r in special_ranges:
        if r["start"] <= line_1based <= r["end"]:
            return True
    return False


def _get_special_range_at(line_1based: int, special_ranges: List[Dict]) -> Dict:
    """获取某行所在的特殊块范围"""
    for r in special_ranges:
        if r["start"] <= line_1based <= r["end"]:
            return r
    return None


def _split_by_paragraphs(
    lines: List[str],
    special_ranges: List[Dict],
    max_chunk_chars: int,
) -> List[Dict]:
    """
    按段落（空行）切分，特殊块保持完整。

    Returns:
        [{"line_start": int, "line_end": int, "text": str}, ...]
        行号从 1 开始
    """
    chunks: List[Dict] = []
    current_lines: List[str] = []
    current_start = 1
    current_char_count = 0

    i = 0
    total = len(lines)

    while i < total:
        line_1based = i + 1

        # 如果当前行在特殊块内，整块收入
        special = _get_special_range_at(line_1based, special_ranges)
        if special is not None:
            # 先 flush 当前累积的普通文本
            if current_lines:
                text = "\n".join(current_lines)
                if text.strip():
                    chunks.append(
                        {
                            "line_start": current_start,
                            "line_end": line_1based - 1,
                            "text": text,
                        }
                    )
                current_lines = []
                current_char_count = 0

            # 收入整个特殊块
            block_start = special["start"]
            block_end = special["end"]
            block_lines = lines[block_start - 1 : block_end]
            block_text = "\n".join(block_lines)
            if block_text.strip():
                chunks.append(
                    {
                        "line_start": block_start,
                        "line_end": block_end,
                        "text": block_text,
                    }
                )

            # 跳过整个特殊块
            i = block_end
            current_start = i + 1
            continue

        line = lines[i]
        is_blank = not line.strip()

        if is_blank:
            # 空行 = 段落边界，flush
            if current_lines:
                text = "\n".join(current_lines)
                if text.strip():
                    chunks.append(
                        {
                            "line_start": current_start,
                            "line_end": i,  # 空行前一行（0-based i 对应 line i+1，但空行本身不算内容）
                            "text": text,
                        }
                    )
                current_lines = []
                current_char_count = 0
            current_start = i + 2  # 下一个非空行（1-based）
            i += 1
            continue

        line_len = len(line)

        # 如果累积字符超限，先 flush
        if current_lines and current_char_count + line_len > max_chunk_chars:
            text = "\n".join(current_lines)
            if text.strip():
                chunks.append(
                    {
                        "line_start": current_start,
                        "line_end": i,  # 当前行之前
                        "text": text,
                    }
                )
            current_lines = []
            current_char_count = 0
            current_start = line_1based

        if not current_lines:
            current_start = line_1based

        current_lines.append(line)
        current_char_count += line_len
        i += 1

    # flush 剩余
    if current_lines:
        text = "\n".join(current_lines)
        if text.strip():
            chunks.append(
                {
                    "line_start": current_start,
                    "line_end": total,
                    "text": text,
                }
            )

    return chunks
