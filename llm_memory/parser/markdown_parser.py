"""
Markdown解析器

用于解析Markdown文档并返回文档块列表。
"""

import re
import hashlib
from typing import List, Dict, Any, Set
from ..models.document_models import DocumentBlock
from ..utils.token_utils import count_tokens


class MarkdownParser:
    """Markdown文档解析器"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS: Set[str] = {'.md', '.txt'}

    # 块大小控制（以token为单位）
    MAX_BLOCK_SIZE = 100  # 最大块大小
    MIN_BLOCK_SIZE = 50   # 最小块大小

    def __init__(self):
        """初始化解析器"""
        pass

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """
        检查是否支持指定的文件扩展名

        Args:
            extension: 文件扩展名（如 '.md'）

        Returns:
            是否支持
        """
        return extension.lower() in cls.SUPPORTED_EXTENSIONS

    def parse(self, markdown_content: str, file_path: str = "") -> List[DocumentBlock]:
        """
        解析Markdown内容

        Args:
            markdown_content: Markdown文本内容
            file_path: 文件路径（用于生成标签）

        Returns:
            文档块列表（所有字段完备，可直接使用）
        """
        lines = markdown_content.split('\n')

        # 第一步：扫描所有标题位置
        headers = self._scan_headers(lines)

        # 第二步：识别章节（标题与标题之间的所有内容）
        sections = self._identify_sections(lines, headers)

        # 第三步：为每个章节提取标签
        for section in sections:
            # 路径标签
            path_tags = self._extract_path_tags(file_path)

            # 标题标签
            header_tags = self._find_header_tags(section['start_line'], headers)

            # 内容标签
            content_tags = self._extract_content_tags(section['content'])

            # 合并所有标签
            section['tags'] = path_tags + header_tags + content_tags

        # 第四步：拆分（只有>MAX_BLOCK_SIZE时才拆分）
        all_blocks = []
        for section in sections:
            blocks = self._split_section(section)
            all_blocks.extend(blocks)

        # 第五步：按起始行排序
        all_blocks.sort(key=lambda b: b['start_line'])

        # 第六步：创建DocumentBlock对象
        document_blocks = []
        for block in all_blocks:
            doc_block = DocumentBlock(
                id="",
                content=block['content'],
                tags=block['tags'],
                tag_vector=[],
                source_file_hash=self._calculate_file_hash(file_path),
                created_at=0,
                related_block_ids=[],
                source_file_path=file_path  # 添加文件路径信息
            )
            document_blocks.append(doc_block)

        # 第七步：建立关联关系
        self._build_related_blocks(document_blocks)

        return document_blocks

    def _scan_headers(self, lines: List[str]) -> List[Dict[str, Any]]:
        """扫描所有标题位置"""
        headers = []
        for i, line in enumerate(lines):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({
                    'level': level,
                    'text': text,
                    'line': i
                })
        return headers

    def _identify_sections(self, lines: List[str], headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别章节（标题与标题之间的所有内容）"""
        sections = []
        header_lines = [h['line'] for h in headers] + [len(lines)]

        for i in range(len(header_lines) - 1):
            start_line = header_lines[i] + 1
            end_line = header_lines[i + 1]

            section_lines = []
            for line_num in range(start_line, end_line):
                if line_num < len(lines):
                    section_lines.append(lines[line_num])

            content = '\n'.join(section_lines).strip()

            if content:
                sections.append({
                    'content': content,
                    'start_line': start_line,
                    'end_line': end_line - 1
                })

        return sections

    def _find_header_tags(self, start_line: int, headers: List[Dict[str, Any]]) -> List[str]:
        """从起始行往上找标题标签"""
        prev_headers = [h for h in headers if h['line'] < start_line]
        if not prev_headers:
            return []

        headers_by_level = {}
        for h in prev_headers:
            headers_by_level[h['level']] = h

        tags = []
        for level in sorted(headers_by_level.keys()):
            tags.append(headers_by_level[level]['text'])

        return tags

    def _extract_path_tags(self, file_path: str) -> List[str]:
        """从文件路径提取标签"""
        tags = []
        path_parts = file_path.replace('\\', '/').split('/')
        for part in path_parts:
            if part:
                clean_part = part.replace('.md', '').replace('.txt', '')
                if clean_part:
                    tags.append(clean_part)
        return tags

    def _extract_content_tags(self, content: str) -> List[str]:
        """从内容中提取标签"""
        tags = []

        # 加粗标签（所有加粗内容）
        bold_matches = re.findall(r'\*\*([^*]+)\*\*', content)
        for match in bold_matches:
            # 去掉所有标点符号
            clean_match = re.sub(r'[^\w\s]', '', match)
            if clean_match and len(clean_match.strip()) > 1:
                tags.append(clean_match.strip())

        # 专有名词标签
        proper_noun_patterns = [
            r'"([^"]+)"',      # 半角双引号
            r'"([^"]+)"',      # 全角双引号
            r'"([^"]+)"',      # 竖版双引号
            r'「([^」]+)」',    # 日式直角引号
            r'『([^』]+)』',    # 日式双直角引号
            r'《([^》]+)》',    # 书名号
            r'〈([^〉]+)〉',    # 尖括号
        ]

        for pattern in proper_noun_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 去掉所有标点符号
                clean_match = re.sub(r'[^\w\s]', '', match)
                if clean_match and len(clean_match.strip()) > 1:
                    tags.append(clean_match.strip())

        # 去重和清理
        tags = list(dict.fromkeys(tags))
        tags = [tag.strip() for tag in tags if tag.strip() and len(tag.strip()) > 1]

        return tags

    def _split_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        拆分章节（独立方法，保证100%继承标签）

        优先按代码块和表格拆分，然后对文本块判断是否需要进一步拆分
        拆分优先级：代码块 → 表格 → \n\n → \n → 句号 → 强制拆分

        Args:
            section: 章节，包含 content, tags, start_line

        Returns:
            块列表，每个块100%继承section的tags
        """
        content = section['content']
        tags = section['tags']  # 原始标签，必须100%继承
        start_line = section['start_line']

        # 第一级拆分：代码块和表格（不管是否超长都要拆分）
        primary_blocks = self._split_by_special_blocks(content, tags, start_line)

        # 第二级拆分：对过长的文本块进一步拆分
        final_blocks = []
        for block in primary_blocks:
            # 代码块和表格不再拆分
            if block.get('is_code') or block.get('is_table'):
                final_blocks.append(block)
            else:
                # 文本块如果还是太长，继续拆分
                if count_tokens(block['content']) > self.MAX_BLOCK_SIZE:
                    sub_blocks = self._split_text_block(block)
                    final_blocks.extend(sub_blocks)
                else:
                    final_blocks.append(block)

        return final_blocks

    def _split_by_special_blocks(self, content: str, tags: List[str], start_line: int) -> List[Dict[str, Any]]:
        """
        第一级拆分：按代码块和表格拆分

        Args:
            content: 内容
            tags: 标签（必须100%继承）
            start_line: 起始行

        Returns:
            块列表，每个块都继承tags
        """
        lines = content.split('\n')
        blocks = []
        i = 0

        while i < len(lines):
            # 识别代码块
            if re.match(r'^```(\w*)$', lines[i]):
                re.match(r'^```(\w*)$', lines[i]).group(1)
                code_lines = [lines[i]]
                i += 1

                while i < len(lines):
                    code_lines.append(lines[i])
                    if re.match(r'^```$', lines[i]):
                        break
                    i += 1

                blocks.append({
                    'content': '\n'.join(code_lines),
                    'tags': tags.copy(),  # 100%继承
                    'start_line': start_line,
                    'is_code': True
                })
                i += 1
                continue

            # 识别表格
            if re.match(r'^\|', lines[i]):
                table_lines = []
                while i < len(lines) and (re.match(r'^\|', lines[i]) or not lines[i].strip()):
                    if lines[i].strip():
                        table_lines.append(lines[i])
                    i += 1

                if table_lines:
                    blocks.append({
                        'content': '\n'.join(table_lines),
                        'tags': tags.copy(),  # 100%继承
                        'start_line': start_line,
                        'is_table': True
                    })
                continue

            # 普通文本
            text_lines = []
            while i < len(lines):
                if re.match(r'^```', lines[i]) or re.match(r'^\|', lines[i]):
                    break
                text_lines.append(lines[i])
                i += 1

            if text_lines:
                text_content = '\n'.join(text_lines).strip()
                if text_content:
                    blocks.append({
                        'content': text_content,
                        'tags': tags.copy(),  # 100%继承
                        'start_line': start_line
                    })

        return blocks

    def _split_text_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        第二级拆分：文本块按 \n\n → \n → 句号 → 强制拆分

        Args:
            block: 文本块，包含 content, tags, start_line

        Returns:
            子块列表，每个子块100%继承block的tags
        """
        content = block['content']
        tags = block['tags']  # 必须100%继承
        start_line = block['start_line']

        # 智能拆分
        parts = self._smart_split(content)
        parts = self._merge_short_blocks(parts)

        # 每个子块都继承标签
        processed_parts = []
        for part in parts:
            if part.strip():
                processed_parts.append({
                    'content': part,
                    'tags': tags.copy(),  # 100%继承
                    'start_line': start_line
                })

        return processed_parts if processed_parts else [block]

    def _smart_split(self, content: str) -> List[str]:
        """
        智能拆分策略：\n\n → \n → 句号 → 强制拆分
        """
        # 1. 按双换行拆分
        parts = content.split('\n\n')
        if all(count_tokens(p) <= self.MAX_BLOCK_SIZE for p in parts):
            return [p for p in parts if p.strip()]

        # 2. 按单换行拆分
        new_parts = []
        for part in parts:
            if count_tokens(part) > self.MAX_BLOCK_SIZE:
                lines = part.split('\n')
                if all(count_tokens(line) <= self.MAX_BLOCK_SIZE for line in lines):
                    new_parts.extend([line for line in lines if line.strip()])
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        parts = [p for p in new_parts if p.strip()]
        if all(count_tokens(p) <= self.MAX_BLOCK_SIZE for p in parts):
            return parts

        # 3. 按句子拆分
        new_parts = []
        for part in parts:
            if count_tokens(part) > self.MAX_BLOCK_SIZE:
                sentences = re.split(r'([。！？.!?])', part)
                sentence_parts = []
                for i in range(0, len(sentences) - 1, 2):
                    sentence_parts.append(sentences[i] + sentences[i + 1])
                if len(sentences) % 2 == 1:
                    sentence_parts.append(sentences[-1])

                merged = []
                current = ""
                for s in sentence_parts:
                    if count_tokens(current + s) <= self.MAX_BLOCK_SIZE:
                        current += s
                    else:
                        if current:
                            merged.append(current)
                        current = s
                if current:
                    merged.append(current)

                new_parts.extend(merged)
            else:
                new_parts.append(part)

        parts = [p for p in new_parts if p.strip()]
        if all(count_tokens(p) <= self.MAX_BLOCK_SIZE for p in parts):
            return parts

        # 4. 强制拆分
        new_parts = []
        for part in parts:
            if count_tokens(part) > self.MAX_BLOCK_SIZE:
                # 计算每个chunk的token数量
                tokens = []
                try:
                    from ..utils.token_utils import get_tokenizer
                    tokenizer = get_tokenizer()
                    tokens = tokenizer.encode(part)
                except Exception:
                    # 回退到字符拆分
                    for i in range(0, len(part), self.MAX_BLOCK_SIZE * 4):
                        chunk = part[i:i + self.MAX_BLOCK_SIZE * 4]
                        if chunk.strip():
                            new_parts.append(chunk)
                    continue

                # 按token拆分
                for i in range(0, len(tokens), self.MAX_BLOCK_SIZE):
                    chunk_tokens = tokens[i:i + self.MAX_BLOCK_SIZE]
                    chunk = tokenizer.decode(chunk_tokens)
                    if chunk.strip():
                        new_parts.append(chunk)
            else:
                new_parts.append(part)

        return [p for p in new_parts if p.strip()]

    def _merge_short_blocks(self, parts: List[str]) -> List[str]:
        """合并过短的块（基于token数量）

        不需要向前找，持续向后合并，直到达到250 token
        """
        if len(parts) <= 1:
            return parts

        merged = []
        i = 0

        while i < len(parts):
            current = parts[i]
            current_tokens = count_tokens(current)

            # 如果当前块的token数小于最小块大小
            if current_tokens < self.MIN_BLOCK_SIZE:
                # 持续向后合并，直到达到最小块大小或无法继续合并
                while i + 1 < len(parts):
                    next_part = parts[i + 1]
                    combined = current + '\n\n' + next_part
                    combined_tokens = count_tokens(combined)

                    # 如果合并后的token数不超过最大块大小，则合并
                    if combined_tokens <= self.MAX_BLOCK_SIZE:
                        current = combined
                        current_tokens = combined_tokens
                        i += 1  # 处理下一个块
                        # 如果已达到最小块大小要求，则停止合并
                        if current_tokens >= self.MIN_BLOCK_SIZE:
                            break
                    else:
                        # 合并后会超出最大块大小，停止合并
                        break

                # 添加合并后的块（或原始小块，如果无法合并）
                merged.append(current)
            else:
                # 当前块大小合适，直接添加
                merged.append(current)

            i += 1

        return merged

    def _build_related_blocks(self, blocks: List[DocumentBlock]) -> None:
        """建立关联关系"""
        for i, block in enumerate(blocks):
            prev_id = blocks[i - 1].id if i > 0 else "none"  # 没有上一个块时用"none"占位
            next_id = blocks[i + 1].id if i < len(blocks) - 1 else "none"  # 没有下一个块时用"none"占位

            block.related_block_ids = [prev_id, next_id]

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        if not file_path:
            return ""

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()