"""
Markdown解析器

用于解析Markdown文档并直接返回NoteData列表。
"""

import re
import uuid
from typing import List, Dict, Any, Set
import jieba.posseg as pseg
from ..models.note_models import NoteData
from ..components.tag_manager import TagManager
from ..utils.token_utils import count_tokens


class MarkdownParser:
    """Markdown文档解析器"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS: Set[str] = {'.md', '.txt'}

    # 块大小控制（以token为单位）
    MAX_BLOCK_SIZE = 500  # 最大块大小
    MIN_BLOCK_SIZE = 250   # 最小块大小

    def __init__(self, tag_manager: TagManager = None):
        """
        初始化解析器

        Args:
            tag_manager: 标签管理器实例（可选）
        """
        self.tag_manager = tag_manager

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

    def parse(self, markdown_content: str, file_id: int, file_path: str = "") -> List[NoteData]:
        """
        解析Markdown内容，直接生成NoteData列表

        Args:
            markdown_content: Markdown文本内容
            file_id: 文件索引ID
            file_path: 文件路径（用于生成标签）

        Returns:
            NoteData列表
        """
        lines = markdown_content.split('\n')

        # 第一步：扫描所有标题位置
        headers = self._scan_headers(lines)

        # 第二步：识别章节（标题与标题之间的所有内容）
        sections = self._identify_sections(lines, headers)

        # 第三步：为每个章节提取标签（收集原始标签字符串）
        all_tags_in_file = set()
        for section in sections:
            # 路径标签
            path_tags = self._extract_path_tags(file_path)

            # 标题标签
            header_tags = self._find_header_tags(section['start_line'], headers)

            # 内容标签
            content_tags = self._extract_content_tags(section['content'])

            # 合并所有标签
            section['tags'] = path_tags + header_tags + content_tags

            # 收集所有标签到集合中（自动去重）
            all_tags_in_file.update(section['tags'])

        # 第四步：拆分（AB两步走，收集所有块）
        all_blocks = []
        for section in sections:
            # A: 初分 - 按硬边界拆分
            a_blocks = self._split_section(section)
            # B: 断长 - 把超长的块打碎
            b_blocks = self._break_long_blocks(a_blocks)
            # 收集所有B表块（不在section内合并）
            all_blocks.extend(b_blocks)

        # 第五步：按起始行排序（确保全局顺序正确）
        all_blocks.sort(key=lambda b: b['start_line'])

        # 第六步：C方法全局贪婪合并（跨section）
        all_blocks = self._merge_short_blocks_global(all_blocks)

        # 第七步：批量转换标签为ID（一次性数据库操作）
        tag_to_id_map = {}
        if self.tag_manager and all_tags_in_file:
            tag_to_id_map = self.tag_manager.get_or_create_tag_ids(list(all_tags_in_file))
            # 构建标签名称到ID的映射字典
            # tag_manager.get_or_create_tag_ids 返回的是ID列表，我们需要构建映射
            tag_names = list(all_tags_in_file)
            tag_to_id_map = {tag_name: tag_id for tag_name, tag_id in zip(tag_names, tag_to_id_map)}
        else:
            tag_to_id_map = {}

        # 第八步：创建NoteData对象（使用批量转换的标签ID）
        note_data_list = []
        for block in all_blocks:
            # 从映射中查找标签ID
            tag_ids = []
            for tag in block['tags']:
                if tag in tag_to_id_map:
                    tag_ids.append(tag_to_id_map[tag])

            # 生成唯一的块ID
            block_id = str(uuid.uuid4())

            note = NoteData.create_file_block(
                block_id=block_id,
                content=block['content'],
                file_id=file_id,
                tag_ids=tag_ids,
                related_ids=[]  # 暂时为空，下一步建立关联
            )
            note_data_list.append(note)

        # 第九步：建立关联关系
        self._build_related_blocks(note_data_list)

        return note_data_list

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

    def _can_be_tag(self, text: str) -> bool:
        """
        检查文本是否适合作为标签

        规则：不允许包含任何标点符号，但允许常见运算符和下划线
        汉字不超过6个

        Args:
            text: 待检查的文本

        Returns:
            是否可以作为标签
        """
        # 允许的字符：中英文、数字、空白、常见运算符 +-*/=<>_
        # 禁止的字符：所有标点符号（包括中英文）
        forbidden_pattern = r'[^\w\s\u4e00-\u9fff+\-*/=<>_]'

        # 如果包含禁止的字符，不能作为标签
        if re.search(forbidden_pattern, text):
            return False

        # 长度检查：至少2个字符
        cleaned_text = text.strip()
        if not cleaned_text or len(cleaned_text) <= 1:
            return False

        # 汉字数量检查：不超过6个汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        if len(chinese_chars) >= 7:
            return False

        return True

    def _extract_path_tags(self, file_path: str) -> List[str]:
        """从文件路径提取标签，只保留raw之后的路径部分"""
        tags = []
        # 统一路径分隔符
        normalized_path = file_path.replace('\\', '/')

        # 查找raw位置
        if '/raw/' in normalized_path:
            # 分割路径，获取raw之后的部分
            raw_index = normalized_path.find('/raw/')
            path_after_raw = normalized_path[raw_index + len('/raw/'):]

            # 分割路径并处理每个部分
            path_parts = path_after_raw.split('/')
            for i, part in enumerate(path_parts):
                if part:
                    # 只对最后一个部分（文件名）去除扩展名
                    if i == len(path_parts) - 1:
                        # 使用 rsplit 从右侧分割，只分割一次，去除扩展名
                        clean_part = part.rsplit('.', 1)[0] if '.' in part else part
                    else:
                        clean_part = part

                    if clean_part:
                        tags.append(clean_part)
        else:
            # 如果没有raw，保持原有逻辑（向后兼容）
            path_parts = normalized_path.split('/')
            for i, part in enumerate(path_parts):
                if part:
                    # 同样只对最后一个部分去除扩展名
                    if i == len(path_parts) - 1:
                        clean_part = part.rsplit('.', 1)[0] if '.' in part else part
                    else:
                        clean_part = part
                    if clean_part:
                        tags.append(clean_part)

        return tags

    def _extract_content_tags(self, content: str) -> List[str]:
        """从内容中提取标签"""
        tags = []

        # 加粗标签提取
        bold_matches = re.findall(r'\*\*([^*]+)\*\*', content)
        for match in bold_matches:
            cleaned_match = match.strip()
            if self._can_be_tag(cleaned_match):
                tags.append(cleaned_match)

        # 专有名词标签提取（修复重复的正则表达式）
        proper_noun_patterns = [
            r'"([^"]+)"',      # 半角双引号
            r'"([^"]+)"',      # 全角双引号
            r'「([^」]+)」',    # 日式直角引号
            r'『([^』]+)』',    # 日式双直角引号
            r'《([^》]+)》',    # 书名号
            r'〈([^〉]+)〉',    # 尖括号
        ]

        for pattern in proper_noun_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                cleaned_match = match.strip()
                if self._can_be_tag(cleaned_match):
                    tags.append(cleaned_match)

        # 对所有已提取的标签进行jieba分词和词性过滤，进一步提取子标签
        # 这是一个过滤过程，保留通过词性检查的词语
        filtered_tags = []
        for existing_tag in tags:
            jieba_tags = self._extract_jieba_tags_from_tag(existing_tag)
            filtered_tags.extend(jieba_tags)

        # 重新设置标签为过滤后的结果
        tags = filtered_tags

        # 去重和清理
        tags = list(dict.fromkeys(tags))  # 去重保持顺序
        tags = [tag.strip() for tag in tags if tag.strip()]  # 清理空白

        return tags

    def _extract_jieba_tags_from_tag(self, tag_text: str) -> List[str]:
        """
        对单个标签文本进行jieba分词和过滤，提取有效的子标签

        Args:
            tag_text: 待分析的标签文本

        Returns:
            过滤后的子标签列表
        """
        tags = []

        # 使用jieba进行分词和词性标注
        words_with_pos = list(pseg.cut(tag_text))

        # 保留适合作为标签的词性（排除动词、数词、量词，只保留名词性内容）
        keep_pos = {
            'n',    # 普通名词
            'nr',   # 人名
            'ns',   # 地名
            'nt',   # 机构团体名
            'nz',   # 其他专有名词
            'vn',   # 名动词（名词化的动词，如"发展"->发展）
            'a',    # 形容词
            't',    # 时间词
        }

        for word, pos in words_with_pos:
            # 跳过标点符号和单字符词
            if len(word.strip()) >= 2 and pos in keep_pos:
                # 应用标签规则检查
                if self._can_be_tag(word):
                    tags.append(word)

        return tags

    def _split_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        A方法：初分 - 按硬边界（代码块、表格）拆分章节

        输入：section = {'content': str, 'tags': List[str], 'start_line': int}
        输出：A表 = [block1, block2, ...]
              每个block = {'content': str, 'tags': List[str], 'start_line': int,
                          'is_code': bool可选, 'is_table': bool可选}

        Args:
            section: 章节，包含 content, tags, start_line

        Returns:
            A表：块列表，每个块100%继承section的tags
        """
        content = section['content']
        tags = section['tags']
        start_line = section['start_line']

        # 调用现有方法，按硬边界拆分
        blocks = self._split_by_special_blocks(content, tags, start_line)

        return blocks

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

    def _break_long_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        B方法：断长 - 把所有超过MAX_BLOCK_SIZE的块打碎

        输入：A表
        输出：B表（所有块 ≤ MAX_BLOCK_SIZE，代码块和表格除外）

        Args:
            blocks: A表，来自_split_section的块列表

        Returns:
            B表：断长后的块列表
        """
        result = []

        for block in blocks:
            content_tokens = count_tokens(block['content'])

            if content_tokens <= self.MAX_BLOCK_SIZE:
                # 不超长，原样保留
                result.append(block)
            else:
                # 超长了
                if block.get('is_code') or block.get('is_table'):
                    # 代码块和表格即使超长也不拆分，保持完整性
                    # 直接加入结果（这是特殊情况，允许超过MAX_BLOCK_SIZE）
                    result.append(block)
                else:
                    # 普通文本块超长，需要打碎
                    # 调用现有的_smart_split方法（完全不修改）
                    parts = self._smart_split(block['content'])

                    # 把parts（字符串列表）转换为block格式
                    for part in parts:
                        if part.strip():
                            result.append({
                                'content': part,
                                'tags': block['tags'].copy(),  # 继承标签
                                'start_line': block['start_line']
                            })

        return result

    def _merge_short_blocks_global(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        C方法：组装 - 贪婪合并策略

        输入：B表
        输出：C表（最终产品，尽可能大的块，但不超过MAX_BLOCK_SIZE）

        贪婪合并策略：
        只要合并后的尺寸不超过 MAX_BLOCK_SIZE，就坚决执行合并。
        这将从根本上最大化地减少小块数量，产出尽可能大的块。

        算法流程：
        1. 初始化：取第一个块作为当前块（current_block）
        2. 单遍扫描：遍历所有剩余块
        3. 决策：
           - 如果合并后 <= MAX_BLOCK_SIZE：执行合并，继续
           - 如果合并后 > MAX_BLOCK_SIZE：当前块完成，开始新块
        4. 收尾：将最后一个当前块加入结果

        关键：
        - 合并标签时去重
        - 计算tokens时包含内容+标签的总tokens

        Args:
            blocks: B表，来自_break_long_blocks的块列表

        Returns:
            C表：组装后的最终块列表
        """
        # 1. 初始化
        if not blocks:
            return []

        final_blocks = []
        current_block = blocks[0].copy()

        # 2. 单遍扫描
        for i in range(1, len(blocks)):
            next_block = blocks[i]

            # 3. 决策与操作：预合并计算
            # 合并内容
            combined_content = current_block['content'] + '\n\n' + next_block['content']

            # 合并标签（去重）
            combined_tags = list(set(current_block['tags']) | set(next_block['tags']))

            # 计算总tokens（内容 + 标签）
            content_tokens = count_tokens(combined_content)
            tags_tokens = count_tokens(' '.join(combined_tags))
            combined_tokens = content_tokens + tags_tokens

            if combined_tokens <= self.MAX_BLOCK_SIZE:
                # 可以合并：更新当前块
                current_block['content'] = combined_content
                current_block['tags'] = combined_tags
            else:
                # 无法合并：当前块完成，加入结果
                final_blocks.append(current_block)
                # 开始新的当前块
                current_block = next_block.copy()

        # 4. 收尾：添加最后一个当前块
        final_blocks.append(current_block)

        return final_blocks

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

    def _build_related_blocks(self, notes: List[NoteData]) -> None:
        """建立关联关系"""
        for i, note in enumerate(notes):
            prev_id = notes[i - 1].id if i > 0 else "none"  # 没有上一个块时用"none"占位
            next_id = notes[i + 1].id if i < len(notes) - 1 else "none"  # 没有下一个块时用"none"占位

            # 更新related_block_ids
            note.related_block_ids = f"{prev_id},{next_id}"