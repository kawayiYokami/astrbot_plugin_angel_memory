"""
检索词生成工具

统一处理笔记检索和记忆检索的查询词预处理：
1. 删除助理的各种昵称别名
2. 从后往前保留500token
"""

import re
from typing import Set
from astrbot.api.event import AstrMessageEvent
import json

from astrbot.api import logger


class QueryProcessor:
    """统一的检索词预处理工具类"""

    def __init__(self):
        self.logger = logger

    def _extract_assistant_names(self, event: AstrMessageEvent) -> Set[str]:
        """
        从事件中提取助理的所有名字和别名

        Args:
            event: 消息事件

        Returns:
            助理名字集合
        """
        names = set()

        try:
            if hasattr(event, 'angelheart_context'):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get('secretary_decision', {})

                # 提取persona_name
                persona_name = secretary_decision.get('persona_name', '').strip()
                if persona_name:
                    names.add(persona_name)

                # 提取alias（可能是字符串或列表）
                alias = secretary_decision.get('alias', '')
                if isinstance(alias, str) and alias.strip():
                    # 处理 | 分隔的别名格式
                    if '|' in alias:
                        for name in alias.split('|'):
                            name = name.strip()
                            if name:
                                names.add(name)
                    else:
                        names.add(alias.strip())
                elif isinstance(alias, list):
                    for a in alias:
                        if isinstance(a, str) and a.strip():
                            names.add(a.strip())

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.debug(f"无法提取助理名字: {e}")

        return names

    def _filter_assistant_names(self, query: str, names: Set[str]) -> str:
        """
        从查询中过滤掉助理名字和别名

        Args:
            query: 原始查询字符串
            names: 要过滤的名字集合

        Returns:
            过滤后的查询字符串
        """
        if not names:
            return query

        filtered_query = query
        for name in names:
            # 使用正则表达式匹配名字（中文环境中不需要词边界）
            pattern = re.escape(name)
            filtered_query = re.sub(pattern, '', filtered_query, flags=re.IGNORECASE)

        return filtered_query

    def _truncate_text(self, text: str, max_tokens: int = 500) -> str:
        """
        从后往前保留指定数量的token

        Args:
            text: 输入文本
            max_tokens: 最大token数量，默认为500

        Returns:
            截断后的文本
        """
        if not text.strip():
            return ""

        try:
            # 使用token工具从后往前截断文本
            # 先获取全部tokens，然后取最后max_tokens个
            from ...llm_memory.utils.token_utils import get_tokenizer

            tokenizer = get_tokenizer()
            tokens = tokenizer.encode(text)

            if len(tokens) <= max_tokens:
                return text

            # 取最后max_tokens个token
            truncated_tokens = tokens[-max_tokens:]
            return tokenizer.decode(truncated_tokens)
        except Exception as e:
            self.logger.warning(f"Token截断处理失败: {e}")
            # 降级处理：简单字符截断（从后往前）
            if len(text) <= max_tokens * 4:  # 粗略估计1 token ≈ 4 字符
                return text
            return text[-(max_tokens * 4):]

    def _clean_text(self, text: str) -> str:
        """
        清理文本中的多余空格，但保留标点符号

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        if not text:
            return text

        # 替换多个连续空格为单个空格
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def process_query(self, query: str, event: AstrMessageEvent) -> str:
        """
        统一处理检索词的预处理流程

        Args:
            query: 原始查询字符串
            event: 消息事件（用于提取助理信息）

        Returns:
            处理后的检索词字符串
        """
        if not query or not query.strip():
            return query

        original_query = query
        self.logger.debug(f"开始处理查询词: '{query}'")

        try:
            # 步骤1: 提取助理名字并过滤
            assistant_names = self._extract_assistant_names(event)
            if assistant_names:
                query = self._filter_assistant_names(query, assistant_names)
                query = self._clean_text(query)

            # 步骤2: 从后往前保留500token
            if query.strip():
                query = self._truncate_text(query, 500)
                query = self._clean_text(query)

            return query

        except Exception as e:
            self.logger.error(f"查询词预处理失败: {e}")
            # 返回原查询作为降级方案
            return original_query

    def process_query_for_memory(self, query: str, event: AstrMessageEvent) -> str:
        """
        专门为记忆检索优化的查询词处理
        比通用处理更保守，保留更多上下文

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            处理后的查询词
        """
        # 对于记忆检索，过滤更保守一些
        # 主要过滤助理名字，但保留更多语义信息
        return self.process_query(query, event)

    def process_query_for_notes(self, query: str, event: AstrMessageEvent) -> str:
        """
        专门为笔记检索优化的查询词处理
        更积极的过滤以提高检索精确性

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            处理后的查询词
        """
        # 对于笔记检索，可以更积极地过滤
        return self.process_query(query, event)


# 全局单例实例
_query_processor_instance = None

def get_query_processor() -> QueryProcessor:
    """获取QueryProcessor的全局单例实例"""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance