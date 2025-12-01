"""
检索词生成工具

统一处理笔记检索和记忆检索的查询词预处理：
1. 删除助理的各种昵称别名
2. 从后往前保留500token
"""

import re
from typing import Set, Tuple, Optional, List
from astrbot.api.event import AstrMessageEvent
import json

from astrbot.api import logger


class QueryProcessor:
    """统一的检索词预处理工具类"""

    def __init__(self):
        self.logger = logger

    def extract_rag_fields(self, event: AstrMessageEvent) -> dict:
        """
        从天使之心上下文中提取RAG字段

        Args:
            event: 消息事件

        Returns:
            包含entities, facts, keywords的字典
        """
        rag_fields = {"entities": [], "facts": [], "keywords": []}

        try:
            if hasattr(event, "angelheart_context") and event.angelheart_context:
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})

                # 提取RAG字段
                for field in rag_fields.keys():
                    field_value = secretary_decision.get(field, [])
                    if isinstance(field_value, list):
                        rag_fields[field] = [str(item).strip() for item in field_value if item and str(item).strip()]
                    elif isinstance(field_value, str) and field_value.strip():
                        rag_fields[field] = [field_value.strip()]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.debug(f"无法提取RAG字段: {e}")

        return rag_fields

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
            if hasattr(event, "angelheart_context"):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})

                # 提取persona_name
                persona_name = secretary_decision.get("persona_name", "").strip()
                if persona_name:
                    names.add(persona_name)

                # 提取alias（可能是字符串或列表）
                alias = secretary_decision.get("alias", "")
                if isinstance(alias, str) and alias.strip():
                    # 处理 | 分隔的别名格式
                    if "|" in alias:
                        for name in alias.split("|"):
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
            filtered_query = re.sub(pattern, "", filtered_query, flags=re.IGNORECASE)

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
            from ...llm_memory.utils.token_utils import truncate_by_tokens_from_end

            return truncate_by_tokens_from_end(text, max_tokens)
        except Exception as e:
            self.logger.warning(f"Token截断处理失败: {e}")
            # 降级处理：简单字符截断（从后往前）
            if len(text) <= max_tokens * 4:  # 粗略估计1 token ≈ 4 字符
                return text
            return text[-(max_tokens * 4) :]

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
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _build_rag_query(self, rag_fields: dict) -> str:
        """
        基于RAG字段构建统一的检索词

        Args:
            rag_fields: 包含entities, facts, keywords的字典

        Returns:
            构建的检索词字符串
        """
        query_parts = []

        # 按优先级组合字段：entities > facts > keywords
        for field_name in ["entities", "facts", "keywords"]:
            field_values = rag_fields.get(field_name, [])
            if field_values:
                query_parts.extend([str(value) for value in field_values if value and str(value).strip()])

        # 用空格连接所有字段值
        rag_query = " ".join(query_parts)

        return rag_query.strip()

    def process_query(self, query: str, event: AstrMessageEvent) -> str:
        """
        统一处理检索词的预处理流程
        优先使用RAG字段，如果不存在则使用原始查询

        Args:
            query: 原始查询字符串
            event: 消息事件（用于提取RAG字段和助理信息）

        Returns:
            处理后的检索词字符串
        """
        if not query or not query.strip():
            return query

        original_query = query

        try:
            # 步骤1: 提取RAG字段并构建检索词
            rag_fields = self.extract_rag_fields(event)
            rag_query = self._build_rag_query(rag_fields)

            # 步骤2: 优先使用RAG查询，如果为空则使用原始查询
            final_query = rag_query if rag_query else original_query

            # --- 新增的逻辑：根据长度决定是否进行进一步预处理 ---
            # 定义一个阈值，如果查询词已经很短，就不需要进一步过滤和截断
            # 这个阈值需要根据实际情况调整。考虑到RAG查询通常是高度精炼的关键词和实体，
            # 短于该阈值的RAG查询，我们认为它已经足够精准，无需额外处理。
            PREPROCESS_THRESHOLD_CHARACTERS = 100 # 例如100个字符，可调整

            # 只有当rag_query非空且其长度小于阈值时，才跳过后续预处理
            if rag_query and len(final_query.strip()) <= PREPROCESS_THRESHOLD_CHARACTERS:
                return final_query

            # --- 原有的逻辑：继续进行预处理 ---
            # 步骤3: 过滤助理名字
            assistant_names = self._extract_assistant_names(event)
            if assistant_names:
                final_query = self._filter_assistant_names(final_query, assistant_names)
                final_query = self._clean_text(final_query)

            # 步骤4: 从后往前保留500token
            if final_query.strip():
                final_query = self._truncate_text(final_query, 500)
                final_query = self._clean_text(final_query)

            return final_query

        except Exception as e:
            self.logger.error(f"查询词预处理失败: {e}")
            # 返回原查询作为降级方案
            return original_query

    async def _precompute_rag_vector(self, rag_query: str, event: AstrMessageEvent) -> Optional[List[float]]:
        """
        预计算RAG查询的向量

        Args:
            rag_query: RAG查询字符串
            event: 消息事件

        Returns:
            预计算的向量，如果失败返回None
        """
        if not rag_query.strip():
            return None

        # 从event中获取plugin_context
        # 假设event中包含plugin_context
        from ..plugin_context import PluginContext
        plugin_context: Optional[PluginContext] = getattr(event, 'plugin_context', None)

        if plugin_context is None:
            self.logger.debug("无法从事件中获取plugin_context，跳过RAG向量预计算")
            return None

        vector_store = plugin_context.get_vector_store()
        if vector_store is None:
            self.logger.debug("plugin_context中未找到有效的vector_store，跳过RAG向量预计算")
            return None

        try:
            # 使用embed_single_document方法进行向量化
            vector = await vector_store.embed_single_document(rag_query, is_query=True)
            return vector
        except Exception as e:
            self.logger.debug(f"RAG向量预计算失败: {e}")
            return None

    def process_query_for_memory(self, query: str, event: AstrMessageEvent) -> str:
        """
        记忆检索的查询词处理

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            处理后的查询词
        """
        # 记忆检索使用统一的RAG字段处理逻辑
        return self.process_query(query, event)

    def process_query_for_notes(self, query: str, event: AstrMessageEvent) -> str:
        """
        笔记检索的查询词处理

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            处理后的查询词
        """
        # 笔记检索也使用统一的RAG字段处理逻辑
        return self.process_query(query, event)

    async def process_query_for_memory_with_vector(self, query: str, event: AstrMessageEvent) -> Tuple[str, Optional[List[float]]]:
        """
        记忆检索的查询词处理（带向量预计算）

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            (处理后的查询词, 预计算的向量)
        """
        processed_query = self.process_query_for_memory(query, event)
        vector = await self._precompute_rag_vector(processed_query, event)
        return processed_query, vector

    async def process_query_for_notes_with_vector(self, query: str, event: AstrMessageEvent) -> Tuple[str, Optional[List[float]]]:
        """
        笔记检索的查询词处理（带向量预计算）

        Args:
            query: 原始查询字符串
            event: 消息事件

        Returns:
            (处理后的查询词, 预计算的向量)
        """
        processed_query = self.process_query_for_notes(query, event)
        vector = await self._precompute_rag_vector(processed_query, event)
        return processed_query, vector


# 全局单例实例
_query_processor_instance = None


def get_query_processor() -> QueryProcessor:
    """获取QueryProcessor的全局单例实例"""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance
