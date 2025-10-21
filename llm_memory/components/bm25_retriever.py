"""
BM25文本检索组件。

提供BM25算法的无状态精排功能，与向量检索形成互补。
"""

from typing import List, Dict, Tuple

# 导入BM25S库作为主要实现
import bm25s
import jieba

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _tokenize_text(text: str) -> List[str]:
    """
    文本分词（支持中英文混合）

    Args:
        text: 待分词的文本

    Returns:
        分词结果列表
    """
    # 使用jieba进行中文分词
    import re

    # 提取英文单词（连续字母）
    english_words = re.findall(r'[a-zA-Z]+', text)

    # 处理中文部分
    chinese_text = re.sub(r'[a-zA-Z]+', ' ', text)
    chinese_words = list(jieba.cut(chinese_text.strip()))

    # 合并并过滤空词
    tokens = chinese_words + english_words
    return [token.strip().lower() for token in tokens if token.strip()]


def rerank_with_bm25(
    query: str,
    candidates: List[Dict[str, str]],
    limit: int = 10
) -> List[Tuple[str, float]]:
    """
    对给定的候选文档列表进行 BM25 精排。
    这是一个无状态的函数，每次调用都会创建一个临时的 BM25 索引。

    Args:
        query: 查询文本
        candidates: 候选文档列表，每个元素应包含 'id' 和 'content' 键。
                  e.g., [{"id": "doc1", "content": "text1"}, ...]
        limit: 返回结果数量限制

    Returns:
        精排后的结果列表，元素为 (doc_id, score) 元组，按分数降序排列。
    """
    if not candidates:
        return []

    try:
        # 1. 准备数据
        doc_ids = [c["id"] for c in candidates]
        corpus = [c["content"] for c in candidates]

        # 过滤掉空的文档
        valid_pairs = [(doc_id, text) for doc_id, text in zip(doc_ids, corpus) if doc_id and text]
        if not valid_pairs:
            logger.warning("所有候选文档的 content 均为空，无法进行 BM25 精排。")
            return []

        final_doc_ids, final_corpus = zip(*valid_pairs)

        # 2. 分词
        tokenized_corpus = [_tokenize_text(doc) for doc in final_corpus]
        tokenized_query = _tokenize_text(query)

        if not tokenized_query:
            logger.warning(f"查询 '{query}' 分词后为空，无法进行 BM25 精排。")
            return []

        # 3. 建立临时索引
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_corpus, show_progress=False)

        # 4. 检索
        results, scores = bm25_index.retrieve([tokenized_query], k=limit)

        # 5. 结果映射
        ranked_results = []
        if results.size > 0:
            for i, idx in enumerate(results[0]):
                if idx < len(final_doc_ids):
                    doc_id = final_doc_ids[idx]
                    score = scores[0][i]
                    ranked_results.append((doc_id, float(score)))

        return ranked_results

    except Exception as e:
        logger.error(f"BM25 精排失败: {e}")
        return []


# 为了向后兼容，保留一个空的 BM25Retriever 类
# 但其所有方法都将被标记为已弃用，并直接返回或记录警告
class BM25Retriever:
    """
    @deprecated
    已弃用：BM25Retriever 类已被无状态的 rerank_with_bm25 函数取代。
    为了向后兼容而保留，但不应再使用。
    """
    def __init__(self, *args, **kwargs):
        logger.warning("BM25Retriever 类已弃用，请使用 rerank_with_bm25 函数。")

    def add_documents(self, *args, **kwargs):
        logger.warning("add_documents 方法已弃用，BM25 索引不再需要预先建立。")
        return True

    def search(self, *args, **kwargs):
        logger.warning("search 方法已弃用，请使用 rerank_with_bm25 函数。")
        return []

    def clear_collection(self, *args, **kwargs):
        logger.warning("clear_collection 方法已弃用。")
        return True

    def is_available(self):
        return True

    def get_collection_names(self):
        return []