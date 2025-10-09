"""
BM25文本检索组件。

提供BM25算法文本检索功能，与向量检索形成互补。
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import threading

# 导入BM25S库作为主要实现
import bm25s
import jieba

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25检索器类。

    提供基于BM25算法的文本检索功能，支持中英文混合分词。
    使用BM25S库实现高性能检索。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        初始化BM25检索器。

        Args:
            k1: BM25 k1参数，控制词频饱和度 (通常1.2-2.0)
            b: BM25 b参数，控制文档长度归一化 (通常0.75)
        """
        self.logger = logger
        self.k1 = k1
        self.b = b

        # 文档存储: collection_name -> {doc_id: text}
        self.documents: Dict[str, Dict[str, str]] = defaultdict(dict)

        # BM25S索引: collection_name -> BM25S对象
        self.bm25_indices: Dict[str, object] = {}

        # 文档顺序映射: collection_name -> [doc_id]
        self.doc_orders: Dict[str, List[str]] = defaultdict(list)

        # BM25S索引的原始文档: collection_name -> [tokenized_doc]
        self.tokenized_docs: Dict[str, List] = defaultdict(list)

        # 线程锁
        self._lock = threading.RLock()

    def _tokenize_text(self, text: str) -> List[str]:
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

    def add_documents(self, collection_name: str, doc_ids: List[str], texts: List[str]) -> bool:
        """
        批量添加文档到索引。

        Args:
            collection_name: 集合名称
            doc_ids: 文档ID列表
            texts: 文档内容列表

        Returns:
            是否成功添加
        """
        if len(doc_ids) != len(texts):
            self.logger.error("文档ID数量与文本数量不匹配")
            return False

        try:
            with self._lock:
                # 批量添加文档到内存
                added_count = 0
                for doc_id, text in zip(doc_ids, texts):
                    if doc_id and text:  # 确保ID和文本都不为空
                        self.documents[collection_name][doc_id] = text
                        if doc_id not in self.doc_orders[collection_name]:
                            self.doc_orders[collection_name].append(doc_id)
                        added_count += 1

                self.logger.debug(f"批量添加了 {added_count} 个文档到 {collection_name}")

                # 单次重建索引（关键优化）
                if added_count > 0:
                    success = self._rebuild_index(collection_name)
                    if success:
                        self.logger.debug(f"BM25索引重建成功: {collection_name}")
                    return success
                else:
                    return True

        except Exception as e:
            self.logger.error(f"批量添加文档到BM25索引失败 {collection_name}: {e}")
            return False

    def add_document(self, collection_name: str, doc_id: str, text: str) -> bool:
        """
        添加单个文档到索引。
        注意：此方法会重建整个集合的索引，适合偶尔添加单个文档的场景。
        对于批量添加，请使用 add_documents 方法。

        Args:
            collection_name: 集合名称
            doc_id: 文档ID
            text: 文档内容

        Returns:
            是否成功添加
        """
        if not doc_id or not text:
            self.logger.warning(f"文档ID或文本为空，跳过添加: {doc_id}")
            return False

        try:
            with self._lock:
                self.documents[collection_name][doc_id] = text
                if doc_id not in self.doc_orders[collection_name]:
                    self.doc_orders[collection_name].append(doc_id)

                # 重建整个集合的索引（保持向后兼容）
                return self._rebuild_index(collection_name)

        except Exception as e:
            self.logger.error(f"添加单个文档到BM25索引失败 {collection_name}: {e}")
            return False

    # 清理完成

    def _rebuild_index(self, collection_name: str) -> bool:
        """
        重建指定集合的BM25索引。

        Args:
            collection_name: 集合名称

        Returns:
            是否成功重建
        """
        try:
            if collection_name not in self.documents or not self.documents[collection_name]:
                self.logger.info(f"集合 {collection_name} 无文档，清空BM25索引")
                self.bm25_indices[collection_name] = None
                self.tokenized_docs[collection_name] = []
                return True

            # 按添加顺序获取所有文档和ID
            corpus = []
            doc_order = []

            for doc_id in self.doc_orders[collection_name]:
                if doc_id in self.documents[collection_name]:
                    text = self.documents[collection_name][doc_id]
                    corpus.append(text)
                    doc_order.append(doc_id)

            if not corpus:
                self.bm25_indices[collection_name] = None
                self.tokenized_docs[collection_name] = []
                return True

            # 分词文档
            tokenized_corpus = [self._tokenize_text(doc) for doc in corpus]
            
            # 保存分词结果用于索引
            self.tokenized_docs[collection_name] = tokenized_corpus

            # 创建BM25S索引
            self.bm25_indices[collection_name] = bm25s.BM25(k1=self.k1, b=self.b)
            self.bm25_indices[collection_name].index(tokenized_corpus, show_progress=False)
            return True

        except Exception as e:
            self.logger.error(f"重建BM25索引失败 {collection_name}: {e}")
            self.bm25_indices[collection_name] = None
            self.tokenized_docs[collection_name] = []
            return False

    def search(self, collection_name: str, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        在指定集合中进行BM25检索。

        Args:
            collection_name: 集合名称
            query: 查询文本
            limit: 返回结果数量限制

        Returns:
            检索结果列表，元素为 (doc_id, score) 元组
        """
        try:
            # 检查索引是否存在
            if (collection_name not in self.bm25_indices or
                self.bm25_indices[collection_name] is None):
                return []

            # 分词查询
            tokenized_query = [self._tokenize_text(query)]  # BM25S expects list of list
            if not tokenized_query[0]:  # 检查内部列表是否为空
                return []

            with self._lock:
                # 使用BM25S进行检索
                results, scores = self.bm25_indices[collection_name].retrieve(
                    tokenized_query, k=limit
                )
                
                # 获取对应的文档ID
                doc_ids = self.doc_orders.get(collection_name, [])
                
                # 将BM25S返回的索引转换为文档ID和分数
                search_results = []
                if results.size > 0:  # 检查是否有结果
                    for i, idx in enumerate(results[0]):  # results[0]是索引数组
                        if idx < len(doc_ids):  # 确保索引不越界
                            doc_id = doc_ids[idx]
                            score = scores[0][i]  # scores[0]是分数数组
                            search_results.append((doc_id, float(score)))

                return search_results

        except Exception as e:
            self.logger.error(f"BM25检索失败 {collection_name}: {e}")
            return []

    def get_document_count(self, collection_name: str) -> int:
        """
        获取集合中的文档数量。

        Args:
            collection_name: 集合名称

        Returns:
            文档数量
        """
        return len(self.documents.get(collection_name, {}))

    # 代码已清理完毕

    def clear_collection(self, collection_name: str) -> bool:
        """
        清空指定集合的所有文档。

        Args:
            collection_name: 集合名称

        Returns:
            是否成功清空
        """
        try:
            with self._lock:
                if collection_name in self.documents:
                    self.documents[collection_name].clear()
                if collection_name in self.doc_orders:
                    self.doc_orders[collection_name].clear()

                self.bm25_indices[collection_name] = None
                self.logger.info(f"BM25集合已清空: {collection_name}")
                return True

        except Exception as e:
            self.logger.error(f"清空BM25集合失败 {collection_name}: {e}")
            return False

    # 移除所有未使用的方法 - 已完成清理

    def is_available(self) -> bool:
        """
        检查BM25功能是否可用。

        Returns:
            是否可用（bm25s库是否安装）
        """
        return True  # BM25S已经导入，所以总是可用

    def get_collection_names(self) -> List[str]:
        """
        获取所有集合名称。

        Returns:
            集合名称列表
        """
        return list(self.documents.keys())