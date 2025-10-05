"""
BM25文本检索组件。

提供BM25算法文本检索功能，与向量检索形成互补。
"""

import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import threading

# 导入日志记录器
from astrbot.api import logger


class BM25Retriever:
    """
    BM25检索器类。

    提供基于BM25算法的文本检索功能，支持中英文混合分词。
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

        # BM25索引: collection_name -> BM25对象
        self.bm25_indices: Dict[str, object] = {}

        # 文档顺序映射: collection_name -> [doc_id]
        self.doc_orders: Dict[str, List[str]] = defaultdict(list)

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
        try:
            # 尝试导入jieba进行中文分词
            import jieba
            import re

            # 提取英文单词（连续字母）
            english_words = re.findall(r'[a-zA-Z]+', text)

            # 处理中文部分
            chinese_text = re.sub(r'[a-zA-Z]+', ' ', text)
            chinese_words = list(jieba.cut(chinese_text.strip()))

            # 合并并过滤空词
            tokens = chinese_words + english_words
            return [token.strip().lower() for token in tokens if token.strip()]

        except ImportError:
            # 如果没有jieba，则使用简单分词
            import re
            # 提取字母和数字作为词汇
            tokens = re.findall(r'[\w]+', text)
            return [token.lower() for token in tokens if token.strip()]

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
                for doc_id, text in zip(doc_ids, texts):
                    if doc_id and text:  # 确保ID和文本都不为空
                        self.documents[collection_name][doc_id] = text
                        if doc_id not in self.doc_orders[collection_name]:
                            self.doc_orders[collection_name].append(doc_id)

                return self._rebuild_index(collection_name)

        except Exception as e:
            self.logger.error(f"添加文档到BM25索引失败 {collection_name}: {e}")
            return False

    def add_document(self, collection_name: str, doc_id: str, text: str) -> bool:
        """
        添加单个文档到索引。

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

                return self._rebuild_index(collection_name)

        except Exception as e:
            self.logger.error(f"添加单个文档到BM25索引失败 {collection_name}: {e}")
            return False

    def update_document(self, collection_name: str, doc_id: str, text: str) -> bool:
        """
        更新索引中的文档。

        Args:
            collection_name: 集合名称
            doc_id: 文档ID
            text: 新的文档内容

        Returns:
            是否成功更新
        """
        return self.add_document(collection_name, doc_id, text)

    def remove_document(self, collection_name: str, doc_id: str) -> bool:
        """
        从索引中删除文档。

        Args:
            collection_name: 集合名称
            doc_id: 文档ID

        Returns:
            是否成功删除
        """
        try:
            with self._lock:
                if collection_name in self.documents and doc_id in self.documents[collection_name]:
                    del self.documents[collection_name][doc_id]

                    # 从文档顺序列表中移除
                    if doc_id in self.doc_orders[collection_name]:
                        self.doc_orders[collection_name].remove(doc_id)

                    return self._rebuild_index(collection_name)
                else:
                    self.logger.warning(f"文档不存在，无法删除: {collection_name}/{doc_id}")
                    return True

        except Exception as e:
            self.logger.error(f"从BM25索引删除文档失败 {collection_name}/{doc_id}: {e}")
            return False

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
                return True

            # 分词文档
            tokenized_corpus = [self._tokenize_text(doc) for doc in corpus]

            # 创建BM25索引
            try:
                from rank_bm25 import BM25Okapi
                self.bm25_indices[collection_name] = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
                self.logger.debug(f"BM25索引重建成功 {collection_name}: {len(corpus)} 个文档")
                return True

            except ImportError:
                self.logger.error("rank_bm25库未安装，无法使用BM25检索功能")
                self.bm25_indices[collection_name] = None
                return False

        except Exception as e:
            self.logger.error(f"重建BM25索引失败 {collection_name}: {e}")
            self.bm25_indices[collection_name] = None
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
            tokenized_query = self._tokenize_text(query)
            if not tokenized_query:
                return []

            with self._lock:
                # 获取BM25分数
                scores = self.bm25_indices[collection_name].get_scores(tokenized_query)

                # 获取对应的文档ID
                doc_ids = self.doc_orders.get(collection_name, [])

                # 组合并排序
                results = list(zip(doc_ids, scores))
                results.sort(key=lambda x: x[1], reverse=True)

                return results[:limit]

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

    def get_document(self, collection_name: str, doc_id: str) -> Optional[str]:
        """
        获取指定集合中的文档内容。

        Args:
            collection_name: 集合名称
            doc_id: 文档ID

        Returns:
            文档内容，如果不存在返回None
        """
        return self.documents.get(collection_name, {}).get(doc_id)

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

    def save_to_disk(self, save_path: str) -> bool:
        """
        保存BM25数据到磁盘。

        Args:
            save_path: 保存路径

        Returns:
            是否成功保存
        """
        try:
            save_data = {
                'documents': dict(self.documents),
                'doc_orders': dict(self.doc_orders),
                'k1': self.k1,
                'b': self.b
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"BM25数据已保存到: {save_path}")
            return True

        except Exception as e:
            self.logger.error(f"保存BM25数据失败: {e}")
            return False

    def load_from_disk(self, save_path: str) -> bool:
        """
        从磁盘加载BM25数据。

        Args:
            save_path: 保存路径

        Returns:
            是否成功加载
        """
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            self.documents = defaultdict(dict, loaded_data.get('documents', {}))
            self.doc_orders = defaultdict(list, loaded_data.get('doc_orders', {}))
            self.k1 = loaded_data.get('k1', 1.2)
            self.b = loaded_data.get('b', 0.75)

            # 重建所有集合的索引
            success_count = 0
            for collection_name in self.documents.keys():
                if self._rebuild_index(collection_name):
                    success_count += 1

            self.logger.info(f"BM25数据加载完成: {save_path}, 成功重建 {success_count}/{len(self.documents)} 个索引")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"加载BM25数据失败: {e}")
            return False

    def is_available(self) -> bool:
        """
        检查BM25功能是否可用。

        Returns:
            是否可用（rank_bm25库是否安装）
        """
        try:
            import importlib.util
            spec = importlib.util.find_spec('rank_bm25')
            return spec is not None
        except ImportError:
            return False

    def get_collection_names(self) -> List[str]:
        """
        获取所有集合名称。

        Returns:
            集合名称列表
        """
        return list(self.documents.keys())