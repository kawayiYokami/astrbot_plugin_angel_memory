"""
向量存储组件。

封装所有与ChromaDB向量数据库和嵌入模型的交互。
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import traceback

# 导入日志记录器
from astrbot.api import logger
from ..models.data_models import BaseMemory
from ..config.system_config import system_config
from .bm25_retriever import BM25Retriever

class VectorStore:
    """
    向量存储类。

    负责记忆的向量化和存储，使用ChromaDB作为后端。
    实现为单例模式，确保全局只加载一次嵌入模型。
    """

    def __init__(self, model_name: str = None, db_path: str = None):
        """
        初始化向量存储。每次调用都会创建一个新的实例。
        不再使用单例模式。

        Args:
            model_name: 嵌入模型名称。如果未提供，则从系统配置中获取。
            db_path: 数据库存储路径。如果未提供，则从系统配置中获取。
        """
        self.logger = logger
        self.model_name = model_name or system_config.embedding_model
        self.db_path = db_path or str(system_config.get_database_path())

        # 加载嵌入模型
        self.logger.info(f"正在为新实例加载嵌入模型: {self.model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info(f"模型加载完成: {self.model_name}")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

        # 创建ChromaDB客户端
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.logger.info(f"已为新实例创建ChromaDB客户端，路径: {self.db_path}")

        # 实例变量
        self.collections = {}

        # ChromaDB是线程安全的，不需要额外的线程锁
        # 移除了 self._db_lock = threading.RLock()

        # BM25检索器组件
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_search_enabled = True
        self.vector_weight = 0.7
        self.bm25_weight = 0.3

        # 初始化BM25检索器
        self._init_bm25_retriever()

    def _post_initialization_verification(self):
        """初始化完成后验证"""
        try:
            self.logger.info("开始初始化后验证...")

            # 验证模型维度
            model_dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"验证模型维度: {model_dimension}")

            # 验证默认集合
            if hasattr(self, 'collection') and self.collection:
                collection_name = self.collection.name
                self.logger.info(f"验证默认集合: {collection_name}")
                self._verify_collection_dimension(self.collection, model_dimension)

            # 验证所有已创建的集合
            for collection_name, collection in self.collections.items():
                self.logger.info(f"验证集合: {collection_name}")
                self._verify_collection_dimension(collection, model_dimension)

            self.logger.info("初始化后验证完成")

        except Exception as e:
            self.logger.error(f"初始化后验证失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")

    def _verify_collection_dimension(self, collection, expected_dimension):
        """
        验证集合维度

        Args:
            collection: ChromaDB集合
            expected_dimension: 期望的维度
        """
        try:
            # 尝试用期望维度的向量查询
            dummy_vector = [0.0] * expected_dimension
            collection.query(
                query_embeddings=[dummy_vector],
                n_results=1
            )
            self.logger.info(f"  ✓ 集合 {collection.name} 维度正确 ({expected_dimension})")
        except Exception as e:
            if "dimension" in str(e).lower():
                self.logger.error(f"  ✗ 集合 {collection.name} 维度不匹配: {e}")
                # 记录集合当前的记录数
                count = collection.count()
                self.logger.info(f"  集合 {collection.name} 当前记录数: {count}")
            else:
                # 其他错误，可能是空集合等
                self.logger.info(f"  ✓ 集合 {collection.name} (新创建或空集合)")

    def set_storage_path(self, new_path: str):
        """
        设置新的存储路径并重新初始化ChromaDB客户端。

        注意：这将创建一个新的ChromaDB客户端，之前的数据仍在原路径中。
        如果需要迁移数据，请手动复制数据库文件。

        Args:
            new_path: 新的存储路径
        """
        try:
            # 创建新的ChromaDB客户端
            self.logger.info(f"正在切换存储路径到: {new_path}")
            new_client = chromadb.PersistentClient(path=new_path)

            # 更新共享客户端
            VectorStore._client = new_client
            self.client = new_client

            # 重新创建集合
            self.collection = self.client.get_or_create_collection(name=self.collection.name)

            self.logger.info(f"存储路径已成功切换到: {new_path}")

        except Exception as e:
            self.logger.error(f"切换存储路径失败: {e}")
            raise

    # ===== 笔记服务专用方法 =====

    def get_note_collection(self, collection_name: str = "notes_collection"):
        """
        获取笔记专用集合

        Args:
            collection_name: 集合名称

        Returns:
            ChromaDB集合对象
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = self.get_or_create_collection_with_dimension_check(collection_name)
        return self.collections[collection_name]


    def remember(self, collection, memory: BaseMemory):
        """
        记住一条新记忆。

        Args:
            collection: 目标 ChromaDB 集合。
            memory: 要记住的记忆对象（任何 BaseMemory 的子类）
        """
        try:
            # 获取用于向量化的语义核心文本
            semantic_core = memory.get_semantic_core()

            # 获取记忆的内容文本（用于文档存储）
            content_text = ""
            if hasattr(memory, 'content'):
                content_text = memory.content
            elif hasattr(memory, 'definition'):
                content_text = memory.definition
            elif hasattr(memory, 'procedure'):
                content_text = memory.procedure
            else:
                content_text = semantic_core  # 兜底方案

            # 使用高级抽象方法存储记忆
            self.upsert_documents(
                collection=collection,
                ids=memory.id,
                embedding_texts=semantic_core,  # 用于向量化的语义核心
                documents=content_text,  # 实际存储的内容
                metadatas=memory.to_dict()
            )

            self.logger.debug(f"向量写入成功 - 记忆ID: {memory.id}, 语义核心: {semantic_core[:100]}...")

            # 透明更新BM25索引
            if self._is_hybrid_search_enabled():
                collection_name = collection.name
                self.bm25_retriever.add_document(collection_name, memory.id, content_text)
                self.logger.debug(f"BM25索引更新成功 - 记忆ID: {memory.id}")

        except Exception:
            # 异常会被装饰器自动记录
            raise  # 重新抛出异常

    def recall(self, collection, query: str, limit: int = 10, where_filter: Optional[dict] = None) -> List[BaseMemory]:
        """
        根据查询回忆相关记忆，支持复杂的元数据过滤。

        Args:
            collection: 目标 ChromaDB 集合。
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器字典 (e.g., {"memory_type": "EventMemory", "is_consolidated": False})

        Returns:
            相关的记忆对象列表（BaseMemory 的子类）
        """
        try:
            # 显式生成查询向量
            query_embedding = self.embed_single_document(query)

            # 构建查询参数
            query_params = {
                "query_embeddings": [query_embedding],  # 使用显式生成的向量
                "n_results": limit
            }

            # 如果提供了过滤器，则添加到查询参数
            if where_filter:
                if len(where_filter) == 1:
                    # 单个条件
                    query_params["where"] = where_filter
                else:
                    # 多个条件，使用 $and 操作符
                    query_params["where"] = {"$and": [{k: v} for k, v in where_filter.items()]}

            # 在ChromaDB中进行向量相似度搜索
            results = collection.query(**query_params)

            # 将结果转换为记忆对象
            vector_results = []
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                for meta in results['metadatas'][0]:
                    if meta:  # 确保元数据不为空
                        vector_results.append(BaseMemory.from_dict(meta))

            # 混合检索：结合BM25结果
            if self._is_hybrid_search_enabled():
                collection_name = collection.name
                # 如果集合还没有同步到BM25，先同步
                if self.bm25_retriever.get_document_count(collection_name) == 0:
                    self._sync_collection_to_bm25(collection_name, collection)

                # BM25检索
                bm25_results = self.bm25_retriever.search(collection_name, query, limit)

                # 融合结果
                final_results = self._merge_results(vector_results, bm25_results, collection)

                self.logger.debug(f"混合检索完成 - 向量结果: {len(vector_results)}, BM25结果: {len(bm25_results)}, 最终结果: {len(final_results)}")
            else:
                final_results = vector_results

        except Exception:
            # 异常会被装饰器自动记录
            raise  # 重新抛出异常

        return final_results

    def update_memory(self, collection, memory_id: str, updates: dict):
        """
        更新记忆的元数据。

        Args:
            collection: 目标 ChromaDB 集合。
            memory_id: 要更新的记忆ID
            updates: 要更新的字段字典 (e.g., {"is_consolidated": True, "strength": 5})
        """
        try:
            # 获取当前记忆的完整信息
            current_data = collection.get(ids=[memory_id])
            if not current_data or not current_data['metadatas']:
                raise ValueError(f"Memory with id {memory_id} not found")

            current_meta = current_data['metadatas'][0]
            current_document = current_data['documents'][0] if current_data['documents'] else ""

            # 从元数据中重新构造语义核心用于向量化
            semantic_core = current_meta.get('judgment', '') + ' ' + current_meta.get('tags', '')

            # 应用更新
            current_meta.update(updates)

            # 使用高级抽象方法重新存储
            self.upsert_documents(
                collection=collection,
                ids=memory_id,
                embedding_texts=semantic_core,  # 重新生成嵌入
                documents=current_document,
                metadatas=current_meta
            )

        except Exception as e:
            self.logger.error(f"更新记忆 {memory_id} 失败: {str(e)}")
            raise

    def delete_memories(self, collection, where_filter: dict, exclude_associations: bool = False):
        """
        根据条件删除记忆。

        Args:
            collection: 目标 ChromaDB 集合。
            where_filter: 删除条件 (e.g., {"strength": {"$lt": 2}})
            exclude_associations: 是否排除关联记忆不删除
        """
        try:
            # 处理多个条件的查询
            if len(where_filter) > 1:
                # 使用 $and 操作符处理多个条件
                base_filter = {"$and": [{k: v} for k, v in where_filter.items()]}
            else:
                base_filter = where_filter

            # 如果需要排除关联，添加额外的条件
            if exclude_associations:
                if "$and" in base_filter:
                    base_filter["$and"].append({"memory_type": {"$ne": "Association"}})
                else:
                    base_filter = {"$and": [base_filter, {"memory_type": {"$ne": "Association"}}]}

            # 获取符合条件的记忆ID
            results = collection.get(where=base_filter)
            ids_to_delete = results['ids']

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                self.logger.info(f"Deleted {len(ids_to_delete)} memories with filter: {where_filter}")

        except Exception as e:
            self.logger.error(f"Failed to delete memories: {str(e)}")
            raise
    def clear_collection(self, collection):
        """清空指定集合。"""
        try:
            collection_name = collection.name
            self.client.delete_collection(collection_name)
            # 重新创建集合，确保 embedding_function 等元数据被保留
            self.get_or_create_collection_with_dimension_check(name=collection_name)
        except Exception as e:
            self.logger.error(f"清空所有记忆失败: {e}")
            raise

    def upsert_documents(
        self,
        collection,
        *,
        ids,
        embedding_texts,
        documents,
        metadatas=None
    ):
        """
        高级 upsert 方法，它接受用于向量化的文本和用于存储的文档作为不同的参数。
        这个方法是向向量数据库添加或更新内容的主要接口。

        Args:
            collection: 目标 ChromaDB 集合。
            ids: 文档ID或ID列表。
            embedding_texts: 用于生成向量的文本或文本列表。
            documents: 实际存储在数据库中的文档内容或列表。
            metadatas: 与文档关联的元数据或元数据列表。
        """
        # 统一处理输入为列表
        ids_list = [ids] if isinstance(ids, str) else ids
        embedding_texts_list = [embedding_texts] if isinstance(embedding_texts, str) else embedding_texts
        documents_list = [documents] if isinstance(documents, str) else documents
        metadatas_list = [metadatas] if isinstance(metadatas, dict) else metadatas

        if not ids_list:
            return

        # 1. 使用内部方法从源文本生成 embeddings
        embeddings = self.embed_documents(embedding_texts_list)

        # 2. 调用底层的 upsert
        upsert_params = {
            'ids': ids_list,
            'embeddings': embeddings,
            'documents': documents_list
        }

        if metadatas_list is not None:
            upsert_params['metadatas'] = metadatas_list

        collection.upsert(**upsert_params)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        使用加载的 sentence-transformer 模型为文档列表生成向量嵌入。

        Args:
            documents: 需要进行向量化的文档字符串列表。

        Returns:
            一个由向量（浮点数列表）组成的列表。
        """
        if not documents:
            return []

        # 最终修复：强制输出为 numpy 数组，然后转换为 list。这是最可靠的方式。
        embeddings_numpy = self.embedding_model.encode(documents, convert_to_numpy=True)

        # numpy 数组总是有 .tolist() 方法
        embeddings = embeddings_numpy.tolist()

        return embeddings

    def embed_single_document(self, document: str) -> List[float]:
        """
        为单个文档生成向量嵌入。
        """
        return self.embed_documents([document])[0]

    def clear_all(self):
        """清空所有记忆。"""
        try:
            collection_name = self.collection.name
            self.client.delete_collection(collection_name)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            self.logger.error(f"清空所有记忆失败: {e}")
            raise

    def _check_collection_dimension(self, collection, expected_dimension):
        """
        检查集合的维度是否与模型匹配

        Args:
            collection: ChromaDB集合
            expected_dimension: 期望的维度
        """
        try:
            # 尝试用期望维度的向量查询
            dummy_vector = [0.0] * expected_dimension
            collection.query(
                query_embeddings=[dummy_vector],
                n_results=1
            )
            return True
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower():
                # 解析错误信息中的实际维度
                actual_dimension = None
                expected_dimension_in_error = None

                # 尝试从错误信息中提取维度信息
                import re
                dimension_pattern = r"dimension of (\d+), got (\d+)"
                match = re.search(dimension_pattern, error_msg.lower())
                if match:
                    expected_dimension_in_error = int(match.group(1))
                    actual_dimension = int(match.group(2))
                    self.logger.error(f"🚨 维度精确不匹配: 集合期望{expected_dimension_in_error}维，提交了{actual_dimension}维")
                else:
                    # 无法解析维度信息，只记录原始错误
                    self.logger.warning(f"集合 {collection.name} 维度不匹配: {e}")

                return False
            else:
                # 其他错误，可能是空集合等
                self.logger.info(f"集合 {collection.name} 维度检查遇到其他错误: {e}")
                return True

    def get_or_create_collection_with_dimension_check(self, name: str):
        """
        获取或创建集合，并检查维度是否匹配

        Args:
            name: 集合名称

        Returns:
            ChromaDB集合对象
        """
        # 获取集合前先记录详细信息
        self.logger.info(f"正在获取或创建集合: {name}")
        self.logger.info(f"客户端信息: {self.client}")

        from chromadb.utils import embedding_functions
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )

        # 输出集合的详细信息
        self.logger.info(f"集合信息 - 名称: {collection.name}, 元数据: {collection.metadata}")

        # 获取模型维度
        model_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(f"模型维度: {model_dimension}")

        # 检查集合维度
        if self._check_collection_dimension(collection, model_dimension):
            self.logger.info(f"集合 {name} 维度匹配 ({model_dimension})")
        else:
            self.logger.error(f"集合 {name} 维度不匹配！期望 {model_dimension} 维度")
            # 记录集合当前的记录数
            try:
                count = collection.count()
                self.logger.info(f"集合 {name} 当前记录数: {count}")

                # 如果集合不为空，这是一个严重问题
                if count > 0:
                    self.logger.error("警告：非空集合维度不匹配，数据可能不一致！")
                else:
                    self.logger.info("空集合维度不匹配，将在首次插入时自动修复")
            except Exception as e:
                self.logger.error(f"获取集合统计信息失败: {e}")

        return collection

    # ===== BM25混合检索集成方法 =====

    def _init_bm25_retriever(self) -> bool:
        """初始化BM25检索器"""
        try:
            self.bm25_retriever = BM25Retriever(k1=1.2, b=0.75)
            if self.bm25_retriever.is_available():
                self.logger.info("BM25检索器初始化成功，混合检索已启用")
                return True
            else:
                self.logger.warning("rank_bm25库未安装，将仅使用向量检索")
                self.hybrid_search_enabled = False
                self.bm25_retriever = None
                return False
        except Exception as e:
            self.logger.error(f"BM25检索器初始化失败: {e}")
            self.bm25_retriever = None
            return False

    def _is_hybrid_search_enabled(self) -> bool:
        """检查是否启用混合检索"""
        return (self.hybrid_search_enabled and
                self.bm25_retriever is not None and
                self.bm25_retriever.is_available() and
                (self.vector_weight > 0 and self.bm25_weight > 0))

    def _merge_results(self, vector_results: List[BaseMemory], bm25_results: List[Tuple[str, float]], collection) -> List[BaseMemory]:
        """融合向量检索和BM25检索结果"""
        if not vector_results and not bm25_results:
            return []

        if not bm25_results or not self._is_hybrid_search_enabled():
            return vector_results

        if not vector_results:
            # 只有BM25结果，需要根据doc_id查找BaseMemory对象
            return self._get_memories_by_ids(collection, [doc_id for doc_id, _ in bm25_results])

        # 创建文档ID到BaseMemory对象的映射
        vector_memories_map = {}
        for memory in vector_results:
            vector_memories_map[memory.id] = memory

        # 标准化分数到[0,1]区间
        vector_scores = {}
        if vector_results:
            # 向量检索结果按相似度排序，分配递减分数
            for i, memory in enumerate(vector_results):
                vector_scores[memory.id] = 1.0 - (i * 0.1)  # 简单线性递减

        bm25_scores = {}
        if bm25_results:
            max_score = max(score for _, score in bm25_results) if bm25_results else 1.0
            for doc_id, score in bm25_results:
                if max_score > 0:
                    bm25_scores[doc_id] = score / max_score
                else:
                    bm25_scores[doc_id] = 0.0

        # 合并分数
        combined_scores = {}
        for doc_id, score in vector_scores.items():
            combined_scores[doc_id] = self.vector_weight * score

        for doc_id, score in bm25_scores.items():
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.bm25_weight * score

        # 添加纯向量检索中存在但BM25中没有的结果
        for doc_id in vector_memories_map:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = self.vector_weight * 0.5  # 给予中等分数

        # 按合并分数排序
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回排序后的BaseMemory对象
        final_memories = []
        memory_ids_to_get = [doc_id for doc_id, _ in sorted_results[:len(vector_results)]]

        # 需要从数据库完整获取这些记忆（因为BM25结果中没有完整的记忆对象）
        if len(memory_ids_to_get) > len(vector_memories_map):
            additional_memories = self._get_memories_by_ids(collection,
                [doc_id for doc_id in memory_ids_to_get if doc_id not in vector_memories_map])
            # 合并结果
            all_memories_map = vector_memories_map.copy()
            for memory in additional_memories:
                all_memories_map[memory.id] = memory
        else:
            all_memories_map = vector_memories_map

        for doc_id, _ in sorted_results[:len(vector_results)]:
            if doc_id in all_memories_map:
                final_memories.append(all_memories_map[doc_id])

        return final_memories

    def _get_memories_by_ids(self, collection, doc_ids: List[str]) -> List[BaseMemory]:
        """根据文档ID列表获取记忆对象"""
        try:
            if not doc_ids:
                return []

            # 使用ChromaDB的get方法获取指定ID的文档
            retrieved_docs = collection.get(ids=doc_ids)
            if not retrieved_docs or not retrieved_docs['metadatas']:
                return []

            memories = []
            for meta in retrieved_docs['metadatas']:
                if meta:
                    memories.append(BaseMemory.from_dict(meta))

            return memories

        except Exception as e:
            self.logger.error(f"根据ID获取记忆失败: {e}")
            return []

    def _sync_collection_to_bm25(self, collection_name: str, collection) -> bool:
        """同步ChromaDB集合数据到BM25索引"""
        if not self._is_hybrid_search_enabled():
            return True

        try:
            # 获取集合中的所有数据
            all_data = collection.get()
            if not all_data or not all_data['documents']:
                return True

            # 准备数据
            doc_ids = all_data['ids']
            texts = all_data['documents']

            # 过滤空文档
            valid_pairs = [(doc_id, text) for doc_id, text in zip(doc_ids, texts) if text]
            if not valid_pairs:
                return True

            valid_ids, valid_texts = zip(*valid_pairs)

            # 批量添加到BM25
            return self.bm25_retriever.add_documents(collection_name, list(valid_ids), list(valid_texts))

        except Exception as e:
            self.logger.error(f"同步集合到BM25失败 {collection_name}: {e}")
            return False

    # ===== 混合检索配置方法 =====

    def enable_hybrid_search(self, vector_weight: float = 0.7, bm25_weight: float = 0.3) -> bool:
        """
        启用混合检索功能。

        Args:
            vector_weight: 向量检索权重 (0.0-1.0)
            bm25_weight: BM25检索权重 (0.0-1.0)

        Returns:
            是否成功启用
        """
        if not (0.0 <= vector_weight <= 1.0 and 0.0 <= bm25_weight <= 1.0):
            self.logger.error("权重参数必须在0.0到1.0之间")
            return False

        if self.bm25_retriever is None or not self.bm25_retriever.is_available():
            self.logger.warning("BM25组件不可用，无法启用混合检索")
            return False

        # 归一化权重
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight

        self.hybrid_search_enabled = True
        self.logger.info(f"混合检索已启用 - 向量权重: {self.vector_weight:.2f}, BM25权重: {self.bm25_weight:.2f}")
        return True

    def disable_hybrid_search(self):
        """禁用混合检索，仅使用向量检索。"""
        self.hybrid_search_enabled = False
        self.logger.info("混合检索已禁用，将仅使用向量检索")

    def set_hybrid_weights(self, vector_weight: float, bm25_weight: float) -> bool:
        """
        设置混合检索权重。

        Args:
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重

        Returns:
            是否成功设置
        """
        if not (0.0 <= vector_weight <= 1.0 and 0.0 <= bm25_weight <= 1.0):
            self.logger.error("权重参数必须在0.0到1.0之间")
            return False

        # 归一化权重
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
        else:
            self.logger.error("权重总和不能为0")
            return False

        self.logger.info(f"混合检索权重已更新 - 向量权重: {self.vector_weight:.2f}, BM25权重: {self.bm25_weight:.2f}")
        return True

    def get_hybrid_search_status(self) -> dict:
        """
        获取混合检索状态信息。

        Returns:
            状态信息字典
        """
        return {
            'hybrid_search_enabled': self.hybrid_search_enabled,
            'bm25_available': self.bm25_retriever is not None and self.bm25_retriever.is_available(),
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'bm25_collections': len(self.bm25_retriever.get_collection_names()) if self.bm25_retriever else 0
        }

    def force_reload_bm25_index(self, collection_name: str, collection = None) -> bool:
        """
        强制重新加载指定集合的BM25索引。

        Args:
            collection_name: 集合名称
            collection: ChromaDB集合对象（可选）

        Returns:
            是否成功重新加载
        """
        if not self._is_hybrid_search_enabled():
            self.logger.warning("混合检索未启用，跳过BM25索引重新加载")
            return False

        try:
            if collection is None:
                collection = self.collections.get(collection_name)
                if collection is None:
                    collection = self.get_or_create_collection_with_dimension_check(collection_name)

            # 清空现有BM25索引
            self.bm25_retriever.clear_collection(collection_name)

            # 重新同步
            return self._sync_collection_to_bm25(collection_name, collection)

        except Exception as e:
            self.logger.error(f"强制重新加载BM25索引失败 {collection_name}: {e}")
            return False

    