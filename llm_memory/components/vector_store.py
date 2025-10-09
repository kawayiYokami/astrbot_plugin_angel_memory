"""
向量存储组件。

封装所有与ChromaDB向量数据库的交互，通过依赖注入支持不同的嵌入提供商。
"""

import chromadb
from typing import List, Optional, Tuple
import traceback
from pathlib import Path
from .embedding_provider import EmbeddingProvider, LocalEmbeddingProvider
from ..utils.path_manager import PathManager
from .tag_manager import TagManager

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from ..models.data_models import BaseMemory
from ..models.note_models import NoteData
from ..config.system_config import system_config
from .bm25_retriever import BM25Retriever

class VectorStore:
    """
    向量存储类。

    负责记忆的向量化和存储，使用ChromaDB作为后端。
    通过依赖注入接收嵌入提供商，实现解耦设计。
    """

    # 类级别的集合缓存，防止重复初始化
    _collection_cache = {}
    _cache_lock = None

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        db_path: Optional[str] = None
    ):
        """
        初始化向量存储。

        Args:
            embedding_provider: 嵌入提供商实例。如果未提供，则使用默认本地模型。
            db_path: 数据库存储路径。如果未提供，则从系统配置中获取。
        """
        self.logger = logger

        # 初始化线程锁（仅首次创建时）
        if VectorStore._cache_lock is None:
            import threading
            VectorStore._cache_lock = threading.RLock()

        # 初始化嵌入提供商
        if embedding_provider is None:
            self.logger.info("未指定嵌入提供商，使用默认本地模型")
            self.embedding_provider = LocalEmbeddingProvider(system_config.embedding_model)
        else:
            self.embedding_provider = embedding_provider
            provider_info = embedding_provider.get_model_info()
            self.logger.info(f"使用嵌入提供商: {provider_info}")

        # 根据提供商类型确定数据库路径
        if db_path is None:
            provider_id = None
            if self.embedding_provider.get_provider_type() == "api":
                provider_info = self.embedding_provider.get_model_info()
                provider_id = provider_info.get("provider_id")

            self.db_path = str(system_config.get_database_path(provider_id))
        else:
            self.db_path = db_path

        # 创建ChromaDB客户端
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.logger.info(f"已创建ChromaDB客户端，路径: {self.db_path}")

        # 实例变量
        self.collections = {}

        # ChromaDB是线程安全的，不需要额外的线程锁
        # 移除了 self._db_lock = threading.RLock()

        # BM25检索器组件 - 延迟初始化
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_search_enabled = False  # ← 默认禁用BM25混合检索
        self.vector_weight = 0.7
        self.bm25_weight = 0.3

        # 初始化BM25检索器（仅在首次时）
        self._lazy_init_bm25_retriever()

        # 懒加载的标签管理器（用于基于 tag_ids 重建标签文本）
        self._tag_manager: Optional[TagManager] = None

    def _get_tag_manager(self) -> Optional[TagManager]:
        """懒加载 TagManager（基于 PathManager 当前 provider 和索引目录）。"""
        if self._tag_manager is not None:
            return self._tag_manager
        try:
            pm = PathManager.get_instance()
            index_dir = str(pm.get_index_dir())
            provider_id = pm.get_current_provider()
            self._tag_manager = TagManager(index_dir, provider_id)
            return self._tag_manager
        except Exception as e:
            self.logger.warning(f"初始化 TagManager 失败，无法为BM25重建标签文本: {e}")
            return None

    def _post_initialization_verification(self):
        """初始化完成后验证"""
        try:
            # 验证嵌入提供商
            if not self.embedding_provider.is_available():
                self.logger.error("嵌入提供商不可用！")
                return

            # 验证默认集合
            if hasattr(self, 'collection') and self.collection:
                self._verify_collection_dimension(self.collection)

            # 验证所有已创建的集合
            for collection_name, collection in self.collections.items():
                self._verify_collection_dimension(collection)

        except Exception as e:
            self.logger.error(f"初始化后验证失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")

    def _verify_collection_dimension(self, collection):
        """
        验证集合维度是否与嵌入提供商匹配

        Args:
            collection: ChromaDB集合
        """
        try:
            # 获取嵌入提供商的维度信息
            provider_info = self.embedding_provider.get_model_info()
            if "dimension" in provider_info:
                expected_dimension = provider_info["dimension"]

                # 尝试用期望维度的向量查询
                dummy_vector = [0.0] * expected_dimension
                collection.query(
                    query_embeddings=[dummy_vector],
                    n_results=1
                )

        except Exception as e:
            if "dimension" in str(e).lower():
                self.logger.error(f"  ✗ 集合 {collection.name} 维度不匹配: {e}")
                # 记录集合当前的记录数
                count = collection.count()
                self.logger.info(f"  集合 {collection.name} 当前记录数: {count}")

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

            # 记忆库不使用BM25索引（高频更新场景）
            # 只有笔记库才在文件扫描完成后统一重建BM25索引
            # 这样避免了每秒上百个记忆更新时的性能瓶颈

        except Exception:
            # 异常会被装饰器自动记录
            raise  # 重新抛出异常

    def recall(self, collection, query: str, limit: int = 10, where_filter: Optional[dict] = None) -> List[BaseMemory]:
        """
        根据查询回忆相关记忆，支持复杂的元数据过滤（同步方法）。

        Args:
            collection: 目标 ChromaDB 集合。
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器字典 (e.g., {"memory_type": "EventMemory", "is_consolidated": False})

        Returns:
            相关的记忆对象列表（BaseMemory 的子类）
        """
        try:
            # 显式生成查询向量（同步调用）
            query_embedding = self.embed_single_document(query)

            # 构建查询参数
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": limit
            }

            # 如果提供了过滤器，则添加到查询参数
            if where_filter:
                if len(where_filter) == 1:
                    query_params["where"] = where_filter
                else:
                    query_params["where"] = {"$and": [{k: v} for k, v in where_filter.items()]}

            # 在ChromaDB中进行向量相似度搜索（数据库内部处理并发）
            results = collection.query(**query_params)

            # 将结果转换为记忆对象
            vector_results = []
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                for meta in results['metadatas'][0]:
                    if meta:
                        vector_results.append(BaseMemory.from_dict(meta))

            # 混合检索（仅笔记系统）
            collection_name = collection.name
            is_note_collection = collection_name.startswith('notes_')

            if is_note_collection and self._is_hybrid_search_enabled():
                # 如果集合还没有同步到BM25，先同步
                if self.bm25_retriever.get_document_count(collection_name) == 0:
                    self._sync_collection_to_bm25(collection_name, collection)

                # BM25检索
                bm25_results = self.bm25_retriever.search(collection_name, query, limit)

                # 融合结果
                final_results = self._merge_results(vector_results, bm25_results, collection)
            else:
                final_results = vector_results

            return final_results

        except Exception:
            raise

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

            # 使用同步方法重新存储
            self.upsert_documents(
                collection=collection,
                ids=memory_id,
                embedding_texts=semantic_core,
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
        metadatas=None,
        _return_timings=False
    ):
        """
        高级 upsert 方法（同步接口），接受用于向量化的文本和用于存储的文档作为不同的参数。

        Args:
            collection: 目标 ChromaDB 集合。
            ids: 文档ID或ID列表。
            embedding_texts: 用于生成向量的文本或文本列表。
            documents: 实际存储在数据库中的文档内容或列表。
            metadatas: 与文档关联的元数据或元数据列表。
            _return_timings: 内部参数，是否返回计时信息
        """
        import time
        timings = {} if _return_timings else None
        
        # 统一处理输入为列表
        ids_list = [ids] if isinstance(ids, str) else ids
        embedding_texts_list = [embedding_texts] if isinstance(embedding_texts, str) else embedding_texts
        documents_list = [documents] if isinstance(documents, str) else documents
        metadatas_list = [metadatas] if isinstance(metadatas, dict) else metadatas

        if not ids_list:
            return timings if _return_timings else None

        # 1. 使用嵌入提供商从源文本生成 embeddings（同步调用）
        t_embed = time.time()
        embeddings = self.embed_documents(embedding_texts_list)
        if _return_timings:
            timings['embed'] = (time.time() - t_embed) * 1000

        # 2. 调用底层的 upsert
        upsert_params = {
            'ids': ids_list,
            'embeddings': embeddings,
        }

        # 笔记主/副集合均不写 documents（正文由 metadata['content'] 提供；副集不存文本）
        try:
            is_note_collection = (collection.name == system_config.notes_main_collection_name or
                                  collection.name == system_config.notes_sub_collection_name)
        except Exception:
            is_note_collection = False

        if not is_note_collection and documents_list is not None:
            upsert_params['documents'] = documents_list

        if metadatas_list is not None:
            upsert_params['metadatas'] = metadatas_list
        
        # 直接upsert（数据库内部处理并发）
        t_db = time.time()
        collection.upsert(**upsert_params)
        if _return_timings:
            timings['db_upsert'] = (time.time() - t_db) * 1000
        
        return timings if _return_timings else None

    async def async_upsert_documents(
        self,
        collection,
        *,
        ids,
        embedding_texts,
        documents,
        metadatas=None
    ):
        """
        异步版本的 upsert 方法（保持兼容性，内部调用同步方法）。

        Args:
            collection: 目标 ChromaDB 集合。
            ids: 文档ID或ID列表。
            embedding_texts: 用于生成向量的文本或文本列表。
            documents: 实际存储在数据库中的文档内容或列表。
            metadatas: 与文档关联的元数据或元数据列表。
        """
        # 在线程池中执行同步upsert
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_call():
            return self.upsert_documents(
                collection=collection,
                ids=ids,
                embedding_texts=embedding_texts,
                documents=documents,
                metadatas=metadatas
            )
        
        await loop.run_in_executor(None, _sync_call)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        使用嵌入提供商为文档列表生成向量嵌入（同步方法）。

        Args:
            documents: 需要进行向量化的文档字符串列表。

        Returns:
            一个由向量（浮点数列表）组成的列表。
        """
        if not documents:
            return []

        # 使用嵌入提供商生成向量（同步调用）
        embeddings = self.embedding_provider.embed_documents_sync(documents)
        return embeddings

    def embed_single_document(self, document: str) -> List[float]:
        """
        为单个文档生成向量嵌入（同步方法）。
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


    def get_or_create_collection_with_dimension_check(self, name: str):
        """
        获取或创建集合，并检查维度是否匹配
        完善的集合缓存机制，防止重复初始化

        Args:
            name: 集合名称

        Returns:
            ChromaDB集合对象
        """
        # 构建更精确的缓存键，包含路径、模型、提供商信息
        provider_info = self.embedding_provider.get_model_info()
        model_name = provider_info.get('model_name', 'default')
        provider_type = self.embedding_provider.get_provider_type()
        
        # 使用绝对路径确保唯一性
        abs_db_path = str(Path(self.db_path).resolve())
        cache_key = f"{abs_db_path}:{name}:{model_name}:{provider_type}"

        with VectorStore._cache_lock:
            if cache_key in VectorStore._collection_cache:
                cached_collection = VectorStore._collection_cache[cache_key]
                # 简单验证缓存集合是否仍然有效
                try:
                    # 尝试访问集合的count方法验证连接
                    cached_collection.count()
                    self.logger.debug(f"✅ 从缓存获取集合: {name}")
                    return cached_collection
                except Exception as e:
                    # 缓存集合无效，删除缓存项
                    self.logger.warning(f"⚠️ 缓存集合无效，重新创建: {name}, 错误: {e}")
                    del VectorStore._collection_cache[cache_key]

        # 获取集合前先记录详细信息
        self.logger.info(f"正在获取或创建集合: {name}")

        from chromadb.utils import embedding_functions

        # 根据嵌入提供商类型创建不同的嵌入函数
        if self.embedding_provider.get_provider_type() == "local":
            # 本地模型使用SentenceTransformerEmbeddingFunction
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_provider.model_name
            )
        else:
            # API提供商使用自定义的嵌入函数
            class APIEmbeddingFunction:
                def __init__(self, provider: EmbeddingProvider):
                    self.provider = provider

                def __call__(self, input_texts):
                    # 这里不能直接使用await，需要在异步环境中调用
                    # 实际使用时会通过其他方式处理
                    raise NotImplementedError("API提供商需要通过异步方式调用")

            embedding_function = None  # API提供商暂时不设置嵌入函数

        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )

        # 检查集合维度（仅对本地模型）
        if self.embedding_provider.get_provider_type() == "local":
            self._verify_collection_dimension(collection)

        # 缓存集合
        with VectorStore._cache_lock:
            VectorStore._collection_cache[cache_key] = collection
            self.logger.debug(f"✅ 集合已缓存: {name}, 缓存大小: {len(VectorStore._collection_cache)}")

        return collection

    @classmethod
    def clear_collection_cache(cls):
        """
        清空集合缓存（用于测试或强制重建）
        """
        with cls._cache_lock:
            cache_size = len(cls._collection_cache)
            cls._collection_cache.clear()
            if hasattr(cls, 'logger'):
                cls.logger.info(f"✅ 集合缓存已清空，清理了 {cache_size} 个缓存项")

    @classmethod
    def get_cache_statistics(cls):
        """
        获取集合缓存统计信息
        
        Returns:
            缓存统计字典
        """
        with cls._cache_lock:
            cache_size = len(cls._collection_cache)
            cache_keys = list(cls._collection_cache.keys())
            
            # 分析缓存键模式
            db_paths = set()
            collections = set()
            models = set()
            
            for key in cache_keys:
                parts = key.split(':')
                if len(parts) >= 4:
                    db_paths.add(parts[0])
                    collections.add(parts[1])
                    models.add(parts[2])
            
            return {
                'cache_size': cache_size,
                'unique_databases': len(db_paths),
                'unique_collections': len(collections),
                'unique_models': len(models),
                'cache_keys': cache_keys
            }

    @classmethod
    def invalidate_cache_by_pattern(cls, pattern: str):
        """
        根据模式失效缓存项
        
        Args:
            pattern: 模式字符串，支持部分匹配
        """
        with cls._cache_lock:
            keys_to_remove = [key for key in cls._collection_cache.keys() if pattern in key]
            
            for key in keys_to_remove:
                del cls._collection_cache[key]
            
            if hasattr(cls, 'logger'):
                cls.logger.info(f"✅ 失效了 {len(keys_to_remove)} 个匹配模式 '{pattern}' 的缓存项")

    # ===== BM25混合检索集成方法 =====

    def _lazy_init_bm25_retriever(self) -> bool:
        """延迟初始化BM25检索器（仅在首次需要时）"""
        if self.bm25_retriever is not None:
            return True  # 已初始化

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

    def _init_bm25_retriever(self) -> bool:
        """保持向后兼容性的BM25检索器初始化方法"""
        return self._lazy_init_bm25_retriever()

    def _is_hybrid_search_enabled(self) -> bool:
        """检查是否启用混合检索"""
        # 确保BM25检索器已初始化
        self._lazy_init_bm25_retriever()
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
        """同步ChromaDB集合数据到BM25索引（副集合按需用 tag_ids 重建标签文本）。"""
        if not self._is_hybrid_search_enabled():
            return True

        try:
            # 获取集合中的所有数据
            all_data = collection.get()
            if not all_data:
                return True

            # 数据拆解
            doc_ids = all_data.get('ids') or []
            documents = all_data.get('documents') or []  # 兼容旧数据，尽量不依赖
            metadatas = all_data.get('metadatas') or []

            pairs = []  # (id, text)
            is_sub = (collection_name == system_config.notes_sub_collection_name)

            if not is_sub:
                # 主集合：直接使用 metadata['content']；如缺失再回退 documents（兼容旧数据）
                for idx, doc_id in enumerate(doc_ids):
                    text = None
                    if idx < len(metadatas) and metadatas[idx]:
                        text = (metadatas[idx] or {}).get('content')
                    if not text and idx < len(documents) and documents[idx]:
                        text = documents[idx]
                    if text:
                        pairs.append((doc_id, text))
            else:
                # 副集合：忽略 documents；优先 metadata.tags_text；否则根据主集合的 tag_ids 重建
                import json
                # 获取主集合实例（用于查 tag_ids）
                try:
                    main_collection = self.get_or_create_collection_with_dimension_check(system_config.notes_main_collection_name)
                except Exception:
                    main_collection = None

                tm = self._get_tag_manager()

                for idx, doc_id in enumerate(doc_ids):
                    text = None
                    meta = (metadatas[idx] or {}) if idx < len(metadatas) else {}
                    # 优先使用已存在的 tags_text（兼容旧数据）
                    text = meta.get('tags_text') if isinstance(meta, dict) else None

                    if not text and main_collection is not None:
                        try:
                            main_data = main_collection.get(ids=[doc_id])
                            main_meta = (main_data.get('metadatas') or [None])[0] if main_data else None
                            tag_ids_val = (main_meta or {}).get('tag_ids')
                            if tag_ids_val is not None and tm is not None:
                                if isinstance(tag_ids_val, str):
                                    try:
                                        tag_ids = json.loads(tag_ids_val)
                                    except Exception:
                                        tag_ids = []
                                else:
                                    tag_ids = tag_ids_val or []
                                tag_names = tm.get_tag_names(tag_ids) if tag_ids else []
                                if tag_names:
                                    text = " ".join(tag_names)
                        except Exception:
                            text = None

                    if text:
                        pairs.append((doc_id, text))

            if not pairs:
                return True

            valid_ids, valid_texts = zip(*pairs)
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

        # 确保BM25检索器已初始化
        if not self._lazy_init_bm25_retriever():
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
        # 确保BM25检索器已初始化
        self._lazy_init_bm25_retriever()
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
        # 确保BM25检索器已初始化
        if not self._lazy_init_bm25_retriever():
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

    @classmethod
    def get_cache_info(cls):
        """获取缓存信息（用于调试）"""
        with cls._cache_lock:
            return {
                'cache_size': len(cls._collection_cache),
                'cached_collections': list(cls._collection_cache.keys())
            }

    # ===== 笔记专用检索方法 =====

    def store_note(self, collection, note: NoteData):
        """
        存储笔记到向量数据库。

        Args:
            collection: 目标 ChromaDB 集合
            note: NoteData 对象
        """
        try:
            # 使用高级抽象方法存储笔记
            self.upsert_documents(
                collection=collection,
                ids=note.id,
                embedding_texts=note.get_embedding_text(),  # 用于向量化的文本
                documents=None,  # 主/副集合均不落盘documents，正文在metadata['content']
                metadatas=note.to_dict()
            )

            # 笔记的BM25索引在文件扫描完成后统一重建，不在这里立即更新
            # 避免每个笔记都触发全量重建，提升批量导入性能

        except Exception:
            raise

    def search_notes(self, collection, query: str, limit: int = 10,
                    where_filter: Optional[dict] = None) -> List[NoteData]:
        """
        搜索笔记，返回 NoteData 对象列表。

        Args:
            collection: 目标 ChromaDB 集合
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器

        Returns:
            相关的笔记对象列表（NoteData）
        """
        try:
            # 显式生成查询向量（同步调用）
            query_embedding = self.embed_single_document(query)

            # 构建查询参数
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": limit
            }

            # 如果提供了过滤器，则添加到查询参数
            if where_filter:
                if len(where_filter) == 1:
                    query_params["where"] = where_filter
                else:
                    query_params["where"] = {"$and": [{k: v} for k, v in where_filter.items()]}

            # 在ChromaDB中进行向量相似度搜索（数据库内部处理并发）
            results = collection.query(**query_params)

            # 将结果转换为笔记对象，并保留相似度分数
            vector_results = []
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                distances = results.get('distances', [[]])[0]
                metadatas = results['metadatas'][0]

                for idx, meta in enumerate(metadatas):
                    if meta:
                        try:
                            note = NoteData.from_dict(meta)
                            # 将距离转换为相似度分数
                            if idx < len(distances):
                                distance = distances[idx]
                                similarity = max(0.0, 1.0 - (distance / 2.0))
                                if idx < 3:
                                    self.logger.debug(f"笔记{idx}: distance={distance:.4f}, similarity={similarity:.4f}")
                                note.similarity = similarity
                            else:
                                note.similarity = 0.0
                            vector_results.append(note)
                        except Exception as e:
                            self.logger.warning(f"无法创建笔记对象，跳过: {e}")
                            continue

            # 混合检索：结合BM25结果
            if self._is_hybrid_search_enabled():
                collection_name = collection.name
                if self.bm25_retriever.get_document_count(collection_name) == 0:
                    self._sync_collection_to_bm25(collection_name, collection)

                bm25_results = self.bm25_retriever.search(collection_name, query, limit)
                final_results = self._merge_note_results(vector_results, bm25_results, collection)
            else:
                final_results = vector_results

            return final_results

        except Exception:
            raise

    def get_notes_by_ids(self, collection, note_ids: List[str]) -> List[NoteData]:
        """
        根据笔记ID列表获取笔记对象。

        Args:
            collection: 目标 ChromaDB 集合
            note_ids: 笔记ID列表

        Returns:
            笔记对象列表
        """
        try:
            if not note_ids:
                return []

            # 使用ChromaDB的get方法获取指定ID的文档
            retrieved_docs = collection.get(ids=note_ids)
            if not retrieved_docs or not retrieved_docs['metadatas']:
                return []

            notes = []
            for meta in retrieved_docs['metadatas']:
                if meta:  # 笔记集合只包含笔记，不需要检查note_type
                    try:
                        note = NoteData.from_dict(meta)
                        notes.append(note)
                    except Exception as e:
                        self.logger.warning(f"无法创建笔记对象，跳过: {e}")
                        continue

            return notes

        except Exception as e:
            self.logger.error(f"根据ID获取笔记失败: {e}")
            return []

    def _merge_note_results(self, vector_results: List[NoteData],
                          bm25_results: List[Tuple[str, float]],
                          collection) -> List[NoteData]:
        """
        融合笔记的向量检索和BM25检索结果。

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            collection: ChromaDB集合

        Returns:
            融合后的笔记列表
        """
        if not vector_results and not bm25_results:
            return []

        if not bm25_results or not self._is_hybrid_search_enabled():
            return vector_results

        if not vector_results:
            # 只有BM25结果，需要根据note_id查找NoteData对象
            return self.get_notes_by_ids(collection, [note_id for note_id, _ in bm25_results])

        # 创建笔记ID到NoteData对象的映射
        vector_notes_map = {}
        for note in vector_results:
            vector_notes_map[note.id] = note

        # 标准化分数到[0,1]区间
        vector_scores = {}
        if vector_results:
            # 向量检索结果按相似度排序，分配递减分数
            for i, note in enumerate(vector_results):
                vector_scores[note.id] = 1.0 - (i * 0.1)  # 简单线性递减

        bm25_scores = {}
        if bm25_results:
            max_score = max(score for _, score in bm25_results) if bm25_results else 1.0
            for note_id, score in bm25_results:
                if max_score > 0:
                    bm25_scores[note_id] = score / max_score
                else:
                    bm25_scores[note_id] = 0.0

        # 合并分数
        combined_scores = {}
        for note_id, score in vector_scores.items():
            combined_scores[note_id] = self.vector_weight * score

        for note_id, score in bm25_scores.items():
            combined_scores[note_id] = combined_scores.get(note_id, 0) + self.bm25_weight * score

        # 添加纯向量检索中存在但BM25中没有的结果
        for note_id in vector_notes_map:
            if note_id not in combined_scores:
                combined_scores[note_id] = self.vector_weight * 0.5  # 给予中等分数

        # 按合并分数排序
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回排序后的NoteData对象
        final_notes = []
        note_ids_to_get = [note_id for note_id, _ in sorted_results[:len(vector_results)]]

        # 需要从数据库完整获取这些笔记（因为BM25结果中没有完整的笔记对象）
        if len(note_ids_to_get) > len(vector_notes_map):
            additional_notes = self.get_notes_by_ids(collection,
                [note_id for note_id in note_ids_to_get if note_id not in vector_notes_map])
            # 合并结果
            all_notes_map = vector_notes_map.copy()
            for note in additional_notes:
                all_notes_map[note.id] = note
        else:
            all_notes_map = vector_notes_map

        for note_id, _ in sorted_results[:len(vector_results)]:
            if note_id in all_notes_map:
                final_notes.append(all_notes_map[note_id])

        return final_notes
