"""
向量存储组件.

封装所有与ChromaDB向量数据库的交互,通过依赖注入支持不同的嵌入提供商.
"""

import chromadb
from typing import List, Optional, Tuple
import traceback
from pathlib import Path
from .embedding_provider import EmbeddingProvider
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
from .bm25_retriever import rerank_with_bm25


class VectorStore:
    """
    向量存储类.

    负责记忆的向量化和存储,使用ChromaDB作为后端.
    通过依赖注入接收嵌入提供商,实现解耦设计.
    """

    # 类级别的集合缓存,防止重复初始化
    _collection_cache = {}
    _cache_lock = None

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        db_path: Optional[str] = None,
    ):
        """
        初始化向量存储.

        Args:
            embedding_provider: 嵌入提供商实例.如果未提供,则使用默认本地模型.
            db_path: 数据库存储路径.如果未提供,则从系统配置中获取.
        """
        self.logger = logger

        # 初始化线程锁(仅首次创建时)
        if VectorStore._cache_lock is None:
            import threading

            VectorStore._cache_lock = threading.RLock()

        # 初始化嵌入提供商
        if embedding_provider is None:
            # 不再强制降级，而是抛出明确错误
            self.logger.error("未指定嵌入提供商，无法初始化VectorStore")
            raise ValueError("VectorStore初始化需要有效的embedding_provider参数")
        else:
            self.embedding_provider = embedding_provider
            provider_info = embedding_provider.get_model_info()
            self.logger.info(f"使用嵌入提供商: {provider_info}")

            # 验证提供商的可用性
            if not embedding_provider.is_available():
                self.logger.error(f"指定的嵌入提供商不可用: {provider_info}")
                raise ValueError(f"指定的嵌入提供商不可用: {provider_info.get('model_name', 'Unknown')}")

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
        self.logger.info(f"已创建ChromaDB客户端,路径: {self.db_path}")

        # 实例变量
        self.collections = {}

        # ChromaDB是线程安全的,不需要额外的线程锁
        # 移除了 self._db_lock = threading.RLock()

        # 混合检索配置 - 强制启用
        self.hybrid_search_enabled = True  # 强制启用混合检索以获得最佳体验
        self.vector_weight = 0.6
        self.bm25_weight = 0.4

        # 懒加载的标签管理器(用于基于 tag_ids 重建标签文本)
        self._tag_manager: Optional[TagManager] = None

    def _get_tag_manager(self) -> Optional[TagManager]:
        """懒加载 TagManager(基于 PathManager 当前 provider 和索引目录)."""
        if self._tag_manager is not None:
            return self._tag_manager
        try:
            pm = PathManager.get_instance()
            index_dir = str(pm.get_index_dir())
            provider_id = pm.get_current_provider()
            self._tag_manager = TagManager(index_dir, provider_id)
            return self._tag_manager
        except Exception as e:
            self.logger.warning(f"初始化 TagManager 失败,无法为BM25重建标签文本: {e}")
            return None

    def _post_initialization_verification(self):
        """初始化完成后验证"""
        try:
            # 验证嵌入提供商
            if not self.embedding_provider.is_available():
                self.logger.error("嵌入提供商不可用！")
                return

            # 验证默认集合
            if hasattr(self, "collection") and self.collection:
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
                collection.query(query_embeddings=[dummy_vector], n_results=1)

        except Exception as e:
            if "dimension" in str(e).lower():
                self.logger.error(f"  ✗ 集合 {collection.name} 维度不匹配: {e}")
                # 记录集合当前的记录数
                count = collection.count()
                self.logger.info(f"  集合 {collection.name} 当前记录数: {count}")

    def set_storage_path(self, new_path: str):
        """
        设置新的存储路径并重新初始化ChromaDB客户端.

        注意:这将创建一个新的ChromaDB客户端,之前的数据仍在原路径中.
        如果需要迁移数据,请手动复制数据库文件.

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
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name
            )

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
            self.collections[collection_name] = (
                self.get_or_create_collection_with_dimension_check(collection_name)
            )
        return self.collections[collection_name]

    async def remember(self, collection, memory: BaseMemory):
        """
        记住一条新记忆.

        Args:
            collection: 目标 ChromaDB 集合.
            memory: 要记住的记忆对象(任何 BaseMemory 的子类)
        """
        # 获取用于向量化的语义核心文本
        semantic_core = memory.get_semantic_core()

        # 获取记忆的内容文本(用于文档存储)
        content_text = ""
        if hasattr(memory, "content"):
            content_text = memory.content
        elif hasattr(memory, "definition"):
            content_text = memory.definition
        elif hasattr(memory, "procedure"):
            content_text = memory.procedure
        else:
            content_text = semantic_core  # 兜底方案

        # 使用高级抽象方法存储记忆
        await self.upsert_documents(
            collection=collection,
            ids=memory.id,
            embedding_texts=semantic_core,  # 用于向量化的语义核心
            documents=content_text,  # 实际存储的内容
            metadatas=memory.to_dict(),
        )

    async def recall(
        self,
        collection,
        query: str,
        limit: int = 10,
        where_filter: Optional[dict] = None,
        similarity_threshold: float = 0.6,
    ) -> List[BaseMemory]:
        """
        根据查询回忆相关记忆,支持复杂的元数据过滤(异步方法).

        Args:
            collection: 目标 ChromaDB 集合.
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器字典 (e.g., {"memory_type": "EventMemory", "is_consolidated": False})
            similarity_threshold: 相似度阈值(0.0-1.0),低于此阈值的结果将被过滤

        Returns:
            相关的记忆对象列表(BaseMemory 的子类)
        """
        # 显式生成查询向量(异步调用)，指明这是查询场景
        query_embedding = await self.embed_single_document(query, is_query=True)

        # --- 处理向量化失败 ---
        if query_embedding is None:
            self.logger.warning(
                f"因向量化失败，查询 '{query[:50]}...' 的回忆流程已中止。"
            )
            return []  # 立即返回空列表

        # 构建查询参数 - 获取更多候选结果用于阈值过滤
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": limit * 3,  # 获取3倍候选结果进行过滤
        }

        # 如果提供了过滤器,则添加到查询参数
        if where_filter:
            if len(where_filter) == 1:
                query_params["where"] = where_filter
            else:
                query_params["where"] = {
                    "$and": [{k: v} for k, v in where_filter.items()]
                }

        # 在ChromaDB中进行向量相似度搜索(数据库内部处理并发)
        results = collection.query(**query_params)

        # 将结果转换为记忆对象,并计算相似度分数进行过滤
        vector_results = []
        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results["metadatas"][0]

            for idx, meta in enumerate(metadatas):
                if meta and idx < len(distances) and idx < len(ids):
                    # 将真实ID添加到meta中
                    meta["id"] = ids[idx]

                    # 计算相似度分数
                    distance = distances[idx]
                    similarity = max(0.0, 1.0 - (distance / 2.0))

                    # 应用相似度阈值过滤
                    if similarity >= similarity_threshold:
                        memory = BaseMemory.from_dict(meta)
                        memory.similarity = similarity  # 设置相似度属性
                        vector_results.append(memory)

                        # 只记录前3个结果的调试信息
                        if len(vector_results) <= 3:
                            self.logger.debug(
                                f"记忆{len(vector_results) - 1}: distance={distance:.4f}, similarity={similarity:.4f}"
                            )

                    # 达到所需数量时停止
                    if len(vector_results) >= limit:
                        break

        # 混合检索:为记忆系统提供BM25精排(强制启用)
        final_results = self._rerank_with_bm25(query, vector_results, collection, limit)

        return final_results

    async def recall_with_vector(
        self,
        collection,
        vector: List[float],
        query: str,
        limit: int = 10,
        where_filter: Optional[dict] = None,
        similarity_threshold: float = 0.6,
    ) -> List[BaseMemory]:
        """
        使用预计算的向量回忆相关记忆,支持复杂的元数据过滤.

        Args:
            collection: 目标 ChromaDB 集合.
            vector: 预计算的查询向量
            query: 原始查询文本，用于BM25精排
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器字典 (e.g., {"memory_type": "EventMemory", "is_consolidated": False})
            similarity_threshold: 相似度阈值(0.0-1.0),低于此阈值的结果将被过滤

        Returns:
            相关的记忆对象列表(BaseMemory 的子类)
        """
        # 构建查询参数 - 获取更多候选结果用于阈值过滤
        query_params = {
            "query_embeddings": [vector],
            "n_results": limit * 3,  # 获取3倍候选结果进行过滤
        }

        # 如果提供了过滤器,则添加到查询参数
        if where_filter:
            if len(where_filter) == 1:
                query_params["where"] = where_filter
            else:
                query_params["where"] = {
                    "$and": [{k: v} for k, v in where_filter.items()]
                }

        # 在ChromaDB中进行向量相似度搜索(数据库内部处理并发)
        results = collection.query(**query_params)

        # 将结果转换为记忆对象,并计算相似度分数进行过滤
        vector_results = []
        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results["metadatas"][0]

            for idx, meta in enumerate(metadatas):
                if meta and idx < len(distances) and idx < len(ids):
                    # 将真实ID添加到meta中
                    meta["id"] = ids[idx]

                    # 计算相似度分数
                    distance = distances[idx]
                    similarity = max(0.0, 1.0 - (distance / 2.0))

                    # 应用相似度阈值过滤
                    if similarity >= similarity_threshold:
                        memory = BaseMemory.from_dict(meta)
                        memory.similarity = similarity  # 设置相似度属性
                        vector_results.append(memory)

                        # 只记录前3个结果的调试信息
                        if len(vector_results) <= 3:
                            self.logger.debug(
                                f"记忆{len(vector_results) - 1}: distance={distance:.4f}, similarity={similarity:.4f}"
                            )

                    # 达到所需数量时停止
                    if len(vector_results) >= limit:
                        break

        # 混合检索:为记忆系统提供BM25精排(强制启用)
        final_results = self._rerank_with_bm25(query, vector_results, collection, limit)

        return final_results

    async def update_memory(self, collection, memory_id, updates: dict = None):
        """
        更新记忆的元数据（支持单个或批量更新）.

        Args:
            collection: 目标 ChromaDB 集合.
            memory_id: 单个记忆ID（str）或批量更新列表（List[Dict]）
                - 单个: memory_id="id123", updates={"strength": 5}
                - 批量: memory_id=[{"id": "id1", "updates": {...}}, {"id": "id2", "updates": {...}}]
            updates: 单个更新时的字段字典 (e.g., {"is_consolidated": True, "strength": 5})
        """
        # 批量更新
        if isinstance(memory_id, list):
            try:
                ids = [item["id"] for item in memory_id]
                current_data = collection.get(ids=ids)

                if not current_data or not current_data["metadatas"]:
                    raise ValueError("No memories found for batch update")

                # 构建 ID 到数据的映射（collection.get 返回顺序不保证与输入一致）
                id_to_data = {}
                for i, mem_id in enumerate(current_data["ids"]):
                    id_to_data[mem_id] = {
                        "metadata": current_data["metadatas"][i],
                        "document": current_data["documents"][i] if current_data["documents"] and i < len(current_data["documents"]) else ""
                    }

                updated_ids = []
                updated_embeddings = []
                updated_documents = []
                updated_metadatas = []

                for item in memory_id:
                    mem_id = item["id"]
                    mem_updates = item["updates"]

                    # 用 ID 查找对应的数据
                    if mem_id not in id_to_data:
                        self.logger.warning(f"跳过记忆 {mem_id} 的批量更新：未找到数据")
                        continue

                    current_meta = id_to_data[mem_id]["metadata"]
                    current_document = id_to_data[mem_id]["document"]

                    semantic_core = current_meta.get("judgment", "") + " " + current_meta.get("tags", "")
                    current_meta.update(mem_updates)

                    updated_ids.append(mem_id)
                    updated_embeddings.append(semantic_core)
                    updated_documents.append(current_document)
                    updated_metadatas.append(current_meta)

                if updated_ids:
                    await self.upsert_documents(
                        collection=collection,
                        ids=updated_ids,
                        embedding_texts=updated_embeddings,
                        documents=updated_documents,
                        metadatas=updated_metadatas,
                    )

            except Exception as e:
                self.logger.error(f"批量更新记忆失败: {str(e)}")
                raise

        # 单个更新（向后兼容）
        else:
            try:
                current_data = collection.get(ids=[memory_id])
                if not current_data or not current_data["metadatas"]:
                    raise ValueError(f"Memory with id {memory_id} not found")

                current_meta = current_data["metadatas"][0]
                current_document = (
                    current_data["documents"][0] if current_data["documents"] else ""
                )

                semantic_core = (
                    current_meta.get("judgment", "") + " " + current_meta.get("tags", "")
                )

                current_meta.update(updates)

                await self.upsert_documents(
                    collection=collection,
                    ids=memory_id,
                    embedding_texts=semantic_core,
                    documents=current_document,
                    metadatas=current_meta,
                )

            except Exception as e:
                self.logger.error(f"更新记忆 {memory_id} 失败: {str(e)}")
                raise

    def delete_memories(
        self, collection, where_filter: dict, exclude_associations: bool = False
    ):
        """
        根据条件删除记忆.

        Args:
            collection: 目标 ChromaDB 集合.
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

            # 如果需要排除关联,添加额外的条件
            if exclude_associations:
                if "$and" in base_filter:
                    base_filter["$and"].append({"memory_type": {"$ne": "Association"}})
                else:
                    base_filter = {
                        "$and": [base_filter, {"memory_type": {"$ne": "Association"}}]
                    }

            # 获取符合条件的记忆ID
            results = collection.get(where=base_filter)
            ids_to_delete = results["ids"]

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                self.logger.info(
                    f"Deleted {len(ids_to_delete)} memories with filter: {where_filter}"
                )

        except Exception as e:
            self.logger.error(f"Failed to delete memories: {str(e)}")
            raise

    def clear_collection(self, collection):
        """清空指定集合."""
        try:
            collection_name = collection.name
            self.client.delete_collection(collection_name)
            # 重新创建集合,确保 embedding_function 等元数据被保留
            self.get_or_create_collection_with_dimension_check(name=collection_name)
        except Exception as e:
            self.logger.error(f"清空所有记忆失败: {e}")
            raise

    async def upsert_documents(
        self,
        collection,
        *,
        ids,
        embedding_texts,
        documents=None,
        metadatas=None,
        _return_timings=False,
    ):
        """
        高级 upsert 方法(异步接口):从文本生成向量并存储到向量数据库.

        这是向量数据库的通用方法,支持不同的使用场景:
        - 记忆系统:documents 参数可选,所有数据可存储在 metadatas 中
        - 笔记系统:documents 参数可选,正文可存储在 metadatas['content'] 中
        - 其他系统:可灵活选择使用 documents 字段

        Args:
            collection: 目标 ChromaDB 集合
            ids: 文档ID或ID列表
            embedding_texts: 用于生成向量的文本或文本列表
            documents: 可选的文档内容或列表(默认 None)
            metadatas: 与文档关联的元数据或元数据列表
            _return_timings: 内部参数,是否返回计时信息
        """
        import time

        timings = {} if _return_timings else None

        # 统一处理输入为列表
        ids_list = [ids] if isinstance(ids, str) else ids
        embedding_texts_list = (
            [embedding_texts] if isinstance(embedding_texts, str) else embedding_texts
        )
        documents_list = (
            [documents]
            if isinstance(documents, str)
            else documents
            if documents is not None
            else None
        )
        metadatas_list = [metadatas] if isinstance(metadatas, dict) else metadatas

        if not ids_list:
            return timings if _return_timings else None

        # 1. 使用嵌入提供商从源文本生成 embeddings(异步调用)
        t_embed = time.time()
        embeddings = await self.embed_documents(embedding_texts_list)
        if _return_timings:
            timings["embed"] = (time.time() - t_embed) * 1000

        # 2. 调用底层的 upsert
        upsert_params = {
            "ids": ids_list,
            "embeddings": embeddings,
        }

        # 只在 documents 不为 None 时才添加到 upsert 参数中
        if documents_list is not None:
            upsert_params["documents"] = documents_list

        if metadatas_list is not None:
            upsert_params["metadatas"] = metadatas_list

        # 直接upsert(数据库内部处理并发)
        t_db = time.time()
        collection.upsert(**upsert_params)
        if _return_timings:
            timings["db_upsert"] = (time.time() - t_db) * 1000

        return timings if _return_timings else None

    async def embed_documents(
        self, documents: List[str], is_query: bool = False, timeout: int = 3
    ) -> Optional[List[List[float]]]:
        """
        使用嵌入提供商为文档列表生成向量嵌入(异步方法).

        Args:
            documents: 需要进行向量化的文档字符串列表.
            is_query: 是否为查询场景. True=性能敏感的查询场景, False=可靠的存储场景.
            timeout: 查询场景下的超时时间(秒), 仅在 is_query=True 时生效.

        Returns:
            一个由向量(浮点数列表)组成的列表, 或在查询失败时返回 None.

        Raises:
            RateLimitExceededError: 仅在存储场景(is_query=False)下, 当重试3次后仍然遇到429错误时抛出
            Exception: 其他错误直接抛出,不重试
        """
        if not documents:
            return []

        # --- 性能敏感的查询路径 ---
        if is_query:
            try:
                # 使用 asyncio.wait_for 实现超时控制
                import asyncio
                embeddings = await asyncio.wait_for(
                    self.embedding_provider.embed_documents(documents),
                    timeout=timeout
                )
                return embeddings
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"查询向量化超时（超过 {timeout} 秒），操作已中断。"
                )
                return None
            except Exception as e:
                self.logger.error(f"查询向量化失败，立即返回。错误: {e}")
                return None

        # --- 可靠的存储路径 (保留现有重试逻辑) ---
        else:
            # 导入自定义异常
            from ..exceptions import RateLimitExceededError
            import asyncio

            # 重试配置(仅针对429错误)
            max_retries = 3
            base_wait_time = 60  # 基础等待时间(秒)

            for attempt in range(max_retries):
                try:
                    # 使用嵌入提供商生成向量(异步调用)
                    embeddings = await self.embedding_provider.embed_documents(documents)
                    return embeddings

                except Exception as e:
                    error_msg = str(e)

                    # 只对429错误进行重试,其他错误直接抛出
                    is_rate_limit = (
                        "429" in error_msg
                        or "rate limit" in error_msg.lower()
                        or "TPM limit" in error_msg
                    )

                    if not is_rate_limit:
                        # 非429错误,直接抛出,不重试
                        self.logger.error(f"❌ 向量化失败: {error_msg}")
                        raise

                    # 429错误的重试逻辑
                    if attempt < max_retries - 1:
                        # 指数退避:60s, 120s, 240s
                        wait_time = base_wait_time * (2**attempt)

                        # 添加随机抖动(±10%)避免同时重试
                        import random

                        jitter = random.uniform(-0.1, 0.1) * wait_time
                        final_wait = int(wait_time + jitter)

                        self.logger.warning(
                            f"⏱️  遇到速率限制 (429错误),"
                            f"等待 {final_wait} 秒后重试 "
                            f"(第 {attempt + 1}/{max_retries} 次)"
                        )

                        await asyncio.sleep(final_wait)
                    else:
                        # 达到最大重试次数,抛出自定义异常
                        self.logger.error(
                            f"❌ 向量化失败:达到最大重试次数 ({max_retries}),"
                            f"速率限制错误: {error_msg}"
                        )
                        raise RateLimitExceededError(
                            f"速率限制重试失败 (尝试了{max_retries}次): {error_msg}",
                            attempts=max_retries,
                        ) from e

            # 理论上不会到这里
            raise Exception("向量化失败:未知错误")

    async def embed_single_document(
        self, document: str, is_query: bool = False, timeout: int = 3
    ) -> Optional[List[float]]:
        """
        为单个文档生成向量嵌入(异步方法).

        Args:
            document: 需要向量化的单个文档字符串.
            is_query: 是否为查询场景.
            timeout: 查询场景下的超时时间(秒).

        Returns:
            单个文档的向量, 或在查询失败时返回 None.
        """
        embeddings = await self.embed_documents(
            [document], is_query=is_query, timeout=timeout
        )
        if embeddings:
            return embeddings[0]
        return None

    async def embed_text_direct(self, text: str, is_query: bool = False, timeout: int = 3) -> Optional[List[float]]:
        """
        直接向量化文本，不经过复杂处理和缓存

        Args:
            text: 需要向量化的文本
            is_query: 是否为查询场景，影响超时和重试策略
            timeout: 查询场景下的超时时间(秒)

        Returns:
            文本的向量表示，失败时返回None
        """
        if not text or not text.strip():
            return None

        # 直接调用底层的嵌入提供商，避免额外处理
        try:
            if is_query:
                # 查询场景：使用超时控制
                import asyncio
                embeddings = await asyncio.wait_for(
                    self.embedding_provider.embed_documents([text]),
                    timeout=timeout
                )
                return embeddings[0] if embeddings else None
            else:
                # 存储场景：直接调用
                embeddings = await self.embedding_provider.embed_documents([text])
                return embeddings[0] if embeddings else None
        except Exception as e:
            self.logger.debug(f"直接向量化失败: {e}")
            return None



    def get_or_create_collection_with_dimension_check(self, name: str):
        """
        获取或创建集合,并检查维度是否匹配
        完善的集合缓存机制,防止重复初始化

        Args:
            name: 集合名称

        Returns:
            ChromaDB集合对象
        """
        # 构建更精确的缓存键,包含路径、模型、提供商信息
        provider_info = self.embedding_provider.get_model_info()
        model_name = provider_info.get("model_name", "default")
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
                    # 缓存集合无效,删除缓存项
                    self.logger.warning(f"⚠️ 缓存集合无效,重新创建: {name}, 错误: {e}")
                    del VectorStore._collection_cache[cache_key]

        # 获取集合前先记录详细信息
        self.logger.info(f"正在获取或创建集合: {name}")

        from chromadb.utils import embedding_functions

        # 根据嵌入提供商类型创建不同的嵌入函数
        if self.embedding_provider.get_provider_type() == "local":
            # 本地模型使用SentenceTransformerEmbeddingFunction
            embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_provider.model_name
                )
            )
        else:
            # API提供商使用自定义的嵌入函数
            class APIEmbeddingFunction:
                def __init__(self, provider: EmbeddingProvider):
                    self.provider = provider

                def __call__(self, input_texts):
                    # 这里不能直接使用await,需要在异步环境中调用
                    # 实际使用时会通过其他方式处理
                    raise NotImplementedError("API提供商需要通过异步方式调用")

            embedding_function = None  # API提供商暂时不设置嵌入函数

        collection = self.client.get_or_create_collection(
            name=name, embedding_function=embedding_function
        )

        # 检查集合维度(仅对本地模型)
        if self.embedding_provider.get_provider_type() == "local":
            self._verify_collection_dimension(collection)

        # 缓存集合
        with VectorStore._cache_lock:
            VectorStore._collection_cache[cache_key] = collection
            self.logger.debug(
                f"✅ 集合已缓存: {name}, 缓存大小: {len(VectorStore._collection_cache)}"
            )

        return collection

    @classmethod
    def clear_collection_cache(cls):
        """
        清空集合缓存(用于测试或强制重建)
        """
        with cls._cache_lock:
            cache_size = len(cls._collection_cache)
            cls._collection_cache.clear()
            if hasattr(cls, "logger"):
                cls.logger.info(f"✅ 集合缓存已清空,清理了 {cache_size} 个缓存项")

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
                parts = key.split(":")
                if len(parts) >= 4:
                    db_paths.add(parts[0])
                    collections.add(parts[1])
                    models.add(parts[2])

            return {
                "cache_size": cache_size,
                "unique_databases": len(db_paths),
                "unique_collections": len(collections),
                "unique_models": len(models),
                "cache_keys": cache_keys,
            }

    @classmethod
    def invalidate_cache_by_pattern(cls, pattern: str):
        """
        根据模式失效缓存项

        Args:
            pattern: 模式字符串,支持部分匹配
        """
        with cls._cache_lock:
            keys_to_remove = [
                key for key in cls._collection_cache.keys() if pattern in key
            ]

            for key in keys_to_remove:
                del cls._collection_cache[key]

            if hasattr(cls, "logger"):
                cls.logger.info(
                    f"✅ 失效了 {len(keys_to_remove)} 个匹配模式 '{pattern}' 的缓存项"
                )

    # ===== BM25混合检索集成方法 =====

    def _rerank_with_bm25(
        self, query: str, vector_results: List[BaseMemory], collection, limit: int
    ) -> List[BaseMemory]:
        """
        使用无状态BM25对向量检索结果进行精排.

        Args:
            query: 查询文本
            vector_results: 向量检索结果
            collection: ChromaDB集合
            limit: 返回结果数量限制

        Returns:
            精排后的记忆列表
        """
        if not vector_results:
            return []

        try:
            # 准备候选文档数据
            candidates = []
            for memory in vector_results:
                # 获取记忆的语义核心文本用于BM25精排
                content = memory.get_semantic_core()
                if content:
                    candidates.append({"id": memory.id, "content": content})

            if not candidates:
                return vector_results

            # 使用无状态BM25函数精排
            bm25_results = rerank_with_bm25(query, candidates, limit)

            # 融合结果
            return self._merge_results(vector_results, bm25_results, collection)

        except Exception as e:
            self.logger.error(f"BM25精排失败: {e}")
            return vector_results

    def _rerank_notes_with_bm25(
        self, query: str, vector_results: List[NoteData], collection, limit: int
    ) -> List[NoteData]:
        """
        使用无状态BM25对笔记向量检索结果进行精排.

        Args:
            query: 查询文本
            vector_results: 向量检索的笔记结果
            collection: ChromaDB集合
            limit: 返回结果数量限制

        Returns:
            精排后的笔记列表
        """
        if not vector_results:
            return []

        try:
            # 准备候选文档数据
            candidates = []
            for note in vector_results:
                # 获取笔记的embedding文本用于BM25精排
                # 注意:NoteData.get_embedding_text()需要tag_names,但这里我们使用content作为fallback
                content = note.content
                if content:
                    candidates.append({"id": note.id, "content": content})

            if not candidates:
                return vector_results

            # 使用无状态BM25函数精排
            bm25_results = rerank_with_bm25(query, candidates, limit)

            # 融合结果
            return self._merge_note_results(vector_results, bm25_results, collection)

        except Exception as e:
            self.logger.error(f"笔记BM25精排失败: {e}")
            return vector_results

    def _merge_results(
        self,
        vector_results: List[BaseMemory],
        bm25_results: List[Tuple[str, float]],
        collection,
    ) -> List[BaseMemory]:
        """融合向量检索和BM25检索结果"""
        if not vector_results and not bm25_results:
            return []

        # 强制启用混合检索,直接进行结果融合

        if not vector_results:
            # 只有BM25结果,需要根据doc_id查找BaseMemory对象
            return self._get_memories_by_ids(
                collection, [doc_id for doc_id, _ in bm25_results]
            )

        # 创建文档ID到BaseMemory对象的映射
        vector_memories_map = {}
        for memory in vector_results:
            vector_memories_map[memory.id] = memory

        # 标准化分数到[0,1]区间
        vector_scores = {}
        if vector_results:
            # 向量检索结果按相似度排序,分配递减分数
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
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0) + self.bm25_weight * score
            )

        # 添加纯向量检索中存在但BM25中没有的结果
        for doc_id in vector_memories_map:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = self.vector_weight

        # 按合并分数排序
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 返回排序后的BaseMemory对象
        final_memories = []
        memory_ids_to_get = [
            doc_id for doc_id, _ in sorted_results[: len(vector_results)]
        ]

        # 需要从数据库完整获取这些记忆(因为BM25结果中没有完整的记忆对象)
        if len(memory_ids_to_get) > len(vector_memories_map):
            additional_memories = self._get_memories_by_ids(
                collection,
                [
                    doc_id
                    for doc_id in memory_ids_to_get
                    if doc_id not in vector_memories_map
                ],
            )
            # 合并结果
            all_memories_map = vector_memories_map.copy()
            for memory in additional_memories:
                all_memories_map[memory.id] = memory
        else:
            all_memories_map = vector_memories_map

        for doc_id, _ in sorted_results[: len(vector_results)]:
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
            if not retrieved_docs or not retrieved_docs["metadatas"]:
                return []

            memories = []
            for meta in retrieved_docs["metadatas"]:
                if meta:
                    memories.append(BaseMemory.from_dict(meta))

            return memories

        except Exception as e:
            self.logger.error(f"根据ID获取记忆失败: {e}")
            return []

    # ===== 混合检索配置方法 =====

    @classmethod
    def get_cache_info(cls):
        """获取缓存信息(用于调试)"""
        with cls._cache_lock:
            return {
                "cache_size": len(cls._collection_cache),
                "cached_collections": list(cls._collection_cache.keys()),
            }

    def shutdown(self):
        """关闭向量存储,释放资源"""
        self.logger.info("正在关闭向量存储...")

        # 关闭嵌入提供商
        if self.embedding_provider:
            self.embedding_provider.shutdown()

        # 清理集合缓存
        self.clear_collection_cache()

        self.logger.info("向量存储已成功关闭")

    # ===== 笔记专用检索方法 =====

    async def store_note(self, collection, note: NoteData):
        """
        存储笔记到向量数据库.

        Args:
            collection: 目标 ChromaDB 集合
            note: NoteData 对象
        """
        # 使用高级抽象方法存储笔记(笔记数据全部存储在 metadata 中)
        await self.upsert_documents(
            collection=collection,
            ids=note.id,
            embedding_texts=note.get_embedding_text(),  # 用于向量化的文本
            metadatas=note.to_dict(),  # 笔记的所有数据都存储在 metadata 中
        )

        # 笔记使用无状态BM25精排,不需要预先建立索引

    async def search_notes_with_vector(
        self,
        collection,
        vector: List[float],
        query: str,
        limit: int = 10,
        where_filter: Optional[dict] = None,
    ) -> List[NoteData]:
        """
        使用预计算的向量搜索笔记,返回 NoteData 对象列表.

        Args:
            collection: 目标 ChromaDB 集合
            vector: 预计算的查询向量
            query: 原始查询文本，用于BM25精排
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器

        Returns:
            相关的笔记对象列表(NoteData)
        """
        # 构建查询参数
        query_params = {"query_embeddings": [vector], "n_results": limit}

        # 如果提供了过滤器,则添加到查询参数
        if where_filter:
            if len(where_filter) == 1:
                query_params["where"] = where_filter
            else:
                query_params["where"] = {
                    "$and": [{k: v} for k, v in where_filter.items()]
                }

        # 在ChromaDB中进行向量相似度搜索(数据库内部处理并发)
        results = collection.query(**query_params)

        # 将结果转换为笔记对象,并保留相似度分数
        vector_results = []
        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            distances = results.get("distances", [[]])[0]
            metadatas = results["metadatas"][0]

            for idx, meta in enumerate(metadatas):
                if meta:
                    try:
                        note = NoteData.from_dict(meta)
                        # 将距离转换为相似度分数
                        if idx < len(distances):
                            distance = distances[idx]
                            similarity = max(0.0, 1.0 - (distance / 2.0))
                            if idx < 3:
                                self.logger.debug(
                                    f"笔记{idx}: distance={distance:.4f}, similarity={similarity:.4f}"
                                )
                            note.similarity = similarity
                        else:
                            note.similarity = 0.0
                        vector_results.append(note)
                    except Exception as e:
                        self.logger.warning(f"无法创建笔记对象,跳过: {e}")
                        self.logger.error(f"导致创建失败的原始 metadata: {meta}")
                        continue

        # 混合检索:结合BM25结果(强制启用)
        final_results = self._rerank_notes_with_bm25(
            query, vector_results, collection, limit
        )

        return final_results

    async def search_notes(
        self,
        collection,
        query: str,
        limit: int = 10,
        where_filter: Optional[dict] = None,
        vector: Optional[List[float]] = None,
    ) -> List[NoteData]:
        """
        搜索笔记,返回 NoteData 对象列表.

        Args:
            collection: 目标 ChromaDB 集合
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器
            vector: 可选的预计算向量,如果提供则直接使用,否则向量化查询文本

        Returns:
            相关的笔记对象列表(NoteData)
        """
        # 如果提供了预计算向量,直接使用向量搜索
        if vector is not None:
            return await self.search_notes_with_vector(
                collection=collection,
                vector=vector,
                limit=limit,
                where_filter=where_filter,
                query=query,  # 传递原始查询文本
            )

        # 显式生成查询向量(异步调用)，指明这是查询场景
        query_embedding = await self.embed_single_document(query, is_query=True)

        # --- 处理向量化失败 ---
        if query_embedding is None:
            self.logger.warning(
                f"因向量化失败，笔记查询 '{query[:50]}...' 的流程已中止。"
            )
            return []  # 立即返回空列表

        # 构建查询参数
        query_params = {"query_embeddings": [query_embedding], "n_results": limit}

        # 如果提供了过滤器,则添加到查询参数
        if where_filter:
            if len(where_filter) == 1:
                query_params["where"] = where_filter
            else:
                query_params["where"] = {
                    "$and": [{k: v} for k, v in where_filter.items()]
                }

        # 在ChromaDB中进行向量相似度搜索(数据库内部处理并发)
        results = collection.query(**query_params)

        # 将结果转换为笔记对象,并保留相似度分数
        vector_results = []
        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            distances = results.get("distances", [[]])[0]
            metadatas = results["metadatas"][0]

            for idx, meta in enumerate(metadatas):
                if meta:
                    try:
                        note = NoteData.from_dict(meta)
                        # 将距离转换为相似度分数
                        if idx < len(distances):
                            distance = distances[idx]
                            similarity = max(0.0, 1.0 - (distance / 2.0))
                            if idx < 3:
                                self.logger.debug(
                                    f"笔记{idx}: distance={distance:.4f}, similarity={similarity:.4f}"
                                )
                            note.similarity = similarity
                        else:
                            note.similarity = 0.0
                        vector_results.append(note)
                    except Exception as e:
                        self.logger.warning(f"无法创建笔记对象,跳过: {e}")
                        self.logger.error(f"导致创建失败的原始 metadata: {meta}")
                        continue

        # 混合检索:结合BM25结果(强制启用)
        final_results = self._rerank_notes_with_bm25(
            query, vector_results, collection, limit
        )

        return final_results

    async def _search_vector_scores(self, collection, query: str, limit: int = 100) -> dict:
        """
        执行向量搜索,只返回 ID 和相似度分数的映射.

        专为副集合设计(该集合不存储 metadata).
        避免了 NoteData 对象构造,性能更高,逻辑更清晰.

        Args:
            collection: 目标 ChromaDB 集合(通常是副集合)
            query: 搜索查询字符串
            limit: 返回结果的最大数量

        Returns:
            {'note_id': similarity_score, ...} 的字典
        """
        try:
            # 显式生成查询向量(异步调用)，指明这是查询场景
            query_embedding = await self.embed_single_document(query, is_query=True)

            # --- 处理向量化失败 ---
            if query_embedding is None:
                self.logger.warning(
                    f"因向量化失败，向量搜索 '{query[:50]}...' 已中止。"
                )
                return {}  # 立即返回空字典

            # 执行向量搜索
            results = collection.query(
                query_embeddings=[query_embedding], n_results=limit
            )

            # 提取 ID 和距离,转换为相似度分数
            scores = {}
            if results and results["ids"] and len(results["ids"]) > 0:
                doc_ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]

                for idx, doc_id in enumerate(doc_ids):
                    if idx < len(distances):
                        distance = distances[idx]
                        # 将距离转换为相似度分数(0到1之间)
                        similarity = max(0.0, 1.0 - (distance / 2.0))
                        scores[doc_id] = similarity

            return scores

        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            return {}

    def get_notes_by_ids(self, collection, note_ids: List[str]) -> List[NoteData]:
        """
        根据笔记ID列表获取笔记对象.

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
            if not retrieved_docs or not retrieved_docs["metadatas"]:
                return []

            notes = []
            for meta in retrieved_docs["metadatas"]:
                if meta:  # 笔记集合只包含笔记,不需要检查note_type
                    try:
                        note = NoteData.from_dict(meta)
                        notes.append(note)
                    except Exception as e:
                        self.logger.warning(f"无法创建笔记对象,跳过: {e}")
                        self.logger.error(f"导致创建失败的原始 metadata: {meta}")
                        continue

            return notes

        except Exception as e:
            self.logger.error(f"根据ID获取笔记失败: {e}")
            return []

    def _merge_note_results(
        self,
        vector_results: List[NoteData],
        bm25_results: List[Tuple[str, float]],
        collection,
    ) -> List[NoteData]:
        """
        融合笔记的向量检索和BM25检索结果.

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            collection: ChromaDB集合

        Returns:
            融合后的笔记列表
        """
        if not vector_results and not bm25_results:
            return []

        # 强制启用混合检索,直接进行结果融合

        if not vector_results:
            # 只有BM25结果,需要根据note_id查找NoteData对象
            return self.get_notes_by_ids(
                collection, [note_id for note_id, _ in bm25_results]
            )

        # 创建笔记ID到NoteData对象的映射
        vector_notes_map = {}
        for note in vector_results:
            vector_notes_map[note.id] = note

        # 标准化分数到[0,1]区间
        vector_scores = {}
        if vector_results:
            # 向量检索结果按相似度排序,分配递减分数
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
            combined_scores[note_id] = (
                combined_scores.get(note_id, 0) + self.bm25_weight * score
            )

        # 添加纯向量检索中存在但BM25中没有的结果
        for note_id in vector_notes_map:
            if note_id not in combined_scores:
                combined_scores[note_id] = self.vector_weight

        # 按合并分数排序
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 返回排序后的NoteData对象
        final_notes = []
        note_ids_to_get = [
            note_id for note_id, _ in sorted_results[: len(vector_results)]
        ]

        # 需要从数据库完整获取这些笔记(因为BM25结果中没有完整的笔记对象)
        if len(note_ids_to_get) > len(vector_notes_map):
            additional_notes = self.get_notes_by_ids(
                collection,
                [
                    note_id
                    for note_id in note_ids_to_get
                    if note_id not in vector_notes_map
                ],
            )
            # 合并结果
            all_notes_map = vector_notes_map.copy()
            for note in additional_notes:
                all_notes_map[note.id] = note
        else:
            all_notes_map = vector_notes_map

        for note_id, _ in sorted_results[: len(vector_results)]:
            if note_id in all_notes_map:
                final_notes.append(all_notes_map[note_id])

        return final_notes
