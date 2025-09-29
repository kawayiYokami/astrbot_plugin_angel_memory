"""
向量存储组件。

封装所有与ChromaDB向量数据库和嵌入模型的交互。
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import threading

import time
from ..models.data_models import BaseMemory
from ..config.system_config import system_config
from ...core.logger import get_logger


class VectorStore:
    """
    向量存储类。

    负责记忆的向量化和存储，使用ChromaDB作为后端。
    实现为单例模式，确保全局只加载一次嵌入模型。
    """

    # 类变量，用于存储单例实例和共享资源
    _instance = None
    _embedding_model = None
    _client = None
    _logger = None
    _initialized = False
    _init_lock = threading.Lock()  # 添加线程锁

    def __new__(cls, *args, **kwargs):
        """
        重写 __new__ 方法以实现线程安全的单例模式。
        """
        if cls._instance is None:
            with cls._init_lock:  # 使用锁确保线程安全
                if cls._instance is None:  # 双重检查
                    cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None, collection_name: str = "memory_collection"):
        """
        初始化向量存储。

        Args:
            model_name: 嵌入模型名称
            collection_name: ChromaDB集合名称
        """
        # 使用锁确保线程安全的初始化检查
        with self._init_lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

        # 设置日志记录器
        self.logger = get_logger()

        self.model_name = model_name or system_config.embedding_model

        # 单例模式：只加载一次嵌入模型和客户端
        if VectorStore._embedding_model is None:
            self.logger.info(f"正在加载嵌入模型: {self.model_name}")
            VectorStore._embedding_model = SentenceTransformer(self.model_name)
        if VectorStore._client is None:
            VectorStore._client = chromadb.PersistentClient(path=str(system_config.get_database_path()))

        self.embedding_model = VectorStore._embedding_model
        self.client = VectorStore._client

        # 为每个实例创建或获取其专属的集合
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self._initialized = True

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

    def remember(self, memory: BaseMemory):
        """
        记住一条新记忆。

        Args:
            memory: 要记住的记忆对象（任何 BaseMemory 的子类）
        """
        try:
            # 获取用于向量化的语义核心文本
            semantic_core = memory.get_semantic_core()

            # 对语义核心进行向量化
            embedding = self.embedding_model.encode(semantic_core).tolist()

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

            # 将记忆存入ChromaDB
            self.collection.upsert(
                ids=[memory.id],
                embeddings=[embedding],
                metadatas=[memory.to_dict()],  # 将整个记忆对象作为元数据存储
                documents=[content_text]
            )

        except Exception as e:
            # 异常会被装饰器自动记录
            raise  # 重新抛出异常

    def recall(self, query: str, limit: int = 10, where_filter: Optional[dict] = None) -> List[BaseMemory]:
        """
        根据查询回忆相关记忆，支持复杂的元数据过滤。

        Args:
            query: 搜索查询字符串
            limit: 返回结果的最大数量
            where_filter: 可选的元数据过滤器字典 (e.g., {"memory_type": "EventMemory", "is_consolidated": False})

        Returns:
            相关的记忆对象列表（BaseMemory 的子类）
        """
        try:
            # 将查询编码为向量
            query_embedding = self.embedding_model.encode(query).tolist()

            # 构建查询参数
            query_params = {
                "query_embeddings": [query_embedding],
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
            results = self.collection.query(**query_params)

            # 将结果转换为记忆对象
            recalled_memories = []
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                for meta in results['metadatas'][0]:
                    if meta:  # 确保元数据不为空
                        recalled_memories.append(BaseMemory.from_dict(meta))

        except Exception as e:
            # 异常会被装饰器自动记录
            raise  # 重新抛出异常

        return recalled_memories

    def update_memory(self, memory_id: str, updates: dict):
        """
        更新记忆的元数据。

        Args:
            memory_id: 要更新的记忆ID
            updates: 要更新的字段字典 (e.g., {"is_consolidated": True, "strength": 5})
        """
        try:
            # 获取当前记忆的完整信息
            current_data = self.collection.get(ids=[memory_id])
            if not current_data or not current_data['metadatas']:
                raise ValueError(f"Memory with id {memory_id} not found")

            current_meta = current_data['metadatas'][0]
            current_document = current_data['documents'][0] if current_data['documents'] else ""

            # 获取当前的嵌入（如果可用）
            if current_data['embeddings'] and len(current_data['embeddings']) > 0:
                current_embedding = current_data['embeddings'][0]
            else:
                # 如果没有嵌入，我们需要重新生成一个
                semantic_core = current_meta.get('memory_type', '') + ' ' + str(current_meta)
                current_embedding = self.embedding_model.encode(semantic_core).tolist()

            # 应用更新
            current_meta.update(updates)

            # 重新存储（ChromaDB的upsert会更新现有条目）
            self.collection.upsert(
                ids=[memory_id],
                embeddings=[current_embedding],  # 保持原有的嵌入
                metadatas=[current_meta],
                documents=[current_document]  # 保持原有的文档
            )

        except Exception as e:
            self.logger.error(f"更新记忆 {memory_id} 失败: {str(e)}")
            raise

    def delete_memories(self, where_filter: dict, exclude_associations: bool = False):
        """
        根据条件删除记忆。

        Args:
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
            results = self.collection.get(where=base_filter)
            ids_to_delete = results['ids']

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self.logger.info(f"Deleted {len(ids_to_delete)} memories with filter: {where_filter}")

        except Exception as e:
            self.logger.error(f"Failed to delete memories: {str(e)}")
            raise

    def clear_all(self):
        """清空所有记忆。"""
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)