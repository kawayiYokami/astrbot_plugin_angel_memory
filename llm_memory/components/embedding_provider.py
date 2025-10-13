"""
嵌入式模型提供商模块

提供统一的嵌入接口抽象，支持本地模型和API提供商。
通过工厂模式实现智能降级和配置驱动。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import threading
import sys
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

# 尝试导入astrbot logger，如果失败则使用标准库logger
try:
    from astrbot.api import logger
except ImportError:
    import logging as logger_module
    logger = logger_module.getLogger(__name__)
    if not logger.handlers:
        handler = logger_module.StreamHandler()
        formatter = logger_module.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_module.INFO)


class EmbeddingCache:
    """
    向量嵌入缓存类

    使用LRU策略，限制内存占用在指定大小以内。
    适用于并发场景下相同文本的重复查询优化。
    """

    def __init__(self, max_memory_mb: float = 100.0, ttl_minutes: int = 30):
        """
        初始化缓存

        Args:
            max_memory_mb: 最大内存占用（MB），默认100MB
            ttl_minutes: 缓存项过期时间（分钟），默认30分钟
        """
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._current_memory_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._ttl_seconds = ttl_minutes * 60  # 转换为秒
        self._cleanup_threshold = 50  # 每次访问50项时执行一次清理检查

    def _estimate_size(self, text: str, embedding: List[float]) -> int:
        """
        估算缓存项的内存大小

        Args:
            text: 文本字符串
            embedding: 向量

        Returns:
            估算的字节数
        """
        # 文本大小（Python字符串开销约为49字节 + 每个字符）
        text_size = sys.getsizeof(text)
        # 向量大小（list开销 + 每个float 8字节）
        embedding_size = sys.getsizeof(embedding) + len(embedding) * 8
        # 时间戳和元数据开销（约64字节）
        metadata_size = 64
        return text_size + embedding_size + metadata_size

    def _is_expired(self, cache_item: Dict[str, Any]) -> bool:
        """
        检查缓存项是否过期
        
        Args:
            cache_item: 缓存项字典
            
        Returns:
            是否过期
        """
        import time
        return time.time() - cache_item['timestamp'] > self._ttl_seconds

    def _cleanup_expired(self, force: bool = False):
        """
        惰性清理过期缓存项
        
        Args:
            force: 是否强制清理所有过期项
        """
        import time
        if not force and self._hit_count % self._cleanup_threshold != 0:
            return
            
        expired_keys = []
        current_time = time.time()
        
        for key, cache_item in self._cache.items():
            if current_time - cache_item['timestamp'] > self._ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            cache_item = self._cache.pop(key)
            old_size = self._estimate_size(key, cache_item['embedding'])
            self._current_memory_bytes -= old_size
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")

    def get(self, text: str) -> Optional[List[float]]:
        """
        从缓存获取向量

        Args:
            text: 文本

        Returns:
            向量，如果不存在或过期返回None
        """
        with self._lock:
            # 惰性清理过期项（每50次访问检查一次）
            self._cleanup_expired()
            
            if text in self._cache:
                cache_item = self._cache[text]
                
                # 检查是否过期
                if self._is_expired(cache_item):
                    # 删除过期项
                    old_size = self._estimate_size(text, cache_item['embedding'])
                    del self._cache[text]
                    self._current_memory_bytes -= old_size
                    self._miss_count += 1
                    return None
                
                self._hit_count += 1
                # 移到最后（最近使用）
                self._cache.move_to_end(text)
                return cache_item['embedding']
            else:
                self._miss_count += 1
                return None

    def put(self, text: str, embedding: List[float]) -> None:
        """
        将向量放入缓存

        Args:
            text: 文本
            embedding: 向量
        """
        import time
        with self._lock:
            # 如果已存在，先删除旧的
            if text in self._cache:
                cache_item = self._cache[text]
                old_size = self._estimate_size(text, cache_item['embedding'])
                self._current_memory_bytes -= old_size
                del self._cache[text]

            # 计算新项大小
            new_size = self._estimate_size(text, embedding)

            # 如果单个项超过最大内存，直接返回不缓存
            if new_size > self._max_memory_bytes:
                return

            # 淘汰旧项直到有足够空间
            while self._current_memory_bytes + new_size > self._max_memory_bytes and self._cache:
                # 删除最旧的项（FIFO）
                oldest_key, oldest_value = self._cache.popitem(last=False)
                oldest_size = self._estimate_size(oldest_key, oldest_value['embedding'])
                self._current_memory_bytes -= oldest_size

            # 添加新项（包含时间戳）
            self._cache[text] = {
                'embedding': embedding,
                'timestamp': time.time()
            }
            self._current_memory_bytes += new_size

    def get_batch(self, texts: List[str]) -> tuple[List[Optional[List[float]]], List[int]]:
        """
        批量获取向量

        Args:
            texts: 文本列表

        Returns:
            (缓存结果列表, 未命中的索引列表)
            缓存结果中，命中的是向量，未命中的是None
        """
        results = []
        missing_indices = []

        with self._lock:
            for i, text in enumerate(texts):
                # 注意：get方法已经包含TTL检查，在锁内部调用可能导致重复清理
                # 为了性能，直接从_cache检查
                cache_item = self._cache.get(text)
                if cache_item and not self._is_expired(cache_item):
                    results.append(cache_item['embedding'])
                    self._hit_count += 1
                    self._cache.move_to_end(text)
                else:
                    results.append(None)
                    self._miss_count += 1
                    if cache_item and self._is_expired(cache_item):
                        # 清理过期项
                        old_size = self._estimate_size(text, cache_item['embedding'])
                        del self._cache[text]
                        self._current_memory_bytes -= old_size
                    missing_indices.append(i)

        return results, missing_indices

    def put_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        批量放入缓存

        Args:
            texts: 文本列表
            embeddings: 向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
            self._hit_count = 0
            self._miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

            # 计算过期项数量
        import time
        current_time = time.time()
        expired_count = sum(1 for item in self._cache.values() 
                          if current_time - item['timestamp'] > self._ttl_seconds)

        return {
            "cache_size": len(self._cache),
            "expired_items": expired_count,
            "memory_usage_mb": self._current_memory_bytes / (1024 * 1024),
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "ttl_minutes": self._ttl_seconds / 60
        }


class EmbeddingProvider(ABC):
    """嵌入提供商抽象基类"""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成向量嵌入

        Args:
            texts: 文档字符串列表

        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型详细信息的字典
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查提供商是否可用

        Returns:
            是否可用
        """
        pass

    @abstractmethod
    def get_provider_type(self) -> str:
        """
        获取提供商类型

        Returns:
            提供商类型标识符
        """
        pass


    @abstractmethod
    def shutdown(self):
        """关闭提供商，释放资源"""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地嵌入模型提供商"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", cache_size_mb: float = 100.0):
        """
        初始化本地嵌入提供商

        Args:
            model_name: 本地模型名称
            cache_size_mb: 缓存大小（MB），默认100MB
        """
        self.model_name = model_name
        self.logger = logger
        self._model: Optional[SentenceTransformer] = None
        self._cache = EmbeddingCache(max_memory_mb=cache_size_mb)
        self._load_model()

    def _load_model(self):
        """加载本地模型"""
        try:
            self.logger.info(f"正在加载本地嵌入模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.logger.info(f"本地嵌入模型加载完成: {self.model_name}")
        except Exception as e:
            self.logger.error(f"本地嵌入模型加载失败: {e}")
            self._model = None

    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """同步方法：为文档列表生成向量嵌入（带缓存）"""
        if not texts:
            return []

        if not self.is_available():
            raise RuntimeError("本地模型不可用")

        # 1. 尝试从缓存获取
        cached_results, missing_indices = self._cache.get_batch(texts)

        # 2. 如果全部命中缓存，直接返回
        if not missing_indices:
            self.logger.debug(f"✅ 缓存全部命中，跳过向量化: {len(texts)}个文本")
            return [r for r in cached_results if r is not None]

        # 3. 对未命中的文本进行向量化
        missing_texts = [texts[i] for i in missing_indices]
        self.logger.debug(f"🔄 缓存部分命中，需要向量化: {len(missing_texts)}/{len(texts)}个文本")

        # 直接同步调用（本地模型）
        new_embeddings = self._model.encode(missing_texts, convert_to_numpy=True)
        new_embeddings_list = new_embeddings.tolist()

        # 4. 将新向量存入缓存
        self._cache.put_batch(missing_texts, new_embeddings_list)

        # 5. 合并结果：组装完整的嵌入列表
        result = []
        new_embedding_iter = iter(new_embeddings_list)
        for cached in cached_results:
            if cached is not None:
                result.append(cached)
            else:
                result.append(next(new_embedding_iter))

        return result

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步方法：为文档列表生成向量嵌入（保持兼容性）"""
        # 在线程池中执行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents_sync, texts)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_available():
            return {
                "model_name": self.model_name,
                "provider_type": "local",
                "status": "unavailable",
                "dimension": 0
            }

        return {
            "model_name": self.model_name,
            "provider_type": "local",
            "status": "available",
            "dimension": self._model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(self._model, 'max_seq_length', None)
        }

    def is_available(self) -> bool:
        """检查提供商是否可用"""
        return self._model is not None

    def get_provider_type(self) -> str:
        """获取提供商类型"""
        return "local"

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self.logger.info("本地提供商缓存已清空")


    def shutdown(self):
        """关闭本地提供商"""
        self.clear_cache()
        self.logger.info(f"本地嵌入提供商 {self.model_name} 已关闭")


class APIEmbeddingProvider(EmbeddingProvider):
    """API嵌入模型提供商"""

    def __init__(self, provider, provider_id: str, cache_size_mb: float = 100.0):
        """
        初始化API嵌入提供商

        Args:
            provider: AstrBot提供商对象
            provider_id: 提供商ID
            cache_size_mb: 缓存大小（MB），默认100MB
        """
        self.provider = provider
        self.provider_id = provider_id
        self.logger = logger
        self._model_info = None
        self._available = None  # None表示未测试，True/False表示测试结果
        self._cache = EmbeddingCache(max_memory_mb=cache_size_mb)
        self.batch_size = 64  # 程序启动时的批量大小，遇到413会减半

        # 创建共享的线程池和事件循环（只创建一次，避免overhead）
        import concurrent.futures
        import threading
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="embed-api")

        # 创建一个专门的线程运行共享event loop
        self._shared_loop = None
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True, name="embed-loop")
        self._loop_thread.start()

        # 等待loop就绪
        import time
        while self._shared_loop is None:
            time.sleep(0.001)

        # 延迟测试可用性，避免在构造函数中进行异步操作
        self.logger.info(f"API嵌入提供商已初始化: {self.provider_id}, 批量大小: {self.batch_size}, 共享loop已启动")

    async def check_availability(self) -> bool:
        """异步检查可用性"""
        if self._available is not None:
            return self._available

        try:
            # 尝试获取一个简单的嵌入来测试可用性
            test_text = "test"
            await self._perform_embedding_test(test_text)
            self._available = True
            self.logger.info(f"API嵌入提供商可用: {self.provider_id}")
            return True
        except Exception as e:
            self.logger.warning(f"API嵌入提供商不可用 {self.provider_id}: {e}")
            self._available = False
            return False

    async def _perform_embedding_test(self, text: str):
        """执行嵌入测试"""
        try:
            await self.provider.get_embedding(text)
        except Exception as e:
            raise e

    def _run_event_loop(self):
        """在专门的线程中运行共享event loop"""
        self._shared_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._shared_loop)
        self._shared_loop.run_forever()

    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """同步方法：为文档列表生成向量嵌入（使用共享event loop）"""
        if not texts:
            return []

        # 在共享event loop中运行异步代码
        future = asyncio.run_coroutine_threadsafe(
            self.embed_documents(texts),
            self._shared_loop
        )
        return future.result()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步方法：为文档列表生成向量嵌入（带缓存，批次内去重，自动分批）"""
        if not texts:
            return []

        if not self.is_available():
            raise RuntimeError(f"API提供商不可用: {self.provider_id}")

        # 1. 尝试从缓存获取
        cached_results, missing_indices = self._cache.get_batch(texts)

        # 2. 如果全部命中缓存，直接返回
        if not missing_indices:
            return [r for r in cached_results if r is not None]

        # 3. 获取未命中的文本
        missing_texts = [texts[i] for i in missing_indices]

        # 4. 批次内去重
        unique_texts, original_to_unique, unique_to_original = self._deduplicate_texts(missing_texts)
        len(missing_texts) - len(unique_texts)

        # 5. 使用当前批量大小处理去重后的文本
        unique_embeddings = await self._get_embeddings_with_batch(unique_texts, self.batch_size)

        # 6. 将去重后的向量回填到原始位置
        full_missing_embeddings = self._map_embeddings_back(unique_embeddings, unique_to_original, len(missing_texts))

        # 7. 将去重后的向量存入缓存
        self._cache.put_batch(unique_texts, unique_embeddings)

        # 8. 合并结果：组装完整的嵌入列表
        result = []
        new_embedding_iter = iter(full_missing_embeddings)
        for cached in cached_results:
            if cached is not None:
                result.append(cached)
            else:
                result.append(next(new_embedding_iter))

        return result

    def _deduplicate_texts(self, texts: List[str]) -> tuple:
        """
        文本去重，保持映射关系

        Args:
            texts: 待去重的文本列表

        Returns:
            tuple: (unique_texts, original_to_unique, unique_to_original)
                - unique_texts: 去重后的文本列表
                - original_to_unique: 原始索引到唯一索引的映射
                - unique_to_original: 唯一文本对应的原始索引列表
        """
        unique_texts = []
        original_to_unique = {}  # 原始索引 -> 唯一索引
        unique_to_original = []  # 唯一索引 -> [原始索引列表]

        for original_idx, text in enumerate(texts):
            if text not in unique_texts:
                # 新的唯一文本
                unique_idx = len(unique_texts)
                unique_texts.append(text)
                original_to_unique[original_idx] = unique_idx
                unique_to_original.append([original_idx])
            else:
                # 重复文本，找到对应的唯一索引
                unique_idx = unique_texts.index(text)
                original_to_unique[original_idx] = unique_idx
                unique_to_original[unique_idx].append(original_idx)

        return unique_texts, original_to_unique, unique_to_original

    def _map_embeddings_back(self, unique_embeddings: List[List[float]],
                             unique_to_original: List[List[int]],
                             original_count: int) -> List[List[float]]:
        """
        将去重后的向量回填到原始位置

        Args:
            unique_embeddings: 去重文本的向量列表
            unique_to_original: 唯一文本对应的原始索引列表
            original_count: 原始文本数量

        Returns:
            List[List[float]]: 回填到原始位置的向量列表
        """
        full_embeddings = [None] * original_count

        for unique_idx, original_indices in enumerate(unique_to_original):
            embedding = unique_embeddings[unique_idx]
            for original_idx in original_indices:
                full_embeddings[original_idx] = embedding

        return full_embeddings

    async def _get_embeddings_with_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """使用指定批量大小并发获取向量嵌入"""

        if batch_size >= len(texts):
            # 单批次处理
            result = await self.provider.get_embeddings(texts)
            return result
        else:
            # 分批并发处理
            tasks = []
            batch_info = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                task = self.provider.get_embeddings(batch)
                tasks.append(task)
                batch_info.append(len(batch))

            # 并发执行所有批次
            batch_results = await asyncio.gather(*tasks)

            # 按顺序拼接结果
            result = []
            for batch_embeddings in batch_results:
                result.extend(batch_embeddings)

            return result

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self._model_info is None:
            meta = self.provider.meta() if hasattr(self.provider, 'meta') else {}
            self._model_info = {
                "provider_id": self.provider_id,
                "provider_type": "api",
                "status": "available" if self._available else "unavailable",
                "meta": meta
            }

        return self._model_info

    def is_available(self) -> bool:
        """检查提供商是否可用"""
        return self._available is True

    def get_provider_type(self) -> str:
        """获取提供商类型"""
        return "api"

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = self._cache.get_stats()
        stats['current_batch_size'] = self.batch_size  # 添加当前批量大小信息
        return stats

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self.logger.info(f"API提供商 {self.provider_id} 缓存已清空")

    def shutdown(self):
        """关闭API提供商，释放资源"""
        self.logger.info(f"正在关闭API嵌入提供商: {self.provider_id}")

        # 关闭共享事件循环
        if self._shared_loop and self._shared_loop.is_running():
            self._shared_loop.call_soon_threadsafe(self._shared_loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
            if self._loop_thread.is_alive():
                self.logger.warning(f"事件循环线程未能及时关闭: {self._loop_thread.name}")

        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=True)

        self.clear_cache()
        self.logger.info(f"API嵌入提供商 {self.provider_id} 已成功关闭")



class EmbeddingProviderFactory:
    """嵌入提供商工厂"""

    def __init__(self, context=None):
        """
        初始化工厂

        Args:
            context: AstrBot上下文，用于获取提供商
        """
        self.context = context
        self.logger = logger

    async def create_provider(
        self,
        provider_id: Optional[str] = None,
        local_model_name: str = "BAAI/bge-small-zh-v1.5"
    ) -> EmbeddingProvider:
        """
        创建嵌入提供商

        Args:
            provider_id: API提供商ID，如果为空则使用本地模型
            local_model_name: 本地模型名称

        Returns:
            嵌入提供商实例
        """
        # 如果没有提供商ID，直接使用本地模型
        if not provider_id:
            self.logger.info("未指定API提供商，使用本地嵌入模型")
            return LocalEmbeddingProvider(local_model_name)

        # 尝试使用API提供商
        if self.context:
            try:
                provider = self.context.get_provider_by_id(provider_id)
                if provider:
                    api_provider = APIEmbeddingProvider(provider, provider_id)

                    # 异步测试API可用性
                    if await api_provider.check_availability():
                        self.logger.info(f"成功使用API嵌入提供商: {provider_id}")
                        return api_provider
                    else:
                        self.logger.warning(f"API提供商不可用，降级到本地模型: {provider_id}")
                        return LocalEmbeddingProvider(local_model_name)
                else:
                    self.logger.warning(f"未找到API提供商，降级到本地模型: {provider_id}")
                    return LocalEmbeddingProvider(local_model_name)
            except Exception as e:
                self.logger.error(f"API提供商测试失败，降级到本地模型: {provider_id}, 错误: {e}")
                return LocalEmbeddingProvider(local_model_name)
        else:
            self.logger.warning("无上下文信息，使用本地嵌入模型")
            return LocalEmbeddingProvider(local_model_name)

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的提供商信息

        Returns:
            可用提供商信息列表
        """
        providers = []

        # 添加本地模型信息
        local_provider = LocalEmbeddingProvider()
        providers.append(local_provider.get_model_info())

        # 添加API提供商信息
        if self.context and hasattr(self.context, 'get_all_embedding_providers'):
            try:
                api_providers = self.context.get_all_embedding_providers()
                for provider in api_providers:
                    try:
                        meta = provider.meta() if hasattr(provider, 'meta') else {}
                        provider_info = {
                            "provider_id": meta.get('id', 'unknown'),
                            "provider_type": "api",
                            "status": "available",
                            "meta": meta
                        }
                        providers.append(provider_info)
                    except Exception as e:
                        self.logger.warning(f"获取API提供商信息失败: {e}")
            except Exception as e:
                self.logger.warning(f"获取API提供商列表失败: {e}")

        return providers