"""
ComponentFactory - 组件工厂

负责统一管理所有核心组件的创建，确保在主线程中创建实例，
避免后台线程和主线程之间的实例不一致问题。
"""

from typing import Dict, Any, Optional
from pathlib import Path

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# 导入核心组件
from ..llm_memory.components.embedding_provider import EmbeddingProviderFactory
from ..llm_memory.components.memory_sql_manager import MemorySqlManager
from ..llm_memory.components.vector_store import VectorStore
from ..llm_memory import CognitiveService
from ..llm_memory.service.note_service import NoteService
from .memory_runtime import SimpleMemoryRuntime, VectorMemoryRuntime
from .deepmind import DeepMind


class ComponentFactory:
    """组件工厂类 - 统一管理所有核心组件的创建"""

    def __init__(self, plugin_context, init_manager=None):
        """
        初始化组件工厂

        Args:
            plugin_context: PluginContext插件上下文（包含所有必要资源）
            init_manager: 初始化管理器（用于标记系统就绪）
        """
        self.plugin_context = plugin_context
        self.context = plugin_context.get_astrbot_context()  # 保持向后兼容
        self.logger = logger
        self._components: Dict[str, Any] = {}
        self._initialized = False
        self.init_manager = init_manager

        # 从PluginContext获取数据目录
        self.data_directory = str(plugin_context.get_index_dir())

        self.logger.info("🏭 ComponentFactory初始化完成")
        self.logger.info(f"   当前提供商: {plugin_context.get_current_provider()}")
        self.logger.info(f"   数据目录: {self.data_directory}")
        self.logger.info(f"   有可用提供商: {plugin_context.has_providers()}")

    async def create_all_components(self, config: dict = None) -> Dict[str, Any]:
        """
        异步创建所有核心组件

        Args:
            config: 插件配置（可选，如果不提供则从PluginContext获取）

        Returns:
            包含所有组件的字典
        """
        if self._initialized:
            return self._components

        # 如果没有提供配置，从PluginContext获取
        if config is None:
            config = self.plugin_context.get_all_config()

        try:
            self.logger.info("🏭 开始创建核心组件...")
            enable_simple_memory = bool(config.get("enable_simple_memory", False))
            memory_sql_manager = self._create_memory_sql_manager()
            self._components["memory_sql_manager"] = memory_sql_manager

            if enable_simple_memory:
                self.logger.info("🧩 检测到 enable_simple_memory=true，使用 SimpleMemoryRuntime")
                memory_runtime = self._create_memory_runtime(
                    cognitive_service=None,
                    memory_sql_manager=memory_sql_manager,
                )
                self._components["memory_runtime"] = memory_runtime

                deepmind = await self._create_deepmind(
                    vector_store=None,
                    note_service=None,
                    memory_runtime=memory_runtime,
                )
                self._components["deepmind"] = deepmind

                self._initialized = True
                self.logger.info("✅ 所有核心组件创建完成")
                self.logger.info("✅ 记忆运行时: SimpleMemoryRuntime")

                if self.init_manager:
                    self.init_manager.mark_ready()
                    self.logger.info("🎉 系统准备就绪！可以开始处理业务请求")

                return self._components

            # 1. 创建嵌入提供商
            embedding_provider = await self._create_embedding_provider()
            self._components["embedding_provider"] = embedding_provider
            self.plugin_context.set_embedding_provider(embedding_provider)

            # API提供商必须在启动期可用；本地提供商允许懒加载
            provider_type = embedding_provider.get_provider_type()
            if provider_type != "local" and not embedding_provider.is_available():
                self.logger.critical(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                self.logger.critical("!!! 核心组件 embedding_provider 加载失败！")
                self.logger.critical(
                    "!!! 当前为上游嵌入提供商模式：提供商不可用、配置错误或凭证异常。"
                )
                self.logger.critical(
                    "!!! 这不是本地模型安装问题。若需本地兜底，请启用 enable_local_embedding。"
                )
                self.logger.critical(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                # 标记初始化完成（以受限模式）并立即返回，不再创建后续组件
                self._initialized = True
                if self.init_manager:
                    self.init_manager.mark_ready()  # 同样需要标记，否则主程序可能卡住
                return self._components
            elif provider_type == "local" and not embedding_provider.is_available():
                self.logger.info(
                    "本地嵌入模型采用懒加载模式：将在首次向量化请求时加载。"
                )

            # 2. 创建向量存储 (只有在 embedding_provider 可用时才会执行)
            vector_store = self._create_vector_store(embedding_provider)
            self._components["vector_store"] = vector_store
            self.plugin_context.set_vector_store(vector_store)

            # 3. 创建认知服务
            cognitive_service = self._create_cognitive_service(vector_store, memory_sql_manager)
            self._components["cognitive_service"] = cognitive_service

            # 4. 创建统一记忆运行时（Phase A: 向量实现）
            memory_runtime = self._create_memory_runtime(
                cognitive_service,
                memory_sql_manager=memory_sql_manager,
            )
            self._components["memory_runtime"] = memory_runtime

            # 5. 创建笔记服务
            note_service = self._create_note_service(vector_store)
            self._components["note_service"] = note_service

            # 6. 创建DeepMind
            deepmind = await self._create_deepmind(
                vector_store, note_service, memory_runtime
            )
            self._components["deepmind"] = deepmind

            # 7. 创建文件监控
            file_monitor = self._create_file_monitor(note_service)
            self._components["file_monitor"] = file_monitor

            # 核心组件已就绪，立即标记初始化完成
            self._initialized = True
            self.logger.info("✅ 所有核心组件创建完成")
            self.logger.info("✅ 记忆运行时: VectorMemoryRuntime")

            # 如果有初始化管理器，立即标记系统准备就绪
            # 此时"电脑已开机"，用户可以开始使用，不需要等待"硬盘整理"（文件监控）
            if self.init_manager:
                self.init_manager.mark_ready()
                self.logger.info("🎉 系统准备就绪！可以开始处理业务请求")

            # 异步启动文件监控（在后台继续运行）
            await self._start_file_monitor(file_monitor)

            return self._components

        except Exception as e:
            self.logger.error(f"❌ 组件创建失败: {e}")
            import traceback

            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    async def _create_embedding_provider(self):
        """创建嵌入提供商"""
        self.logger.info("📚 创建嵌入提供商...")

        embedding_provider_id = self.plugin_context.get_embedding_provider_id()
        self.logger.info(f"🔧 配置的嵌入式模型提供商ID: '{embedding_provider_id}'")

        factory = EmbeddingProviderFactory(self.context)
        embedding_provider = await factory.create_provider(
            embedding_provider_id,
            enable_local_embedding=self.plugin_context.get_enable_local_embedding(),
        )

        provider_info = embedding_provider.get_model_info()
        self.logger.info(f"✅ 嵌入提供商创建完成: {provider_info}")

        return embedding_provider

    def _create_vector_store(self, embedding_provider):
        """创建向量存储"""
        self.logger.info("🗄️ 创建向量存储...")

        # 使用PluginContext的ChromaDB路径
        db_path = str(self.plugin_context.get_chroma_db_path())
        self.logger.info(f"📁 使用数据库路径: {db_path}")

        rerank_provider = self._resolve_rerank_provider()

        vector_store = VectorStore(
            embedding_provider=embedding_provider,
            db_path=db_path,
            rerank_provider=rerank_provider,
        )

        # 获取提供商类型用于日志
        provider_type = embedding_provider.get_provider_type()
        provider_info = embedding_provider.get_model_info()

        if provider_type == "api":
            provider_id = provider_info.get("provider_id", "unknown")
            self.logger.info(f"✅ 向量存储创建完成 (使用API提供商: {provider_id})")
        else:
            model_name = provider_info.get("model_name", "unknown")
            self.logger.info(f"✅ 向量存储创建完成 (使用本地模型: {model_name})")

        return vector_store

    def _resolve_rerank_provider(self) -> Optional[Any]:
        """
        解析上游重排提供商（可选）。

        优先级：
        1. 配置项 rerank_provider_id（如果存在）
        2. 配置项 provider_id（LLM 提供商ID）
        3. 上游 context 中第一个具备 rerank() 方法的提供商
        """
        try:
            rerank_provider_id = self.plugin_context.get_config("rerank_provider_id", None)
            llm_provider_id = self.plugin_context.get_llm_provider_id()
            candidate_ids = [
                pid
                for pid in [
                    rerank_provider_id,
                    llm_provider_id,
                ]
                if pid
            ]

            # 先按显式 ID 查找
            for pid in candidate_ids:
                if hasattr(self.context, "get_rerank_provider_by_id"):
                    provider = self.context.get_rerank_provider_by_id(pid)
                    if provider and hasattr(provider, "rerank"):
                        self.logger.info(f"✅ 使用上游重排提供商: {pid}")
                        return provider

                if hasattr(self.context, "get_provider_by_id"):
                    provider = self.context.get_provider_by_id(pid)
                    if provider and hasattr(provider, "rerank"):
                        self.logger.info(f"✅ 使用上游可重排提供商: {pid}")
                        return provider

            # 再从列表中兜底挑选
            if hasattr(self.context, "get_all_rerank_providers"):
                providers = self.context.get_all_rerank_providers() or []
                for p in providers:
                    if hasattr(p, "rerank"):
                        self.logger.info("✅ 使用上游重排提供商: <auto>")
                        return p

            if hasattr(self.context, "get_all_providers"):
                providers = self.context.get_all_providers() or []
                for p in providers:
                    if hasattr(p, "rerank"):
                        self.logger.info("✅ 使用上游可重排提供商: <auto>")
                        return p

            self.logger.info("ℹ️ 未启用记忆二阶段重排，当前使用 Chroma 向量相似度排序结果")
            return None
        except Exception as e:
            self.logger.warning(f"解析上游重排提供商失败，自动降级为 Chroma 向量相似度排序: {e}")
            return None

    def _create_cognitive_service(self, vector_store, memory_sql_manager: MemorySqlManager = None):
        """创建认知服务"""
        self.logger.info("🧠 创建认知服务...")

        cognitive_service = CognitiveService(
            vector_store=vector_store,
            memory_sql_manager=memory_sql_manager,
        )
        self.logger.info("✅ 认知服务创建完成")

        return cognitive_service

    def _create_memory_sql_manager(self) -> MemorySqlManager:
        """创建 SQL 记忆管理器（两种运行时共用）。"""
        simple_db_path = self.plugin_context.get_simple_memory_db_path()
        manager = MemorySqlManager(simple_db_path)
        self.logger.info(f"✅ SQL记忆管理器创建完成: {simple_db_path}")
        return manager

    def _create_memory_runtime(self, cognitive_service, memory_sql_manager: MemorySqlManager):
        """创建统一记忆运行时。"""
        self.logger.info("🧩 创建统一记忆运行时...")

        if self.plugin_context.get_config("enable_simple_memory", False):
            runtime = SimpleMemoryRuntime(memory_sql_manager)
            self.logger.info("✅ 统一记忆运行时创建完成 (SimpleMemoryRuntime)")
            return runtime

        if cognitive_service is None:
            raise RuntimeError("向量模式下创建 memory_runtime 失败：cognitive_service 不可用。")

        runtime = VectorMemoryRuntime(cognitive_service)
        self.logger.info("✅ 统一记忆运行时创建完成 (VectorMemoryRuntime)")
        return runtime

    def _create_note_service(self, vector_store):
        """创建笔记服务"""
        self.logger.info("📝 创建笔记服务...")

        # 使用PluginContext模式创建NoteService
        note_service = NoteService.from_plugin_context(self.plugin_context)
        # 设置VectorStore
        note_service.set_vector_store(vector_store)

        self.logger.info("✅ 笔记服务创建完成")

        return note_service

    async def _create_deepmind(self, vector_store, note_service, memory_runtime):
        """创建DeepMind"""
        self.logger.info("🤖 创建DeepMind...")

        # 从PluginContext获取LLM提供商ID
        llm_provider_id = self.plugin_context.get_llm_provider_id()

        # 创建配置对象
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

        config_obj = Config(self.plugin_context.get_all_config())

        deepmind = DeepMind(
            config=config_obj,
            context=self.context,
            vector_store=vector_store,
            note_service=note_service,
            plugin_context=self.plugin_context, # 传递plugin_context
            provider_id=llm_provider_id,
            memory_runtime=memory_runtime,  # 使用统一记忆运行时
        )

        self.logger.info("✅ DeepMind创建完成")
        return deepmind

    def _create_file_monitor(self, note_service):
        """创建文件监控"""
        self.logger.info("📂 创建文件监控组件...")

        # 导入相关模块
        from ..core.file_monitor import FileMonitorService

        # 使用PathManager获取正确的索引目录
        data_directory = str(self.plugin_context.get_index_dir())

        self.logger.info(f"📁 使用数据目录: {data_directory}")
        self.logger.info(f"📁 当前工作目录: {Path.cwd()}")

        # 创建文件监控服务
        file_monitor = FileMonitorService(
            data_directory=data_directory,
            note_service=note_service,  # 传入已创建的note_service实例
            config=self.plugin_context.config,  # 传入配置
        )

        self.logger.info(
            f"✅ 文件监控组件创建完成 (提供商: {self.plugin_context.get_current_provider()})"
        )
        return file_monitor

    async def _start_file_monitor(self, file_monitor):
        """启动文件监控服务（内部同步执行）"""
        try:
            # 直接调用同步方法（在线程池中执行，避免阻塞event loop）
            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, file_monitor.start_monitoring)
            self.logger.info("📂 文件监控服务已启动")

        except Exception as e:
            self.logger.error(f"启动文件监控服务失败: {e}")
            # 文件监控失败不应该中断整个初始化流程

    def get_components(self) -> Dict[str, Any]:
        """获取已创建的组件"""
        return self._components.copy()

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def reset(self):
        """重置工厂状态（用于测试）"""
        self._components.clear()
        self._initialized = False

    def shutdown(self):
        """关闭所有组件，释放资源"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        self.logger.info("🏭 开始关闭所有核心组件...")

        # 按创建顺序的逆序关闭组件
        component_shutdown_order = [
            "file_monitor",
            "deepmind",
            "memory_runtime",
            "memory_sql_manager",
            "note_service",
            "cognitive_service",
            "vector_store",
            "embedding_provider",
        ]

        def _try_shutdown_component(component_name: str, component: Any) -> None:
            shutdown_method = None
            for method_name in ("shutdown", "close", "stop", "dispose"):
                if hasattr(component, method_name):
                    shutdown_method = getattr(component, method_name)
                    break

            if shutdown_method is None:
                return

            try:
                self.logger.info(f"正在关闭组件: {component_name}...")
                result = shutdown_method()
                if hasattr(result, "__await__"):
                    try:
                        asyncio.get_running_loop()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(asyncio.run, result).result()
                    except RuntimeError:
                        asyncio.run(result)
                self.logger.info(f"✅ 组件 {component_name} 已关闭")
            except Exception as e:
                self.logger.error(f"❌ 关闭组件 {component_name} 失败: {e}")

        for component_name in component_shutdown_order:
            component = self._components.get(component_name)
            if component:
                _try_shutdown_component(component_name, component)
                self._components.pop(component_name, None)

        self._components.clear()
        self._initialized = False
        self.logger.info("✅ 所有核心组件已成功关闭")
