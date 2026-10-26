"""
ComponentFactory - 组件工厂

负责统一管理所有核心组件的创建，确保在主线程中创建实例，
避免后台线程和主线程之间的实例不一致问题。

存储与检索由 p-memory 承担：本层只负责配置数据目录、注册提供商回调，
并把记忆运行时、笔记索引服务、DeepMind 组装起来。
"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from ..llm_memory.components.embedding_provider import EmbeddingProviderFactory
from ..llm_memory.components.memory_sql_manager import MemorySqlManager
from ..llm_memory.service.memory_decay_policy import (
    MemoryDecayConfig,
    build_decay_config,
)
from ..pmemory import MemoryStore, pmemory
from .deepmind import DeepMind
from .memory_runtime import PMemoryMemoryRuntime
from .services.legacy_memory_migration import LegacyMemoryMigration
from .services.note_index_service import NoteIndexService

# 重排回调的总 token 预算，与 story 侧一致
RERANK_MAX_TOKENS_TOTAL = 8192


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

        self.logger.debug("ComponentFactory初始化完成")
        self.logger.debug(f"当前提供商: {plugin_context.get_current_provider()}")

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
            self.logger.debug("开始创建核心组件...")
            decay_config = build_decay_config(config)
            rerank_provider = self._resolve_rerank_provider()
            rerank_max_docs = self._resolve_rerank_max_docs()
            rerank_max_tokens = self._resolve_rerank_max_tokens()

            # 账本（用户/群 first_seen/last_seen、标签），记忆本体不在 SQL 里
            memory_sql_manager = self._create_memory_sql_manager()
            self._components["memory_sql_manager"] = memory_sql_manager

            # 1. 嵌入提供商（可选，失败则记忆检索走纯全文）
            embedding_provider = await self._try_create_embedding_provider()

            # 2. 配置 p-memory 单例
            self._configure_pmemory(
                embedding_provider=embedding_provider,
                rerank_provider=rerank_provider,
                rerank_max_docs=rerank_max_docs,
                rerank_max_tokens=rerank_max_tokens,
            )

            # 3. 记忆运行时
            memory_store = MemoryStore()
            self._components["memory_store"] = memory_store
            memory_runtime = self._create_memory_runtime(memory_store, decay_config)
            self._components["memory_runtime"] = memory_runtime

            # 4. 笔记索引服务
            note_index = self._create_note_index_service()
            self._components["note_index"] = note_index

            # 5. 潜意识
            deepmind = await self._create_deepmind(note_index, memory_runtime)
            self._components["deepmind"] = deepmind

            # 6. 旧记忆自动迁移（后台补齐，用户无需操作）
            memory_center = Path(self.plugin_context.get_memory_center_dir())
            migration = LegacyMemoryMigration(
                memory_store=memory_store,
                memory_sql_manager=memory_sql_manager,
                state_dir=memory_center,
                backup_dir=memory_center / "backups",
                logger=self.logger,
            )
            self._components["legacy_migration"] = migration

            # 核心组件已就绪，立即标记初始化完成
            self._initialized = True
            self.logger.debug("所有核心组件创建完成")
            self.logger.debug(f"记忆运行时: p-memory 目录={pmemory.root}")

            # 如果有初始化管理器，立即标记系统准备就绪
            if self.init_manager:
                self.init_manager.mark_ready()

            # 异步启动笔记扫描（后台首次全量导入）
            migration.start()
            note_index.start()

            return self._components

        except Exception as e:
            self.logger.error(f"组件创建失败: {e}", exc_info=True)
            raise

    # === 配置解析 ===

    def _resolve_rerank_max_docs(self) -> int:
        try:
            return int(self.plugin_context.get_rerank_max_docs())
        except Exception:
            return 64

    def _resolve_rerank_max_tokens(self) -> int:
        try:
            return int(self.plugin_context.get_rerank_max_tokens_per_doc())
        except Exception:
            return 1024

    def _resolve_raw_dir(self) -> Optional[Path]:
        try:
            return self.plugin_context.get_path_manager().get_raw_dir()
        except Exception as e:
            self.logger.warning(f"无法解析 raw 目录，笔记导入将跳过: {e}")
            return None

    @staticmethod
    def _build_decay_policy(decay_config: MemoryDecayConfig) -> Dict[str, Any]:
        """把旧三档参数翻译成 p-memory 的 DecayPolicy。

        旧实现用 `cycle_tier0_days / (forget_speed * tier0_forget_speed)` 折算周期，
        库侧只接受整数天数，这里沿用同一口径把速度合进周期。
        """
        speed = max(0.01, float(decay_config.forget_speed)) * max(
            0.01, float(decay_config.tier0_forget_speed)
        )
        cycle_days = max(1, int(round(float(decay_config.cycle_tier0_days) / speed)))
        tier0 = float(decay_config.tier0_threshold)
        tier1 = max(float(decay_config.tier1_threshold), tier0 + 0.001)
        return {
            "tier0_threshold": tier0,
            "tier1_threshold": tier1,
            "useful_score_boost": max(0.01, float(decay_config.consolidate_speed)),
            "strength_boost": 1,
            "tier0_cycle_days": cycle_days,
        }

    def _configure_pmemory(
        self,
        *,
        embedding_provider,
        rerank_provider,
        rerank_max_docs: int,
        rerank_max_tokens: int,
    ) -> None:
        """数据目录放在 memory_center 同级，向量按 space 存，换供应商只新增 space。"""
        root = Path(self.plugin_context.get_memory_center_dir()).parent / "pmemory"
        raw_dir = self._resolve_raw_dir()
        pmemory.configure(
            root=root,
            raw_dir=raw_dir,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            rerank_options={
                "max_tokens_total": RERANK_MAX_TOKENS_TOTAL,
                "max_candidates": int(rerank_max_docs),
                "max_tokens_per_doc": int(rerank_max_tokens),
            },
        )
        self.logger.debug(f"p-memory 已配置: 目录={root} 笔记根目录={raw_dir}")

        # 配置里出现的 scope 一并登记，退休与画像才能跨 scope 看到它们
        try:
            pmemory.register_scopes(self.plugin_context.get_conversation_scope_map().values())
        except Exception as e:
            self.logger.warning(f"登记记忆 scope 失败: {e}")

    # === 组件创建 ===

    async def _try_create_embedding_provider(self):
        """创建嵌入提供商；不可用时返回 None，由库降级为纯全文检索。"""
        embedding_provider_id = self.plugin_context.get_embedding_provider_id()
        enable_local = self.plugin_context.get_enable_local_embedding()
        if not str(embedding_provider_id or "").strip() and not enable_local:
            self.logger.info("未启用嵌入能力，记忆检索走纯全文")
            return None

        self.logger.debug(f"创建嵌入提供商: '{embedding_provider_id}'")
        try:
            provider = await self._create_embedding_provider()
        except Exception as e:
            self.logger.warning(f"嵌入提供商初始化失败，记忆检索走纯全文: {e}")
            return None

        self._components["embedding_provider"] = provider
        self.plugin_context.set_embedding_provider(provider)
        return provider

    async def _create_embedding_provider(self):
        """创建嵌入提供商"""
        embedding_provider_id = self.plugin_context.get_embedding_provider_id()
        factory = EmbeddingProviderFactory(self.context)
        embedding_provider = await factory.create_provider(
            embedding_provider_id,
            enable_local_embedding=self.plugin_context.get_enable_local_embedding(),
            batch_size=self.plugin_context.get_embedding_batch_size(),
            timeout=60,
        )
        self.logger.debug(f"嵌入提供商创建完成: {embedding_provider.get_model_info()}")
        return embedding_provider

    def _resolve_rerank_provider(self) -> Optional[Any]:
        """
        解析上游重排提供商（可选）。

        优先级：
        1. 配置项 rerank_provider_id（如果存在）
        2. 配置项 provider_id（LLM 提供商ID）
        3. 上游 context 中第一个具备 rerank() 方法的提供商
        """
        try:
            rerank_provider_id = self.plugin_context.get_rerank_provider_id()
            llm_provider_id = self.plugin_context.get_llm_provider_id()
            candidate_ids = [
                pid
                for pid in [
                    rerank_provider_id,
                    llm_provider_id,
                ]
                if pid
            ]

            for pid in candidate_ids:
                if hasattr(self.context, "get_rerank_provider_by_id"):
                    provider = self.context.get_rerank_provider_by_id(pid)
                    if provider and hasattr(provider, "rerank"):
                        self.logger.info(f"使用上游重排提供商: {pid}")
                        return provider

                if hasattr(self.context, "get_provider_by_id"):
                    provider = self.context.get_provider_by_id(pid)
                    if provider and hasattr(provider, "rerank"):
                        self.logger.info(f"使用上游可重排提供商: {pid}")
                        return provider

            if hasattr(self.context, "get_all_rerank_providers"):
                for p in self.context.get_all_rerank_providers() or []:
                    if hasattr(p, "rerank"):
                        self.logger.info("使用上游重排提供商: <auto>")
                        return p

            if hasattr(self.context, "get_all_providers"):
                for p in self.context.get_all_providers() or []:
                    if hasattr(p, "rerank"):
                        self.logger.info("使用上游可重排提供商: <auto>")
                        return p

            self.logger.info("未配置重排提供商，检索按库内排序返回")
            return None
        except Exception as e:
            self.logger.warning(f"解析上游重排提供商失败，检索不重排: {e}")
            return None

    def _create_memory_sql_manager(self) -> MemorySqlManager:
        """创建账本管理器（仅承载账本与迁移源数据）。"""
        simple_db_path = self.plugin_context.get_simple_memory_db_path()
        manager = MemorySqlManager(simple_db_path)
        self.logger.debug(f"SQL记忆管理器创建完成: {simple_db_path}")
        return manager

    def _create_memory_runtime(
        self,
        memory_store: MemoryStore,
        decay_config: Optional[MemoryDecayConfig] = None,
    ) -> PMemoryMemoryRuntime:
        """创建统一记忆运行时。"""
        policy = self._build_decay_policy(decay_config or MemoryDecayConfig())
        runtime = PMemoryMemoryRuntime(memory_store, decay_policy=policy)
        self.logger.debug(f"统一记忆运行时创建完成 (p-memory) 衰减策略={policy}")
        return runtime

    def _create_note_index_service(self) -> NoteIndexService:
        """创建笔记索引服务。"""
        raw_dir = self._resolve_raw_dir()
        if raw_dir is None:
            raw_dir = Path(self.data_directory)
            self.logger.warning(f"笔记根目录回退为数据目录: {raw_dir}")
        service = NoteIndexService(raw_dir)
        self.logger.debug(f"笔记索引服务创建完成: {raw_dir}")
        return service

    async def _create_deepmind(self, note_index, memory_runtime):
        """创建DeepMind"""
        self.logger.debug("创建DeepMind...")

        llm_provider_id = self.plugin_context.get_llm_provider_id()

        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

        config_obj = Config(self.plugin_context.get_all_config())

        deepmind = DeepMind(
            config=config_obj,
            context=self.context,
            note_service=note_index,
            plugin_context=self.plugin_context,
            provider_id=llm_provider_id,
            memory_runtime=memory_runtime,
        )

        self.logger.debug("DeepMind创建完成")
        return deepmind

    # === 状态 ===

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

        self.logger.debug("开始关闭所有核心组件...")

        component_shutdown_order = [
            "legacy_migration",
            "note_index",
            "deepmind",
            "memory_runtime",
            "memory_sql_manager",
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
                self.logger.debug(f"正在关闭组件: {component_name}...")
                result = shutdown_method()
                if hasattr(result, "__await__"):
                    try:
                        asyncio.get_running_loop()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(asyncio.run, result).result()
                    except RuntimeError:
                        asyncio.run(result)
                self.logger.debug(f"组件 {component_name} 已关闭")
            except Exception as e:
                self.logger.error(f"关闭组件 {component_name} 失败: {e}", exc_info=True)

        for component_name in component_shutdown_order:
            component = self._components.get(component_name)
            if component:
                _try_shutdown_component(component_name, component)
                self._components.pop(component_name, None)

        try:
            pmemory.close()
        except Exception as e:
            self.logger.error(f"关闭 p-memory 失败: {e}", exc_info=True)

        self._components.clear()
        self._initialized = False
        self.logger.debug("所有核心组件已成功关闭")
