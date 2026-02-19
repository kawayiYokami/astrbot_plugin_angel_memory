"""
BackgroundInitializer - 后台初始化器（异步版本）

使用 asyncio.create_task() 在后台异步执行初始化任务。
所有核心组件在当前运行事件循环中创建，避免跨线程/跨事件循环问题。
"""

import asyncio
from .initialization_manager import InitializationManager
from .component_factory import ComponentFactory
from .migrations.memory_scope_migration import MemoryScopeMigration

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class BackgroundInitializer:
    """后台初始化器 - 使用 asyncio 异步初始化"""

    def __init__(
        self, init_manager: InitializationManager, config: dict, plugin_context
    ):
        """
        初始化后台初始化器

        Args:
            init_manager: 初始化状态管理器（专注于状态管理）
            config: 插件配置（在主线程中获取）
            plugin_context: PluginContext实例（与主线程共享）
        """
        self.init_manager = init_manager
        self.background_task = None
        self.context = init_manager.context
        self.logger = logger
        self.config = config
        self.plugin_context = plugin_context
        self._migration_tasks = []
        self._post_init_tasks = []

        self.logger.info(f"📋 后台初始化器接收配置: {list(self.config.keys())}")
        self.logger.info(
            f"📋 后台初始化器使用数据目录: {plugin_context.get_index_dir()}"
        )

        # 直接使用主线程的PluginContext创建ComponentFactory
        self.component_factory = ComponentFactory(
            self.plugin_context, init_manager=self.init_manager
        )
        self.plugin_context.set_component_factory(self.component_factory)
        self.logger.debug("BackgroundInitializer初始化完成 - 共享主线程PluginContext")

    def start_background_initialization(self):
        """启动后台初始化任务（纯 asyncio，无线程回退）"""
        try:
            # 检查是否有运行中的事件循环
            asyncio.get_running_loop()
        except RuntimeError as e:
            error_msg = (
                "BackgroundInitializer 需要运行中的 asyncio 事件循环。\n"
                "请确保在 async 上下文中调用此方法。\n"
                "如果您看到此错误，说明 AstrBot 的异步环境未正确初始化。"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # 按照官方推荐使用 asyncio.create_task()
        self.background_task = asyncio.create_task(self._background_initialization())
        self.logger.info("🚀 后台初始化任务已启动（asyncio）")

    async def _background_initialization(self):
        """异步后台初始化任务"""
        try:
            self.logger.info("🚀 启动异步后台初始化...")

            # 异步等待提供商就绪（不阻塞事件循环）
            should_initialize = (
                await self.init_manager.wait_for_providers_and_initialize_async()
            )

            if should_initialize:
                # 在当前事件循环中创建所有组件
                await self._perform_initialization_async()
            else:
                self.logger.info("⏹️ 初始化被中断")
                return

            self.logger.info("✅ 异步后台初始化完成")

        except Exception as e:
            self.logger.error(f"❌ 异步后台初始化失败: {e}")
            import traceback
            self.logger.error(f"异常详情: {traceback.format_exc()}")
            try:
                self.init_manager.mark_failed(str(e))
            except Exception:
                pass

    async def _perform_initialization_async(self):
        """执行真正的初始化工作（同一事件循环）"""
        self.logger.info("🤖 开始执行完整的系统初始化...")

        try:
            # 配置已经在主线程中获取，直接使用
            self.logger.info(f"📋 使用配置: {list(self.config.keys())}")

            components = await self.component_factory.create_all_components(self.config)
            self.logger.info("✅ 所有组件在当前事件循环中创建完成")

            # 后台迁移：补齐历史缺失 memory_scope 的记录（不阻塞启动）
            cognitive_service = components.get("cognitive_service")
            if cognitive_service and hasattr(cognitive_service, "main_collection"):
                async def _run_memory_scope_migration():
                    try:
                        migrator = MemoryScopeMigration(self.logger)
                        await migrator.migrate_missing_memory_scope(
                            collection=cognitive_service.main_collection
                        )
                    except Exception as e:
                        self.logger.error(f"❌ memory_scope 后台迁移失败: {e}", exc_info=True)

                migration_task = asyncio.create_task(_run_memory_scope_migration())
                self._migration_tasks.append(migration_task)
                self.logger.info("🛠️ memory_scope 后台迁移任务已调度（异步分离）")
            else:
                if bool(self.config.get("enable_simple_memory", False)):
                    self.logger.info("ℹ️ 当前为简化记忆模式，已跳过向量 memory_scope 迁移。")
                else:
                    self.logger.warning("⚠️ memory_scope 迁移跳过：cognitive_service/main_collection 不可用")

            embedding_provider = components.get("embedding_provider")
            if embedding_provider and hasattr(embedding_provider, 'clear_and_disable_cache'):
                embedding_provider.clear_and_disable_cache()
                self.logger.info("🗑️ 嵌入缓存已清理并禁用（节省内存）")

            deepmind = components.get("deepmind")
            if deepmind and deepmind.is_enabled():
                memory_behavior = self.config.get("memory_behavior", {})
                if isinstance(memory_behavior, dict):
                    sleep_interval = int(memory_behavior.get("sleep_interval", 3600))
                else:
                    sleep_interval = int(self.config.get("sleep_interval", 3600))

                async def _trigger_sleep_once_after_init():
                    try:
                        self.logger.info(
                            f"[simple_backup] trigger_sleep_after_init provider={self.plugin_context.get_current_provider()}"
                        )
                        if sleep_interval > 0:
                            await deepmind.check_and_sleep_if_needed(sleep_interval)
                        else:
                            await deepmind._sleep()
                    except Exception as e:
                        self.logger.error(f"初始化后触发睡眠失败: {e}", exc_info=True)

                task = asyncio.create_task(_trigger_sleep_once_after_init())
                self._post_init_tasks.append(task)
            else:
                self.logger.warning("⚠️ DeepMind未启用")

        except Exception as e:
            self.logger.error(f"❌ 系统初始化失败: {e}")
            import traceback

            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    def get_initialized_components(self):
        """获取已初始化的组件（向后兼容）"""
        return self.component_factory.get_components()

    def get_component_factory(self):
        """获取组件工厂"""
        return self.component_factory

    def shutdown(self):
        """关闭后台初始化器和所有组件"""
        self.logger.info("后台初始化器正在关闭...")

        # 取消后台初始化任务（如果仍在运行）
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            self.logger.info("后台初始化任务已取消")

        # 取消后台迁移任务（如果仍在运行）
        for task in self._migration_tasks:
            if task and not task.done():
                task.cancel()
        if self._migration_tasks:
            self.logger.info("后台迁移任务已取消")

        for task in self._post_init_tasks:
            if task and not task.done():
                task.cancel()
        if self._post_init_tasks:
            self.logger.info("初始化后后台任务已取消")

        # 关闭所有由ComponentFactory创建的组件
        if self.component_factory:
            self.component_factory.shutdown()

        self.logger.info("后台初始化器已成功关闭")
