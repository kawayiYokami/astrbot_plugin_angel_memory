"""
BackgroundInitializer - 后台初始化器（异步版本）

使用 asyncio.create_task() 在后台异步执行初始化任务。
按照 AstrBot 官方推荐的异步架构设计。
"""

import asyncio
from .initialization_manager import InitializationManager
from .component_factory import ComponentFactory

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

        self.logger.info(f"📋 后台初始化器接收配置: {list(self.config.keys())}")
        self.logger.info(
            f"📋 后台初始化器使用数据目录: {plugin_context.get_index_dir()}"
        )

        # 直接使用主线程的PluginContext创建ComponentFactory
        self.component_factory = ComponentFactory(
            self.plugin_context, init_manager=self.init_manager
        )
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

            # 等待提供商就绪（在线程池中执行同步方法）
            should_initialize = await asyncio.to_thread(
                self.init_manager.wait_for_providers_and_initialize
            )

            if should_initialize:
                # 开始真正的初始化（在线程池中执行）
                await asyncio.to_thread(self._perform_initialization)
            else:
                self.logger.info("⏹️ 初始化被中断")
                return

            self.logger.info("✅ 异步后台初始化完成")

        except Exception as e:
            self.logger.error(f"❌ 异步后台初始化失败: {e}")
            import traceback
            self.logger.error(f"异常详情: {traceback.format_exc()}")

    def _perform_initialization(self):
        """执行真正的初始化工作"""
        self.logger.info("🤖 开始执行完整的系统初始化...")

        try:
            # 配置已经在主线程中获取，直接使用
            self.logger.info(f"📋 使用配置: {list(self.config.keys())}")

            # 2. 在主线程的事件循环中创建所有组件
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                components = loop.run_until_complete(
                    self.component_factory.create_all_components(self.config)
                )
                self.logger.info("✅ 所有组件在后台线程中创建完成")

                # 清理并禁用嵌入缓存（初始化完成后节省内存）
                embedding_provider = components.get("embedding_provider")
                if embedding_provider and hasattr(embedding_provider, 'clear_and_disable_cache'):
                    embedding_provider.clear_and_disable_cache()
                    self.logger.info("🗑️ 嵌入缓存已清理并禁用（节省内存）")

                # 3. DeepMind初始化时已经执行了记忆巩固，这里不需要重复执行
                deepmind = components.get("deepmind")
                if deepmind and deepmind.is_enabled():
                    self.logger.info(
                        "🧠 DeepMind已在初始化时完成记忆巩固，跳过重复巩固"
                    )
                else:
                    self.logger.warning("⚠️ DeepMind未启用")

            finally:
                loop.close()

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

        # 关闭所有由ComponentFactory创建的组件
        if self.component_factory:
            self.component_factory.shutdown()

        self.logger.info("后台初始化器已成功关闭")
