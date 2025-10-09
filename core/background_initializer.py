"""
BackgroundInitializer - 后台初始化器

负责在后台线程中执行初始化任务，但实例由主线程统一管理。
后台线程只负责执行初始化逻辑，不拥有任何组件实例。
"""

import threading
import asyncio
from .initialization_manager import InitializationManager
from .component_factory import ComponentFactory
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BackgroundInitializer:
    """后台初始化器 - 仅负责初始化逻辑，不拥有实例"""

    def __init__(self, init_manager: InitializationManager, config: dict, plugin_context, data_directory: str = None):
        """
        初始化后台初始化器

        Args:
            init_manager: 初始化状态管理器（专注于状态管理）
            config: 插件配置（在主线程中获取）
            plugin_context: PluginContext实例（与主线程共享）
            data_directory: 数据目录路径（由main传入，向后兼容）
        """
        self.init_manager = init_manager
        self.background_thread = None
        self.context = init_manager.context
        self.logger = logger
        self.config = config
        self.plugin_context = plugin_context
        self.data_directory = data_directory

        self.logger.info(f"📋 后台初始化器接收配置: {list(self.config.keys())}")
        if self.data_directory:
            self.logger.info(f"📋 后台初始化器接收数据目录: {self.data_directory}")

        # 直接使用主线程的PluginContext创建ComponentFactory
        self.component_factory = ComponentFactory(self.plugin_context, init_manager=self.init_manager)
        self.logger.debug("BackgroundInitializer初始化完成 - 共享主线程PluginContext")

    def start_background_initialization(self):
        """启动后台初始化线程"""
        self.background_thread = threading.Thread(
            target=self._initialization_worker,
            daemon=True,
            name="BackgroundInitializer"
        )
        self.background_thread.start()
        self.logger.info("🚀 后台初始化线程已启动")

    def _initialization_worker(self):
        """后台初始化工作线程"""
        try:
            self.logger.info("🚀 启动后台初始化工作线程...")

            # 等待提供商就绪
            should_initialize = self.init_manager.wait_for_providers_and_initialize()

            if should_initialize:
                # 开始真正的初始化
                self._perform_initialization()
            else:
                self.logger.info("⏹️ 初始化被中断")
                return

            self.logger.info("✅ 后台初始化工作完成")

        except Exception as e:
            self.logger.error(f"❌ 后台初始化失败: {e}")
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

                # 3. DeepMind初始化时已经执行了记忆巩固，这里不需要重复执行
                deepmind = components.get("deepmind")
                if deepmind and deepmind.is_enabled():
                    self.logger.info("🧠 DeepMind已在初始化时完成记忆巩固，跳过重复巩固")
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