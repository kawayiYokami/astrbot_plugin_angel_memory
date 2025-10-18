"""
PluginManager - 插件管理器

负责插件的业务逻辑处理，集成初始化管理和后台初始化。
"""

from .initialization_manager import InitializationManager
from .background_initializer import BackgroundInitializer
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class PluginManager:
    """插件管理器"""

    def __init__(self, plugin_context):
        """
        初始化插件管理器

        Args:
            plugin_context: PluginContext插件上下文对象（包含所有必要资源）
        """
        self.plugin_context = plugin_context
        self.context = plugin_context.get_astrbot_context()  # 保持向后兼容
        self.logger = logger
        self.config = plugin_context.get_all_config()  # 从PluginContext获取配置

        # 初始化管理器（专注于状态管理，使用AstrBot Context进行提供商检测）
        self.init_manager = InitializationManager(self.context)

        # 后台初始化器（共享主线程的PluginContext）
        data_directory = plugin_context.base_data_dir
        self.background_initializer = BackgroundInitializer(self.init_manager, self.config, plugin_context, data_directory)

        # 主线程组件实例（将在初始化完成后由主插件设置）
        self.main_thread_components = {}

        # 启动后台初始化
        self.background_initializer.start_background_initialization()

        self.logger.info("🚀 Angel Memory Plugin 管理器已启动")
        self.logger.info(f"   当前提供商: {plugin_context.get_current_provider()}")
        self.logger.info(f"   数据目录: {plugin_context.get_index_dir()}")
        self.logger.info(f"   有可用提供商: {plugin_context.has_providers()}")
        self.logger.info("   初始化架构: PluginContext + InitializationManager 协作模式")
    async def handle_llm_request(self, event, request, event_plugin_context=None):
        """
        处理LLM请求

        Args:
            event: 消息事件对象
            request: LLM请求对象
            event_plugin_context: 事件专用的PluginContext（可选）

        Returns:
            dict: 处理结果
        """

        # 等待系统准备就绪
        if not self.init_manager.wait_until_ready(timeout=30):
            self.logger.info("⏳ 系统正在初始化中，LLM请求将跳过")
            return {
                "status": "waiting",
                "message": "系统正在初始化中，请稍候..."
            }

        # 系统准备就绪，正常处理业务
        return await self._process_llm_request(event, request, event_plugin_context)

    async def handle_llm_response(self, event, response, event_plugin_context=None):
        """
        处理LLM响应

        Args:
            event: 消息事件对象
            response: LLM响应对象
            event_plugin_context: 事件专用的PluginContext（可选）

        Returns:
            dict: 处理结果
        """

        # 等待系统准备就绪
        if not self.init_manager.wait_until_ready(timeout=30):
            self.logger.info("⏳ 系统正在初始化中，LLM响应将跳过")
            return {
                "status": "waiting",
                "message": "系统正在初始化中，请稍候..."
            }

        # 系统准备就绪，正常处理业务
        return await self._process_llm_response(event, response, event_plugin_context)

    def set_main_thread_components(self, components: dict):
        """
        设置主线程组件实例

        Args:
            components: 主线程创建的组件字典
        """
        self.main_thread_components = components
    async def _process_llm_request(self, event, request, event_plugin_context=None):
        """
        处理LLM请求的具体逻辑

        Args:
            event: 消息事件对象
            request: LLM请求对象
            event_plugin_context: 事件专用的PluginContext（可选）

        Returns:
            dict: 处理结果
        """
        self.logger.debug("开始执行 _process_llm_request")
        try:
            # 优先使用主线程组件，其次使用后台初始化的组件
            deepmind = self.main_thread_components.get("deepmind")
            if not deepmind:
                # 如果主线程组件还没设置，使用后台组件（向后兼容）
                components = self.background_initializer.get_initialized_components()
                deepmind = components.get("deepmind")

            if deepmind:
                self.logger.debug("找到 DeepMind 组件，开始执行 organize_and_inject_memories")
                # 直接使用 await 处理异步任务
                await deepmind.organize_and_inject_memories(event, request)
                self.logger.debug("organize_and_inject_memories 执行完成")

                return {
                    "status": "success",
                    "message": "LLM请求处理完成",
                    "request_type": "llm"
                }
            else:
                self.logger.warning("DeepMind组件尚未初始化完成")
                return {
                    "status": "waiting",
                    "message": "DeepMind组件尚未初始化完成"
                }

        except Exception as e:
            self.logger.error(f"LLM请求处理失败: {e}")
            return {
                "status": "error",
                "message": f"LLM请求处理失败: {str(e)}"
            }

    async def _process_llm_response(self, event, response, event_plugin_context=None):
        """
        处理LLM响应的具体逻辑

        Args:
            event: 消息事件对象
            response: LLM响应对象
            event_plugin_context: 事件专用的PluginContext（可选）

        Returns:
            dict: 处理结果
        """
        self.logger.debug("开始执行 _process_llm_response")
        try:
            # 优先使用主线程组件，其次使用后台初始化的组件
            deepmind = self.main_thread_components.get("deepmind")
            if not deepmind:
                # 如果主线程组件还没设置，使用后台组件（向后兼容）
                components = self.background_initializer.get_initialized_components()
                deepmind = components.get("deepmind")

            if deepmind:
                self.logger.debug("找到 DeepMind 组件，开始执行 async_analyze_and_update_memory")
                # 调用异步分析方法
                await deepmind.async_analyze_and_update_memory(event, response)
                self.logger.debug("async_analyze_and_update_memory 执行完成")

                return {
                    "status": "success",
                    "message": "LLM响应处理完成",
                    "response_type": "llm"
                }
            else:
                self.logger.warning("DeepMind组件尚未初始化完成")
                return {
                    "status": "waiting",
                    "message": "DeepMind组件尚未初始化完成"
                }

        except Exception as e:
            self.logger.error(f"LLM响应处理失败: {e}")
            return {
                "status": "error",
                "message": f"LLM响应处理失败: {str(e)}"
            }

    def get_initialized_components(self):
        """
        获取已初始化的组件（供主插件使用）

        Returns:
            dict: 已初始化的组件
        """
        return self.background_initializer.get_initialized_components()

    def get_status(self):
        """
        获取插件状态

        Returns:
            dict: 状态信息
        """
        try:
            providers = self.context.get_all_providers()
            has_providers = len(providers) > 0
            provider_count = len(providers)

            status = {
                "state": self.init_manager.get_current_state().value,
                "ready": self.init_manager.is_ready(),
                "has_providers": has_providers,
                "provider_count": provider_count
            }

            self.logger.debug(f"📊 插件状态查询: {status}")
            return status

        except Exception as e:
            self.logger.error(f"❌ 获取插件状态失败: {e}")
            return {
                "state": "error",
                "ready": False,
                "has_providers": False,
                "provider_count": 0,
                "error": str(e)
            }

    def is_ready(self):
        """
        检查插件是否准备就绪

        Returns:
            bool: 是否准备就绪
        """
        ready = self.init_manager.is_ready()
        self.logger.debug(f"📋 插件就绪状态检查: {ready}")
        return ready

    def shutdown(self):
        """关闭插件管理器和所有后台服务"""
        self.logger.info("插件管理器正在关闭...")

        # 关闭后台初始化器
        if self.background_initializer:
            self.background_initializer.shutdown()

        self.logger.info("插件管理器已成功关闭")
