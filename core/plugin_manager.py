"""
PluginManager - 插件管理器

负责插件的业务逻辑处理，集成初始化管理和后台初始化。
"""

from .initialization_manager import InitializationManager, InitializationState
from .background_initializer import BackgroundInitializer
from astrbot.api import logger

class PluginManager:
    """插件管理器"""

    def __init__(self, context):
        """
        初始化插件管理器

        Args:
            context: AstrBot上下文对象
        """
        self.context = context
        self.logger = logger
        self.init_manager = InitializationManager(context)
        self.background_initializer = BackgroundInitializer(self.init_manager)

        # 启动后台初始化
        self.background_initializer.start_background_initialization()

        self.logger.info("🚀 Angel Memory Plugin 管理器已启动，等待提供商...")

    def handle_message_event(self, event):
        """
        处理消息事件

        Args:
            event: 消息事件对象

        Returns:
            dict: 处理结果
        """
        self.logger.info("📥 收到消息事件，检查系统状态...")

        # 等待系统准备就绪
        if not self.init_manager.wait_until_ready(timeout=30):
            self.logger.info("⏳ 系统正在初始化中，消息事件将跳过")
            return {
                "status": "waiting",
                "message": "系统正在初始化中，请稍候..."
            }

        # 系统准备就绪，正常处理业务
        self.logger.info("✅ 系统准备就绪，开始处理消息事件")
        return self._process_message_event(event)

    def handle_llm_request(self, event, request):
        """
        处理LLM请求

        Args:
            event: 消息事件对象
            request: LLM请求对象

        Returns:
            dict: 处理结果
        """
        self.logger.info("📥 收到LLM请求，检查系统状态...")

        # 等待系统准备就绪
        if not self.init_manager.wait_until_ready(timeout=30):
            self.logger.info("⏳ 系统正在初始化中，LLM请求将跳过")
            return {
                "status": "waiting",
                "message": "系统正在初始化中，请稍候..."
            }

        # 系统准备就绪，正常处理业务
        self.logger.info("✅ 系统准备就绪，开始处理LLM请求")
        return self._process_llm_request(event, request)

    def _process_message_event(self, event):
        """
        处理消息事件的具体逻辑

        Args:
            event: 消息事件对象

        Returns:
            dict: 处理结果
        """
        # 这里会集成现有的消息事件处理逻辑
        # 目前返回模拟结果
        self.logger.debug("🔧 执行消息事件处理逻辑")
        return {
            "status": "success",
            "message": "消息事件处理完成",
            "event_type": "message"
        }

    def _process_llm_request(self, event, request):
        """
        处理LLM请求的具体逻辑

        Args:
            event: 消息事件对象
            request: LLM请求对象

        Returns:
            dict: 处理结果
        """
        # 这里会集成现有的LLM请求处理逻辑
        # 目前返回模拟结果
        self.logger.debug("🔧 执行LLM请求处理逻辑")
        return {
            "status": "success",
            "message": "LLM请求处理完成",
            "request_type": "llm"
        }

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