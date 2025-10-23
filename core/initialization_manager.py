"""
InitializationManager - 初始化状态管理器

专注于管理系统初始化状态，提供提供商检测、状态管理和同步机制。
与PluginContext协作，PluginContext负责资源管理，InitializationManager负责状态管理。
"""

from enum import Enum
from threading import RLock, Event
import time

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class InitializationState(Enum):
    """初始化状态枚举"""

    NOT_STARTED = "not_started"
    WAITING_FOR_PROVIDERS = "waiting_for_providers"
    INITIALIZING = "initializing"
    READY = "ready"


class InitializationManager:
    """初始化状态管理器"""

    def __init__(self, context):
        """
        初始化管理器

        Args:
            context: AstrBot上下文对象，用于检查提供商
        """
        self.context = context
        self.current_state = InitializationState.NOT_STARTED
        self.state_lock = RLock()
        self.ready_event = Event()
        self.logger = logger
        self.logger.debug(
            "InitializationManager初始化完成 - 专注于状态管理和提供商检测"
        )

    def wait_for_providers_and_initialize(self, check_interval=10):
        """
        等待提供商并初始化

        Args:
            check_interval: 检查提供商的间隔时间（秒）

        Returns:
            bool: True=应该开始初始化, False=被中断（目前不会发生）
        """
        self.transition_to(InitializationState.WAITING_FOR_PROVIDERS)
        self.logger.info("🔍 开始等待LLM提供商...")

        while True:
            try:
                # 获取LLM提供商
                llm_providers = self.context.get_all_providers()
                llm_provider_count = len(llm_providers)

                # 获取嵌入式模型提供商
                embedding_providers = (
                    self.context.get_all_embedding_providers()
                    if hasattr(self.context, "get_all_embedding_providers")
                    else []
                )
                embedding_provider_count = len(embedding_providers)

                total_providers = llm_provider_count + embedding_provider_count
                self.logger.info(
                    f"📊 检查提供商状态: 发现 {llm_provider_count} 个LLM提供商, {embedding_provider_count} 个嵌入提供商"
                )

                if total_providers > 0:
                    # 收集所有提供商信息
                    provider_info = []

                    # 处理LLM提供商
                    if llm_providers:
                        llm_ids = [p.meta().id for p in llm_providers]
                        provider_info.append(f"LLM: {', '.join(llm_ids)}")

                    # 处理嵌入提供商
                    if embedding_providers:
                        embedding_ids = [p.meta().id for p in embedding_providers]
                        provider_info.append(f"嵌入: {', '.join(embedding_ids)}")

                    self.logger.info(
                        f"✅ 检测到提供商: {' | '.join(provider_info)}，开始初始化"
                    )
                    self.transition_to(InitializationState.INITIALIZING)
                    return True
                else:
                    self.logger.info("⏳ 暂无提供商，10秒后再次检查...")

            except Exception as e:
                self.logger.error(f"❌ 检查提供商时出错: {e}")

            # 等待10秒再检查
            time.sleep(check_interval)

    def mark_ready(self):
        """标记为准备就绪"""
        self.transition_to(InitializationState.READY)
        self.ready_event.set()
        self.logger.info("🎉 系统准备就绪！可以开始处理业务请求")

    def wait_until_ready(self, timeout=None):
        """
        等待系统准备就绪

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            bool: True=准备就绪, False=超时
        """
        result = self.ready_event.wait(timeout)
        if result:
            self.logger.info("✅ 系统等待完成：准备就绪")
        else:
            self.logger.warning(f"⏰ 系统等待超时：{timeout}秒")
        return result

    def transition_to(self, new_state):
        """
        状态转换

        Args:
            new_state: 新状态
        """
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self.logger.info(f"🔄 状态从 {old_state.value} 切换到 {new_state.value}")

    def is_ready(self):
        """检查是否准备就绪"""
        ready = self.ready_event.is_set()
        self.logger.debug(f"📋 系统就绪状态检查: {ready}")
        return ready

    def get_current_state(self):
        """获取当前状态"""
        current_state = self.current_state
        self.logger.debug(f"📋 当前状态: {current_state.value}")
        return current_state
