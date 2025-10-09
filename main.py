"""
AstrBot Angel Memory Plugin

基于双层认知架构的AI记忆系统插件，为AstrBot提供记忆能力。
实现观察→回忆→反馈→睡眠的完整认知工作流。

采用新的懒加载+后台预初始化架构，实现极速启动和智能提供商等待。
"""

from astrbot.api.star import Context, Star
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools

# 导入核心模块
from .core.plugin_manager import PluginManager
from .core.config import MemoryConfig
from .core.initialization_manager import InitializationState
from pathlib import Path

class AngelMemoryPlugin(Star):
    """天使记忆插件主类

    集成DeepMind记忆系统和多格式文档处理能力，为AstrBot提供完整的记忆功能。

    新架构特点：
    - 极速启动：毫秒级启动，所有耗时操作移至后台
    - 智能等待：后台自动检测提供商，有提供商时自动初始化
    - 无降级逻辑：没有提供商时插件不会被调用，无需特殊处理
    - 状态同步：业务请求自动等待初始化完成

    插件启动时只创建实例和启动后台线程，terminate时安全清理资源。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        # 保存配置
        self.config = config or {}

        # 使用 astrbot.api 的 logger
        self.logger = logger

        # 1. 在主类中安全地获取数据目录
        plugin_data_dir_str = str(StarTools.get_data_dir())

        # 2. 将获取到的目录传递给 MemoryConfig
        self.memory_config = MemoryConfig(self.config, data_dir=plugin_data_dir_str)

        # 3. 初始化插件管理器（极速启动）
        self.plugin_manager = PluginManager(context)

        # 记录数据路径以验证配置
        self.logger.info(f"天使记忆数据路径设置为: {Path(self.memory_config.data_directory).resolve()}")
        self.logger.info("Angel Memory Plugin 实例创建完成，后台初始化已启动")

        # 初始化原有组件的占位符（将在后台初始化中真正创建）
        self.deepmind = None
        self.note_service = None
        self.file_monitor = None

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE | filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def on_message_event(self, event: AstrMessageEvent):
        """
        事件到达时立即注入初始记忆

        Args:
            event: 消息事件
        """
        try:
            # 使用插件管理器处理事件
            result = self.plugin_manager.handle_message_event(event)

            if result["status"] == "waiting":
                self.logger.info("系统正在初始化中，跳过此次事件注入")
                return
            elif result["status"] == "success":
                # 这里应该调用实际的deepmind逻辑，暂时用占位符
                if self.deepmind:
                    await self.deepmind.inject_initial_memories(event)
                else:
                    self.logger.debug("DeepMind 组件尚未初始化完成")
            else:
                self.logger.error(f"事件处理失败: {result.get('message', '未知错误')}")

        except Exception as e:
            self.logger.error(f"EVENT_INJECT failed: {e}")

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        LLM调用前整理记忆并注入到请求中

        Args:
            event: 消息事件
            request: LLM请求对象
        """
        try:
            # 使用插件管理器处理请求
            result = self.plugin_manager.handle_llm_request(event, request)

            if result["status"] == "waiting":
                self.logger.info("系统正在初始化中，跳过此次LLM请求处理")
                return
            elif result["status"] == "success":
                # 这里应该调用实际的deepmind逻辑，暂时用占位符
                if self.deepmind:
                    await self.deepmind.organize_and_inject_memories(event, request)
                else:
                    self.logger.debug("DeepMind 组件尚未初始化完成")
            else:
                self.logger.error(f"LLM请求处理失败: {result.get('message', '未知错误')}")

        except Exception as e:
            self.logger.error(f"LLM_REQUEST failed: {e}")

    async def terminate(self) -> None:
        """插件卸载时的清理工作"""
        try:
            # 停止文件监控（如果存在）
            if self.file_monitor:
                self.file_monitor.stop_monitoring()

            # 停止定期睡眠（如果存在）
            if self.deepmind:
                self.deepmind.stop_sleep()

            # 获取最终状态
            status = self.plugin_manager.get_status() if self.plugin_manager else {"state": "unknown"}
            self.logger.info(f"Angel Memory Plugin 正在关闭，最终状态: {status.get('state', 'unknown')}")

        except Exception as e:
            self.logger.error(f"Angel Memory Plugin: 插件卸载清理失败: {e}")

    def get_plugin_status(self):
        """
        获取插件状态（用于调试）

        Returns:
            dict: 插件状态信息
        """
        if not self.plugin_manager:
            return {"status": "not_initialized"}

        return self.plugin_manager.get_status()