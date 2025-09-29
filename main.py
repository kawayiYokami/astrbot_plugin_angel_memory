"""
AstrBot Angel Memory Plugin

基于双层认知架构的AI记忆系统插件，为AstrBot提供记忆能力。
实现观察→回忆→反馈→睡眠的完整认知工作流。
"""

from astrbot.api.star import Context, Star
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools

# 导入核心模块
from .core import DeepMind
from .core.config import MemoryConfig
from .core.logger import set_logger
from .llm_memory.config.system_config import system_config
from pathlib import Path
class AngelMemoryPlugin(Star):
    """天使记忆插件主类"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        # 使用 astrbot.api 的 logger
        self.logger = logger

        # 设置全局 logger 容器，供所有子模块使用
        set_logger(self.logger)

        # 1. 在主类中安全地获取数据目录
        plugin_data_dir_str = str(StarTools.get_data_dir())

        # 2. 将获取到的目录传递给 MemoryConfig
        memory_config = MemoryConfig(config, data_dir=plugin_data_dir_str)

        # 配置 llm_memory 的 system_config
        # 1. 设置巩固间隔（将秒转换为小时）
        system_config.consolidation_interval_hours = memory_config.get_sleep_interval() // 3600

        # 2. 设置存储目录
        plugin_data_dir = Path(memory_config.get_data_directory())
        system_config.storage_dir = plugin_data_dir
        system_config.index_dir = plugin_data_dir / "index"

        # 3. 确保目录存在
        system_config.ensure_directories_exist()

        # 4. 获取记忆整理 LLM 提供商ID
        provider_id = memory_config.get_provider_id()

        # 初始化核心组件
        self.deepmind = DeepMind(system_config, self.context, provider_id)

        # 记录数据路径以验证配置
        self.logger.info(f"天使记忆数据路径设置为: {system_config.storage_dir.resolve()}")

        self.logger.info("Angel Memory Plugin 初始化完成")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE | filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def on_message_event(self, event: AstrMessageEvent):
        """
        事件到达时立即注入初始记忆

        Args:
            event: 消息事件
        """
        try:
            self.deepmind.inject_initial_memories(event)
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
            await self.deepmind.organize_and_inject_memories(event, request)
        except Exception as e:
            self.logger.error(f"LLM_REQUEST failed: {e}")

    async def terminate(self):
        """插件卸载时的清理工作"""
        try:
            # 停止定期睡眠
            self.deepmind.stop_sleep()
            self.logger.info("Angel Memory Plugin 正在关闭")
        except Exception as e:
            self.logger.error(f"Angel Memory Plugin: 插件卸载清理失败: {e}")
