"""
AstrBot Angel Memory Plugin

基于双层认知架构的AI记忆系统插件，为AstrBot提供记忆能力。
实现观察→回忆→反馈→睡眠的完整认知工作流。

采用新的懒加载+后台预初始化架构，实现极速启动和智能提供商等待。
"""

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.core.star.star_tools import StarTools
import asyncio

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# 导入版本检查相关模块
import subprocess
import sys
import pkg_resources

# 导入核心模块
from .core.plugin_manager import PluginManager
from .core.plugin_context import PluginContextFactory
from .tools.core_memory_remember import CoreMemoryRememberTool
from .tools.core_memory_recall import CoreMemoryRecallTool


def ensure_chromadb_version():
    """确保 chromadb 版本不低于 1.2.1"""
    MINIMUM_CHROMADB_VERSION = "1.2.1"
    logger.info("开始检查 chromadb 版本...")

    try:
        # 获取当前安装的 chromadb 版本
        current_version = pkg_resources.get_distribution("chromadb").version
        logger.info(f"当前安装的 chromadb 版本: {current_version}")

        if pkg_resources.parse_version(current_version) < pkg_resources.parse_version(
            MINIMUM_CHROMADB_VERSION
        ):
            logger.warning(
                f"chromadb 版本过低 (当前: {current_version}, 最低要求: {MINIMUM_CHROMADB_VERSION})，将升级到最新版本。"
            )
            _upgrade_chromadb()
        else:
            logger.info(f"chromadb 版本检查通过 (版本: {current_version})")

    except pkg_resources.DistributionNotFound:
        logger.warning(
            f"chromadb 未安装，将安装最新版本（不低于 {MINIMUM_CHROMADB_VERSION}）。"
        )
        _upgrade_chromadb()
    except Exception as e:
        logger.error(f"检查 chromadb 版本时出错: {e}")
        logger.warning("无法验证 chromadb 版本，插件可能无法正常工作")

def _upgrade_chromadb():
    """升级 chromadb 到最新版本"""
    try:
        logger.info("正在升级 chromadb 到最新版本...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "chromadb"]
        )
        logger.info("chromadb 升级成功。强烈建议重启应用程序以加载新版本的库。")
    except subprocess.CalledProcessError as e:
        logger.error(f"升级 chromadb 失败: {e}")
        raise


@register(
    "astrbot_plugin_angel_memory",
    "kawayiYokami",
    "天使的记忆，让astrbot拥有记忆维护系统和开箱即用的知识库检索",
    "0.4.1",
    "https://github.com/kawayiYokami/astrbot_plugin_angel_memory"
)
class AngelMemoryPlugin(Star):
    """天使记忆插件主类

    集成DeepMind记忆系统和多格式文档处理能力，为AstrBot提供完整的记忆功能。

    新架构特点：
    - 极速启动：毫秒级启动，所有耗时操作移至后台
    - 智能等待：后台自动检测提供商，有提供商时自动初始化
    - 统一实例管理：所有核心实例在主线程创建，后台线程通过依赖注入使用
    - 无重复初始化：彻底解决重复初始化和实例不一致问题
    - 线程安全：避免跨线程使用异步组件的竞态条件

    插件启动时创建核心实例并启动后台线程，terminate时安全清理资源。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        # 确保 chromadb 版本在初始化开始时检查
        ensure_chromadb_version()

        # 使用 astrbot.api 的 logger
        self.logger = logger

        # 1. 获取插件数据目录（在main.py中获取）
        data_dir = StarTools.get_data_dir("astrbot_plugin_angel_memory")
        self.logger.info(f"获取到插件数据目录: {data_dir}")

        # 2. 创建统一的PluginContext，包含所有必要资源
        self.plugin_context = PluginContextFactory.create_from_initialization(
            context, config or {}, data_dir
        )

        # 2. 核心实例占位符（将在后台初始化完成后通过ComponentFactory创建）
        self.vector_store = None
        self.cognitive_service = None
        self.deepmind = None
        self.note_service = None
        self.file_monitor = None

        # 3. 在主线程获取完整配置（包含提供商信息）
        self._load_complete_config()

        # 4. 初始化插件管理器（极速启动）- 只传递PluginContext
        self.plugin_manager = PluginManager(self.plugin_context)

        # 5. 注册LLM工具
        self.llm_tools_enabled = True  # 标记LLM工具是否启用
        try:
            self.context.add_llm_tools(CoreMemoryRememberTool(), CoreMemoryRecallTool())
            self.logger.info("✅ 已注册 core_memory_remember 和 core_memory_recall 工具。")
        except AttributeError as e:
            self.llm_tools_enabled = False
            self.logger.error(f"❌ 注册LLM工具失败，context可能不支持add_llm_tools方法: {e}", exc_info=True)
            self.logger.warning("⚠️ LLM工具功能已禁用，插件将继续以基础模式运行")
        except Exception as e:
            self.llm_tools_enabled = False
            self.logger.error(f"❌ 注册LLM工具时发生异常: {e}", exc_info=True)
            self.logger.warning("⚠️ LLM工具功能已禁用，插件将继续以基础模式运行")
        self.logger.info(
            f"天使记忆数据路径设置为: {self.plugin_context.get_index_dir().resolve()}"
        )
        self.logger.info(
            f"Angel Memory Plugin 实例创建完成 (提供商: {self.plugin_context.get_current_provider()}), 后台初始化已启动"
        )

    def _load_complete_config(self):
        """在主线程检查配置项"""
        try:
            config = self.plugin_context.get_all_config()
            self.logger.info(f"📋 插件配置加载完成: {list(config.keys())}")

            # 检查关键配置
            embedding_provider_id = self.plugin_context.get_embedding_provider_id()
            if embedding_provider_id:
                self.logger.info(f"✅ 检测到嵌入提供商配置: {embedding_provider_id}")
            else:
                self.logger.info(
                    "ℹ️ 未配置嵌入提供商ID (astrbot_embedding_provider_id)，将使用本地模型"
                )

            llm_provider_id = self.plugin_context.get_llm_provider_id()
            if llm_provider_id:
                self.logger.info(f"✅ 检测到LLM提供商配置: {llm_provider_id}")
            else:
                self.logger.info(
                    "ℹ️ 未配置LLM提供商ID (provider_id)，将使用基础记忆功能"
                )

            # 检查提供商可用性
            if self.plugin_context.has_providers():
                self.logger.info("✅ 检测到可用的提供商")
            else:
                self.logger.info("ℹ️ 未检测到可用提供商，将使用本地模式")

        except (AttributeError, KeyError, TypeError) as e:
            self.logger.error(f"❌ 配置检查失败: {e}")

    def update_components(self):
        """更新组件引用（在初始化完成后调用）"""
        if self.plugin_manager:
            # 从后台初始化器获取组件工厂
            component_factory = (
                self.plugin_manager.background_initializer.get_component_factory()
            )

            # 设置ComponentFactory引用到PluginContext
            self.plugin_context.set_component_factory(component_factory)

            # 获取所有组件
            components = component_factory.get_components()

            # 更新主线程组件引用
            self.vector_store = components.get("vector_store")
            self.cognitive_service = components.get("cognitive_service")
            self.deepmind = components.get("deepmind")
            self.note_service = components.get("note_service")
            self.file_monitor = components.get("file_monitor")

            # 将主线程组件设置给PluginManager
            main_components = {
                "vector_store": self.vector_store,
                "cognitive_service": self.cognitive_service,
                "deepmind": self.deepmind,
                "note_service": self.note_service,
                "file_monitor": self.file_monitor,
            }
            self.plugin_manager.set_main_thread_components(main_components)

    @filter.on_llm_request(priority=40)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        LLM调用前整理记忆并注入到请求中

        Args:
            event: 消息事件
            request: LLM请求对象
        """
        self.logger.debug("开始执行 on_llm_request")
        try:
            # 检查LLM工具是否可用
            if not self.are_llm_tools_enabled():
                self.logger.debug("LLM工具未启用，跳过LLM请求处理")
                return

            # 更新组件引用
            self.update_components()
            self.logger.debug("组件引用已更新")

            # 使用共享的PluginContext处理请求
            result = await self.plugin_manager.handle_llm_request(
                event, request, self.plugin_context
            )
            self.logger.debug(f"handle_llm_request 返回结果: {result}")

            if result["status"] == "waiting":
                self.logger.info("系统正在初始化中，跳过此次LLM请求处理")
                return
            elif result["status"] == "success":
                self.logger.debug("LLM请求处理完成")
            else:
                self.logger.error(
                    f"LLM请求处理失败: {result.get('message', '未知错误')}"
                )

        except (AttributeError, ValueError, RuntimeError) as e:
            self.logger.error(f"LLM_REQUEST failed: {e}")

    @filter.on_llm_response(priority=-100)
    async def on_llm_response(self, event: AstrMessageEvent, response):
        """
        LLM调用后捕获响应数据，存储到event上下文中

        Args:
            event: 消息事件
            response: LLM响应对象
        """
        self.logger.debug("开始执行 on_llm_response - 捕获响应数据")
        try:
            # 将响应数据存储到event上下文中，供after_message_sent使用
            if hasattr(event, "angelmemory_context"):
                try:
                    import json
                    import time

                    context_data = json.loads(event.angelmemory_context)
                    # 添加响应数据
                    context_data["llm_response"] = {
                        "completion_text": getattr(response, "completion_text", str(response))
                        if response
                        else "",
                        "timestamp": time.time(),
                    }
                    event.angelmemory_context = json.dumps(context_data)
                    self.logger.debug("LLM响应数据已存储到event上下文")
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    self.logger.warning(f"存储响应数据失败: {e}")

        except Exception as e:
            self.logger.error(f"on_llm_response failed: {e}")

    @filter.after_message_sent(priority=-100)
    async def after_message_sent(self, event: AstrMessageEvent):
        """
        消息发送后执行记忆整理，不阻塞主线程

        Args:
            event: 消息事件
        """
        self.logger.debug("开始执行 after_message_sent - 记忆整理")
        try:
            # 检查LLM工具是否可用
            if not self.are_llm_tools_enabled():
                self.logger.debug("LLM工具未启用，跳过记忆整理")
                return

            # 更新组件引用
            self.update_components()

            # 检查是否有需要处理的记忆数据
            if not hasattr(event, "angelmemory_context"):
                self.logger.debug("没有记忆上下文，跳过记忆整理")
                return

            # 将记忆整理任务提交到事件循环，但不等待其完成，以避免阻塞主事件流程
            asyncio.create_task(
                self.plugin_manager.handle_memory_consolidation(
                    event, self.plugin_context
                )
            )
            self.logger.debug("记忆整理任务已提交至后台，不等待完成。")

        except Exception as e:
            self.logger.error(f"after_message_sent failed: {e}")

    async def terminate(self) -> None:
        """插件卸载时的清理工作"""
        try:
            self.logger.info("Angel Memory Plugin 正在关闭...")

            # 停止核心服务
            if self.plugin_manager:
                self.plugin_manager.shutdown()

            # 获取最终状态
            status = (
                self.plugin_manager.get_status()
                if self.plugin_manager
                else {"state": "unknown"}
            )
            self.logger.info(
                f"Angel Memory Plugin 已关闭，最终状态: {status.get('state', 'unknown')}"
            )

        except (AttributeError, RuntimeError) as e:
            self.logger.error(f"Angel Memory Plugin: 插件卸载清理失败: {e}")

    def get_plugin_status(self):
        """
        获取插件状态（用于调试）

        Returns:
            dict: 插件状态信息
        """
        if not self.plugin_manager:
            return {"status": "not_initialized"}

        status = self.plugin_manager.get_status()
        # 添加PluginContext信息
        status.update(
            {
                "plugin_context": {
                    "current_provider": self.plugin_context.get_current_provider(),
                    "has_providers": self.plugin_context.has_providers(),
                    "index_dir": str(self.plugin_context.get_index_dir()),
                    "embedding_provider_id": self.plugin_context.get_embedding_provider_id(),
                    "llm_provider_id": self.plugin_context.get_llm_provider_id(),
                    "llm_tools_enabled": self.are_llm_tools_enabled(),
                }
            }
        )
        return status

    def get_plugin_context(self):
        """
        获取PluginContext实例（用于测试和调试）

        Returns:
            PluginContext: 插件上下文实例
        """
        return self.plugin_context

    def are_llm_tools_enabled(self):
        """
        检查LLM工具是否已成功启用

        Returns:
            bool: 如果LLM工具已启用返回True，否则返回False
        """
        return getattr(self, 'llm_tools_enabled', False)
