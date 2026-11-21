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
import copy

from astrbot.api import logger

# 导入核心模块
from .core.plugin_manager import PluginManager
from .core.plugin_context import PluginContextFactory
from .core.version import PLUGIN_VERSION
from .pmemory import pmemory
from .tools.angel_remember import CoreMemoryRememberTool
from .tools.angel_recall import CoreMemoryRecallTool
from .tools.angel_note_read import NoteRecallTool
from .tools.angel_note_create import NoteCreateTool


@register(
    "astrbot_plugin_angel_memory",
    "kawayiYokami",
    "天使的记忆，让astrbot拥有记忆维护系统和开箱即用的知识库检索",
    PLUGIN_VERSION,
    "https://github.com/kawayiYokami/astrbot_plugin_angel_memory"
)
class AngelMemoryPlugin(Star):
    """天使记忆插件主类

    集成DeepMind记忆系统和多格式文档处理能力，为AstrBot提供完整的记忆功能。

    新架构特点：
    - 极速启动：毫秒级启动，所有耗时操作移至后台
    - 智能等待：后台自动检测提供商，有提供商时自动初始化
    - 统一实例管理：核心实例在后台异步任务中于同一事件循环创建
    - 无重复初始化：彻底解决重复初始化和实例不一致问题
    - 线程安全：避免跨线程使用异步组件的竞态条件

    插件启动后异步初始化核心实例，terminate时安全清理资源。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        # 使用 astrbot.api 的 logger
        self.logger = logger

        # 1. 获取插件数据目录（在main.py中获取）
        data_dir = StarTools.get_data_dir("astrbot_plugin_angel_memory")
        self.logger.debug(f"获取到插件数据目录: {data_dir}")

        # 2. 创建统一的PluginContext，包含所有必要资源
        self.plugin_context = PluginContextFactory.create_from_initialization(
            context, config or {}, data_dir
        )

        # 2. 核心实例占位符（将在后台初始化完成后通过ComponentFactory创建）
        self.deepmind = None
        self.memory_runtime = None
        self.memory_store = None
        self.note_index = None
        # 会话ID日志提示：插件启动后每个会话只提示一次（群聊/私聊统一）
        self._conversation_id_logged_once: set[str] = set()
        self._background_tasks: set[asyncio.Task] = set()
        self._is_terminating: bool = False

        # 3. 在主线程获取完整配置（包含提供商信息）
        self._load_complete_config()

        # 4. 初始化插件管理器（极速启动）- 只传递PluginContext
        self.plugin_manager = PluginManager(self.plugin_context)

        # 5. 注册LLM工具
        self.llm_tools_enabled = True  # 标记LLM工具是否启用
        self._llm_tools = []  # 持有引用，供配置保存钩子动态增删 NoteCreateTool
        try:
            llm_tools = [
                CoreMemoryRememberTool(),
                CoreMemoryRecallTool(),
                NoteRecallTool(),
            ]
            note_config = self.plugin_context.get_config("note_assistant", {}) or {}
            if note_config.get("enable_create", True):
                llm_tools.append(NoteCreateTool())

            self._llm_tools = llm_tools
            self.context.add_llm_tools(*llm_tools)
            registered_names = "、".join(tool.name for tool in llm_tools)
            self.logger.debug(f"已注册 LLM 工具：{registered_names}")
        except AttributeError as e:
            self.llm_tools_enabled = False
            self.logger.error(f"注册LLM工具失败，context可能不支持add_llm_tools方法: {e}", exc_info=True)
            self.logger.warning("LLM工具功能已禁用，插件将继续以基础模式运行")
        except Exception as e:
            self.llm_tools_enabled = False
            self.logger.error(f"注册LLM工具时发生异常: {e}", exc_info=True)
            self.logger.warning("LLM工具功能已禁用，插件将继续以基础模式运行")
        self.logger.debug(
            f"天使记忆数据路径设置为: {self.plugin_context.get_index_dir().resolve()}"
        )
        self.logger.debug(
            f"天使记忆实例创建完成 数据目录={self.plugin_context.get_index_dir().resolve()} "
            f"提供商={self.plugin_context.get_current_provider()} "
            f"嵌入={self.plugin_context.get_embedding_provider_id() or '无（纯全文）'} 后台初始化已启动"
        )

        # 6. 注册 WebUI API 路由
        try:
            from .web_api import register_all_routes
            register_all_routes(self.context, self.plugin_context, plugin=self)
            self.logger.debug("已注册 WebUI API 路由（Plugin Pages）")
        except Exception as e:
            self.logger.warning(f"WebUI API 路由注册失败（不影响核心功能）: {e}")

    def on_config_saved(self):
        """WebUI 保存全局配置后的热生效钩子。

        PluginContext.config 持引用、组件动态读取，多数配置自动生效，
        scope map 缓存已由 update_config 刷新；此处只同步启动时固化的
        NoteCreateTool 注册状态（note_assistant.enable_create）。
        DeepMind 的 provider_id 与反思调度参数按次现读，同样即时生效。
        embedding_provider_id / rerank_provider_id 绑定在初始化时创建的
        组件实例上，改动仍需重载插件。
        """
        if not self.llm_tools_enabled:
            self.logger.warning("AngelMemory: LLM 工具未启用，跳过 NoteCreateTool 同步")
            return
        note_config = self.plugin_context.get_config("note_assistant", {}) or {}
        enabled = bool(note_config.get("enable_create", True))
        registered = any(t.name == "angel_note_create" for t in self._llm_tools)
        try:
            if enabled and not registered:
                tool = NoteCreateTool()
                self.context.add_llm_tools(tool)
                self._llm_tools.append(tool)
                self.logger.info("AngelMemory: 已热注册 LLM 工具 angel_note_create")
            elif not enabled and registered:
                self.context.provider_manager.llm_tools.remove_func("angel_note_create")
                self._llm_tools = [t for t in self._llm_tools if t.name != "angel_note_create"]
                self.logger.info("AngelMemory: 已热移除 LLM 工具 angel_note_create")
        except Exception as e:
            self.logger.error(f"AngelMemory: 同步 NoteCreateTool 失败: {e}", exc_info=True)

    def _load_complete_config(self):
        """在主线程检查配置项"""
        try:
            config = self.plugin_context.get_all_config()
            self.logger.debug(f"插件配置加载完成: {list(config.keys())}")

            # 检查关键配置
            embedding_provider_id = self.plugin_context.get_embedding_provider_id()
            if embedding_provider_id:
                self.logger.debug(f"检测到嵌入提供商配置: {embedding_provider_id}")
            else:
                self.logger.debug(
                    "未配置嵌入提供商ID，将按能力自动降级为 BM25-only（向量非必须）"
                )

            llm_provider_id = self.plugin_context.get_llm_provider_id()
            if llm_provider_id:
                self.logger.debug(f"检测到LLM提供商配置: {llm_provider_id}")
            else:
                self.logger.debug(
                    "未配置LLM提供商ID (provider_id)，将使用基础记忆功能"
                )

            # 检查提供商可用性
            if self.plugin_context.has_providers():
                self.logger.debug("检测到可用的提供商")
            else:
                self.logger.debug("未检测到可用提供商，将使用本地模式")

        except (AttributeError, KeyError, TypeError) as e:
            self.logger.error(f"配置检查失败: {e}", exc_info=True)

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
            self.deepmind = components.get("deepmind")
            self.memory_runtime = components.get("memory_runtime")
            self.memory_store = components.get("memory_store")
            self.note_index = components.get("note_index")

            # 将主线程组件设置给PluginManager
            main_components = {
                "deepmind": self.deepmind,
                "memory_runtime": self.memory_runtime,
                "memory_store": self.memory_store,
                "note_index": self.note_index,
                "deepmind": self.deepmind,
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
        await self._log_event_persona(event)
        await self._log_group_id_once(event)
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

            # 注入完成后的请求才是模型真正看到的那一份，在这里存快照供反思配对
            self._remember_provider_request(event, request)

        except (AttributeError, ValueError, RuntimeError) as e:
            self.logger.error(f"LLM请求处理异常: {e}", exc_info=True)

    async def _log_event_persona(self, event: AstrMessageEvent) -> None:
        """每条消息记录一次当前事件的人格名，便于排障。"""
        try:
            conversation_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            persona_name = await self.plugin_context.get_event_persona_name(event)
            self.logger.debug(
                f"[事件人格] 会话ID={conversation_id or '(空)'} 当前人格={persona_name or '(空)'}"
            )
        except Exception as e:
            self.logger.debug(f"事件人格日志记录失败（已忽略）: {e}")

    async def _log_group_id_once(self, event: AstrMessageEvent) -> None:
        """插件启动后每个会话仅记录一次会话ID，便于用户确认配置键。"""
        try:
            conversation_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            if not conversation_id:
                return
            if conversation_id in self._conversation_id_logged_once:
                return

            self._conversation_id_logged_once.add(conversation_id)
            persona_name = await self.plugin_context.get_event_persona_name(event)
            resolved_scope, matched_by, matched_key = (
                self.plugin_context.resolve_memory_scope_with_source(
                    conversation_id, persona_name=persona_name
                )
            )
            scope_mode = self.plugin_context.get_conversation_scope_mode()
            match_desc = {
                "persona": "人格键",
                "conversation": "会话ID键",
                "default": "默认规则",
            }.get(matched_by, matched_by)
            if scope_mode == "group":
                scope_key_example = f'{{"{conversation_id}": "家人"}}'
            else:
                scope_key_example = '{"女友": "恋爱"}'
            self.logger.debug(
                f"[会话分类提示] 分柜方式={scope_mode} 当前人格={persona_name or '(空)'} "
                f"当前会话ID={conversation_id} 命中来源={match_desc} 命中键={matched_key} "
                f"目标scope={resolved_scope}。注意：以下仅为配置示例，不会自动写入。"
                f"conversation_scope_map 示例：{scope_key_example}"
            )
        except Exception as e:
            self.logger.debug(f"会话ID日志记录失败（已忽略）: {e}")

    @filter.on_agent_done(priority=-100)
    async def on_agent_done(self, event: AstrMessageEvent, run_context, response):
        """
        Agent 跑完一轮后成对记录本次请求与回复，供反思取最近 N 份使用。

        选 on_agent_done 而不是 on_llm_response：多带一个 run_context，
        resp 取不到时可以从 run_context.messages 兜底。

        Args:
            event: 消息事件
            run_context: 本轮 agent 运行上下文
            response: LLM响应对象
        """
        import json
        import time

        try:
            resp_text = self._resolve_agent_response_text(run_context, response)

            # 仍写回 llm_response：记忆整理链要靠它取得本轮 response
            if hasattr(event, "angelmemory_context"):
                try:
                    context_data = json.loads(event.angelmemory_context)
                    context_data["llm_response"] = {
                        "completion_text": resp_text,
                        "timestamp": time.time(),
                    }
                    event.angelmemory_context = json.dumps(context_data)
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    self.logger.warning(f"存储响应数据失败: {e}")

            request = event.get_extra("angel_provider_request") or event.get_extra(
                "provider_request"
            )
            sender_id, sender_name = self._event_sender_identity(event)
            event.set_extra(
                "angel_reflection_turn",
                {
                    "req": request,
                    "resp": resp_text,
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "timestamp": time.time(),
                },
            )
            self.logger.debug(
                f"[反思调度] 本轮请求与回复已配对 req={'有' if request is not None else '无'} "
                f"resp_len={len(resp_text)}"
            )
        except Exception as e:
            self.logger.error(f"Agent 完成钩子异常: {e}", exc_info=True)

    @staticmethod
    def _event_sender_identity(event: AstrMessageEvent) -> tuple:
        """取这条消息的发送者身份，供反思剧本署名用。"""
        sender_id = ""
        sender_name = ""

        get_sender_id = getattr(event, "get_sender_id", None)
        if callable(get_sender_id):
            try:
                sender_id = str(get_sender_id() or "").strip()
            except Exception:
                sender_id = ""
        if not sender_id:
            sender_id = str(getattr(event, "sender_id", "") or "").strip()

        get_sender_name = getattr(event, "get_sender_name", None)
        if callable(get_sender_name):
            try:
                sender_name = str(get_sender_name() or "").strip()
            except Exception:
                sender_name = ""
        if not sender_name:
            sender_name = str(getattr(event, "sender_name", "") or "").strip()

        return sender_id, sender_name

    @classmethod
    def _resolve_agent_response_text(cls, run_context, response) -> str:
        """优先取 completion_text，为空则从 run_context.messages 里最后一条 assistant 兜底。"""
        text = ""
        if response is not None:
            text = str(getattr(response, "completion_text", "") or "").strip()
        if text:
            return text

        messages = getattr(run_context, "messages", None) or []
        for message in reversed(list(messages)):
            if str(getattr(message, "role", "") or "").strip() != "assistant":
                continue
            candidate = cls._agent_message_text(getattr(message, "content", ""))
            if candidate.strip():
                return candidate.strip()
        return ""

    @staticmethod
    def _agent_message_text(content) -> str:
        """把 agent 消息内容压成纯文本：兼容字符串、文本块列表与对象。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                    continue
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            return " ".join(parts)
        return str(content or "")

    @staticmethod
    def _remember_provider_request(event: AstrMessageEvent, request) -> None:
        """
        把本轮请求复制一份存进 event，供 on_agent_done 配对使用。

        contexts 是共享的同一个 list，后续轮次还会往里 append，
        只存引用等于让快照自己变，所以必须复制顶层 list；
        里面每条 dict 是历史消息、不会再被改写，不需要深拷贝。
        """
        try:
            snapshot = copy.copy(request)
            if isinstance(getattr(request, "contexts", None), list):
                snapshot.contexts = list(request.contexts)
            event.set_extra("angel_provider_request", snapshot)
        except Exception as e:
            logger.debug(f"[反思调度] 请求快照失败: {e}")

    @filter.after_message_sent(priority=-100)
    async def after_message_sent(self, event: AstrMessageEvent):
        """
        消息发送后执行记忆整理，不阻塞主线程

        Args:
            event: 消息事件
        """
        self.logger.debug("开始执行 after_message_sent - 记忆整理")
        try:
            if self._is_terminating:
                self.logger.debug("插件正在关闭，跳过记忆整理任务提交")
                return

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
            task = asyncio.create_task(
                self.plugin_manager.handle_memory_consolidation(
                    event, self.plugin_context
                )
            )
            self._track_background_task(task)
            self.logger.debug("记忆整理任务已提交至后台，不等待完成。")

        except Exception as e:
            self.logger.error(f"记忆整理提交异常: {e}", exc_info=True)

    def _track_background_task(self, task: asyncio.Task) -> None:
        """追踪后台任务，便于 terminate 阶段统一取消并等待收束。"""
        self._background_tasks.add(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            self._background_tasks.discard(done_task)
            try:
                if done_task.cancelled():
                    return
                exc = done_task.exception()
                if exc is not None:
                    self.logger.error(f"后台任务异常退出: {exc}", exc_info=True)
            except Exception:
                pass

        task.add_done_callback(_cleanup)

    async def terminate(self) -> None:
        """插件卸载时的清理工作"""
        self._is_terminating = True
        try:
            self.logger.debug("Angel Memory Plugin 正在关闭...")

            # 先停止插件内自行提交的后台任务，避免与组件关闭并发冲突。
            pending_tasks = [t for t in self._background_tasks if not t.done()]
            if pending_tasks:
                self.logger.debug(f"检测到待收束后台任务: {len(pending_tasks)} 个，开始取消")
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                self.logger.debug("插件内后台任务已收束")

            # 停笔记扫描线程：它持有 NotesStore，库关掉之后下一轮扫描会撞上
            # ClosedError 并降级，但线程本身还在空转，这里先停干净。
            # 用 getattr：测试会用 object.__new__ 造实例，属性不一定存在。
            note_index = getattr(self, "note_index", None)
            if note_index is not None:
                try:
                    await asyncio.to_thread(note_index.stop)
                    self.logger.debug("笔记索引线程已停止")
                except Exception as e:
                    self.logger.error(f"笔记索引线程停止失败: {e}", exc_info=True)

            # 先给 p-memory 的后台线程发停止信号，但不关库：后面还要停组件，
            # 这段时间足够向量同步线程自己退出，close() 时就不用干等满超时。
            try:
                pmemory.request_stop()
            except Exception as e:
                self.logger.error(f"通知 p-memory 停止失败: {e}", exc_info=True)

            # 停止核心服务
            if self.plugin_manager:
                await self.plugin_manager.shutdown()

            # 获取最终状态
            status = (
                self.plugin_manager.get_status()
                if self.plugin_manager
                else {"state": "unknown"}
            )
            self.logger.info(
                f"Angel Memory Plugin 已关闭，最终状态: {status.get('state', 'unknown')}"
            )

        except Exception as e:
            self.logger.error(f"Angel Memory Plugin: 插件卸载清理失败: {e}", exc_info=True)
        finally:
            await self._close_pmemory()

    async def _close_pmemory(self) -> None:
        """无条件关闭 p-memory 知识库，释放 native 目录锁。

        关闭链（plugin_manager → background_initializer → component_factory）任一环
        没建好或抛异常，都靠这里兜底。pmemory.close() 自身幂等，重复调用无害。
        """
        try:
            await asyncio.to_thread(pmemory.close)
            self.logger.info("p-memory 知识库已关闭")
        except Exception as e:
            self.logger.error(f"p-memory 知识库关闭失败: {e}", exc_info=True)

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
