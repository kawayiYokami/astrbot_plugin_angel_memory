"""
DeepMind潜意识核心模块

这是AI的潜意识系统，在后台默默帮你管理记忆：
- 看到消息时自动回忆相关内容
- 筛选出有用的记忆喂给主意识
- 定期整理记忆，让重要内容不容易忘记
- 就像人睡觉时整理记忆一样
"""

import asyncio
import re
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from .soul.soul_state import SoulState
from .utils.memory_id_resolver import MemoryIDResolver
from ..llm_memory.utils.json_parser import JsonParser
from .session_memory import SessionMemoryManager
from .memory_runtime import MemoryRuntime
from .utils import SmallModelPromptBuilder, MemoryInjector
from .utils.query_processor import get_query_processor
from .services.retrieval_service import DeepMindRetrievalService
from .services.injection_service import DeepMindInjectionService
from .services.sleep_service import DeepMindSleepService
from .services.user_profile_service import UserProfileService
from .utils.feedback_queue import get_feedback_queue
from .reflection_trace import ReflectionTraceBuffer
from .sleep_schedule import seconds_until_next_sleep, should_run_sleep, today_key
from .trace_store import TraceStore

from astrbot.api import logger

# 框架识别图片失败时塞进内容块的原文，反思时靠它认出「有图但没识别出来」
_IMAGE_CAPTION_FAILED = "[Image Captioning Failed]"


@dataclass
class ReflectionInput:
    """反思入口的纯数据载体（与 AstrMessageEvent 解耦）。"""

    session_id: str
    memory_scope: str
    persona_name: str
    latest_user_text: str
    latest_assistant_text: str
    secretary_decision: Dict[str, Any] = field(default_factory=dict)
    chat_records: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: Dict[str, Any] = field(default_factory=dict)


class DeepMind:
    """AI的潜意识系统

    这是AI的潜意识，配合主意识（LLM）工作：
    - 观察阶段：注意用户说了什么，开始回忆相关内容
    - 回忆阶段：从记忆库里找出有用的信息
    - 反馈阶段：好记性加强，坏记性淘汰
    - 睡眠阶段：定期整理记忆，巩固重要内容

    潜意识的工作方式：
    1. 只管会话状态，具体记忆存储交给底层系统
    2. 出错了也不影响你正常聊天
    3. 没有配置AI助手时自动关闭，不消耗资源
    4. 像人的潜识一样，默默工作，不打扰主意识思考
    """

    def __init__(
        self,
        config,
        context,
        note_service,
        plugin_context, # 新增
        memory_runtime: MemoryRuntime,
        provider_id: str = "",
    ):
        """
        初始化AI的潜意识系统

        Args:
            config: 配置信息（比如多久睡一次觉整理记忆）
            context: 聊天机器人的大脑（主意识）
            note_service: 笔记管理器（重要信息专门存放处）
            plugin_context: 插件上下文（新增）
            memory_runtime: 统一记忆运行时（必填）
            provider_id: AI助手的ID，没有的话潜意识就睡觉不干活
        """
        self.config = config
        self.memory_system = memory_runtime
        self.note_service = note_service
        self.context = context
        # provider_id 每次使用时现读（见下方同名属性），这里只留无 plugin_context 时的兜底
        self._provider_id_fallback = str(provider_id or "").strip()
        self.plugin_context = plugin_context # 保存引用

        # 从全局容器获取logger
        self.logger = logger
        self.json_parser = JsonParser()
        # 反思归纳过程追踪（内存保留最近 10 次，供 WebUI 查看与反馈）
        self.reflection_trace = ReflectionTraceBuffer(maxlen=10)
        # 过程追踪持久化（反思与睡眠共用一份 JSONL，追加写入、永不自动删除）
        self.trace_store = TraceStore(self._resolve_trace_base_dir(plugin_context))

        # 记忆行为与笔记参数不在此处快照，全部按次现读（见下方同名属性）

        # 初始化短期记忆，容量倍数每次淘汰前现读，改配置立刻生效
        self.session_memory_manager = SessionMemoryManager(
            capacity_provider=lambda: self.short_term_memory_capacity
        )

        # 睡眠调度状态：每天 04:00 一次，按运行环境本地时间，不做启动补偿
        self._sleep_tick_task: Optional[asyncio.Task] = None
        self._sleep_stop_event = asyncio.Event()
        self._last_sleep_day = ""

        # 初始化工具类
        self.prompt_builder = SmallModelPromptBuilder()
        self.memory_injector = MemoryInjector()
        self.query_processor = get_query_processor()
        self.retrieval_service = DeepMindRetrievalService(self)
        self.injection_service = DeepMindInjectionService(self)
        self.user_profile_service = UserProfileService(
            memory_store=self.plugin_context.get_component("memory_store"),
            logger=self.logger,
        )
        self.sleep_service = DeepMindSleepService(self)
        self._reflection_state_lock = asyncio.Lock()
        self._reflection_states: Dict[str, Dict[str, Any]] = {}
        self._reflection_tick_task: Optional[asyncio.Task] = None
        self._reflection_stop_event = asyncio.Event()
        # 反思调度三项参数同样现读，见下方 _reflection_turn_threshold 等属性

        # 初始化灵魂状态管理器
        try:
            # 将配置对象传递给 SoulState
            self.soul = SoulState(config=self.config)
        except Exception as e:
            self.logger.error(f"灵魂状态管理器初始化失败: {e}")
            self.soul = None

    @property
    def provider_id(self) -> str:
        """LLM 提供商 ID。实时读取配置，WebUI 保存后立即生效，无需重载。"""
        if self.plugin_context is None:
            return self._provider_id_fallback
        try:
            return str(self.plugin_context.get_llm_provider_id() or "").strip()
        except Exception:
            return self._provider_id_fallback

    def _read_group_value(self, section: str, key: str, default: Any) -> Any:
        """实时读取分组配置；分组里没有该项时回落到构造时的扁平旧格式。"""
        if self.plugin_context is not None:
            group = self.plugin_context.get_config(section, {}) or {}
            if isinstance(group, dict) and key in group:
                return group[key]
        return getattr(self.config, key, default)

    def _read_reflection_param(self, key: str, default: int, minimum: int) -> int:
        """实时读取反思调度参数，缺失或非法时回落默认值。"""
        raw = self._read_group_value("memory_behavior", key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
        if value <= 0:
            value = default
        return max(minimum, value)

    @property
    def _reflection_turn_threshold(self) -> int:
        """攒够多少次机器人回话后触发一次反思。"""
        return self._read_reflection_param("reflection_turn_threshold", 6, 1)

    @property
    def _reflection_idle_seconds(self) -> int:
        """空闲多少秒后由调度 tick 触发一次反思。"""
        return self._read_reflection_param("reflection_idle_seconds", 600, 1)

    @property
    def _reflection_tick_seconds(self) -> int:
        """反思调度 tick 的检查间隔。"""
        return self._read_reflection_param("reflection_tick_seconds", 600, 10)

    @property
    def min_message_length(self) -> int:
        """触发记忆处理的最小消息字符数。"""
        raw = self._read_group_value("memory_behavior", "min_message_length", 5)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 5
        return max(0, value)

    @property
    def short_term_memory_capacity(self) -> float:
        """短期记忆槽的容量倍数，每种记忆类型的槽位都乘上它。"""
        raw = self._read_group_value("memory_behavior", "short_term_memory_capacity", 1.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1.0
        if value <= 0:
            value = 1.0
        return min(10.0, value)

    @property
    def note_inject_top_k(self) -> int:
        """笔记注入条数。"""
        raw = self._read_group_value("note_assistant", "top_k", 3)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 3
        return max(0, value)

    @property
    def note_candidate_top_k(self) -> int:
        """笔记候选条数，固定为注入条数的七倍。"""
        return self.note_inject_top_k * 7

    @property
    def note_recall_enabled(self) -> bool:
        """是否允许自动检索并注入笔记。"""
        return bool(self._read_group_value("note_assistant", "enable_recall", True))

    def is_enabled(self) -> bool:
        """检查记忆系统是否可用"""
        return self.memory_system is not None

    def should_process_message(self, event: AstrMessageEvent) -> bool:
        """
        判断这条消息是否值得记住

        Args:
            event: 用户发来的消息

        Returns:
            True=值得记住，False=不用记
        """
        if not self.is_enabled():
            return False

        # 从事件中提取消息文本
        message_text = self._extract_message_text(event)
        if not message_text:
            return False

        message_text = message_text.strip()

        # 检查最小消息长度
        if len(message_text) < self.min_message_length:
            return False

        # 忽略纯指令消息（以/开头）
        if message_text.startswith("/"):
            return False

        return True

    def _parse_memory_context(
        self, event: AstrMessageEvent
    ) -> Optional[Dict[str, Any]]:
        """
        解析事件中的记忆上下文数据

        Args:
            event: 消息事件

        Returns:
            包含 session_id, query, user_list 的字典，解析失败返回 None
        """
        return self.retrieval_service.parse_memory_context(event)

    async def _retrieve_memories_and_notes(
        self, event: AstrMessageEvent, query: str
    ) -> Dict[str, Any]:
        """
        检索长期记忆和候选笔记

        Args:
            event: 消息事件
            query: 查询字符串

        Returns:
            包含 long_term_memories, candidate_notes, note_id_mapping, secretary_decision 的字典
        """
        return await self.retrieval_service.retrieve_memories_and_notes(
            event, query
        )



    async def _inject_memories_to_request(
        self,
        request: ProviderRequest,
        session_id: str,
        note_context: str,
        soul_state_values: Optional[Dict[str, Any]] = None,
        message_id: str = "",
        message_order=None,
    ) -> list:
        """
        将记忆、笔记和灵魂状态统一注入到LLM请求中（使用 extra_user_content_parts）

        返回真正挂进请求的那批记忆。
        """
        return await self.injection_service.inject_memories_to_request(
            request,
            session_id,
            note_context,
            soul_state_values,
            message_id=message_id,
            message_order=message_order,
        )

    def _clean_note_content(self, content: str) -> str:
        """
        清理笔记内容，保留单个换行符，去除双换行符

        Args:
            content: 原始笔记内容

        Returns:
            清理后的笔记内容（保留\n，去除\n\n）
        """
        # 去除首尾空白
        content = content.strip()

        # 将所有连续的换行符（包括空行）替换为单个换行符
        import re
        content = re.sub(r'\n+', '\n', content)

        return content

    async def organize_and_inject_memories(
        self, event: AstrMessageEvent, request: ProviderRequest
    ):
        """
        潜意识的核心工作：整理相关记忆并结合灵魂状态，喂给主意识。
        """
        # 将plugin_context注入到event中，供QueryProcessor使用
        event.plugin_context = self.plugin_context

        session_id = self._get_session_id(event)

        # 1. 从 event.angelheart_context 中获取对话历史
        # 天使之心送来的是「截到秘书判断点为止」的完整快照，里面没有已处理/未处理的分界，
        # 整份记录就是本轮该看的对话，因此不再按 is_processed 过滤。
        chat_records: List[Dict[str, Any]] = []
        context_chat_records: List[Dict[str, Any]] = []
        secretary_decision = {}
        if hasattr(event, "angelheart_context"):
            try:
                angelheart_data = json.loads(event.angelheart_context)
                chat_records = angelheart_data.get("chat_records", []) or []
                if not isinstance(chat_records, list):
                    chat_records = []
                context_chat_records = [
                    msg for msg in chat_records if isinstance(msg, dict)
                ]
                secretary_decision = angelheart_data.get("secretary_decision", {}) or {}
            except (json.JSONDecodeError, KeyError, TypeError):
                self.logger.error(f"为会话 {session_id} 解析 angelheart_context 失败")

        # 2. 从 secretary_decision 构建查询字符串
        query = ""
        user_list = []

        if secretary_decision:
            # 从 secretary_decision 构建查询词
            topic = secretary_decision.get("topic", "")
            entities = secretary_decision.get("entities", [])
            facts = secretary_decision.get("facts", [])
            keywords = secretary_decision.get("keywords", [])

            # 构建查询词：主题 + 实体 + 关键事实 + 关键词
            query_parts = []
            if topic:
                query_parts.append(topic)
            if entities:
                query_parts.extend(entities)
            if facts:
                query_parts.extend(facts[:3])  # 限制事实数量
            if keywords:
                query_parts.extend(keywords[:3])  # 限制关键词数量

            query = " ".join(query_parts)
        else:
            # 降级到原始逻辑
            if not context_chat_records:
                message_text = self._extract_message_text(event)
                query = message_text if message_text else ""
            else:
                query, user_list = self.prompt_builder.format_chat_records(context_chat_records)

        # 3. 检索长期记忆和笔记
        # 召回与注入全是查询和拼装，一次模型都不调，因此不依赖 provider_id。
        # 只有「整理 / 反思」才需要模型，守卫加在那几处（见第 9 步与
        # async_analyze_and_update_memory），不能因为没配模型就把召回一起停掉。
        retrieval_data = await self._retrieve_memories_and_notes(event, query)

        long_term_memories = retrieval_data["long_term_memories"]
        candidate_notes = retrieval_data["candidate_notes"]
        core_topic = retrieval_data["core_topic"]

        # 4. 将检索到的长期记忆填入会话工作缓存，连同带入它的消息一起绑定
        current_message_id = self._event_message_id(event)
        current_message_text = str(self._extract_message_text(event) or "").strip()
        if long_term_memories and self.memory_system:
            self.session_memory_manager.add_memories_to_session(
                session_id,
                long_term_memories,
                bound_message_id=current_message_id,
                bound_message_text=current_message_text,
            )

        # 5. 构建笔记上下文（复用NoteContextBuilder）
        note_context = ""
        if candidate_notes:
            from .utils.note_context_builder import NoteContextBuilder

            # Top-K 注入策略：不再按 token 预算裁剪
            selected_notes = candidate_notes[: max(0, int(self.note_inject_top_k))]

            # 使用 NoteContextBuilder 来构建最终的上下文
            if selected_notes:
                # builder 现在返回包含时效性警告的、格式化的笔记列表
                note_context = NoteContextBuilder.build_candidate_list_for_prompt(selected_notes)

                self.logger.debug(
                    f"笔记上下文构建完成：{len(selected_notes)}条笔记，"
                    f"注入上限K={self.note_inject_top_k}"
                )

        # 6. 获取灵魂状态倾向
        soul_state_values = None
        if hasattr(self, "soul") and self.soul:
            try:
                soul_state_values = self.soul.get_scores()
            except Exception as e:
                self.logger.warning(f"获取灵魂状态倾向失败: {e}")

        # 7. 刷新当前批次用户画像通道
        memory_scope = "public"
        try:
            profile_refresh_started_at = time.time()
            memory_sql_manager = self.plugin_context.get_component("memory_sql_manager")
            memory_store = self.plugin_context.get_component("memory_store")
            if memory_store is not None:
                self.user_profile_service.set_memory_store(memory_store)
                # 登记当前事件用户与群聊到账本
                ledger_sender_id, ledger_sender_name = self._get_event_sender_identity(event)
                if ledger_sender_id and ledger_sender_id.lower() not in {
                    "unknown", "user", "assistant", "none", "null", "0",
                }:
                    try:
                        ledger_platform = ""
                        if hasattr(event, "get_platform_name"):
                            try:
                                ledger_platform = str(event.get_platform_name() or "").strip()
                            except Exception:
                                ledger_platform = ""
                        ledger_group_id = ""
                        if hasattr(event, "get_group_id"):
                            try:
                                ledger_group_id = str(event.get_group_id() or "").strip()
                            except Exception:
                                ledger_group_id = ""

                        def _register_ledger() -> None:
                            memory_sql_manager.upsert_user(
                                ledger_platform, ledger_sender_id, ledger_sender_name
                            )
                            if ledger_group_id:
                                memory_sql_manager.upsert_group(ledger_group_id)

                        await asyncio.to_thread(_register_ledger)
                    except Exception as ledger_e:
                        ledger_elapsed_ms = int((time.time() - profile_refresh_started_at) * 1000)
                        self.logger.warning(
                            f"[用户账本] 登记失败 任务=事件身份登记 触发条件=对话注入前 "
                            f"sender_id={ledger_sender_id} 耗时毫秒={ledger_elapsed_ms} 错误={ledger_e}",
                            exc_info=True,
                        )
            memory_scope = await self.plugin_context.resolve_memory_scope_from_event(event)
            await self.user_profile_service.refresh_session_profiles(
                session_id=session_id,
                chat_records=context_chat_records,
                fallback_sender_id=str(getattr(event, "sender_id", "") or ""),
                fallback_sender_name=str(getattr(event, "sender_name", "") or ""),
                memory_scope=memory_scope,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - profile_refresh_started_at) * 1000)
            self.logger.warning(
                f"[用户画像] 失败 任务=刷新当前批次画像 触发条件=对话注入前 "
                f"session={session_id} 当前批次消息数={len(context_chat_records)} "
                f"耗时毫秒={elapsed_ms} 错误={e}",
                exc_info=True,
            )

        # 8. 注入记忆、笔记和灵魂状态到请求（注入源是短期记忆，不重新检索）
        alive_ids = self._alive_message_ids(context_chat_records, current_message_id)
        if alive_ids is not None:
            departed_items = (
                self.session_memory_manager.prune_absent_messages_from_session(
                    session_id, alive_ids
                )
            )
            if departed_items:
                self.logger.debug(
                    f"[短期记忆] 清理离场记忆条目 session={session_id} count={len(departed_items)}"
                )
                # 离场反思要把记忆送去模型做强化/合并，没配核心模型就跳过，
                # 否则只会提交一个注定失败的后台任务。
                if self.provider_id:
                    await self._reflect_departed_memories(
                        session_id,
                        memory_scope,
                        current_message_id,
                        departed_items,
                    )
                else:
                    self.logger.debug(
                        f"[反思调度] 未配置核心模型，跳过离场记忆送审 "
                        f"session={session_id} 记忆数={len(departed_items)}"
                    )
        injected_memories = await self._inject_memories_to_request(
            request,
            session_id,
            note_context,
            soul_state_values,
            message_id=current_message_id,
            message_order=self._message_order(context_chat_records),
        )

        # 9. (异步任务所需) 将原始上下文数据存入event.angelmemory_context
        #     账目按实际挂进请求的那批记忆写，反思强化对象与模型看到的一致
        try:
            injected_memories = list(injected_memories or [])
            memory_sql_manager = self.plugin_context.get_component("memory_sql_manager")
            memory_id_mapping = (
                memory_sql_manager.build_id_mapping([mem.id for mem in injected_memories])
                if memory_sql_manager
                else MemoryIDResolver.generate_id_mapping([mem.to_dict() for mem in injected_memories], "id")
            )
            angelmemory_context = {
                "memories": self._memories_to_json(self.session_memory_manager.get_session_memories(session_id)),
                "recall_query": query,
                "recall_time": time.time(),
                "session_id": session_id,
                "user_list": user_list,
                "raw_chat_records": context_chat_records,
                "raw_memories": [memory.to_dict() for memory in injected_memories],
                "raw_notes": candidate_notes,
                "core_topic": core_topic,
                "memory_id_mapping": memory_id_mapping,
                "note_id_mapping": {}
            }
            event.angelmemory_context = json.dumps(angelmemory_context)
        except Exception as e:
            self.logger.error(f"保存原始上下文数据以供异步分析失败: {e}")

    def _extract_message_text(self, event: AstrMessageEvent) -> Optional[str]:
        """
        从事件中提取消息文本 (使用标准方法)

        Args:
            event: 消息事件

        Returns:
            消息文本，如果无法提取则返回None
        """
        try:
            # 参照 AngelHeart 的正确实现
            return event.get_message_outline()
        except Exception as e:
            self.logger.error(f"潜意识: 调用 event.get_message_outline() 失败: {e}")
            return None

    @staticmethod
    def _event_message_id(event: AstrMessageEvent) -> str:
        """当前这条用户消息的 ID，用于注入幂等判定。"""
        message_obj = getattr(event, "message_obj", None)
        return str(getattr(message_obj, "message_id", "") or "").strip()

    @staticmethod
    def _chat_record_message_id(record: Dict[str, Any]) -> str:
        """账本记录里带入这条消息的原始消息 ID。"""
        return str(record.get("source_message_id") or "").strip()

    def _latest_user_message(self, context_data) -> tuple:
        """反思链路里触发本次回复的那条用户消息，返回 (消息 ID, 消息纯文本)。"""
        records = (context_data or {}).get("raw_chat_records") or []
        for record in reversed(records):
            if str(record.get("role") or "") != "user":
                continue
            message_id = self._chat_record_message_id(record)
            if message_id:
                return message_id, self.injection_service._plain_text(
                    record.get("content")
                )
        return "", ""

    def _alive_message_ids(
        self, chat_records: List[Dict[str, Any]], current_message_id: str
    ) -> Optional[set]:
        """本次请求里真实存在的用户消息 ID 集合。

        有天使之心账本时是账本里的用户消息，没有账本时只剩当前这一条——老记忆因此
        找不到绑定消息，随下一次清理离开短期记忆。判定不了当前消息 ID 时返回 None，
        表示不做清理。
        """
        current = str(current_message_id or "").strip()
        if not current:
            return None
        alive = {current}
        for record in chat_records or []:
            if str(record.get("role") or "") != "user":
                continue
            message_id = self._chat_record_message_id(record)
            if message_id:
                alive.add(message_id)
        return alive

    def _message_order(self, chat_records: List[Dict[str, Any]]) -> List[tuple]:
        """账本里用户消息的顺序表，元素为 (消息 ID, 消息纯文本)。"""
        plain = self.injection_service._plain_text
        order = []
        for record in chat_records or []:
            if str(record.get("role") or "") != "user":
                continue
            message_id = self._chat_record_message_id(record)
            if not message_id:
                continue
            order.append((message_id, plain(record.get("content"))))
        return order

    async def _reflect_departed_memories(
        self,
        session_id: str,
        memory_scope: str,
        current_message_id: str,
        departed_items: List[Any],
    ) -> None:
        """绑定的消息离场时，把那条消息和它带回的记忆一起送去反思。"""
        if not departed_items or not self.memory_system:
            return

        memories = []
        for item in departed_items:
            memory = getattr(item, "memory", None)
            if memory is None:
                continue
            memory_id = str(getattr(memory, "id", "") or "").strip()
            if memory_id and all(
                str(getattr(existing, "id", "") or "") != memory_id
                for existing in memories
            ):
                memories.append(memory)
        if not memories:
            return

        texts: List[str] = []
        for item in departed_items:
            text = str(getattr(item, "bound_message_text", "") or "").strip()
            if text and text not in texts:
                texts.append(text)
        message_text = "\n".join(texts)
        records = (
            [
                {
                    "role": "user",
                    "content": message_text,
                    "source_message_id": current_message_id,
                }
            ]
            if message_text
            else []
        )

        memory_sql_manager = self.plugin_context.get_component("memory_sql_manager")
        memory_id_mapping = (
            memory_sql_manager.build_id_mapping([memory.id for memory in memories])
            if memory_sql_manager
            else MemoryIDResolver.generate_id_mapping(
                [memory.to_dict() for memory in memories], "id"
            )
        )
        reflection_input = ReflectionInput(
            session_id=session_id,
            memory_scope=memory_scope,
            persona_name="",
            latest_user_text=message_text,
            latest_assistant_text="",
            chat_records=list(records),
            memory_context={
                "session_id": session_id,
                "query": message_text,
                "user_list": [],
                "raw_chat_records": list(records),
                "raw_memories": [memory.to_dict() for memory in memories],
                "raw_notes": [],
                "core_topic": "",
                "memory_id_mapping": memory_id_mapping,
                "note_id_mapping": {},
            },
        )

        submitted = await get_feedback_queue().submit(
            {
                "feedback_fn": self._execute_async_analysis_task,
                "session_id": session_id,
                "payload": {
                    "reflection_input": reflection_input,
                    "historical_chat_text_override": message_text,
                },
            }
        )
        self.logger.info(
            f"[反思调度] 离场记忆送审 session={session_id} "
            f"消息数={len(texts)} 记忆数={len(memories)} "
            f"结果={'已提交' if submitted else '提交失败'}"
        )

    def _get_event_sender_identity(self, event: AstrMessageEvent) -> tuple[str, str]:
        sender_id = ""
        sender_name = ""

        get_sender_id = getattr(event, "get_sender_id", None)
        if callable(get_sender_id):
            try:
                sender_id = str(get_sender_id() or "").strip()
            except Exception as e:
                self.logger.error(f"[反思调度] 获取事件发送者ID失败: {e}")
        if not sender_id:
            sender_id = str(getattr(event, "sender_id", "") or "").strip()

        get_sender_name = getattr(event, "get_sender_name", None)
        if callable(get_sender_name):
            try:
                sender_name = str(get_sender_name() or "").strip()
            except Exception as e:
                self.logger.error(f"[反思调度] 获取事件发送者昵称失败: {e}")
        if not sender_name:
            sender_name = str(getattr(event, "sender_name", "") or "").strip()

        return sender_id, sender_name

    def _get_session_id(self, event: AstrMessageEvent) -> str:
        """获取会话ID，统一使用 event.unified_msg_origin"""
        try:
            session_id = str(event.unified_msg_origin)
            if not session_id:
                self.logger.error("event.unified_msg_origin 为空值，无法处理会话！")
                raise ValueError(
                    "Cannot process event with an empty session ID from unified_msg_origin"
                )
            return session_id
        except AttributeError:
            self.logger.error(
                "事件中缺少 'event.unified_msg_origin' 属性，无法确定会话ID！"
            )
            raise

    def _memories_to_json(self, memories: List) -> List[Dict[str, Any]]:
        """
        将记忆对象统一转换为JSON格式

        适用于BaseMemory、MemoryItem等所有记忆对象类型

        Args:
            memories: 记忆对象列表

        Returns:
            JSON格式的记忆列表
        """
        memories_json = []
        for memory in memories:
            # 处理枚举类型的memory_type（兼容BaseMemory和MemoryItem）
            raw_memory_type = getattr(memory, "memory_type", "")
            memory_type = (
                raw_memory_type.value
                if hasattr(raw_memory_type, "value")
                else raw_memory_type
            )

            memory_data = {
                "id": getattr(memory, "id", ""),
                "type": memory_type,
            }
            if hasattr(memory, "priority_points"):
                memory_data["priority_points"] = getattr(memory, "priority_points", 0)
            if hasattr(memory, "added_at"):
                memory_data["added_at"] = getattr(memory, "added_at", 0.0)
            if hasattr(memory, "user_id"):
                memory_data["user_id"] = getattr(memory, "user_id", "")
            for field_name in ("strength", "judgment", "reasoning", "tags"):
                if hasattr(memory, field_name):
                    memory_data[field_name] = getattr(memory, field_name)
            memories_json.append(memory_data)

        return memories_json

    # _resolve_memory_ids 方法已移至 MemoryIDResolver 类中

    async def start_sleep_scheduler(self) -> None:
        """启动睡眠调度循环：每天 04:00 执行一次，不做启动补偿。"""
        if not self.is_enabled():
            self.logger.info("[睡眠] 调度 跳过（DeepMind 未启用）")
            return
        if self._sleep_tick_task and not self._sleep_tick_task.done():
            return
        self._sleep_stop_event.clear()
        self._sleep_tick_task = asyncio.create_task(self._sleep_tick_loop())
        next_at = time.time() + seconds_until_next_sleep()
        self.logger.debug(
            "[睡眠] 调度 开始 下次执行="
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_at))}"
        )

    async def _sleep_tick_loop(self) -> None:
        """常驻循环：等到下一个 04:00 触发一次睡眠，漏跑不补偿。"""
        while not self._sleep_stop_event.is_set():
            delay = seconds_until_next_sleep()
            try:
                await asyncio.wait_for(self._sleep_stop_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            if self._sleep_stop_event.is_set():
                return
            await self._run_scheduled_sleep()

    async def _run_scheduled_sleep(self) -> None:
        if not should_run_sleep(self._last_sleep_day):
            self.logger.info(f"[睡眠] 调度 跳过（今日已执行）date={today_key()}")
            return
        self.logger.debug(f"[睡眠] 调度 触发 date={today_key()}")
        try:
            await self.sleep_service.sleep()
        finally:
            self._last_sleep_day = today_key()

    def _complete_reflection_trace(
        self,
        trace_id: Optional[int],
        *,
        status: str,
        response: str = "",
        result: Any = None,
        error: str = "",
    ) -> None:
        """回填内存 trace 并落盘一份持久化记录，落盘失败不影响主流程。"""
        self.reflection_trace.complete(
            trace_id,
            status=status,
            response=response,
            result=result,
            error=error,
        )
        if trace_id is None:
            return
        try:
            entry = next(
                (
                    item
                    for item in self.reflection_trace.snapshot()
                    if item.get("id") == trace_id
                ),
                None,
            )
            if entry is None:
                return
            self.trace_store.append_sync(
                {
                    "type": "reflection",
                    "status": entry.get("status", status),
                    "timestamp": entry.get("timestamp", 0.0),
                    "session_id": entry.get("session_id", ""),
                    "provider_id": entry.get("provider_id", ""),
                    "prompt": entry.get("prompt", ""),
                    "system_prompt": entry.get("system_prompt", ""),
                    "response": entry.get("response", ""),
                    "result": entry.get("result"),
                    "error": entry.get("error", ""),
                }
            )
        except Exception as e:
            self.logger.warning(f"[过程追踪] 反思记录落盘失败: {e}")

    @staticmethod
    def _resolve_trace_base_dir(plugin_context) -> str:
        try:
            return str(plugin_context.get_memory_center_dir())
        except Exception:
            return "."

    async def _sleep(self):
        """AI睡觉整理记忆：重要内容加强，无用内容清理"""
        await self.sleep()

    async def sleep(self):
        """公共睡眠入口，供外部组件触发睡眠流程。"""
        await self.sleep_service.sleep()

    def shutdown(self):
        """关闭潜意识系统，让AI好好休息"""
        try:
            self._reflection_stop_event.set()
            if self._reflection_tick_task and not self._reflection_tick_task.done():
                self._reflection_tick_task.cancel()
            self._sleep_stop_event.set()
            if self._sleep_tick_task and not self._sleep_tick_task.done():
                self._sleep_tick_task.cancel()
        except Exception:
            pass

        # 停止记忆整理任务
        from .utils.feedback_queue import stop_feedback_queue

        stop_feedback_queue()

    async def async_analyze_and_update_memory(self, event: AstrMessageEvent, response):
        """
        异步分析并更新记忆系统

        Args:
            event: 消息事件
            response: 主LLM的响应
        """
        # 整理新记忆整条链（攒轮次、调度 tick、反思送审）最终都要调模型识别该记什么。
        # 没配核心模型就直接收工：召回与注入不走这里，不受影响。
        if not self.provider_id:
            self.logger.debug("[反思调度] 未配置核心模型，跳过本次记忆整理")
            return

        # 获取会话ID
        session_id = self._get_session_id(event)
        await self._ensure_reflection_tick_task()
        await self._record_reflection_turn(event, response, session_id)
        await self._trigger_reflection_if_needed(session_id, reason="count")

    async def _ensure_reflection_tick_task(self) -> None:
        if self._reflection_tick_task and not self._reflection_tick_task.done():
            return
        self._reflection_stop_event.clear()
        self._reflection_tick_task = asyncio.create_task(self._reflection_tick_loop())
        self.logger.debug(
            f"[反思调度] 开始 tick={self._reflection_tick_seconds}s "
            f"idle={self._reflection_idle_seconds}s turns={self._reflection_turn_threshold}"
        )

    async def _reflection_tick_loop(self) -> None:
        while not self._reflection_stop_event.is_set():
            try:
                await asyncio.sleep(self._reflection_tick_seconds)
            except asyncio.CancelledError:
                break

            now = time.time()
            async with self._reflection_state_lock:
                session_ids = [
                    sid
                    for sid, state in self._reflection_states.items()
                    if int(state.get("pending_turns", 0)) > 0
                    and not bool(state.get("processing", False))
                    and (now - float(state.get("last_activity_at", 0.0)))
                    >= float(self._reflection_idle_seconds)
                ]
            if session_ids:
                self.logger.debug(
                    f"[反思调度] tick扫描命中会话数={len(session_ids)} "
                    f"idle阈值={self._reflection_idle_seconds}s"
                )

            for sid in session_ids:
                await self._trigger_reflection_if_needed(sid, reason="idle")

    async def _build_reflection_input(
        self,
        event: AstrMessageEvent,
        response,
        session_id: str,
    ) -> ReflectionInput:
        """
        从主线程事件中提取反思所需最小数据结构，避免反思执行阶段依赖 AstrMessageEvent。
        """
        turn = event.get_extra("angel_reflection_turn") or {}
        if not isinstance(turn, dict):
            turn = {}
        latest_user_text = self._request_user_text(turn.get("req"))
        latest_assistant_text = str(turn.get("resp") or "").strip()

        secretary_decision: Dict[str, Any] = {}
        if hasattr(event, "angelheart_context") and getattr(event, "angelheart_context", None):
            try:
                angelheart_data = json.loads(event.angelheart_context)
                sd = angelheart_data.get("secretary_decision", {}) or {}
                if isinstance(sd, dict):
                    secretary_decision = sd
            except (json.JSONDecodeError, TypeError, KeyError):
                secretary_decision = {}

        memory_context: Dict[str, Any] = {}
        if hasattr(event, "angelmemory_context") and getattr(event, "angelmemory_context", None):
            try:
                context_data = json.loads(event.angelmemory_context)
                if isinstance(context_data, dict):
                    memory_context = {
                        "session_id": context_data.get("session_id", session_id),
                        "query": context_data.get("recall_query", ""),
                        "user_list": context_data.get("user_list", []),
                        "raw_chat_records": context_data.get("raw_chat_records", []),
                        "raw_memories": context_data.get("raw_memories", []),
                        "raw_notes": context_data.get("raw_notes", []),
                        "core_topic": context_data.get("core_topic", ""),
                        "memory_id_mapping": context_data.get("memory_id_mapping", {}),
                        "note_id_mapping": context_data.get("note_id_mapping", {}),
                    }
            except (json.JSONDecodeError, TypeError, KeyError):
                memory_context = {}

        try:
            memory_scope = await self.plugin_context.resolve_memory_scope_from_event(event)
        except Exception:
            memory_scope = "public"

        try:
            persona_name = await self.plugin_context.get_event_persona_name(event)
        except Exception:
            persona_name = ""

        return ReflectionInput(
            session_id=session_id,
            memory_scope=memory_scope,
            persona_name=persona_name,
            latest_user_text=latest_user_text,
            latest_assistant_text=latest_assistant_text,
            secretary_decision=secretary_decision,
            memory_context=memory_context,
        )

    @staticmethod
    def _turn_memories(reflection_input) -> List[Dict[str, Any]]:
        """取这一轮真正注入给模型的记忆批次。"""
        context_data = getattr(reflection_input, "memory_context", {}) or {}
        if not isinstance(context_data, dict):
            return []
        candidate = context_data.get("raw_memories", [])
        if not isinstance(candidate, list):
            return []
        return [memory for memory in candidate if isinstance(memory, dict)]

    @staticmethod
    def _merge_turn_memories(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把多轮检索到的记忆按 id 合并去重，保持首次出现的顺序。

        同一份记忆在多个轮次被召回只保留一条；没有 id 的条目用内容指纹兜底，
        避免把它们全折叠成一条。
        """
        merged: List[Dict[str, Any]] = []
        seen = set()
        for turn in turns or []:
            if not isinstance(turn, dict):
                continue
            for memory in turn.get("memories", []) or []:
                if not isinstance(memory, dict):
                    continue
                key = str(memory.get("id", "") or "").strip()
                if not key:
                    key = "fp:" + json.dumps(
                        memory, sort_keys=True, ensure_ascii=False, default=str
                    )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(memory)
        return merged

    def _request_user_text(self, request) -> str:
        """取一轮的用户输入：优先 prompt，其次附件转出来的文本。

        contexts 不能当兜底：纯媒体轮 message_str 为空时，
        contexts 里最后一条 user 是上一轮那个人的话，会张冠李戴。
        extra_user_content_parts 也不能整段拼：知识库结果、系统提醒、
        引用块都塞在里面，只有附件转出来的那份才是用户这轮发来的东西。
        """
        if request is None:
            return ""
        prompt = str(getattr(request, "prompt", "") or "").strip()
        if prompt:
            return prompt

        parts = getattr(request, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            return ""
        collected: List[str] = []
        for part in parts:
            text = ""
            if isinstance(part, dict):
                if str(part.get("type", "") or "") == "text":
                    text = str(part.get("text", "") or "")
            elif str(getattr(part, "type", "") or "") == "text":
                text = str(getattr(part, "text", "") or "")
            attachment = self._attachment_text(text)
            if attachment:
                collected.append(attachment)
        return " ".join(collected)

    @staticmethod
    def _attachment_text(text: str) -> str:
        """只认附件转出来的文字，别的内容块一律不算用户输入。

        框架把图片描述写成 <image_caption>...</image_caption>；同一字段里还混着
        [Related Knowledge Base Results]、<system_reminder>、<Quoted Message>、
        <selected_excerpt>，那些都不是用户说的话，混进反思会认错人。
        """
        raw = str(text or "").strip()
        if not raw:
            return ""
        captions = [
            item.strip()
            for item in re.findall(r"<image_caption>(.*?)</image_caption>", raw, re.S)
            if item.strip()
        ]
        if captions:
            return " ".join(captions)
        if raw == _IMAGE_CAPTION_FAILED:
            return raw
        return ""

    def _turns_to_chat_records(
        self, turns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """把成对记录摊成 format_chat_records 认得的 chat_records。

        一份记录产出「用户 → 我」两条；助理那条加 1 毫秒，
        保证同一轮内先用户后回复，且不影响跨轮排序。
        """
        records: List[Dict[str, Any]] = []
        for turn in turns or []:
            if not isinstance(turn, dict):
                continue
            timestamp = float(turn.get("timestamp", 0.0) or 0.0)
            user_text = self._request_user_text(turn.get("req"))
            resp_text = str(turn.get("resp") or "").strip()
            if user_text:
                records.append(
                    {
                        "role": "user",
                        "content": user_text,
                        "sender_id": str(turn.get("sender_id") or "").strip(),
                        "sender_name": str(turn.get("sender_name") or "").strip(),
                        "timestamp": timestamp,
                        "is_structured_toolcall": False,
                    }
                )
            else:
                self.logger.debug(
                    f"[反思调度] 本轮没有用户文本，跳过 user 条目 "
                    f"session={turn.get('session_id', '')} timestamp={timestamp}"
                )
            if resp_text:
                records.append(
                    {
                        "role": "assistant",
                        "content": resp_text,
                        "sender_id": "assistant",
                        "sender_name": "助理",
                        "timestamp": timestamp + 0.001,
                        "is_structured_toolcall": False,
                    }
                )
        return records

    def _reflection_window_limit(self) -> int:
        """反思窗口保留上限：至少 2 倍阈值且不少于 20 份。

        留 2 倍是为了空闲触发时窗口不短于阈值；下限 20 防止阈值很小时
        窗口过窄，长会话也不会让窗口无限增长。
        """
        return max(int(self._reflection_turn_threshold) * 2, 20)

    async def _record_reflection_turn(
        self, event: AstrMessageEvent, response, session_id: str
    ) -> None:
        """把本轮 (req, resp) 成对记进该会话的滑动窗口。

        一次请求 = 一份记录。不做合并、不去重、不排序：
        turns 按写入顺序追加，反思时直接取最近 N 份。
        """
        reflection_input = await self._build_reflection_input(
            event, response, session_id
        )
        turn = event.get_extra("angel_reflection_turn") or {}
        if not isinstance(turn, dict):
            turn = {}
        request = turn.get("req")
        resp_text = str(turn.get("resp") or "").strip()

        if request is None and not resp_text:
            self.logger.warning(
                f"[反思调度] 本轮请求与回复均缺失，不记录 会话={session_id} "
                f"原因=既没有 provider_request 也没有回复文本"
            )
            return

        record = {
            "session_id": session_id,
            "timestamp": float(turn.get("timestamp") or time.time()),
            "req": request,
            "resp": resp_text,
            "sender_id": str(turn.get("sender_id") or "").strip(),
            "sender_name": str(turn.get("sender_name") or "").strip(),
            "memories": self._turn_memories(reflection_input),
        }
        now = time.time()
        limit = self._reflection_window_limit()

        async with self._reflection_state_lock:
            state = self._reflection_states.setdefault(
                session_id,
                {
                    "pending_turns": 0,
                    "last_activity_at": 0.0,
                    "turns": [],
                    "latest_input": None,
                    "processing": False,
                },
            )
            turns = list(state.get("turns", []))
            turns.append(record)
            if len(turns) > limit:
                turns = turns[-limit:]
            state["turns"] = turns
            state["pending_turns"] = int(state.get("pending_turns", 0)) + 1
            state["last_activity_at"] = now
            state["latest_input"] = reflection_input
            self.logger.debug(
                f"[反思调度] 记录本轮 session={session_id} "
                f"pending_turns={state['pending_turns']} turns={len(turns)}/{limit}"
            )

    async def _trigger_reflection_if_needed(self, session_id: str, reason: str) -> bool:
        now = time.time()
        payload: Dict[str, Any] = {}
        async with self._reflection_state_lock:
            state = self._reflection_states.get(session_id)
            if not state:
                return False
            if bool(state.get("processing", False)):
                return False

            pending_turns = int(state.get("pending_turns", 0))
            if pending_turns <= 0:
                return False
            idle_elapsed = now - float(state.get("last_activity_at", 0.0))
            meets_count = pending_turns >= int(self._reflection_turn_threshold)
            meets_idle = idle_elapsed >= float(self._reflection_idle_seconds)
            if not (meets_count or meets_idle):
                if reason == "count":
                    self.logger.debug(
                        f"[反思调度] 未触发 session={session_id} "
                        f"pending={pending_turns}/{self._reflection_turn_threshold} "
                        f"idle={int(idle_elapsed)}s/{self._reflection_idle_seconds}s"
                    )
                return False

            state["processing"] = True
            latest_input = state.get("latest_input")
            if latest_input is None:
                state["processing"] = False
                return False
            window = int(self._reflection_turn_threshold)
            recent_turns = list(state.get("turns", []))[-window:]
            payload = {
                "reflection_input": latest_input,
                "recent_turns": recent_turns,
                "merged_memories": self._merge_turn_memories(recent_turns),
                "consumed_turns": pending_turns,
            }
            state["pending_turns"] = 0

        historical_chat_text = ""
        try:
            records = self._turns_to_chat_records(payload.get("recent_turns", []))
            historical_chat_text, _ = self.prompt_builder.format_chat_records(records)
        except Exception as e:
            self.logger.warning(f"格式化反思聊天记录失败，降级为空: {e}")
            historical_chat_text = ""

        self.logger.debug(
            f"[反思调度] 触发 reason={reason} session={session_id} "
            f"pending={payload.get('consumed_turns', 0)} "
            f"取用轮次={len(payload.get('recent_turns', []))} "
            f"合并记忆={len(payload.get('merged_memories', []))}"
        )
        success = await get_feedback_queue().submit(
            {
                "feedback_fn": self._execute_async_analysis_task,
                "session_id": session_id,
                "payload": {
                    "reflection_input": payload.get("reflection_input"),
                    "historical_chat_text_override": historical_chat_text,
                    "raw_memories_override": payload.get("merged_memories", []),
                },
            }
        )

        async with self._reflection_state_lock:
            state = self._reflection_states.get(session_id)
            if not state:
                return bool(success)
            state["processing"] = False
            # 执行失败只回滚待反思计数：turns 是滑动窗口，取用时并不消费它。
            if not bool(success):
                state["pending_turns"] = int(state.get("pending_turns", 0)) + int(
                    payload.get("consumed_turns", 0) or 0
                )
                self.logger.warning(
                    f"[反思调度] 执行失败后回滚 session={session_id} "
                    f"pending_turns={state['pending_turns']} "
                    f"turns={len(state.get('turns', []))}"
                )
            else:
                self.logger.debug(f"[反思调度] 完成 session={session_id}")
        return bool(success)

    async def _execute_async_analysis_task(
        self,
        reflection_input: ReflectionInput,
        historical_chat_text_override: str = "",
        raw_memories_override: List[Dict[str, Any]] = None,
    ):
        """
        异步执行的记忆分析任务

        Args:
            reflection_input: 反思输入纯数据
        """
        trace_id = None
        try:
            session_id = str(getattr(reflection_input, "session_id", "") or "").strip()
            if not session_id:
                return False
            self.logger.debug(
                f"[反思执行] 开始 session={session_id} "
                f"user_len={len(str(getattr(reflection_input, 'latest_user_text', '') or ''))} "
                f"assistant_len={len(str(getattr(reflection_input, 'latest_assistant_text', '') or ''))}"
            )

            context_data = getattr(reflection_input, "memory_context", {}) or {}
            if not isinstance(context_data, dict):
                context_data = {}

            query = str(context_data.get("query", "") or "")
            historical_chat_text = str(historical_chat_text_override or "").strip()
            if not historical_chat_text.strip():
                historical_chat_text = query
            if not historical_chat_text.strip():
                historical_chat_text = str(
                    getattr(reflection_input, "latest_user_text", "") or ""
                ).strip()

            # 获取原始记忆数据：优先用多轮合并去重后的那批，
            # 否则 N 轮反思只能看到最后一轮检索出来的记忆，前几轮的拿不到反馈。
            if raw_memories_override is not None:
                raw_memories_data = list(raw_memories_override)
            else:
                raw_memories_data = context_data.get("raw_memories", [])
            core_topic = context_data.get("core_topic", "")

            # 将原始数据转换为记忆对象
            from ..llm_memory.models.data_models import BaseMemory

            long_term_memories = []
            for memory_dict in raw_memories_data:
                try:
                    memory = BaseMemory.from_dict(memory_dict)
                    if memory:
                        long_term_memories.append(memory)
                except Exception as e:
                    self.logger.error(f"转换记忆对象失败: {e}")

            # 获取主LLM的最终回答
            response_text = str(
                getattr(reflection_input, "latest_assistant_text", "") or ""
            )


            # 从上下文数据中获取ID映射表
            memory_id_mapping = context_data.get("memory_id_mapping", {})

            # 构建反思提示词（只传递记忆数据，不传递笔记）
            # system 放固定指南、user 放本次数据，便于命中提示词缓存
            system_prompt, prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompts(
                historical_query=historical_chat_text,
                main_llm_response=response_text,
                raw_memories=long_term_memories,
                core_topic=core_topic,
                memory_id_mapping=memory_id_mapping,
                config=self.config,
            )

            self.logger.debug(
                f"[反思执行] 提示词 session={session_id} "
                f"system_len={len(system_prompt)} prompt_len={len(prompt)}"
            )
            trace_id = self.reflection_trace.record_start(
                session_id, self.provider_id, prompt, system_prompt
            )

            # 调用小模型进行分析（在后台线程中同步调用）
            provider = self.context.get_provider_by_id(self.provider_id)
            if not provider:
                self.logger.error(
                    f"找不到提供者: {self.provider_id}，会话: {session_id}"
                )
                self._complete_reflection_trace(
                    trace_id,
                    status="failed",
                    error=f"找不到提供者: {self.provider_id}",
                )
                return

            try:
                # 直接异步调用，无需检查
                llm_response = await provider.text_chat(
                    prompt=prompt, system_prompt=system_prompt
                )
            except Exception as e:
                self.logger.error(
                    f"会话 {session_id} 的LLM调用失败，跳过记忆整理: {e}",
                    exc_info=True,
                )
                self._complete_reflection_trace(
                    trace_id, status="failed", error=f"LLM调用失败: {e}"
                )
                return

            if not llm_response or not getattr(llm_response, "completion_text", ""):
                self.logger.error(f"会话 {session_id} 的LLM API调用失败")
                self._complete_reflection_trace(
                    trace_id, status="failed", error="LLM API调用返回空响应"
                )
                return

            # 提取响应文本
            response_text = llm_response.completion_text

            # 解析完整的结构化输出
            full_json_data = self.json_parser.extract_json(response_text)

            if not isinstance(full_json_data, dict):
                self.logger.error(
                    f"会话 {session_id} 的JSON解析失败或未返回字典"
                )
                self._complete_reflection_trace(
                    trace_id, status="failed", error="JSON解析失败或未返回字典"
                )
                return

            # 提取 feedback_data
            feedback_data = full_json_data.get("feedback_data", {})

            # --- 灵魂状态更新 (Feedback Loop) ---
            if hasattr(self, "soul") and self.soul and "soul_state" in full_json_data:
                soul_scores = full_json_data.get("soul_state") or {}
                if isinstance(soul_scores, dict) and soul_scores:
                    self.soul.reflect(soul_scores)
                    self.logger.debug(
                        f"灵魂反思: {self.soul.get_state_description()}"
                    )
                else:
                    self.logger.warning("反思结果里的 soul_state 不是有效字典，跳过情绪更新")

            # ID解析：使用映射表将LLM返回的短ID翻译回长ID
            memory_id_mapping = context_data.get("memory_id_mapping", {})

            if "useful_memory_ids" in feedback_data:
                # 使用映射表将短ID翻译回长ID
                short_ids = feedback_data.get("useful_memory_ids", [])
                long_ids = [
                    memory_id_mapping.get(short_id, short_id) for short_id in short_ids
                ]
                feedback_data["useful_memory_ids"] = long_ids
            else:
                self.logger.error(
                    "feedback_data中没有useful_memory_ids字段"
                )

            if not isinstance(feedback_data, dict):
                self.logger.error(
                    f"feedback_data不是字典类型，实际类型: {type(feedback_data)}，内容: {feedback_data}"
                )
                feedback_data = {}  # 安全降级

            # === 新的简化接口实现 ===

            # --- 开始最终修正 ---

            memory_actions_raw = feedback_data.get("memory_actions", [])
            memory_actions_normalized = MemoryIDResolver.normalize_memory_actions_format(
                memory_actions_raw, self.logger
            )

            for action in memory_actions_normalized:
                if action.get("action") not in {"merge", "updata"}:
                    continue
                source_memory_ids = action.get("source_memory_ids", [])
                action["source_memory_ids"] = [
                    memory_id_mapping.get(memory_id, memory_id)
                    for memory_id in source_memory_ids
                ]

            # --- 记忆写入：附上生成这条记忆时的情绪读数 ---
            if hasattr(self, "soul") and self.soul and memory_actions_normalized:
                soul_scores = self.soul.get_scores()
                for action in memory_actions_normalized:
                    if not isinstance(action, dict):
                        continue
                    metadata = action.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                        action["metadata"] = metadata
                    metadata["soul_state"] = dict(soul_scores)

            # --- 修正结束 ---

            # 3. 调用封装好的 feedback 接口，并使用"转换后"的扁平列表
            #    (以及我们之前讨论过的，让 feedback 返回新创建的对象)
            newly_created_memories = []
            if self.memory_system:
                # 从反思输入载体获取 persona_name
                persona_name = getattr(reflection_input, "persona_name", "")
                memory_scope = self.plugin_context.resolve_memory_scope(
                    session_id, persona_name=persona_name
                )
                # 直接异步调用
                newly_created_memories = await self.memory_system.feedback(
                    useful_memory_ids=feedback_data.get("useful_memory_ids", []),
                    recalled_memory_ids=[mem.id for mem in (long_term_memories or []) if getattr(mem, "id", None)],
                    memory_actions=memory_actions_normalized,
                    memory_scope=memory_scope,
                )

            # 2. 更新会话工作缓存
            # 获取有用的旧记忆
            useful_ids = feedback_data.get("useful_memory_ids", [])
            useful_long_term_memories = [
                mem for mem in long_term_memories if mem.id in useful_ids
            ]

            # 先按这一轮的有用判定结算生命值，被判定有用的加分，其余减分，归零的当场淘汰
            died_items = self.session_memory_manager.evaluate_feedback_in_session(
                session_id, useful_ids
            )
            if died_items:
                self.logger.debug(
                    f"[短期记忆] 生命值耗尽淘汰 session={session_id} count={len(died_items)}"
                )

            # 将有用的旧记忆和全新的记忆合并，一起放入短期记忆，绑定触发这次回复的消息
            memories_for_session = useful_long_term_memories + newly_created_memories
            if memories_for_session:
                bound_message_id, bound_message_text = self._latest_user_message(
                    context_data
                )
                self.session_memory_manager.add_memories_to_session(
                    session_id,
                    memories_for_session,
                    bound_message_id=bound_message_id,
                    bound_message_text=bound_message_text,
                )
            self.logger.info(
                f"[反思执行] 完成 session={session_id} "
                f"useful={len(useful_ids)} new={len(newly_created_memories)}"
            )
            self._complete_reflection_trace(
                trace_id,
                status="success",
                response=response_text,
                result=full_json_data,
            )

        except Exception as e:
            self._complete_reflection_trace(
                trace_id,
                status="failed",
                error=f"反思执行异常: {e}",
            )

            self.logger.error(f"异步记忆分析失败 会话={session_id} 错误={e}", exc_info=True)
            return False
        return True
