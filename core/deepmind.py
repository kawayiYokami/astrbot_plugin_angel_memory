"""
DeepMind潜意识核心模块

这是AI的潜意识系统，在后台默默帮你管理记忆：
- 看到消息时自动回忆相关内容
- 筛选出有用的记忆喂给主意识
- 定期整理记忆，让重要内容不容易忘记
- 就像人睡觉时整理记忆一样
"""

import time
import json
from typing import List, Dict, Any, Optional
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
from .services.feedback_service import DeepMindFeedbackService
from .services.sleep_service import DeepMindSleepService

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


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

    # 潜意识回忆的规则
    CHAINED_RECALL_PER_TYPE_LIMIT = 7  # 每种记忆最多想7条，防止信息过载
    CHAINED_RECALL_FINAL_LIMIT = 7  # 最终给主意识最多7条记忆

    def __init__(
        self,
        config,
        context,
        vector_store,
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
            vector_store: 记忆数据库（存所有长期记忆的地方）
            note_service: 笔记管理器（重要信息专门存放处）
            plugin_context: 插件上下文（新增）
            memory_runtime: 统一记忆运行时（必填）
            provider_id: AI助手的ID，没有的话潜意识就睡觉不干活
        """
        self.config = config
        self.memory_system = memory_runtime
        self.note_service = note_service
        self.context = context
        self.vector_store = vector_store
        self.provider_id = provider_id
        self.plugin_context = plugin_context # 保存引用

        # 从全局容器获取logger
        self.logger = logger
        self.json_parser = JsonParser()

        # 获取配置值（嵌套配置）
        # 当前格式：memory_behavior.*, note_topk.*

        # 记忆行为参数
        memory_behavior = getattr(config, "memory_behavior", {})
        self.min_message_length = memory_behavior.get("min_message_length") if isinstance(memory_behavior, dict) else getattr(config, "min_message_length", 5)
        self.short_term_memory_capacity = memory_behavior.get("short_term_memory_capacity") if isinstance(memory_behavior, dict) else getattr(config, "short_term_memory_capacity", 1.0)
        self.sleep_interval = memory_behavior.get("sleep_interval") if isinstance(memory_behavior, dict) else getattr(config, "sleep_interval", 3600)

        # 笔记 Top-K 参数（候选固定为注入的 7 倍）
        note_topk = getattr(config, "note_topk", {})
        note_top_k = (
            int(note_topk.get("top_k", 8))
            if isinstance(note_topk, dict)
            else int(getattr(config, "note_top_k", 8))
        )
        if note_top_k < 0:
            note_top_k = 0
        self.note_inject_top_k = note_top_k
        self.note_candidate_top_k = note_top_k * 7

        # 初始化短期记忆管理器
        self.session_memory_manager = SessionMemoryManager(
            capacity_multiplier=self.short_term_memory_capacity
        )

        # 睡眠状态管理
        self.last_sleep_time = None  # 上次睡眠时间戳

        # 初始化工具类
        self.prompt_builder = SmallModelPromptBuilder()
        self.memory_injector = MemoryInjector()
        self.query_processor = get_query_processor()
        self.retrieval_service = DeepMindRetrievalService(self)
        self.injection_service = DeepMindInjectionService(self)
        self.feedback_service = DeepMindFeedbackService(self)
        self.sleep_service = DeepMindSleepService(self)

        # 初始化灵魂状态管理器
        try:
            # 将配置对象传递给 SoulState
            self.soul = SoulState(config=self.config)
        except Exception as e:
            self.logger.error(f"灵魂状态管理器初始化失败: {e}")
            self.soul = None

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
        self, event: AstrMessageEvent, query: str, precompute_vectors: bool = False
    ) -> Dict[str, Any]:
        """
        检索长期记忆和候选笔记

        Args:
            event: 消息事件
            query: 查询字符串
            precompute_vectors: 是否预计算向量

        Returns:
            包含 long_term_memories, candidate_notes, note_id_mapping, secretary_decision 的字典
        """
        return await self.retrieval_service.retrieve_memories_and_notes(
            event, query, precompute_vectors
        )



    def _normalize_soul_value(self, dimension: str, value: float) -> float:
        """将灵魂状态的物理值归一化到 [0, 1] 区间"""
        return self.injection_service.normalize_soul_value(dimension, value)

    def _create_tendency_bar(self, normalized_value: float) -> str:
        """创建一个10格的文本进度条"""
        return self.injection_service.create_tendency_bar(normalized_value)

    def _inject_memories_to_request(
        self, request: ProviderRequest, session_id: str, note_context: str, soul_state_values: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        将记忆、笔记和灵魂状态统一注入到LLM请求中（使用 extra_user_content_parts）
        """
        self.injection_service.inject_memories_to_request(
            request, session_id, note_context, soul_state_values
        )

    async def _update_memory_system(
        self, feedback_data: Dict[str, Any], long_term_memories: List, session_id: str
    ) -> None:
        """
        更新短期记忆并将长期反馈任务加入后台队列

        Args:
            feedback_data: LLM反馈数据
            long_term_memories: 长期记忆列表
            session_id: 会话ID
        """
        await self.feedback_service.update_memory_system(
            feedback_data, long_term_memories, session_id
        )

    async def _execute_feedback_task(
        self,
        useful_memory_ids: List[str],
        new_memories: List[Dict[str, Any]],
        merge_groups: List[List[str]],
        session_id: str,
    ) -> None:
        """异步执行的长期记忆反馈。"""

        await self.feedback_service.execute_feedback_task(
            useful_memory_ids, new_memories, merge_groups, session_id
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
        chat_records = []
        if hasattr(event, "angelheart_context"):
            try:
                angelheart_data = json.loads(event.angelheart_context)
                chat_records = angelheart_data.get("chat_records", [])
            except (json.JSONDecodeError, KeyError):
                self.logger.error(f"为会话 {session_id} 解析 angelheart_context 失败")

        # 2. 从 secretary_decision 构建查询字符串
        query = ""
        user_list = []

        # 尝试从 secretary_decision 中提取关键信息
        secretary_decision = {}
        if hasattr(event, "angelheart_context"):
            try:
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})
            except (json.JSONDecodeError, KeyError):
                self.logger.error(f"为会话 {session_id} 解析 secretary_decision 失败")

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
            if not chat_records:
                message_text = self._extract_message_text(event)
                query = message_text if message_text else ""
            else:
                query, user_list = self.prompt_builder.format_chat_records(chat_records)

        # 3. 如果未配置 provider_id，跳过记忆整理
        if not self.provider_id:
            return

        # 4. 检索长期记忆和笔记
        retrieval_data = await self._retrieve_memories_and_notes(event, query, precompute_vectors=True)

        long_term_memories = retrieval_data["long_term_memories"]
        candidate_notes = retrieval_data["candidate_notes"]
        core_topic = retrieval_data["core_topic"]

        # 5. 将检索到的长期记忆填入短期记忆
        if long_term_memories and self.memory_system:
            self.session_memory_manager.add_memories_to_session(session_id, long_term_memories)

        # 6. 构建笔记上下文（复用NoteContextBuilder）
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

        # 7. 获取灵魂状态值
        soul_state_values = None
        if hasattr(self, "soul") and self.soul:
            try:
                soul_state_values = {
                    "RecallDepth": self.soul.get_value("RecallDepth"),
                    "ImpressionDepth": self.soul.get_value("ImpressionDepth"),
                    "ExpressionDesire": self.soul.get_value("ExpressionDesire"),
                    "Creativity": self.soul.get_value("Creativity")
                }
            except Exception as e:
                self.logger.warning(f"获取灵魂状态值失败: {e}")

        # 8. 注入记忆、笔记和灵魂状态到请求
        self._inject_memories_to_request(request, session_id, note_context, soul_state_values)

        # 9. (异步任务所需) 将原始上下文数据存入event.angelmemory_context
        try:
            memory_id_mapping = MemoryIDResolver.generate_id_mapping([mem.to_dict() for mem in long_term_memories], "id")
            angelmemory_context = {
                "memories": self._memories_to_json(self.session_memory_manager.get_session_memories(session_id)),
                "recall_query": query,
                "recall_time": time.time(),
                "session_id": session_id,
                "user_list": user_list,
                "raw_memories": [memory.to_dict() for memory in long_term_memories],
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
            memory_type = (
                memory.memory_type.value
                if hasattr(memory.memory_type, "value")
                else memory.memory_type
            )

            memory_data = {
                "id": memory.id,
                "type": memory_type,
                "strength": memory.strength,
                "judgment": memory.judgment,
                "reasoning": memory.reasoning,
                "tags": memory.tags,
            }
            memories_json.append(memory_data)

        return memories_json

    # _resolve_memory_ids 方法已移至 MemoryIDResolver 类中

    async def check_and_sleep_if_needed(self, sleep_interval: int) -> bool:
        """
        检查是否需要睡眠，如果需要则触发睡眠

        Args:
            sleep_interval: 睡眠间隔（秒），0表示禁用睡眠

        Returns:
            bool: 是否执行了睡眠
        """
        return await self.sleep_service.check_and_sleep_if_needed(sleep_interval)

    async def _sleep(self):
        """AI睡觉整理记忆：重要内容加强，无用内容清理"""
        await self.sleep()

    async def sleep(self):
        """公共睡眠入口，供外部组件触发睡眠流程。"""
        await self.sleep_service.sleep()

    def shutdown(self):
        """关闭潜意识系统，让AI好好休息"""

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
        # 获取会话ID
        session_id = self._get_session_id(event)

        await self.feedback_service.submit_async_analysis_task(
            event, response, session_id
        )

    def _serialize_event_data(self, event: AstrMessageEvent) -> Dict:
        """序列化事件数据以便在后台线程中使用"""
        try:
            # 提取事件中的关键数据
            event_data = {
                "angelmemory_context": getattr(event, "angelmemory_context", None),
                "angelheart_context": getattr(event, "angelheart_context", None),
                "unified_msg_origin": getattr(event, "unified_msg_origin", None),
            }

            # 如果有 angelmemory_context，尝试解析它以确保数据完整性
            if event_data["angelmemory_context"]:
                try:
                    context_json = json.loads(event_data["angelmemory_context"])
                    event_data["angelmemory_context_parsed"] = context_json
                except (json.JSONDecodeError, TypeError):
                    pass

            return event_data
        except Exception as e:
            self.logger.error(f"序列化事件数据失败: {e}")
            return {}

    def _serialize_response_data(self, response) -> Dict:
        """序列化响应数据以便在后台线程中使用"""
        try:
            # 提取响应中的关键数据
            response_data = {
                "completion_text": getattr(response, "completion_text", str(response))
                if response
                else ""
            }
            return response_data
        except Exception as e:
            self.logger.error(f"序列化响应数据失败: {e}")
            return {"completion_text": ""}

    async def _execute_async_analysis_task(
        self, event_data: Dict, response_data: Dict, session_id: str
    ):
        """
        异步执行的记忆分析任务

        Args:
            event_data: 序列化的事件数据
            response_data: 序列化的响应数据
            session_id: 会话ID
        """
        try:
            # 重构事件对象的部分数据用于处理
            class SimpleEvent:
                def __init__(self, data, plugin_context):
                    self.angelmemory_context = data.get("angelmemory_context")
                    self.angelheart_context = data.get("angelheart_context")
                    self.unified_msg_origin = data.get("unified_msg_origin")
                    # 注入plugin_context，确保QueryProcessor能够访问
                    self.plugin_context = plugin_context

            event = SimpleEvent(event_data, self.plugin_context)

            # 获取原始上下文数据
            context_data = self._parse_memory_context(event)
            if not context_data:
                return

            query = context_data["query"]

            # 获取原始记忆数据
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
            response_text = response_data.get("completion_text", "")


            # 从上下文数据中获取ID映射表
            memory_id_mapping = context_data.get("memory_id_mapping", {})

            # 构建反思提示词（只传递记忆数据，不传递笔记）
            prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompt(
                historical_query=query,
                main_llm_response=response_text,
                raw_memories=long_term_memories,
                core_topic=core_topic,
                memory_id_mapping=memory_id_mapping,
                config=self.config,
            )


            # 调用小模型进行分析（在后台线程中同步调用）
            provider = self.context.get_provider_by_id(self.provider_id)
            if not provider:
                self.logger.error(
                    f"找不到提供者: {self.provider_id}，会话: {session_id}"
                )
                return

            try:
                # 直接异步调用，无需检查
                llm_response = await provider.text_chat(prompt=prompt)
            except Exception as e:
                self.logger.error(
                    f"会话 {session_id} 的LLM调用失败，跳过记忆整理: {e}"
                )
                return

            if not llm_response or not getattr(llm_response, "completion_text", ""):
                self.logger.error(f"会话 {session_id} 的LLM API调用失败")
                return

            # 提取响应文本
            response_text = llm_response.completion_text

            # 解析完整的结构化输出
            full_json_data = self.json_parser.extract_json(response_text)

            if not isinstance(full_json_data, dict):
                self.logger.error(
                    f"会话 {session_id} 的JSON解析失败或未返回字典"
                )
                return

            # 提取 feedback_data
            feedback_data = full_json_data.get("feedback_data", {})

            # --- 灵魂状态更新 (Feedback Loop) ---
            if hasattr(self, "soul") and self.soul and "soul_state_code" in full_json_data:
                state_code = full_json_data.get("soul_state_code", "0000")
                if len(state_code) == 4:
                    try:
                        # 使用新的原子化接口（4位代码一次性调整）
                        # 代码位对应: RecallDepth, ImpressionDepth, ExpressionDesire, Creativity
                        # 0000 颓废: 话少(Expression-), 死板(Creativity-), 不查历史(Recall-), 拒绝新知(Impression-)
                        # 1111 觉醒: 话多(Expression+), 飞升(Creativity+), 查阅历史(Recall+), 吸收新知(Impression+)

                        self.soul.adjust(state_code, mode="reflect")
                        self.logger.info(f"🧘 灵魂反思 ({state_code}): {self.soul.get_state_description()}")
                    except ValueError:
                        self.logger.warning(f"无效的灵魂状态代码: {state_code}")

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

            # 1. 从 feedback_data 中获取原始的、按类型分组的新记忆字典
            new_memories_raw = feedback_data.get("new_memories", {})

            # 2. 调用已有的工具函数，将其转换为底层服务期望的"扁平列表"格式
            new_memories_normalized = MemoryIDResolver.normalize_new_memories_format(
                new_memories_raw, self.logger
            )

            # --- 记忆生成限制 (基于灵魂 ImpressionDepth) ---
            if hasattr(self, "soul") and self.soul and new_memories_normalized:
                # 获取允许生成的最大数量
                impression_limit = int(self.soul.get_value("ImpressionDepth"))
                original_count = len(new_memories_normalized)

                # 截断列表
                if original_count > impression_limit:
                    new_memories_normalized = new_memories_normalized[:impression_limit]
                    self.logger.info(f"✂️ 记忆截断: 灵魂仅允许记录 {impression_limit} 条 (原 {original_count} 条)")

                # 为每条新记忆注入当前的灵魂快照
                snapshot = self.soul.get_snapshot()
                for mem in new_memories_normalized:
                    mem["state_snapshot"] = snapshot

            # --- 修正结束 ---

            # 3. 调用封装好的 feedback 接口，并使用"转换后"的扁平列表
            #    (以及我们之前讨论过的，让 feedback 返回新创建的对象)
            newly_created_memories = []
            if self.memory_system:
                memory_scope = self.plugin_context.resolve_memory_scope(session_id)
                # 直接异步调用
                newly_created_memories = await self.memory_system.feedback(
                    useful_memory_ids=feedback_data.get("useful_memory_ids", []),
                    new_memories=new_memories_normalized,  # <--- 使用转换后的数据
                    merge_groups=feedback_data.get("merge_groups", []),
                    memory_scope=memory_scope,
                )

            # 2. 更新短期记忆
            # 获取有用的旧记忆
            useful_ids = feedback_data.get("useful_memory_ids", [])
            useful_long_term_memories = [
                mem for mem in long_term_memories if mem.id in useful_ids
            ]

            # 将有用的旧记忆和全新的记忆合并，一起放入短期记忆
            memories_for_session = useful_long_term_memories + newly_created_memories
            if memories_for_session:
                self.session_memory_manager.add_memories_to_session(
                    session_id, memories_for_session
                )

        except Exception as e:
            import traceback

            self.logger.error(f"异步记忆分析失败 - 会话={session_id}: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
