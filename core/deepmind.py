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
import threading
from typing import List, Dict, Any, Optional
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from ..llm_memory import CognitiveService
from ..llm_memory.utils.json_parser import JsonParser
from .session_memory import SessionMemoryManager
from .utils import SmallModelPromptBuilder, MemoryInjector, MemoryIDResolver
from .utils.feedback_queue import get_feedback_queue
from .utils.note_context_builder import NoteContextBuilder
from .utils.query_processor import get_query_processor
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
    CHAINED_RECALL_FINAL_LIMIT = 7     # 最终给主意识最多7条记忆
    NOTE_CANDIDATE_COUNT = 50          # 先找50条笔记，让小AI帮忙筛选有用的

    def __init__(self, config, context, vector_store, note_service, provider_id: str = "", cognitive_service=None):
        """
        初始化AI的潜意识系统

        Args:
            config: 配置信息（比如多久睡一次觉整理记忆）
            context: 聊天机器人的大脑（主意识）
            vector_store: 记忆数据库（存所有长期记忆的地方）
            note_service: 笔记管理器（重要信息专门存放处）
            provider_id: AI助手的ID，没有的话潜意识就睡觉不干活
            cognitive_service: 记忆管理服务（可选，如果外面已经有了就直接用）
        """
        self.config = config
        self.memory_system: Optional[CognitiveService] = cognitive_service  # 使用注入的实例
        self.note_service = note_service
        self.context = context
        self.vector_store = vector_store
        self.provider_id = provider_id
        # 从全局容器获取logger
        self.logger = logger
        self.json_parser = JsonParser()

        # 获取配置值
        self.min_message_length = getattr(config, 'min_message_length', 5)
        self.short_term_memory_capacity = getattr(config, 'short_term_memory_capacity', 1.0)
        self.sleep_interval = getattr(config, 'sleep_interval', 3600)  # 默认1小时
        self.small_model_note_budget = getattr(config, 'small_model_note_budget', 8000)
        self.large_model_note_budget = getattr(config, 'large_model_note_budget', 12000)

        # 初始化短期记忆管理器
        self.session_memory_manager = SessionMemoryManager(
            capacity_multiplier=self.short_term_memory_capacity
        )

        # 初始化工具类
        self.prompt_builder = SmallModelPromptBuilder()
        self.memory_injector = MemoryInjector()
        self.query_processor = get_query_processor()

        # 睡眠相关
        self._sleep_timer = None
        self._stop_sleep_event = threading.Event()  # 使用Event替代布尔标志，避免竞态条件

        # 初始化记忆系统（如果没有通过依赖注入提供）
        self._init_memory_system()

    def _init_memory_system(self) -> None:
        """初始化记忆系统"""
        # 如果已经有了注入的认知服务，直接使用
        if self.memory_system is not None:
            self.logger.info("Using injected CognitiveService instance")
            # 初始化时执行一次睡眠
            self._sleep()
            # 启动定期睡眠
            self._start_periodic_sleep()
            return

        # 否则创建新的认知服务实例（向后兼容）
        try:
            self.memory_system = CognitiveService(vector_store=self.vector_store)
            self.logger.info("Memory system initialized successfully")

            # 初始化时执行一次睡眠
            self._sleep()

            # 启动定期睡眠
            self._start_periodic_sleep()

        except Exception as e:
            self.logger.error(f"Memory system initialization failed: {e}")
            self.memory_system = None

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
            self.logger.debug("消息被过滤: 记忆系统未启用")
            return False

        # 从事件中提取消息文本
        message_text = self._extract_message_text(event)
        if not message_text:
            self.logger.debug("消息被过滤: 文本为空")
            return False

        message_text = message_text.strip()

        # 检查最小消息长度
        if len(message_text) < self.min_message_length:
            self.logger.debug(f"消息被过滤: 长度过短 ({len(message_text)} < {self.min_message_length})")
            return False

        # 忽略纯指令消息（以/开头）
        if message_text.startswith('/'):
            self.logger.debug("消息被过滤: 以'/'开头")
            return False

        return True

    def inject_initial_memories(self, event: AstrMessageEvent):
        """
        事件到达时注入初始记忆

        Args:
            event: 消息事件
        """
        if not self.should_process_message(event):
            return

        session_id = self._get_session_id(event)

        try:
            # 1. 从天使之心获取完整对话历史
            chat_records = []
            if hasattr(event, 'angelheart_context'):
                try:
                    angelheart_data = json.loads(event.angelheart_context)
                    chat_records = angelheart_data.get('chat_records', [])
                except (json.JSONDecodeError, KeyError):
                    self.logger.warning(f"Failed to parse angelheart_context for session {session_id}")

            # 初始化 user_list 为空列表
            user_list = []

            # 如果没有对话记录，使用当前消息文本
            if not chat_records:
                message_text = self._extract_message_text(event)
                query = message_text if message_text else ""
            else:
               # 2. 格式化对话历史为查询字符串
               query, user_list = self.prompt_builder.format_chat_records(chat_records)

            # 3. 读取主意识的短期记忆（供其他模块参考，不进行长期记忆召回）
            session_memories = self.session_memory_manager.get_session_memories(session_id)

            # 4. 转换为JSON格式
            memories_json = self._memories_to_json(session_memories)

            # 注入到事件中（使用angelmemory_context）
            event.angelmemory_context = json.dumps({
                'memories': memories_json,
                'recall_query': query,  # 保留查询字符串供后续使用
                'recall_time': time.time(),
                'session_id': session_id,
                'user_list': user_list  # 将新生成的"用户清单"存入上下文
            })

        except Exception as e:
            self.logger.error(f"Memory recall failed for session {session_id}: {e}")

    def _parse_memory_context(self, event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        """
        解析事件中的记忆上下文数据

        Args:
            event: 消息事件

        Returns:
            包含 session_id, query, user_list 的字典，解析失败返回 None
        """
        if not hasattr(event, 'angelmemory_context'):
            return None

        try:
            context_data = json.loads(event.angelmemory_context)
            return {
                'session_id': context_data['session_id'],
                'query': context_data.get('recall_query', ''),
                'user_list': context_data.get('user_list', [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse memory context: {e}")
            return None

    def _retrieve_memories_and_notes(
        self,
        event: AstrMessageEvent,
        query: str
    ) -> Dict[str, Any]:
        """
        检索长期记忆和候选笔记

        Args:
            event: 消息事件
            query: 查询字符串

        Returns:
            包含 long_term_memories, candidate_notes, note_id_mapping, secretary_decision 的字典
        """
        from .timing_diagnostics import timing_log, log_checkpoint

        log_checkpoint("开始链式召回")

        # 1. 预处理记忆检索查询词
        memory_query = self.query_processor.process_query_for_memory(query, event)

        # 1. 使用链式召回从长期记忆检索相关记忆
        with timing_log("链式召回(chained_recall)", threshold_ms=5000):
            long_term_memories = self.memory_system.chained_recall(
                query=memory_query,
                per_type_limit=self.CHAINED_RECALL_PER_TYPE_LIMIT,
                final_limit=self.CHAINED_RECALL_FINAL_LIMIT
            )

        log_checkpoint(f"链式召回完成，获得{len(long_term_memories)}条记忆")

        # 2. 获取 secretary_decision 信息
        secretary_decision = {}
        try:
            if hasattr(event, 'angelheart_context'):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get('secretary_decision', {})
        except (json.JSONDecodeError, KeyError):
            self.logger.debug("无法获取 secretary_decision 信息")

        # 3. 优先使用天使之心核心话题进行笔记检索
        core_topic = self._extract_core_topic(event)
        note_query = core_topic if core_topic and core_topic.strip() else query

        # 应用统一的检索词预处理
        note_query = self.query_processor.process_query_for_notes(note_query, event)

        if core_topic and core_topic.strip():
            self.logger.info(f"使用核心话题进行笔记检索: {note_query}")
        else:
            self.logger.debug(f"未找到核心话题，使用聊天记录查询: {note_query}")

        log_checkpoint("开始笔记检索")

        # 4. 获取候选笔记（用于小模型的选择）
        with timing_log("笔记检索(search_notes_by_token_limit)", threshold_ms=5000):
            candidate_notes = self.note_service.search_notes_by_token_limit(
                query=note_query,
                max_tokens=self.small_model_note_budget,
                recall_count=self.NOTE_CANDIDATE_COUNT
            )

        log_checkpoint(f"笔记检索完成，获得{len(candidate_notes)}条笔记")

        # 5. 创建短ID到完整ID的映射（用于后续上下文扩展）
        note_id_mapping = {}
        for note in candidate_notes:
            note_id = note.get('id')
            if note_id:
                short_id = MemoryIDResolver.generate_short_id(note_id)
                note_id_mapping[short_id] = note_id
            else:
                self.logger.warning(f"🔍 [DEBUG] 跳过无ID的笔记: {note}")

        # 6. 创建短期记忆ID映射表（用于解析 useful_memory_ids）
        memory_id_mapping = {}
        if long_term_memories:
            memory_id_mapping = MemoryIDResolver.generate_id_mapping(
                [memory.to_dict() for memory in long_term_memories], 'id'
            )
        else:
            self.logger.warning("🔍 [DEBUG] 没有长期记忆，memory_id_mapping为空")

        return {
            'long_term_memories': long_term_memories,
            'candidate_notes': candidate_notes,
            'note_id_mapping': note_id_mapping,
            'memory_id_mapping': memory_id_mapping,
            'secretary_decision': secretary_decision,
            'core_topic': core_topic
        }

    async def _filter_memories_with_llm(
        self,
        query: str,
        long_term_memories: List,
        user_list: List,
        candidate_notes: List,
        secretary_decision: Dict,
        note_id_mapping: Dict[str, str],
        memory_id_mapping: Dict[str, str],
        session_id: str,
        core_topic: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        使用LLM筛选记忆和笔记

        Args:
            query: 查询字符串
            long_term_memories: 长期记忆列表
            user_list: 用户列表
            candidate_notes: 候选笔记列表
            secretary_decision: 秘书决策信息
            note_id_mapping: 笔记ID映射
            memory_id_mapping: 短期记忆ID映射
            session_id: 会话ID
            core_topic: 当前对话的核心话题

        Returns:
            包含 feedback_data, useful_note_short_ids, note_context 的字典，失败返回 None
        """
        # 1. 构建小模型提示词
        prompt = self.prompt_builder.build_memory_prompt(
            query, long_term_memories, user_list, candidate_notes, secretary_decision, core_topic
        )

        # 调试日志：记录提示词和候选笔记
        if not candidate_notes:
            self.logger.warning("🔍 [DEBUG] 候选笔记为空！")

        # 输出小模型的请求内容
        self.logger.debug(f"Small model request content for session {session_id}:\n{prompt}")

        # 2. 获取 LLM 提供商
        provider = self.context.get_provider_by_id(self.provider_id)
        if not provider:
            self.logger.error(f"Provider not found: {self.provider_id} for session {session_id}")
            return None

        # 3. 调用 LLM（添加5秒超时）
        import asyncio
        try:
            llm_response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt),
                timeout=30.0  # 30秒超时，适合实时对话
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"⏱️ LLM调用超时(30秒) for session {session_id}，跳过记忆整理")
            return None

        if not llm_response or not llm_response.completion_text:
            self.logger.error(f"LLM API call failed for session {session_id}")
            return None

        # 4. 提取响应文本
        response_text = llm_response.completion_text

        # 输出小模型的原始响应
        self.logger.info(f"Small model raw response for session {session_id}:\n{response_text}")

        # 5. 解析完整的结构化输出
        full_json_data = self.json_parser.extract_json(response_text)
        self.logger.debug(f"Parsed full_json_data: {full_json_data}")

        if not isinstance(full_json_data, dict):
            self.logger.error(f"JSON parsing failed or did not return a dict for session {session_id}")
            return None

        # 6. 分别提取 useful_notes 和 feedback_data
        useful_note_short_ids = full_json_data.get('useful_notes', [])
        feedback_data = full_json_data.get('feedback_data', {})

        # 7. ID解析：使用正确的映射解析ID (现在作用于 feedback_data)
        if 'useful_memory_ids' in feedback_data:
            # 使用短期记忆映射解析 useful_memory_ids
            feedback_data['useful_memory_ids'] = MemoryIDResolver.resolve_memory_ids(
                feedback_data.get('useful_memory_ids', []),
                long_term_memories,
                self.logger
            )
        else:
            self.logger.warning("🔍 [DEBUG] feedback_data中没有useful_memory_ids字段")

        if not isinstance(feedback_data, dict):
            self.logger.error(f"feedback_data is not a dict, it's {type(feedback_data)}: {feedback_data}")
            feedback_data = {}  # 安全降级

        # 8. 处理选中的笔记ID，进行上下文扩展
        note_context = ""
        if useful_note_short_ids:

            # 进行上下文扩展
            note_context = NoteContextBuilder.expand_context_from_note_ids(
                useful_note_short_ids,
                self.note_service,
                self.large_model_note_budget,
                note_id_mapping
            )

        return {
            'feedback_data': feedback_data,
            'useful_note_short_ids': useful_note_short_ids,
            'note_context': note_context
        }

    def _inject_memories_to_request(
        self,
        request: ProviderRequest,
        session_id: str,
        note_context: str
    ) -> None:
        """
        将记忆注入到LLM请求中

        Args:
            request: LLM请求对象
            session_id: 会话ID
            note_context: 笔记上下文
        """
        # 1. 从短期记忆推送给主意识（潜意识筛选后的精选记忆）
        short_term_memories = self.session_memory_manager.get_session_memories(session_id)

        memory_context = self.memory_injector.format_fifo_memories_for_prompt(short_term_memories)

        # 2. 合并记忆和笔记上下文
        if memory_context or note_context:
            combined_context = ""
            if memory_context:
                combined_context += memory_context
            if note_context:
                if combined_context:
                    combined_context += "\n\n---\n\n"
                combined_context += f"相关笔记上下文：\n{note_context}"

            # 3. 注入到系统提示词
            request.system_prompt = self.memory_injector.inject_into_system_prompt(
                request.system_prompt,
                combined_context
            )

    def _update_memory_system(
        self,
        feedback_data: Dict[str, Any],
        long_term_memories: List,
        session_id: str
    ) -> None:
        """
        更新短期记忆并将长期反馈任务加入后台队列

        Args:
            feedback_data: LLM反馈数据
            long_term_memories: 长期记忆列表
            session_id: 会话ID
        """
        useful_memory_ids = feedback_data.get('useful_memory_ids', [])
        new_memories_raw = feedback_data.get('new_memories', {})
        merge_groups_raw = feedback_data.get('merge_groups', [])

        # 1. 同步更新短期记忆，确保本次请求即可使用
        if useful_memory_ids:
            memory_map = {memory.id: memory for memory in long_term_memories}
            useful_long_term_memories = [
                memory_map[memory_id]
                for memory_id in useful_memory_ids
                if memory_id in memory_map
            ]

            if useful_long_term_memories:
                self.session_memory_manager.add_memories_to_session(session_id, useful_long_term_memories)
                self.logger.debug(
                    "潜意识筛选：%d条有用记忆进入短期记忆",
                    len(useful_long_term_memories)
                )

        # 2. 后台异步处理长期记忆反馈
        new_memories = MemoryIDResolver.normalize_new_memories_format(new_memories_raw, self.logger)
        merge_groups = MemoryIDResolver.normalize_merge_groups_format(merge_groups_raw)

        if useful_memory_ids or new_memories or merge_groups:
            task_payload = {
                'feedback_fn': self._execute_feedback_task,
                'session_id': session_id,
                'payload': {
                    'useful_memory_ids': list(useful_memory_ids),
                    'new_memories': new_memories,
                    'merge_groups': merge_groups,
                    'session_id': session_id
                }
            }
            get_feedback_queue().submit(task_payload)
            self.logger.debug(
                "已提交记忆反馈任务（session=%s, useful=%d, new=%d, merge=%d）",
                session_id,
                len(useful_memory_ids),
                len(new_memories),
                len(merge_groups)
            )
        else:
            self.logger.debug("记忆反馈无待处理内容，跳过")

        self.logger.info(
            "记忆整理提交完成（会话 %s）：潜意识筛选出 %d 条有用记忆进入短期记忆",
            session_id,
            len(useful_memory_ids)
        )

    def _execute_feedback_task(
        self,
        useful_memory_ids: List[str],
        new_memories: List[Dict[str, Any]],
        merge_groups: List[List[str]],
        session_id: str
    ) -> None:
        """后台线程执行的长期记忆反馈。"""
        self.logger.debug(
            "[feedback_queue] session=%s 开始处理反馈: useful=%d new=%d merge=%d",
            session_id,
            len(useful_memory_ids),
            len(new_memories),
            len(merge_groups)
        )

        self.memory_system.feedback(
            useful_memory_ids=useful_memory_ids,
            new_memories=new_memories,
            merge_groups=merge_groups
        )

        self.logger.debug("[feedback_queue] session=%s 反馈任务完成", session_id)

    def _extract_core_topic(self, event: AstrMessageEvent) -> str:
        """
        从天使之心上下文中提取核心话题

        Args:
            event: 消息事件

        Returns:
            核心话题字符串，如果没有则返回空字符串
        """
        try:
            if hasattr(event, 'angelheart_context'):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get('secretary_decision', {})
                core_topic = secretary_decision.get('topic', '')

                if core_topic and core_topic.strip():
                    return core_topic.strip()
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.debug(f"无法提取核心话题: {e}")

        return ""

    async def organize_and_inject_memories(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        潜意识的核心工作：整理相关记忆喂给主意识

        工作流程：
        1. 从记忆库找出相关的内容
        2. 问小AI哪些信息有用
        3. 把有用的记忆包装成记忆包
        4. 喂给主意识（LLM）帮助他思考

        Args:
            event: 用户的消息（触发回忆的线索）
            request: 即将发给主意识的请求（我们要往里面塞记忆）
        """
        from .timing_diagnostics import timing_log, log_checkpoint

        # 如果未配置 provider_id，跳过记忆整理
        if not self.provider_id:
            return

        # 解析记忆上下文数据
        with timing_log("解析记忆上下文", threshold_ms=10):
            context_data = self._parse_memory_context(event)
            if not context_data:
                return

        session_id = context_data['session_id']
        query = context_data['query']
        user_list = context_data['user_list']

        log_checkpoint(f"开始检索记忆 - session={session_id}")

        # 检索长期记忆和候选笔记
        with timing_log("检索长期记忆和笔记", threshold_ms=1000):
            retrieval_data = self._retrieve_memories_and_notes(event, query)
        long_term_memories = retrieval_data['long_term_memories']
        candidate_notes = retrieval_data['candidate_notes']
        note_id_mapping = retrieval_data['note_id_mapping']
        memory_id_mapping = retrieval_data['memory_id_mapping']
        secretary_decision = retrieval_data['secretary_decision']
        core_topic = retrieval_data['core_topic']

        try:
            # 使用LLM筛选记忆和笔记
            filter_result = await self._filter_memories_with_llm(
                query, long_term_memories, user_list, candidate_notes,
                secretary_decision, note_id_mapping, memory_id_mapping, session_id, core_topic
            )

            if not filter_result:
                return

            feedback_data = filter_result['feedback_data']
            note_context = filter_result['note_context']

            # 将反馈数据保存到事件中，供后续更新使用
            event.memory_feedback = {
                'feedback_data': feedback_data,
                'session_id': session_id
            }

            # 更新记忆系统（将筛选出的记忆同步加入短期记忆）
            self._update_memory_system(feedback_data, long_term_memories, session_id)

            # 注入记忆到请求（从短期记忆中读取并注入）
            self._inject_memories_to_request(request, session_id, note_context)

        except Exception as e:
            import traceback
            self.logger.error(f"Memory organization failed for session {session_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")



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
            self.logger.warning(f"Deep思维: 调用 event.get_message_outline() 失败: {e}")
            return None

    def _get_session_id(self, event: AstrMessageEvent) -> str:
        """获取会话ID，统一使用 event.unified_msg_origin"""
        try:
            session_id = str(event.unified_msg_origin)
            if not session_id:
                self.logger.error("event.unified_msg_origin 为空值，无法处理会话！")
                raise ValueError("Cannot process event with an empty session ID from unified_msg_origin")
            return session_id
        except AttributeError:
            self.logger.error("事件中缺少 'event.unified_msg_origin' 属性，无法确定会话ID！")
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
            memory_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else memory.memory_type

            memory_data = {
                'id': memory.id,
                'type': memory_type,
                'strength': memory.strength,
                'judgment': memory.judgment,
                'reasoning': memory.reasoning,
                'tags': memory.tags
            }
            memories_json.append(memory_data)

        return memories_json


    # _resolve_memory_ids 方法已移至 MemoryIDResolver 类中

    def _sleep(self):
        """AI睡觉整理记忆：重要内容加强，无用内容清理"""
        if not self.is_enabled():
            return

        try:
            self.memory_system.consolidate_memories()
            self.logger.info("记忆巩固完成")
        except Exception as e:
            self.logger.error(f"记忆巩固失败: {e}")

    def _start_periodic_sleep(self):
        """启动定期睡觉：像人一样按时整理记忆"""
        if not self.is_enabled():
            return

        sleep_interval = self.sleep_interval
        if sleep_interval <= 0:
            return

        def sleep_worker():
            # 使用Event.wait()替代time.sleep()，可以立即响应停止信号
            while not self._stop_sleep_event.wait(timeout=sleep_interval):
                self._sleep()

        self._sleep_timer = threading.Thread(target=sleep_worker, daemon=True)
        self._sleep_timer.start()
        self.logger.info(f"启动定期睡眠，间隔: {sleep_interval}秒")

    def stop_sleep(self):
        """停止定期睡眠"""
        self._stop_sleep_event.set()  # 设置事件，通知线程停止
        if self._sleep_timer and self._sleep_timer.is_alive():
            self._sleep_timer.join(timeout=5)
        self.logger.info("定期睡眠已停止")

    def shutdown(self):
        """关闭潜意识系统，让AI好好休息"""
        self.logger.info("正在关闭AI的潜意识...")

        # 停止定期睡觉
        self.stop_sleep()

        # 停止记忆整理任务
        from .utils.feedback_queue import stop_feedback_queue
        stop_feedback_queue(timeout=5)

        self.logger.info("AI潜意识已休息，下次再见！")
