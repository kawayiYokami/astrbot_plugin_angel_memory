"""
DeepMind核心模块

负责记忆整理的核心逻辑：召回、筛选、格式化和更新记忆。
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
from astrbot.api import logger


class DeepMind:
    """DeepMind记忆整理核心类

    负责实现双层认知架构中的意识层，处理整个记忆工作流：
    - 观察阶段：事件驱动记忆召回
    - 回忆阶段：筛选并注入相关记忆
    - 反馈阶段：更新记忆网络权重
    - 睡眠阶段：定期巩固和优化记忆

    相比底层llm_memory系统，DeepMind主要负责会话级别的记忆管理、
    记忆筛选逻辑、以及用户事件响应。

    设计原则：
    1. 插件级别仅处理会话状态和配置
    2. 记忆存储和索引交给llm_memory子系统
    3. 错误容忍机制确保不影响主流程
    4. 通过provider_id开关决定是否使用LLM进行记忆整理
    """

    def __init__(self, config, context, vector_store, note_service, provider_id: str = ""):
        """
        初始化DeepMind

        Args:
            config: 配置管理器
            context: AstrBot上下文对象
            vector_store: 共享的VectorStore实例
            note_service: 笔记服务实例
            provider_id: LLM提供商ID，留空则跳过记忆整理
        """
        self.config = config
        self.memory_system: Optional[CognitiveService] = None
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

        # 初始化短期记忆管理器
        self.session_memory_manager = SessionMemoryManager(
            capacity_multiplier=self.short_term_memory_capacity
        )

        # 初始化工具类
        self.prompt_builder = SmallModelPromptBuilder()
        self.memory_injector = MemoryInjector()

        # 睡眠相关
        self._sleep_timer = None
        self._stop_sleep = False

        # 初始化记忆系统
        self._init_memory_system()

    def _init_memory_system(self) -> None:
        """初始化记忆系统"""
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
        判断是否应该处理此消息

        Args:
            event: 消息事件

        Returns:
            是否应该处理
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
            memories_json = self._session_memories_to_json(session_memories)

            # 注入到事件中（使用angelmemory_context）
            event.angelmemory_context = json.dumps({
                'memories': memories_json,
                'recall_query': query,  # 保留查询字符串供后续使用
                'recall_time': time.time(),
                'session_id': session_id,
                'user_list': user_list  # 将新生成的"用户清单"存入上下文
            })

            self.logger.info(f"读取短期记忆：{len(session_memories)}条（供其他模块参考）")

            # debug级别显示具体记忆内容
            if session_memories:
                memory_summaries = [f'"{m.judgment}"' for m in session_memories[:3]]
                self.logger.debug(f"短期记忆详情：{', '.join(memory_summaries)}{'...' if len(session_memories) > 3 else ''}")

        except Exception as e:
            self.logger.error(f"Memory recall failed for session {session_id}: {e}")

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
                    self.logger.info(f"识别到核心话题用于笔记检索: {core_topic}")
                    return core_topic.strip()
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.debug(f"无法提取核心话题: {e}")

        return ""

    async def organize_and_inject_memories(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        整理记忆并注入到LLM请求中

        Args:
            event: 消息事件
            request: LLM请求对象
        """
        # 如果未配置 provider_id，跳过记忆整理
        if not self.provider_id:
            return

        if not hasattr(event, 'angelmemory_context'):
            return

        try:
            # 解析angelmemory_context
            context_data = json.loads(event.angelmemory_context)
            session_id = context_data['session_id']
            query = context_data.get('recall_query', '')
            user_list = context_data.get('user_list', [])  # 从上下文中恢复"用户清单"
        except (json.JSONDecodeError, KeyError):
            return

        # 使用链式召回从长期记忆检索相关记忆
        long_term_memories = self.memory_system.chained_recall(query=query, per_type_limit=7, final_limit=7)

        # 获取 secretary_decision 信息
        secretary_decision = {}
        try:
            if hasattr(event, 'angelheart_context'):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get('secretary_decision', {})
        except (json.JSONDecodeError, KeyError):
            self.logger.debug("无法获取 secretary_decision 信息")

        try:
            # 优先使用天使之心核心话题进行笔记检索
            note_query = self._extract_core_topic(event)
            if not note_query or note_query.strip() == "":
                # 如果没有核心话题，回退使用聊天记录查询
                note_query = query
                self.logger.debug(f"未找到核心话题，使用聊天记录查询: {note_query}")
            else:
                self.logger.info(f"使用核心话题进行笔记检索: {note_query}")

            # 获取候选笔记（用于小模型的选择）
            candidate_notes = self.note_service.search_notes_by_token_limit(
                query=note_query,
                max_tokens=self.config.small_model_note_budget,  # 使用配置的Token预算
                recall_count=50  # 提供足够多的候选片段供选择
            )

            # 创建短ID到完整ID的映射（用于后续上下文扩展）
            note_id_mapping = {}
            for note in candidate_notes:
                short_id = MemoryIDResolver.generate_short_id(note['id'])
                note_id_mapping[short_id] = note['id']

            # 构建小模型提示词
            prompt = self.prompt_builder.build_memory_prompt(query, long_term_memories, user_list, candidate_notes, secretary_decision)

            # 输出小模型的请求内容
            self.logger.debug(f"Small model request content for session {session_id}:\n{prompt}")

            # 获取 LLM 提供商
            provider = self.context.get_provider_by_id(self.provider_id)
            if not provider:
                self.logger.error(f"Provider not found: {self.provider_id} for session {session_id}")
                return

            # 调用 LLM
            llm_response = await provider.text_chat(prompt=prompt)

            if not llm_response or not llm_response.completion_text:
                self.logger.error(f"LLM API call failed for session {session_id}")
                return

            # 提取响应文本
            response_text = llm_response.completion_text

            # 输出小模型的原始响应
            self.logger.info(f"Small model raw response for session {session_id}:\n{response_text}")

            # 解析完整的结构化输出
            full_json_data = self.json_parser._safe_extract_json(response_text)
            self.logger.debug(f"Parsed full_json_data: {full_json_data}")

            if not isinstance(full_json_data, dict):
                self.logger.error(f"JSON parsing failed or did not return a dict for session {session_id}")
                return

            # 分别提取 useful_notes 和 feedback_data
            useful_note_short_ids = full_json_data.get('useful_notes', [])
            feedback_data = full_json_data.get('feedback_data', {})

            if not isinstance(feedback_data, dict):
                self.logger.error(f"feedback_data is not a dict, it's {type(feedback_data)}: {feedback_data}")
                feedback_data = {} # 安全降级

            # ID解析：将短ID转换为完整ID (现在作用于 feedback_data)
            if 'useful_memory_ids' in feedback_data:
                feedback_data['useful_memory_ids'] = MemoryIDResolver.resolve_memory_ids(
                    feedback_data.get('useful_memory_ids', []),
                    long_term_memories,
                    self.logger
                )

            # 处理选中的笔记ID，进行上下文扩展 (现在作用于顶层提取的 useful_note_short_ids)
            note_context = ""
            if useful_note_short_ids:
                # 进行上下文扩展
                from .utils.note_context_builder import NoteContextBuilder
                note_context = NoteContextBuilder.expand_context_from_note_ids(
                    useful_note_short_ids,
                    self.note_service,
                    self.config.large_model_note_budget,  # 使用配置的Token预算
                    note_id_mapping  # 传递短ID到完整ID的映射
                )

            # 将反馈数据保存到事件中，供后续更新使用
            event.memory_feedback = {
                'feedback_data': feedback_data,
                'session_id': session_id
            }

            # 4. 从短期记忆推送给主意识（潜意识筛选后的精选记忆）
            short_term_memories = self.session_memory_manager.get_session_memories(session_id)
            memory_context = self.memory_injector.format_fifo_memories_for_prompt(short_term_memories)
            if memory_context or note_context:
                combined_context = ""
                if memory_context:
                    combined_context += memory_context
                if note_context:
                    if combined_context:
                        combined_context += "\n\n---\n\n"
                    combined_context += f"相关笔记上下文：\n{note_context}"

                request.system_prompt = self.memory_injector.inject_into_system_prompt(
                    request.system_prompt,
                    combined_context
                )

            # 更新长期记忆系统
            useful_memory_ids = feedback_data.get('useful_memory_ids', [])
            new_memories_raw = feedback_data.get('new_memories', {})
            merge_groups_raw = feedback_data.get('merge_groups', [])

            # 使用 MemoryIDResolver 处理数据格式转换
            new_memories = MemoryIDResolver.normalize_new_memories_format(new_memories_raw, self.logger)
            merge_groups = MemoryIDResolver.normalize_merge_groups_format(merge_groups_raw)

            if useful_memory_ids or new_memories or merge_groups:
                # 调用 feedback（不传 query 参数）
                self.logger.debug(f"Calling feedback with: useful_ids={len(useful_memory_ids)}, new_memories={len(new_memories)}, merge_groups={len(merge_groups)}")
                self.memory_system.feedback(
                    useful_memory_ids=useful_memory_ids,
                    new_memories=new_memories,
                    merge_groups=merge_groups
                )

                # 3.2 把潜意识筛选的有用记忆加入短期记忆
                if useful_memory_ids:
                    useful_long_term_memories = []
                    for memory_id in useful_memory_ids:
                        for memory in long_term_memories:
                            if memory.id == memory_id:
                                useful_long_term_memories.append(memory)
                                break

                    if useful_long_term_memories:
                        self.session_memory_manager.add_memories_to_session(session_id, useful_long_term_memories)
                        self.logger.debug(f"潜意识筛选：{len(useful_long_term_memories)}条有用记忆进入短期记忆")

            self.logger.info(f"记忆整理完成（会话 {session_id}）：潜意识筛选出 {len(feedback_data.get('useful_memory_ids', []))} 条有用记忆进入短期记忆")

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
        """获取会话ID"""
        # 尝试从事件中获取发送者ID和群组ID
        sender_id = "unknown"
        group_id = "private"

        # 尝试获取发送者ID
        if hasattr(event, 'sender_id'):
            sender_id = str(event.sender_id)
        elif hasattr(event, 'user_id'):
            sender_id = str(event.user_id)

        # 尝试获取群组ID
        if hasattr(event, 'group_id'):
            group_id = str(event.group_id)
        elif hasattr(event, 'channel_id'):
            group_id = str(event.channel_id)

        return f"{sender_id}_{group_id}"

    def _base_memories_to_json(self, memories: List) -> List[Dict[str, Any]]:
        """将BaseMemory对象转换为JSON格式"""
        memories_json = []
        for memory in memories:
            memory_data = {
                'id': memory.id,
                'type': memory.memory_type.value,
                'strength': memory.strength,
                'judgment': memory.judgment,
                'reasoning': memory.reasoning,
                'tags': memory.tags
            }
            memories_json.append(memory_data)

        return memories_json

    def _session_memories_to_json(self, memories: List) -> List[Dict[str, Any]]:
        """将短期记忆对象转换为JSON格式"""
        memories_json = []
        for memory in memories:
            memory_data = {
                'id': memory.id,
                'type': memory.memory_type.value,
                'strength': memory.strength,
                'judgment': memory.judgment,
                'reasoning': memory.reasoning,
                'tags': memory.tags
            }
            memories_json.append(memory_data)

        return memories_json


    # _resolve_memory_ids 方法已移至 MemoryIDResolver 类中

    def _sleep(self):
        """执行记忆巩固（睡眠）"""
        if not self.is_enabled():
            return

        try:
            self.memory_system.consolidate_memories()
            self.logger.info("记忆巩固完成")
        except Exception as e:
            self.logger.error(f"记忆巩固失败: {e}")

    def _start_periodic_sleep(self):
        """启动定期睡眠定时器"""
        if not self.is_enabled():
            return

        sleep_interval = self.config.sleep_interval  # 该属性已经是秒
        if sleep_interval <= 0:
            return

        def sleep_worker():
            while not self._stop_sleep:
                time.sleep(sleep_interval)
                if not self._stop_sleep:
                    self._sleep()

        self._sleep_timer = threading.Thread(target=sleep_worker, daemon=True)
        self._sleep_timer.start()
        self.logger.info(f"启动定期睡眠，间隔: {sleep_interval}秒")

    def stop_sleep(self):
        """停止定期睡眠"""
        self._stop_sleep = True
        if self._sleep_timer and self._sleep_timer.is_alive():
            self._sleep_timer.join(timeout=5)
        self.logger.info("定期睡眠已停止")
