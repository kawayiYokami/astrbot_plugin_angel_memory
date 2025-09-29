"""
DeepMind核心模块

负责记忆整理的核心逻辑：召回、筛选、格式化和更新记忆。
"""

import time
import re
import json
import os
import threading
from typing import List, Dict, Any, Optional, Tuple
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse

from ..llm_memory import CognitiveService
from ..llm_memory.models.data_models import BaseMemory
from ..llm_memory.utils.json_parser import JsonParser
from ..llm_memory.config.system_config import MemorySystemConfig
from .session_memory import SessionMemoryManager
from .utils import SmallModelPromptBuilder, MemoryInjector
from .logger import get_logger


class DeepMind:
    """DeepMind记忆整理核心类"""

    def __init__(self, config: MemorySystemConfig, context, provider_id: str = ""):
        """
        初始化DeepMind

        Args:
            config: 配置管理器
            context: AstrBot上下文对象
            provider_id: LLM提供商ID，留空则跳过记忆整理
        """
        self.config = config
        self.memory_system: Optional[CognitiveService] = None
        self.context = context
        self.provider_id = provider_id
        # 从全局容器获取logger
        self.logger = get_logger()
        self.json_parser = JsonParser()

        # 初始化短期记忆管理器
        self.session_memory_manager = SessionMemoryManager(
            capacity_multiplier=getattr(config, 'short_term_memory_capacity', 1)
        )

        # 初始化工具类
        self.prompt_builder = SmallModelPromptBuilder()
        self.memory_injector = MemoryInjector()

        # 睡眠相关
        self._sleep_timer = None
        self._stop_sleep = False

        # 初始化记忆系统
        self._init_memory_system()

    def _init_memory_system(self):
        """初始化记忆系统"""
        try:
            self.memory_system = CognitiveService()
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
        if len(message_text) < getattr(self.config, 'min_message_length', 5):
            self.logger.debug(f"消息被过滤: 长度过短 ({len(message_text)} < {getattr(self.config, 'min_message_length', 5)})")
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

            # 如果没有对话记录，使用当前消息文本
            if not chat_records:
                message_text = self._extract_message_text(event)
                query = message_text if message_text else ""
            else:
               # 2. 格式化对话历史为查询字符串
               query, user_list = self.prompt_builder.format_chat_records(chat_records)

            # 3. 使用格式化的查询进行链式回忆
            memories = self.memory_system.chained_recall(query=query)

            # 4. 将召回的记忆加入短期记忆
            self.session_memory_manager.add_memories_to_session(session_id, memories)

            # 5. 从短期记忆中获取记忆
            session_memories = self.session_memory_manager.get_session_memories(session_id)

            # 6. 转换为JSON格式
            memories_json = self._session_memories_to_json(session_memories)

            # 注入到事件中（使用angelmemory_context）
            event.angelmemory_context = json.dumps({
                'memories': memories_json,
                'recall_query': query,
                'recall_time': time.time(),
                'session_id': session_id,
                'user_list': user_list  # 将新生成的“用户清单”存入上下文
            })

            self.logger.info(f"Memory recall completed for session {session_id}: recalled {len(session_memories)} memories")

        except Exception as e:
            self.logger.error(f"Memory recall failed for session {session_id}: {e}")

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
            user_list = context_data.get('user_list', [])  # 从上下文中恢复“用户清单”
        except (json.JSONDecodeError, KeyError):
            return

        # 从短期记忆获取记忆
        session_memories = self.session_memory_manager.get_session_memories(session_id)

        try:
            # 构建小模型提示词
            prompt = self.prompt_builder.build_memory_prompt(query, session_memories, user_list)

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

            # 解析结构化输出
            feedback_data = self.json_parser.parse_llm_response(response_text)
            self.logger.debug(f"Parsed feedback_data type: {type(feedback_data)}, value: {feedback_data}")

            if feedback_data is None:
                self.logger.error(f"JSON parsing failed for session {session_id}")
                return

            # 确保 feedback_data 是字典
            if not isinstance(feedback_data, dict):
                self.logger.error(f"feedback_data is not a dict, it's {type(feedback_data)}: {feedback_data}")
                return

            # ID解析：将短ID转换为完整ID
            if 'useful_memory_ids' in feedback_data:
                feedback_data['useful_memory_ids'] = self._resolve_memory_ids(
                    feedback_data['useful_memory_ids'],
                    session_memories
                )

            # 将反馈数据保存到事件中，供后续更新使用
            event.memory_feedback = {
                'feedback_data': feedback_data,
                'session_id': session_id
            }

            # 注入有用的记忆到提示词
            memory_context = self.memory_injector.format_memories_for_prompt(feedback_data, session_memories)
            if memory_context:
                request.system_prompt = self.memory_injector.inject_into_system_prompt(
                    request.system_prompt,
                    memory_context
                )

            # 更新长期记忆系统
            useful_memory_ids = feedback_data.get('useful_memory_ids', [])
            new_memories_raw = feedback_data.get('new_memories', {})
            merge_groups_raw = feedback_data.get('merge_groups', [])

            # 转换 new_memories 格式：从字典（按类型分组）转换为列表
            new_memories = []
            if isinstance(new_memories_raw, dict):
                for memory_type, memories in new_memories_raw.items():
                    if isinstance(memories, list):
                        for memory in memories:
                            # 检查 memory 是否是字典
                            if not isinstance(memory, dict):
                                self.logger.warning(f"Skipping non-dict memory in {memory_type}: {type(memory)} - {memory}")
                                continue
                            # 添加类型字段
                            memory['type'] = memory_type
                            new_memories.append(memory)
            elif isinstance(new_memories_raw, list):
                # 如果已经是列表，直接使用
                new_memories = new_memories_raw

            self.logger.debug(f"Converted new_memories: {new_memories}")

            # 转换 merge_groups 格式：从对象列表提取 ids 字段
            merge_groups = []
            if isinstance(merge_groups_raw, list):
                for group in merge_groups_raw:
                    if isinstance(group, dict) and 'ids' in group:
                        merge_groups.append(group['ids'])
                    elif isinstance(group, list):
                        # 如果已经是列表格式，直接使用
                        merge_groups.append(group)

            if useful_memory_ids or new_memories or merge_groups:
                # 调用 feedback（不传 query 参数）
                self.logger.debug(f"Calling feedback with: useful_ids={len(useful_memory_ids)}, new_memories={len(new_memories)}, merge_groups={len(merge_groups)}")
                self.memory_system.feedback(
                    useful_memory_ids=useful_memory_ids,
                    new_memories=new_memories,
                    merge_groups=merge_groups
                )

                # 更新短期记忆：重新召回并添加到会话
                # 这样可以确保短期记忆中包含最新的记忆
                updated_memories = self.memory_system.chained_recall(query=query)
                self.session_memory_manager.add_memories_to_session(session_id, updated_memories)

            self.logger.info(f"Memory organization completed for session {session_id}: injected {len(feedback_data.get('useful_memory_ids', []))} useful memories")

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

    def _session_memories_to_json(self, memories: List) -> List[Dict[str, Any]]:
        """将短期记忆对象转换为JSON格式"""
        memories_json = []
        for memory in memories:
            memory_data = {
                'id': memory.id,
                'type': memory.type,
                'strength': memory.strength,
                'judgment': memory.judgment,
                'reasoning': memory.reasoning,
                'tags': memory.tags
            }
            memories_json.append(memory_data)

        return memories_json

    def _resolve_memory_ids(self, short_ids: List[str], memories: List) -> List[str]:
        """
        将短ID转换为完整ID

        Args:
            short_ids: 短ID列表（如 ["a1b2c3", "d4e5f6"]）
            memories: 记忆列表

        Returns:
            完整ID列表
        """
        resolved_ids = []

        for short_id in short_ids:
            # 在记忆中查找匹配的完整ID
            for memory in memories:
                if memory.id.startswith(short_id):
                    resolved_ids.append(memory.id)
                    break
            else:
                # 如果没有找到匹配的ID，记录警告但继续处理
                self.logger.warning(f"未找到匹配的完整ID: {short_id}")

        return resolved_ids

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

        sleep_interval = self.config.consolidation_interval_hours * 3600  # 转换为秒
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
