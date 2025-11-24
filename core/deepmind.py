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
from .utils.memory_id_resolver import MemoryIDResolver
from ..llm_memory import CognitiveService
from ..llm_memory.utils.json_parser import JsonParser
from .session_memory import SessionMemoryManager
from .utils import SmallModelPromptBuilder, MemoryInjector
from .utils.feedback_queue import get_feedback_queue
from .utils.query_processor import get_query_processor
from .config import MemoryConstants

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
    NOTE_CANDIDATE_COUNT = 50  # 先找50条笔记，让小AI帮忙筛选有用的

    def __init__(
        self,
        config,
        context,
        vector_store,
        note_service,
        plugin_context, # 新增
        provider_id: str = "",
        cognitive_service=None,
    ):
        """
        初始化AI的潜意识系统

        Args:
            config: 配置信息（比如多久睡一次觉整理记忆）
            context: 聊天机器人的大脑（主意识）
            vector_store: 记忆数据库（存所有长期记忆的地方）
            note_service: 笔记管理器（重要信息专门存放处）
            plugin_context: 插件上下文（新增）
            provider_id: AI助手的ID，没有的话潜意识就睡觉不干活
            cognitive_service: 记忆管理服务（可选，如果外面已经有了就直接用）
        """
        self.config = config
        self.memory_system: Optional[CognitiveService] = (
            cognitive_service  # 使用注入的实例
        )
        self.note_service = note_service
        self.context = context
        self.vector_store = vector_store
        self.provider_id = provider_id
        self.plugin_context = plugin_context # 保存引用

        # 从全局容器获取logger
        self.logger = logger
        self.json_parser = JsonParser()

        # 获取配置值
        self.min_message_length = getattr(config, "min_message_length", 5)
        self.short_term_memory_capacity = getattr(
            config, "short_term_memory_capacity", 1.0
        )
        self.sleep_interval = getattr(config, "sleep_interval", 3600)  # 默认1小时
        self.small_model_note_budget = getattr(config, "small_model_note_budget", 8000)
        self.large_model_note_budget = getattr(config, "large_model_note_budget", 12000)

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

        # 初始化记忆系统（如果没有通过依赖注入提供）
        self._init_memory_system()

    def _init_memory_system(self) -> None:
        """初始化记忆系统"""
        # 如果已经有了注入的认知服务，直接使用
        if self.memory_system is not None:
            return

        # 否则创建新的认知服务实例（向后兼容）
        try:
            self.memory_system = CognitiveService(vector_store=self.vector_store)

        except Exception as e:
            self.logger.error(f"记忆系统初始化失败: {e}")
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
        if not hasattr(event, "angelmemory_context"):
            return None

        try:
            context_data = json.loads(event.angelmemory_context)
            return {
                "session_id": context_data["session_id"],
                "query": context_data.get("recall_query", ""),
                "user_list": context_data.get("user_list", []),
                # 添加原始数据字段
                "raw_memories": context_data.get("raw_memories", []),
                "raw_notes": context_data.get("raw_notes", []),
                "core_topic": context_data.get("core_topic", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"解析记忆上下文失败: {e}")
            return None

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
        # 1. 预处理记忆检索查询词
        if precompute_vectors:
            memory_query, memory_vector = await self.query_processor.process_query_for_memory_with_vector(query, event)
        else:
            memory_query = self.query_processor.process_query_for_memory(query, event)
            memory_vector = None

        # 1. 使用链式召回从长期记忆检索相关记忆
        long_term_memories = []
        if self.memory_system:
            try:
                # 从 angelheart_context 获取实体列表 (通过 QueryProcessor)
                rag_fields = self.query_processor.extract_rag_fields(event)
                entities = rag_fields.get("entities", [])

                # 安全获取 handlers
                handlers = None
                if hasattr(self.memory_system, 'memory_handler_factory') and self.memory_system.memory_handler_factory:
                    handlers = getattr(self.memory_system.memory_handler_factory, 'handlers', None)
                
                # 调用新的 chained_recall
                long_term_memories = await self.memory_system.chained_recall(
                    query=memory_query,
                    entities=entities,
                    per_type_limit=self.CHAINED_RECALL_PER_TYPE_LIMIT,
                    final_limit=self.CHAINED_RECALL_FINAL_LIMIT,
                    memory_handlers=handlers,
                    event=event,
                    vector=memory_vector,
                )

            except Exception as e:
                self.logger.error(f"链式召回失败，跳过记忆检索: {e}")
                long_term_memories = []

        # 2. 获取 secretary_decision 信息
        secretary_decision = {}
        try:
            if hasattr(event, "angelheart_context"):
                angelheart_data = json.loads(event.angelheart_context)
                secretary_decision = angelheart_data.get("secretary_decision", {})
        except (json.JSONDecodeError, KeyError):
            self.logger.error("无法获取 secretary_decision 信息")

        # 3. 使用统一的RAG字段进行笔记检索
        # 直接使用原始query，让QueryProcessor统一处理RAG字段
        if precompute_vectors:
            note_query, note_vector = await self.query_processor.process_query_for_notes_with_vector(query, event)
        else:
            note_query = self.query_processor.process_query_for_notes(query, event)
            note_vector = None

        # 4. 获取候选笔记（用于小模型的选择）
        candidate_notes = []
        if self.note_service:
            candidate_notes = await self.note_service.search_notes_by_token_limit(
                query=note_query,
                max_tokens=self.small_model_note_budget,
                recall_count=self.NOTE_CANDIDATE_COUNT,
                vector=note_vector  # 传递预计算向量
            )

        # 5. 创建短ID到完整ID的映射（用于后续上下文扩展）
        note_id_mapping = {}
        for note in candidate_notes:
            note_id = note.get("id")
            if note_id:
                short_id = MemoryIDResolver.generate_short_id(note_id)
                note_id_mapping[short_id] = note_id

        # 6. 创建短期记忆ID映射表（用于解析 useful_memory_ids）
        memory_id_mapping = {}
        if long_term_memories:
            memory_id_mapping = MemoryIDResolver.generate_id_mapping(
                [memory.to_dict() for memory in long_term_memories], "id"
            )

        return {
            "long_term_memories": long_term_memories,
            "candidate_notes": candidate_notes,
            "note_id_mapping": note_id_mapping,
            "memory_id_mapping": memory_id_mapping,
            "secretary_decision": secretary_decision,
            "core_topic": secretary_decision.get("topic", ""),
        }



    def _inject_memories_to_request(
        self, request: ProviderRequest, session_id: str, note_context: str
    ) -> None:
        """
        将记忆注入到LLM请求中

        Args:
            request: LLM请求对象
            session_id: 会话ID
            note_context: 笔记上下文
        """

        # 1. 从短期记忆推送给主意识（潜意识筛选后的精选记忆）
        short_term_memories = self.session_memory_manager.get_session_memories(
            session_id
        )

        memory_context = self.memory_injector.format_session_memories_for_prompt(
            short_term_memories
        )

        # 2. 合并记忆和笔记上下文
        if memory_context or note_context:
            # 注入记忆内容作为用户消息
            if memory_context:
                request.contexts.append({
                    "role": "user",
                    "content": f"[RAG-记忆] 相关记忆参考:\n{memory_context}"
                })

            # 注入笔记内容作为用户消息
            if note_context:
                request.contexts.append({
                    "role": "user",
                    "content": f"[RAG-笔记] 相关笔记参考:\n{note_context}"
                })

        else:
            pass

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
        useful_memory_ids = feedback_data.get("useful_memory_ids", [])
        new_memories_raw = feedback_data.get("new_memories", {})
        merge_groups_raw = feedback_data.get("merge_groups", [])

        # 1. 处理有用的旧记忆
        useful_long_term_memories = []
        if useful_memory_ids:
            memory_map = {memory.id: memory for memory in long_term_memories}
            useful_long_term_memories = [
                memory_map[memory_id]
                for memory_id in useful_memory_ids
                if memory_id in memory_map
            ]

        # 2. 处理新生成的记忆
        new_memories_normalized = MemoryIDResolver.normalize_new_memories_format(
            new_memories_raw, self.logger
        )
        new_memory_objects = []
        if new_memories_normalized:
            from ..llm_memory.models.data_models import BaseMemory, MemoryType

            for mem_dict in new_memories_normalized:
                try:
                    # 创建一个字典副本以进行修改
                    init_data = mem_dict.copy()

                    # 将 'type' 键重命名为 'memory_type' 并转换为枚举类型
                    if "type" in init_data:
                        init_data["memory_type"] = MemoryType(init_data.pop("type"))

                    # 现在，init_data 中的键与构造函数完全匹配
                    new_memory_objects.append(BaseMemory(**init_data))
                except Exception as e:
                    self.logger.warning(f"为新记忆创建BaseMemory对象失败: {e}")

        # 3. 更新短期记忆：添加新记忆，评估现有记忆，清理死亡记忆
        useful_memory_ids = [memory.id for memory in useful_long_term_memories]
        self.session_memory_manager.update_session_memories(
            session_id, new_memory_objects, useful_memory_ids
        )

        # 5. 后台异步处理长期记忆反馈
        merge_groups = MemoryIDResolver.normalize_merge_groups_format(merge_groups_raw)

        if useful_memory_ids or new_memories_normalized or merge_groups:
            task_payload = {
                "feedback_fn": self._execute_feedback_task,
                "session_id": session_id,
                # 将所有数据都放在顶层，与 'feedback_fn' 同级
                "useful_memory_ids": list(useful_memory_ids),
                "new_memories": new_memories_normalized,
                "merge_groups": merge_groups,
                # 'payload' 字段可以保留并传入 session_id，因为 _execute_feedback_task 会用到
                "payload": {"session_id": session_id},
            }
            await get_feedback_queue().submit(task_payload)
        else:
            pass

    async def _execute_feedback_task(
        self,
        useful_memory_ids: List[str],
        new_memories: List[Dict[str, Any]],
        merge_groups: List[List[str]],
        session_id: str,
    ) -> None:
        """异步执行的长期记忆反馈。"""

        # 检查 memory_system 是否可用
        if self.memory_system is not None:
            # 直接异步调用
            await self.memory_system.feedback(
                useful_memory_ids=useful_memory_ids,
                new_memories=new_memories,
                merge_groups=merge_groups,
            )
        else:
            self.logger.error("记忆系统不可用，跳过反馈")

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
        潜意识的核心工作：整理相关记忆喂给主意识

        工作流程：
        1. 从记忆库找出相关的内容
        2. 直接注入原始内容（极速响应改造）
        3. 把有用的记忆包装成记忆包
        4. 喂给主意识（LLM）帮助他思考

        Args:
            event: 用户的消息（触发回忆的线索）
            request: 即将发给主意识的请求（我们要往里面塞记忆）
        """
        session_id = self._get_session_id(event)

        # 1. 从 event.angelheart_context 中获取对话历史
        chat_records = []
        if hasattr(event, "angelheart_context"):
            try:
                angelheart_data = json.loads(event.angelheart_context)
                chat_records = angelheart_data.get("chat_records", [])
            except (json.JSONDecodeError, KeyError):
                self.logger.error(
                    f"为会话 {session_id} 解析 angelheart_context 失败"
                )

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
        event.angelmemory_context = json.dumps(
            {
                "memories": memories_json,
                "recall_query": query,  # 保留查询字符串供后续使用
                "recall_time": time.time(),
                "session_id": session_id,
                "user_list": user_list,  # 将新生成的"用户清单"存入上下文
            }
        )

        # 如果未配置 provider_id，跳过记忆整理
        if not self.provider_id:
            return

        # 解析记忆上下文数据
        context_data = self._parse_memory_context(event)
        if not context_data:
            return

        session_id = context_data["session_id"]
        query = context_data["query"]

        # 核心修复：将 plugin_context 附加到 event 对象
        event.plugin_context = self.plugin_context

        # 检索长期记忆和候选笔记
        retrieval_data = await self._retrieve_memories_and_notes(event, query, precompute_vectors=True)
        long_term_memories = retrieval_data["long_term_memories"]
        candidate_notes = retrieval_data["candidate_notes"]
        core_topic = retrieval_data["core_topic"]


        try:
            # 直接将检索到的长期记忆填入短期记忆，由session_memory按生命值淘汰
            if long_term_memories and self.memory_system:
                self.session_memory_manager.add_memories_to_session(
                    session_id, long_term_memories
                )

            # 直接注入原始笔记内容（不经过小模型筛选）
            note_context = ""
            if candidate_notes:
                # 构建笔记上下文，限制token数量
                from ..llm_memory.utils.token_utils import count_tokens

                current_tokens = 0
                selected_notes = []

                for note in candidate_notes:
                    note_content = note.get("content", "")
                    note_tokens = count_tokens(note_content)

                    # 检查是否超出大模型笔记预算
                    if current_tokens + note_tokens <= self.large_model_note_budget:
                        selected_notes.append(note)
                        current_tokens += note_tokens
                    else:
                        break

                # 构建笔记上下文
                if selected_notes:
                    # 使用新的方法构建笔记上下文，避免模型误解标签为引用
                    note_context_parts = []
                    for note in selected_notes:
                        content = note.get("content", "")
                        tags = note.get("tags", [])

                        # 清理笔记内容（去除所有空行）
                        cleaned_content = self._clean_note_content(content)

                        if tags:
                            # 如果有标签，构建新的引言格式
                            tags_str = ", ".join(tags)
                            intro_str = f"关于({tags_str})的笔记："
                            note_context_parts.append(f"{intro_str} {cleaned_content}")
                        else:
                            # 如果没有标签，直接添加内容
                            note_context_parts.append(cleaned_content)

                    # 合并所有笔记，只在开头添加一次时效性提醒
                    time_warning = "[注意：以下笔记内容可能不具备时效性，请勿作为最新消息看待]\n"
                    note_context = time_warning + "\n\n".join(note_context_parts)

            # 生成并传递ID映射表

            # 为记忆和笔记分别生成 ID => 短ID 的映射
            memory_id_mapping = MemoryIDResolver.generate_id_mapping(
                [mem.to_dict() for mem in long_term_memories], "id"
            )
            note_id_mapping = MemoryIDResolver.generate_id_mapping(
                candidate_notes, "id"
            )

            # 将原始上下文数据存入event.angelmemory_context，供异步分析使用
            try:
                angelmemory_context = (
                    json.loads(event.angelmemory_context)
                    if hasattr(event, "angelmemory_context")
                    and event.angelmemory_context
                    else {}
                )
                angelmemory_context["raw_memories"] = [
                    memory.to_dict() if hasattr(memory, "to_dict") else {}
                    for memory in long_term_memories
                ]
                angelmemory_context["raw_notes"] = candidate_notes
                angelmemory_context["core_topic"] = core_topic
                # 把ID映射表也一起存进去
                angelmemory_context["memory_id_mapping"] = memory_id_mapping
                angelmemory_context["note_id_mapping"] = note_id_mapping
                event.angelmemory_context = json.dumps(angelmemory_context)
            except Exception as e:
                self.logger.error(f"保存原始上下文数据失败: {e}")

            # 注入记忆到请求（从短期记忆中读取并注入）
            self._inject_memories_to_request(request, session_id, note_context)

        except Exception as e:
            import traceback

            self.logger.error(
                f"会话 {session_id} 的记忆组织失败: {e}"
            )
            self.logger.error(f"错误详情: {traceback.format_exc()}")

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
        # 如果睡眠间隔为0，表示禁用睡眠
        if sleep_interval <= 0:
            return False

        import time

        current_time = time.time()

        # 首次运行，立即睡眠
        if self.last_sleep_time is None:
            await self._sleep()
            self.last_sleep_time = current_time
            return True

        # 计算距离上次睡眠的时间
        time_since_last_sleep = current_time - self.last_sleep_time

        # 检查是否到达睡眠间隔
        if time_since_last_sleep >= sleep_interval:
            await self._sleep()
            self.last_sleep_time = current_time
            return True
        else:
            # 未到睡眠时间
            return False

    async def _sleep(self):
        """AI睡觉整理记忆：重要内容加强，无用内容清理"""
        if not self.is_enabled():
            return

        import time

        start_time = time.time()

        try:
            # 检查 memory_system 是否可用
            if self.memory_system is not None:
                await self.memory_system.consolidate_memories()
                elapsed_time = time.time() - start_time
            else:
                self.logger.error(
                    "记忆系统不可用，跳过巩固"
                )
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"记忆巩固失败（耗时 {elapsed_time:.2f} 秒）: {e}")

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

        # 直接将任务提交到后台队列，不等待LLM响应
        task_payload = {
            "feedback_fn": self._execute_async_analysis_task,
            "session_id": session_id,
            "payload": {
                "event_data": self._serialize_event_data(event),
                "response_data": self._serialize_response_data(response),
                "session_id": session_id,
            },
            # 添加这些字段以确保任务能被正确刷新到队列中
            # 即使是空列表，也能确保任务被处理
            "useful_memory_ids": [],  # 这些字段是为了确保任务能被反馈队列正确处理
            "new_memories": [],
            "merge_groups": [],
        }

        # 提交到反馈队列后台执行
        await get_feedback_queue().submit(task_payload)

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
                def __init__(self, data):
                    self.angelmemory_context = data.get("angelmemory_context")
                    self.angelheart_context = data.get("angelheart_context")
                    self.unified_msg_origin = data.get("unified_msg_origin")

            event = SimpleEvent(event_data)

            # 获取原始上下文数据
            context_data = self._parse_memory_context(event)
            if not context_data:
                return

            query = context_data["query"]

            # 获取原始记忆和笔记数据
            raw_memories_data = context_data.get("raw_memories", [])
            raw_notes_data = context_data.get("raw_notes", [])
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
            note_id_mapping = context_data.get("note_id_mapping", {})

            # 构建反思提示词（使用模块化的提示词构建器，现在展示短ID）
            prompt = SmallModelPromptBuilder.build_post_hoc_analysis_prompt(
                historical_query=query,
                main_llm_response=response_text,
                raw_memories=long_term_memories,
                raw_notes=raw_notes_data,
                core_topic=core_topic,
                memory_id_mapping=memory_id_mapping,  # 传递记忆ID映射表
                note_id_mapping=note_id_mapping,  # 传递笔记ID映射表
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

            # ID解析：使用映射表将LLM返回的短ID翻译回长ID
            memory_id_mapping = context_data.get("memory_id_mapping", {})
            note_id_mapping = context_data.get("note_id_mapping", {})

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

            # --- 修正结束 ---

            # 3. 调用封装好的 feedback 接口，并使用"转换后"的扁平列表
            #    (以及我们之前讨论过的，让 feedback 返回新创建的对象)
            newly_created_memories = []
            if self.memory_system:
                # 直接异步调用
                newly_created_memories = await self.memory_system.feedback(
                    useful_memory_ids=feedback_data.get("useful_memory_ids", []),
                    new_memories=new_memories_normalized,  # <--- 使用转换后的数据
                    merge_groups=feedback_data.get("merge_groups", []),
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
