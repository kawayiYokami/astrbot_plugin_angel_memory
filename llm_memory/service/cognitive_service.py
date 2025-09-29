"""
认知服务 - 统一的多类型记忆管理。

这是一个更高层次的服务，统一管理五种不同类型的记忆存储和检索。
为每种记忆类型提供专门的接口，同时支持跨记忆类型的复杂查询。
"""

from typing import List, Optional

from ..models.data_models import BaseMemory, ValidationError
from ..components.vector_store import VectorStore
from ..components.association_manager import AssociationManager
from ..config.system_config import system_config
from .memory_handlers import MemoryHandlerFactory
from .memory_manager import MemoryManager
from ...core.logger import get_logger


class CognitiveService:
    """
    认知服务类（Cognitive Service）- 统一的多类型记忆管理（中文核心概念）

    中文定义：统一管理五种记忆类型的存储和检索，为每种记忆提供专门的接口
    英文翻译：Unified management of five types of memory storage and retrieval, providing dedicated interfaces for each memory type

    功能特点：
    - 统一接口：为不同类型的记忆提供一致的访问方式
    - 协调管理：协调不同类型的知识存储和检索
    - 系统大脑：作为整个记忆系统的控制中心
    - 清醒睡眠：支持清醒模式（学习强化）和睡眠模式（巩固遗忘）
    """

    def __init__(self):
        """
        初始化认知服务。

        为每种记忆类型创建独立的向量存储实例。
        """
        # 设置日志记录器
        self.logger = get_logger()

        # 创建统一的向量存储实例，用于存储所有类型的记忆
        self.main_store = VectorStore(collection_name=system_config.collection_name)

        # 创建关联管理器
        self.association_manager = AssociationManager(self.main_store)

        # 创建记忆处理器工厂
        self.memory_handler_factory = MemoryHandlerFactory(self.main_store)

        # 创建记忆管理器
        self.memory_manager = MemoryManager(self.main_store, self.association_manager)

        # 执行数据库健康性检查
        self.memory_manager.health_check()

        # 记录初始化状态以验证VectorStore
        self.logger.info(f"认知服务初始化完成。向量存储客户端: {self.main_store.client}")

    # ===== 存储管理接口 =====

    def set_storage_path(self, new_path: str):
        """
        设置记忆系统的新存储路径。

        这个方法会更新所有向量存储的路径，包括主存储和关联管理器使用的存储。

        注意：切换路径后，之前的数据仍在原路径中。如果需要迁移数据，
        请手动复制数据库文件到新路径。

        Args:
            new_path: 新的存储路径（可以是绝对路径或相对路径）
        """
        try:
            # 更新主存储路径
            self.main_store.set_storage_path(new_path)

            # 更新系统配置中的路径
            from pathlib import Path
            system_config.index_dir = Path(new_path).parent

            self.logger.info(f"记忆系统存储路径已更新到: {new_path}")

        except Exception as e:
            self.logger.error(f"更新存储路径失败: {e}")
            raise

    # ===== 记忆接口 =====

    def remember(self, memory_type: str, judgment: str, reasoning: str, tags: List[str]) -> str:
        """
        记住一条记忆。

        Args:
            memory_type: 记忆类型（event/knowledge/skill/emotional/task）
            judgment: 论断
            reasoning: 解释
            tags: 标签列表

        Returns:
            创建的记忆ID
        """
        handler = self.memory_handler_factory.get_handler(memory_type)
        return handler.remember(judgment, reasoning, tags)

    def recall(self, memory_type: str, query: str, limit: int = 10, include_consolidated: bool = True) -> List[BaseMemory]:
        """
        回忆记忆。

        Args:
            memory_type: 记忆类型（event/knowledge/skill/emotional/task）
            query: 搜索查询
            limit: 返回数量限制
            include_consolidated: 是否包含已巩固记忆

        Returns:
            记忆列表
        """
        handler = self.memory_handler_factory.get_handler(memory_type)
        return handler.recall(query, limit, include_consolidated)

    # ===== 高级记忆功能 =====

    def comprehensive_recall(self, query: str, fresh_limit: int = None, consolidated_limit: int = None) -> List[BaseMemory]:
        """实现双轨检索：同时从新鲜记忆和已巩固记忆中检索相关内容"""
        return self.memory_manager.comprehensive_recall(query, fresh_limit, consolidated_limit)

    def consolidate_memories(self):
        """执行记忆巩固过程（睡眠模式）"""
        return self.memory_manager.consolidate_memories()

    def chained_recall(self, query: str, per_type_limit: int = 7, final_limit: int = 7) -> List[BaseMemory]:
        """链式多通道回忆 - 基于关联网络的多轮回忆"""
        memory_handlers = self.memory_handler_factory.handlers
        return self.memory_manager.chained_recall(query, per_type_limit, final_limit, memory_handlers)

    def feedback(
        self,
        useful_memory_ids: List[str] = None,
        new_memories: List[dict] = None,
        merge_groups: List[List[str]] = None
    ):
        """统一反馈接口 - 处理回忆后的反馈（核心工作流）"""
        memory_handlers = self.memory_handler_factory.handlers
        return self.memory_manager.process_feedback(useful_memory_ids, new_memories, merge_groups, memory_handlers)


    # ===== 管理功能 =====

    def clear_all_memories(self):
        """
        清空所有记忆。

        这是一个危险操作，会永久删除所有存储的记忆。
        """
        self.main_store.clear_all()
        self.logger.info("所有记忆已被清空。")

    @staticmethod
    def get_prompt() -> str:
        """
        获取记忆系统使用指南的提示词。

        下游模块可以将此提示词加入到系统提示词中，
        AI就能知道如何维护记忆系统。

        Returns:
            记忆系统使用指南的完整内容
        """
        import os
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts',
            'memory_system_guide.md'
        )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "记忆系统提示词文件未找到，请检查 llm_memory/prompts/memory_system_guide.md 是否存在。"
        except Exception as e:
            return f"读取记忆系统提示词失败: {str(e)}"
