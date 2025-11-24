"""
插件上下文 - 包装AstrBot Context并统一管理插件资源

提供统一的资源获取接口，隐藏复杂的依赖关系，
让下游服务只需要依赖PluginContext即可获得所有必要资源。
"""

from typing import Any, Dict, Optional
from pathlib import Path

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from ..llm_memory.utils.path_manager import PathManager
from ..llm_memory.components.vector_store import VectorStore
from ..llm_memory.components.embedding_provider import EmbeddingProvider


class PluginContext:
    """
    插件上下文 - 包装AstrBot Context

    统一管理插件所需的所有资源，包括：
    - AstrBot Context（包装）
    - PathManager（路径管理）
    - 插件配置
    - 供应商信息
    """

    def __init__(
        self, astrbot_context, config: Dict[str, Any], base_data_dir: str = None
    ):
        """
        初始化插件上下文

        Args:
            astrbot_context: AstrBot的Context对象
            config: 插件配置字典
            base_data_dir: 基础数据目录（从main.py传入）
        """
        self.astrbot_context = astrbot_context
        self.config = config or {}
        self.base_data_dir = base_data_dir
        self.logger = logger

        # 同步资源
        self.path_manager: PathManager = None
        # 异步资源，由ComponentFactory创建后设置
        self.embedding_provider: EmbeddingProvider = None
        self.vector_store: VectorStore = None

        # 初始化插件资源
        self._setup_plugin_resources()

        self.logger.info(
            f"PluginContext初始化完成 (提供商: {self.get_embedding_provider_id()})"
        )

    def _setup_plugin_resources(self):
        """初始化插件专用资源"""
        try:
            # 获取基础数据目录
            base_data_dir = self._get_base_data_dir()

            # 获取嵌入提供商ID
            embedding_provider_id = self.get_embedding_provider_id()

            # 创建并配置PathManager
            self.path_manager = PathManager()
            self.path_manager.set_provider(embedding_provider_id, base_data_dir)

            self.logger.debug(
                f"插件资源初始化完成: 提供商={embedding_provider_id}, 数据目录={base_data_dir}"
            )

        except Exception as e:
            self.logger.error(f"插件资源初始化失败: {e}")
            raise

    def _get_base_data_dir(self) -> str:
        """获取基础数据目录"""
        # 优先使用传入的数据目录（从main.py获取）
        if self.base_data_dir:
            return self.base_data_dir

        # 其次使用配置中的数据目录
        data_directory = self.config.get("data_directory")
        if data_directory:
            return str(data_directory)

        # 没有提供数据目录，抛出异常强制调用方传入
        raise ValueError(
            "未提供数据目录！请确保在初始化PluginContext时传入base_data_dir参数，"
            "或在配置中设置data_directory。数据目录应由外部传入，不应自动推测。"
        )

    # === AstrBot Context 代理方法 ===

    def get_all_providers(self):
        """获取所有LLM提供商（代理到AstrBot Context）"""
        return self.astrbot_context.get_all_providers()

    def get_all_embedding_providers(self):
        """获取所有嵌入提供商（代理到AstrBot Context）"""
        if hasattr(self.astrbot_context, "get_all_embedding_providers"):
            return self.astrbot_context.get_all_embedding_providers()
        return []

    def get_astrbot_context(self):
        """获取原始AstrBot Context对象"""
        return self.astrbot_context

    # === 插件资源获取方法 ===

    def get_path_manager(self) -> PathManager:
        """获取路径管理器"""
        return self.path_manager

    def get_embedding_provider_id(self) -> str:
        """获取嵌入提供商ID"""
        return self.config.get("astrbot_embedding_provider_id", "local")
    def get_enable_local_embedding(self) -> bool:
        """获取是否启用本地嵌入模型"""
        return self.config.get("enable_local_embedding", False)

    def get_llm_provider_id(self) -> str:
        """获取LLM提供商ID"""
        return self.config.get("provider_id", "")

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()

    # === 便捷的资源获取方法 ===

    def get_index_dir(self) -> Path:
        """获取索引目录"""
        return self.path_manager.get_index_dir()

    def get_tag_db_path(self) -> Path:
        """获取标签数据库路径"""
        return self.path_manager.get_tag_db_path()

    def get_file_db_path(self) -> Path:
        """获取文件数据库路径"""
        return self.path_manager.get_file_db_path()

    def get_chroma_db_path(self) -> Path:
        """获取Chroma数据库路径"""
        return self.path_manager.get_chroma_db_path()

    def get_current_provider(self) -> str:
        """获取当前供应商ID"""
        return self.path_manager.get_current_provider()

    def get_embedding_provider(self) -> Optional[EmbeddingProvider]:
        """获取由ComponentFactory创建的嵌入提供商实例"""
        return self.embedding_provider

    def get_vector_store(self) -> Optional[VectorStore]:
        """获取由ComponentFactory创建的向量存储实例"""
        return self.vector_store

    def set_embedding_provider(self, provider: EmbeddingProvider):
        """由ComponentFactory设置嵌入提供商实例"""
        self.embedding_provider = provider

    def set_vector_store(self, store: VectorStore):
        """由ComponentFactory设置向量存储实例"""
        self.vector_store = store

    # === 验证方法 ===

    def has_embedding_providers(self) -> bool:
        """检查是否有嵌入提供商"""
        embedding_providers = self.get_all_embedding_providers()
        return len(embedding_providers) > 0

    def has_llm_providers(self) -> bool:
        """检查是否有LLM提供商"""
        llm_providers = self.get_all_providers()
        return len(llm_providers) > 0

    def has_providers(self) -> bool:
        """检查是否有任何提供商"""
        return self.has_embedding_providers() or self.has_llm_providers()

    # === 更新方法 ===

    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        self.logger.debug(f"PluginContext配置已更新: {list(new_config.keys())}")

    def update_from_event(self, event_context):
        """从事件Context更新（如果需要）"""
        # 保存原始AstrBot Context
        self.astrbot_context = event_context
        self.logger.debug("PluginContext已更新事件Context")

    def __str__(self) -> str:
        """字符串表示"""
        return f"PluginContext(provider={self.get_current_provider()}, has_providers={self.has_providers()})"

    def __repr__(self) -> str:
        """详细表示"""
        return (
            f"PluginContext("
            f"embedding_provider='{self.get_embedding_provider_id()}', "
            f"llm_provider='{self.get_llm_provider_id()}', "
            f"has_providers={self.has_providers()}, "
            f"index_dir='{self.get_index_dir()}')"
            f")"
        )


class PluginContextFactory:
    """插件上下文工厂 - 提供统一的Context创建接口"""

    @staticmethod
    def create_from_initialization(
        astrbot_context, config: Dict[str, Any], base_data_dir: str = None
    ) -> PluginContext:
        """
        从初始化创建PluginContext

        Args:
            astrbot_context: AstrBot的Context对象
            config: 插件配置
            base_data_dir: 基础数据目录（从main.py传入）

        Returns:
            PluginContext实例
        """
        return PluginContext(astrbot_context, config, base_data_dir)

    @staticmethod
    def create_from_event(event, base_config: Dict[str, Any]) -> PluginContext:
        """
        从事件创建PluginContext

        Args:
            event: AstrBot事件对象
            base_config: 基础配置

        Returns:
            PluginContext实例
        """
        return PluginContext(event.context, base_config)

    @staticmethod
    def create_mock_context(config: Dict[str, Any] = None) -> PluginContext:
        """
        创建模拟的PluginContext（用于测试）

        Args:
            config: 模拟配置

        Returns:
            模拟的PluginContext实例
        """
        if config is None:
            config = {"astrbot_embedding_provider_id": "test", "provider_id": "test"}

        # 创建模拟的AstrBot Context
        class MockAstrbotContext:
            def get_all_providers(self):
                return []

            def get_all_embedding_providers(self):
                return []

        mock_context = MockAstrbotContext()
        return PluginContext(mock_context, config)
