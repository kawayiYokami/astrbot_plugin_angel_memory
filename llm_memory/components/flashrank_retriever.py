"""
FlashRank 重排组件。

使用轻量级 Cross-Encoder (MultiBERT-L-12) 进行语义重排，替代传统的 BM25。
基于 ONNX Runtime，针对 CPU 推理进行了极致优化。
"""

from typing import List, Dict, Tuple

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# 尝试导入 FlashRank
try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False


class FlashRankRetriever:
    """
    FlashRank 重排器封装类。
    实现了单例模式以复用模型实例（虽然 FlashRank 本身也是轻量的）。
    """

    _instance = None
    _ranker = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FlashRankRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化 FlashRank 模型"""
        if self._initialized:
            return

        self._initialized = True

        if not HAS_FLASHRANK:
            logger.warning("FlashRank 未安装，无法使用重排功能。请运行: pip install flashrank")
            return

        try:
            model_name = "ms-marco-MultiBERT-L-12"  # 默认模型

            # 初始化 Ranker (FlashRank 会自动下载模型)

            # 初始化 Ranker (FlashRank 会自动下载模型)
            # MultiBERT-L-12 是目前 FlashRank 支持的模型中对中文语义理解最好的
            # 虽然速度比 MiniLM 慢一倍，但为了检索质量，这是值得的默认选择
            self._ranker = Ranker(model_name=model_name)
            logger.info(f"FlashRank 模型初始化完成 ({model_name})")
        except Exception as e:
            logger.error(f"FlashRank 初始化失败: {e}")
            self._ranker = None

    def rerank(
        self, query: str, candidates: List[Dict[str, str]], limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        对候选文档进行语义重排。

        Args:
            query: 查询文本
            candidates: 候选文档列表，每个元素需包含 'id' 和 'content'
            limit: 返回结果数量限制

        Returns:
            重排后的列表: [(doc_id, score), ...]
        """
        if not candidates:
            return []

        if not HAS_FLASHRANK or self._ranker is None:
            logger.error("FlashRank 不可用，跳过重排")
            return []

        try:
            # 1. 格式化输入数据
            passages = []
            for c in candidates:
                # FlashRank 需要 id 和 text 字段
                # meta 字段是可选的，可以放入原始 metadata
                passages.append({
                    "id": str(c.get("id", "")),
                    "text": str(c.get("content", "")),
                    "meta": c.get("metadata", {})
                })

            # 2. 构建请求
            request = RerankRequest(query=query, passages=passages)

            # 3. 执行重排
            results = self._ranker.rerank(request)

            # 4. 提取结果
            ranked_results = []
            for res in results[:limit]:
                # FlashRank 返回的结果包含 score (0-1之间)
                ranked_results.append((res["id"], float(res["score"])))

            return ranked_results

        except Exception as e:
            logger.error(f"FlashRank 重排执行失败: {e}")
            return []

    @classmethod
    def is_available(cls) -> bool:
        """检查 FlashRank 是否可用"""
        return HAS_FLASHRANK