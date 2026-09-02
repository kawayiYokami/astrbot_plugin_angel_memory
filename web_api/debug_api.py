"""
功能调试 API（嵌入 / 重排 探针）

仅做单次覆盖，不落盘、不污染 retrieval.* 全局配置。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class DebugAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_embedding_provider(self) -> Optional[Any]:
        try:
            provider = self.plugin_context.get_embedding_provider()
            if provider is not None:
                return provider
        except Exception:
            pass
        try:
            provider = self.plugin_context.get_component("embedding_provider")
            if provider is not None:
                return provider
        except Exception:
            pass
        return None

    def _get_rerank_provider(self) -> Optional[Any]:
        # 优先从已创建的组件实例中取
        for comp_name in ("memory_sql_manager", "note_chunk_search", "vector_store"):
            try:
                comp = self.plugin_context.get_component(comp_name)
                if comp is None:
                    continue
                # vector_store 可能是 FaissVectorStore / SqliteVectorStore
                rp = getattr(comp, "_rerank_provider", None)
                if rp is not None and hasattr(rp, "rerank"):
                    return rp
                rp = getattr(comp, "rerank_provider", None)
                if rp is not None and hasattr(rp, "rerank"):
                    return rp
                # HybridRetrievalEngine 内部
                hybrid = getattr(comp, "_hybrid_engine", None)
                if hybrid is not None:
                    rp2 = getattr(hybrid, "rerank_provider", None)
                    if rp2 is not None and hasattr(rp2, "rerank"):
                        return rp2
            except Exception:
                continue
        # 回退到上游 context 查找
        try:
            ctx = self.plugin_context.get_astrbot_context()
            # 显式 ID
            try:
                rerank_id = self.plugin_context.get_rerank_provider_id()
                if rerank_id and hasattr(ctx, "get_rerank_provider_by_id"):
                    p = ctx.get_rerank_provider_by_id(rerank_id)
                    if p and hasattr(p, "rerank"):
                        return p
                llm_id = self.plugin_context.get_llm_provider_id()
                if llm_id and hasattr(ctx, "get_provider_by_id"):
                    p = ctx.get_provider_by_id(llm_id)
                    if p and hasattr(p, "rerank"):
                        return p
            except Exception:
                pass
            if hasattr(ctx, "get_all_rerank_providers"):
                for p in ctx.get_all_rerank_providers() or []:
                    if hasattr(p, "rerank"):
                        return p
            if hasattr(ctx, "get_all_providers"):
                for p in ctx.get_all_providers() or []:
                    if hasattr(p, "rerank"):
                        return p
        except Exception:
            pass
        return None

    @staticmethod
    def _clamp_timeout(raw: Any, default: int = 5) -> int:
        try:
            v = int(raw) if raw is not None else default
        except Exception:
            return default
        return max(5, min(120, v))

    @staticmethod
    def _clamp_batch(raw: Any, default: int = 50) -> int:
        try:
            v = int(raw) if raw is not None else default
        except Exception:
            return default
        return max(1, v)

    @staticmethod
    def _extract_rerank_items(resp: Any) -> List[Dict[str, Any]]:
        if resp is None:
            return []
        items = None
        if isinstance(resp, dict):
            if resp.get("code") not in (None, 0, 200, "0", "200"):
                return []
            items = resp.get("results")
            if items is None:
                data = resp.get("data")
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("results") or data.get("items")
        elif isinstance(resp, list):
            items = resp
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in items:
            if isinstance(it, dict):
                out.append(it)
            else:
                out.append(
                    {
                        "index": getattr(it, "index", None),
                        "score": getattr(it, "score", None),
                        "id": getattr(it, "id", None),
                    }
                )
        return out

    async def embedding_probe(self):
        """POST /debug/embedding  嵌入探针（单次覆盖批次/超时）"""
        try:
            data = await request.get_json(silent=True) or {}
        except Exception:
            data = {}

        # 兼容前端两种传参：texts: string[] 或 text: 多行字符串
        raw_texts = data.get("texts")
        if raw_texts is None:
            raw_text = str(data.get("text", "") or "")
            raw_texts = [line.strip() for line in raw_text.splitlines() if line.strip()] if raw_text else []

        if isinstance(raw_texts, str):
            raw_texts = [line.strip() for line in raw_texts.splitlines() if line.strip()]

        texts: List[str] = []
        if isinstance(raw_texts, list):
            for item in raw_texts:
                t = str(item or "").strip()
                if t:
                    texts.append(t)
        if not texts:
            return jsonify({"error": "texts 不能为空（至少 1 行）", "total": 0}), 400

        if len(texts) > 200:
            return jsonify({"error": f"文本过多（{len(texts)} 行），上限 200 行，请分批测试", "total": len(texts)}), 400

        batch_size = self._clamp_batch(data.get("batch_size"), 50)
        timeout = self._clamp_timeout(data.get("timeout"), 5)

        provider = self._get_embedding_provider()
        if provider is None:
            return jsonify({"error": "嵌入提供商不可用（未配置或未初始化）", "has_provider": False}), 503

        # 手动按 batch_size 分批，避免篡改 provider.batch_size 全局状态
        batches: List[List[str]] = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        provider_info = {}
        try:
            if hasattr(provider, "get_model_info"):
                provider_info = provider.get_model_info() or {}
        except Exception:
            provider_info = {}

        overall_start = time.time()
        batch_results: List[Dict[str, Any]] = []
        dimensions: List[int] = []
        failed = 0
        timed_out_batches = 0
        preview_embeddings: List[Dict[str, Any]] = []

        # 整体上限 90s，防止大批次卡死
        overall_budget = 90

        for idx, batch in enumerate(batches):
            elapsed_budget = time.time() - overall_start
            if elapsed_budget >= overall_budget:
                batch_results.append(
                    {
                        "batch_index": idx,
                        "batch_size": len(batch),
                        "elapsed_ms": 0,
                        "success": False,
                        "timed_out": False,
                        "error": f"整体探针超时（>{overall_budget}s），剩余 {len(batches)-idx} 批未执行",
                    }
                )
                failed += len(batches) - idx
                break

            # 单批超时取用户指定 timeout
            start = time.time()
            try:
                # provider.embed_documents 自带缓存与 413 自动减半
                coro = provider.embed_documents(batch)
                vectors = await asyncio.wait_for(coro, timeout=timeout)
                elapsed_ms = int((time.time() - start) * 1000)
                if not vectors or not isinstance(vectors, list):
                    raise RuntimeError("提供商返回为空")
                dim = 0
                try:
                    dim = len(vectors[0]) if vectors and isinstance(vectors[0], (list, tuple)) else 0
                    if dim:
                        dimensions.append(dim)
                except Exception:
                    pass
                # 仅保留前 3 条预览，避免响应过大
                for b_idx, vec in enumerate(vectors[:3]):
                    try:
                        preview_embeddings.append(
                            {
                                "global_index": idx * batch_size + b_idx,
                                "dimension": len(vec) if isinstance(vec, (list, tuple)) else 0,
                                "preview": (vec[:3] if isinstance(vec, (list, tuple)) and len(vec) >= 3 else vec),
                            }
                        )
                    except Exception:
                        pass
                batch_results.append(
                    {
                        "batch_index": idx,
                        "batch_size": len(batch),
                        "elapsed_ms": elapsed_ms,
                        "success": True,
                        "timed_out": False,
                        "dimension": dim,
                    }
                )
            except asyncio.TimeoutError:
                elapsed_ms = int((time.time() - start) * 1000)
                failed += 1
                timed_out_batches += 1
                batch_results.append(
                    {
                        "batch_index": idx,
                        "batch_size": len(batch),
                        "elapsed_ms": elapsed_ms,
                        "success": False,
                        "timed_out": True,
                        "error": f"单批超时 provider 超时={timeout}s",
                    }
                )
                logger.warning(f"[WebUI-探针] 嵌入单批超时 batch={idx} size={len(batch)} timeout={timeout}s")
            except Exception as e:
                elapsed_ms = int((time.time() - start) * 1000)
                failed += 1
                err_msg = str(e)[:500]
                # 识别常见 413/429 便于前端提示
                err_kind = "unknown"
                lower = err_msg.lower()
                if "413" in err_msg or "batch size" in lower:
                    err_kind = "batch_too_large_413"
                elif "429" in err_msg or "rate limit" in lower or "throttling" in lower:
                    err_kind = "rate_limit_429"
                elif "timeout" in lower:
                    err_kind = "timeout"
                batch_results.append(
                    {
                        "batch_index": idx,
                        "batch_size": len(batch),
                        "elapsed_ms": elapsed_ms,
                        "success": False,
                        "timed_out": False,
                        "error": err_msg,
                        "error_kind": err_kind,
                    }
                )
                logger.warning(f"[WebUI-探针] 嵌入单批失败 batch={idx} 异常={e}")

        overall_elapsed = int((time.time() - overall_start) * 1000)
        # 统计维度一致性
        dim_set = sorted(set(dimensions)) if dimensions else []
        dimension = dim_set[0] if len(dim_set) == 1 else (dim_set[0] if dim_set else 0)

        return jsonify(
            {
                "total": len(texts),
                "batches": len(batches),
                "batch_size": batch_size,
                "timeout": timeout,
                "elapsed_ms": overall_elapsed,
                "avg_ms_per_text": round(overall_elapsed / max(1, len(texts)), 2),
                "dimension": dimension,
                "dimension_set": dim_set,
                "failed_batches": failed,
                "timed_out_batches": timed_out_batches,
                "provider": {
                    "provider_id": provider_info.get("provider_id") or getattr(provider, "provider_id", ""),
                    "model_name": provider_info.get("model_name") or provider_info.get("model") or "",
                    "type": getattr(provider, "get_provider_type", lambda: "")() if hasattr(provider, "get_provider_type") else "",
                },
                "batch_details": batch_results,
                "preview": preview_embeddings[:5],
            }
        )

    async def rerank_probe(self):
        """POST /debug/rerank  重排探针（单次覆盖超时）"""
        try:
            data = await request.get_json(silent=True) or {}
        except Exception:
            data = {}

        query = str(data.get("query", "") or "").strip()
        raw_docs = data.get("documents")
        if raw_docs is None:
            raw_text = str(data.get("text", "") or "")
            raw_docs = [line.strip() for line in raw_text.splitlines() if line.strip()] if raw_text else []
        if isinstance(raw_docs, str):
            raw_docs = [line.strip() for line in raw_docs.splitlines() if line.strip()]

        documents: List[str] = []
        if isinstance(raw_docs, list):
            for item in raw_docs:
                t = str(item or "").strip()
                if t:
                    documents.append(t)

        if not query:
            return jsonify({"error": "query 不能为空"}), 400
        if not documents:
            return jsonify({"error": "documents 不能为空（至少 1 行）"}), 400
        if len(documents) > 100:
            return jsonify({"error": f"documents 过多（{len(documents)} 行），上限 100 行"}), 400

        timeout = self._clamp_timeout(data.get("timeout"), 5)

        provider = self._get_rerank_provider()
        if provider is None or not hasattr(provider, "rerank"):
            return jsonify(
                {
                    "has_rerank": False,
                    "error": "未配置重排提供商，将降级为 BM25/融合（本次探针无重排可测）",
                    "query": query,
                    "documents": documents,
                    "timeout": timeout,
                }
            ), 200

        provider_id = ""
        try:
            provider_id = str(getattr(provider, "provider_id", "") or getattr(provider, "id", "") or "")
        except Exception:
            provider_id = ""

        start = time.time()
        timed_out = False
        error = None
        raw_resp = None
        try:
            method = getattr(provider, "rerank")
            resp = method(query=query, documents=documents)
            if inspect.isawaitable(resp):
                try:
                    raw_resp = await asyncio.wait_for(resp, timeout=timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    raw_resp = None
            else:
                raw_resp = resp
        except Exception as e:
            error = str(e)[:800]
            logger.warning(f"[WebUI-探针] 重排失败 query={query[:40]} 异常={e}")

        elapsed_ms = int((time.time() - start) * 1000)

        if timed_out:
            return jsonify(
                {
                    "has_rerank": True,
                    "provider_id": provider_id,
                    "query": query,
                    "documents": documents,
                    "timeout": timeout,
                    "elapsed_ms": elapsed_ms,
                    "timed_out": True,
                    "error": f"重排超时（>{timeout}s），已降级",
                    "scores": [],
                }
            )

        if error:
            return jsonify(
                {
                    "has_rerank": True,
                    "provider_id": provider_id,
                    "query": query,
                    "timeout": timeout,
                    "elapsed_ms": elapsed_ms,
                    "timed_out": False,
                    "error": error,
                    "scores": [],
                }
            )

        items = self._extract_rerank_items(raw_resp)
        # 归一化为 {index, score} 方便前端展示
        scores: List[Dict[str, Any]] = []
        for it in items:
            try:
                idx = it.get("index")
                if idx is None:
                    # 尝试按 id/text 反推 index
                    continue
                idx = int(idx)
                if idx < 0 or idx >= len(documents):
                    continue
                sc = it.get("score", it.get("relevance_score", it.get("relevanceScore", 0)))
                try:
                    sc = float(sc)
                except Exception:
                    sc = 0.0
                scores.append({"index": idx, "score": sc, "document": documents[idx][:120]})
            except Exception:
                continue

        # 若 provider 未返回 index，仅返回原始 items
        if not scores and items:
            # 尝试直接取 relevance_score 排序
            for idx, it in enumerate(items[: len(documents)]):
                try:
                    sc = float(it.get("relevance_score", it.get("score", 0)) or 0)
                except Exception:
                    sc = 0.0
                scores.append({"index": idx, "score": sc, "document": documents[idx][:120]})

        scores.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

        return jsonify(
            {
                "has_rerank": True,
                "provider_id": provider_id,
                "query": query,
                "timeout": timeout,
                "elapsed_ms": elapsed_ms,
                "timed_out": False,
                "scores": scores,
                "raw_item_count": len(items),
            }
        )
