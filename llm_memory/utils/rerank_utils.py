"""
重排结果解析工具

统一上游 rerank 返回形态，屏蔽不同实现之间的差异。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _read_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_rerank_items(rerank_resp: Any) -> List[Dict[str, Any]]:
    """把上游重排返回统一成 ``[{"index", "id", "score", "text"}]``。

    上游 provider 返回 AstrBot 的 ``RerankResult`` 对象（字段为
    ``relevance_score``），也可能返回原始 JSON dict（字段 ``relevance_score``
    或 ``score``）；两种形态都要能取到分数，缺失时保留 ``None`` 交由调用方决定降级。
    """
    if rerank_resp is None:
        return []

    items: Any = None
    if isinstance(rerank_resp, dict):
        if rerank_resp.get("code") not in (None, 0, 200, "0", "200"):
            return []
        items = rerank_resp.get("results")
        if items is None:
            data = rerank_resp.get("data")
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("results") or data.get("items")
    elif isinstance(rerank_resp, list):
        items = rerank_resp

    if not isinstance(items, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in items:
        if item is None:
            continue
        score = _coerce_float(_read_field(item, "relevance_score"))
        if score is None:
            score = _coerce_float(_read_field(item, "score"))
        normalized.append(
            {
                "index": _coerce_index(_read_field(item, "index")),
                "id": _read_field(item, "id"),
                "score": score,
                "text": _read_field(item, "text") or _read_field(item, "document"),
            }
        )
    return normalized
