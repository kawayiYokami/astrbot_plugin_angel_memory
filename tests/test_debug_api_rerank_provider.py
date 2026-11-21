"""重排探针按配置 ID 取提供商：框架没有 get_rerank_provider_by_id，得走 get_provider_by_id。

真实 AstrBot 4.30 的 Context 只有 get_provider_by_id / get_all_providers，
后者还只返回文本生成 provider，扫不到重排提供商。早先探针把按 ID 查找整段嵌在
`hasattr(ctx, "get_rerank_provider_by_id")` 里，那个判断恒为 False，
于是配置里配好的重排提供商从来没被查过，探针恒报「未配置重排提供商」。
"""

from astrbot_plugin_angel_memory.web_api.debug_api import DebugAPI


class _Provider:
    """提供商替身：带 rerank 的才具备重排能力。"""

    def __init__(self, provider_id, supports_rerank=False):
        self.provider_id = provider_id
        if supports_rerank:

            async def rerank(query, documents, top_n=None):
                return []

            self.rerank = rerank


class _Host:
    """宿主替身：与真实框架同形，没有 get_rerank_provider_by_id / get_all_rerank_providers。"""

    def __init__(self, providers):
        self._by_id = {p.provider_id: p for p in providers}
        # 真实框架的 get_all_providers() 只返回文本生成 provider，不含重排提供商
        self._chat_only = [p for p in providers if not hasattr(p, "rerank")]

    def get_provider_by_id(self, provider_id):
        return self._by_id.get(provider_id)

    def get_all_providers(self):
        return list(self._chat_only)


class _FutureHost(_Host):
    """假设未来框架补上专用入口，探针应优先用它。"""

    def __init__(self, providers, rerank_providers):
        super().__init__(providers)
        self._rerank_only = list(rerank_providers)

    def get_rerank_provider_by_id(self, provider_id):
        for p in self._rerank_only:
            if p.provider_id == provider_id:
                return p
        return None

    def get_all_rerank_providers(self):
        return list(self._rerank_only)


class _PluginContext:
    def __init__(self, host, rerank_id="", llm_id=""):
        self._host = host
        self._rerank_id = rerank_id
        self._llm_id = llm_id

    def get_astrbot_context(self):
        return self._host

    def get_rerank_provider_id(self):
        return self._rerank_id

    def get_llm_provider_id(self):
        return self._llm_id


def _api(providers, rerank_id="", llm_id=""):
    return DebugAPI(_PluginContext(_Host(providers), rerank_id, llm_id))


def test_resolves_configured_rerank_provider_without_dedicated_getter():
    """宿主没有专用入口时，仍按配置的 rerank_provider_id 取到重排提供商。"""
    host_providers = [
        _Provider("siliconflow-chat"),
        _Provider("bge-reranker", supports_rerank=True),
    ]
    api = _api(host_providers, rerank_id="bge-reranker", llm_id="siliconflow-chat")

    provider = api._get_rerank_provider()

    assert provider is not None
    assert provider.provider_id == "bge-reranker"
    # 前提：真实框架没有专用入口，回归点就在这条路径上
    assert not hasattr(api.plugin_context.get_astrbot_context(), "get_rerank_provider_by_id")


def test_falls_back_to_llm_provider_when_rerank_id_unset():
    """未配重排 ID 时用对话模型兜底：它没有 rerank，全量提供商里也扫不到重排，返回 None。"""
    host_providers = [
        _Provider("siliconflow-chat"),
        _Provider("bge-reranker", supports_rerank=True),
    ]
    api = _api(host_providers, rerank_id="", llm_id="siliconflow-chat")

    assert api._get_rerank_provider() is None


def test_rerank_id_pointing_at_missing_provider_returns_none():
    """配了 ID 但提供商不存在时不抛异常，返回 None。"""
    api = _api([_Provider("siliconflow-chat")], rerank_id="gone", llm_id="siliconflow-chat")

    assert api._get_rerank_provider() is None


def test_no_rerank_capable_provider_returns_none():
    """全都不带 rerank 时返回 None，探针据此报降级。"""
    api = _api([_Provider("chat-a"), _Provider("chat-b")], rerank_id="chat-a", llm_id="chat-a")

    assert api._get_rerank_provider() is None


def test_prefers_dedicated_getter_when_framework_provides_one():
    """未来框架若补上专用入口，仍应能按 ID 取到重排提供商。"""
    rerank_provider = _Provider("bge-reranker", supports_rerank=True)
    host = _FutureHost([_Provider("siliconflow-chat"), rerank_provider], [rerank_provider])
    api = DebugAPI(_PluginContext(host, rerank_id="bge-reranker", llm_id="siliconflow-chat"))

    assert api._get_rerank_provider() is rerank_provider


class _RerankResult:
    """框架 RerankResult 的同形替身：只有 index / relevance_score。"""

    def __init__(self, index, relevance_score):
        self.index = index
        self.relevance_score = relevance_score


def test_extract_reads_relevance_score_from_framework_rerank_result():
    """框架 RerankResult 没有 score 字段，归一化后必须把 relevance_score 带出来。"""
    raw = [_RerankResult(2, 0.87), _RerankResult(0, 0.12)]

    items = DebugAPI._extract_rerank_items(raw)

    assert [DebugAPI._pick_score(it) for it in items] == [0.87, 0.12]


def test_pick_score_falls_back_when_score_key_is_none():
    """dict 里 score 键存在但值为 None 时要回退到 relevance_score，不能给 0。"""
    assert DebugAPI._pick_score({"index": 0, "score": None, "relevance_score": 0.53}) == 0.53


def test_pick_score_keeps_zero_as_valid_value():
    """0.0 是有效分数，不能被当成缺失值跳过。"""
    assert DebugAPI._pick_score({"index": 0, "score": 0.0, "relevance_score": 0.9}) == 0.0


def test_pick_score_returns_zero_when_no_score_field():
    """三个键都没有时返回 0.0，不抛异常。"""
    assert DebugAPI._pick_score({"index": 1}) == 0.0
