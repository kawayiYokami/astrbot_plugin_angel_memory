"""重排模型候选列表按能力筛：带 rerank() 的提供商才算候选。

框架的 Context 没有 get_all_rerank_providers，取它恒为空，
所以候选只能从 get_all_providers() 里按能力筛，与 component_factory 解析重排提供商同口径。
"""

from types import SimpleNamespace

from astrbot_plugin_angel_memory.web_api import plugin_config_api as api


class _Provider:
    """提供商替身：带 rerank 的才具备重排能力。"""

    def __init__(self, provider_id, supports_rerank=False):
        self._id = provider_id
        if supports_rerank:

            def rerank(*args, **kwargs):
                return []

            self.rerank = rerank

    def meta(self):
        return SimpleNamespace(id=self._id)


class _Host:
    """宿主上下文替身：故意不提供 get_all_rerank_providers，真实框架也没有这个方法。"""

    def __init__(self, providers, embedding=()):
        self._providers = list(providers)
        self._embedding = list(embedding)

    def get_all_providers(self):
        return list(self._providers)

    def get_all_embedding_providers(self):
        return list(self._embedding)


def _handler(providers, embedding=()):
    handler = api.PluginConfigAPI.__new__(api.PluginConfigAPI)
    handler._ctx = SimpleNamespace(astrbot_context=_Host(providers, embedding), config={})
    handler._plugin = None
    handler._config = {}
    handler._schema = {}
    return handler


def _providers_payload(handler):
    captured = {}
    original = api.json_response
    api.json_response = lambda payload: captured.setdefault("payload", payload)
    try:
        handler.get_config()
    finally:
        api.json_response = original
    return captured["payload"]["data"]["providers"]


def test_rerank_candidates_only_include_capable_providers():
    """三个聊天提供商里两个带 rerank，候选必须正好是那两个。"""
    host_providers = [
        _Provider("siliconflow-chat"),
        _Provider("siliconflow-rerank", supports_rerank=True),
        _Provider("jina-rerank", supports_rerank=True),
    ]

    providers = _providers_payload(_handler(host_providers))

    assert providers["chat"] == ["siliconflow-chat", "siliconflow-rerank", "jina-rerank"]
    assert providers["rerank"] == ["siliconflow-rerank", "jina-rerank"]


def test_rerank_candidates_empty_when_no_provider_supports_it():
    """一个都不带 rerank 时返回空数组，且不抛异常。"""
    host_providers = [_Provider("siliconflow-chat"), _Provider("deepseek-chat")]

    providers = _providers_payload(_handler(host_providers))

    assert providers["rerank"] == []
    assert providers["chat"] == ["siliconflow-chat", "deepseek-chat"]


def test_rerank_candidates_survive_host_without_rerank_getter():
    """宿主连 get_all_rerank_providers 都没有时，仍能筛出候选。"""
    host_providers = [_Provider("bge-reranker", supports_rerank=True)]
    handler = _handler(host_providers)
    assert not hasattr(handler._ctx.astrbot_context, "get_all_rerank_providers")

    providers = _providers_payload(handler)

    assert providers["rerank"] == ["bge-reranker"]


def test_embedding_candidates_still_come_from_embedding_getter():
    """嵌入候选仍走 get_all_embedding_providers，不被本次改动影响。"""
    providers = _providers_payload(_handler([_Provider("chat-a")], embedding=[_Provider("bge-m3")]))

    assert providers["embedding"] == ["bge-m3"]
    assert providers["rerank"] == []
