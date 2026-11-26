"""重排与嵌入候选列表的取法。

框架按 ProviderType 分类装载：RerankProvider 只进 provider_manager.rerank_provider_insts，
不进 provider_insts。所以从 get_all_providers() 里按能力筛永远筛不出重排模型——
候选必须在框架的重排实例列表上取。老框架没有这个列表时退回按能力筛。
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
    """宿主替身：重排实例挂在 provider_manager 上，与真实框架一致。"""

    def __init__(self, chat=(), embedding=(), rerank=None, with_manager=True):
        self._chat = list(chat)
        self._embedding = list(embedding)
        if with_manager:
            self.provider_manager = SimpleNamespace(rerank_provider_insts=list(rerank or ()))

    def get_all_providers(self):
        return list(self._chat)

    def get_all_embedding_providers(self):
        return list(self._embedding)


def _handler(host):
    handler = api.PluginConfigAPI.__new__(api.PluginConfigAPI)
    handler._ctx = SimpleNamespace(astrbot_context=host, config={})
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


def test_rerank_candidates_come_from_framework_rerank_instances():
    """候选只认框架 rerank_provider_insts；chat 列表里带 rerank() 的不算。"""
    host = _Host(
        chat=[_Provider("deepseek-chat"), _Provider("sneaky-chat", supports_rerank=True)],
        rerank=[_Provider("siliconflow-rerank"), _Provider("jina-rerank")],
    )

    providers = _providers_payload(_handler(host))

    assert providers["rerank"] == ["siliconflow-rerank", "jina-rerank"]
    assert providers["chat"] == ["deepseek-chat", "sneaky-chat"]


def test_rerank_candidates_fall_back_to_capability_scan_on_old_framework():
    """宿主没有 provider_manager 时，退回按 rerank() 能力筛。"""
    host = _Host(
        chat=[_Provider("plain-chat"), _Provider("bge-reranker", supports_rerank=True)],
        with_manager=False,
    )

    providers = _providers_payload(_handler(host))

    assert providers["rerank"] == ["bge-reranker"]


def test_rerank_candidates_empty_when_nothing_configured():
    """一个重排模型都没配时返回空数组，且不抛异常。"""
    providers = _providers_payload(_handler(_Host()))

    assert providers["rerank"] == []
    assert providers["chat"] == []


def test_embedding_candidates_are_deduplicated():
    """同一嵌入实例被重复装载时，候选只出现一次。"""
    host = _Host(
        embedding=[_Provider("bge-m3"), _Provider("bge-m3"), _Provider("other-embed")],
    )

    providers = _providers_payload(_handler(host))

    assert providers["embedding"] == ["bge-m3", "other-embed"]
