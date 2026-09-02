import asyncio
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT.parent))

from astrbot_plugin_angel_memory.core.rerank_failover import FailoverRerankProvider


class _FailingProvider:
    def __init__(self):
        self.calls = 0

    async def rerank(self, **kwargs):
        self.calls += 1
        raise RuntimeError("remote unavailable")


class _WorkingProvider:
    def __init__(self):
        self.calls = 0

    async def rerank(self, **kwargs):
        self.calls += 1
        return [{"index": 0, "relevance_score": 0.9}]


class _CapturingProvider(_WorkingProvider):
    def __init__(self):
        super().__init__()
        self.documents = []
        self.top_n = None

    async def rerank(self, **kwargs):
        self.documents = kwargs["documents"]
        self.top_n = kwargs["top_n"]
        return await super().rerank(**kwargs)


class _ServerError(RuntimeError):
    status = 500


class _OverflowThenWorkingProvider(_WorkingProvider):
    def __init__(self):
        super().__init__()
        self.document_lengths = []

    async def rerank(self, **kwargs):
        self.calls += 1
        self.document_lengths.append([len(doc) for doc in kwargs["documents"]])
        if self.calls == 1:
            raise _ServerError("Internal Server Error")
        return [{"index": 0, "relevance_score": 0.9}]


class _ProviderWithoutTopN:
    def __init__(self):
        self.calls = 0

    async def rerank(self, query, documents):
        self.calls += 1
        assert query == "query"
        assert documents == ["one", "two"]
        return [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ]


def test_rerank_falls_back_after_primary_failure():
    primary = _FailingProvider()
    fallback = _WorkingProvider()
    provider = FailoverRerankProvider(
        [("remote", primary), ("local", fallback)],
        failure_threshold=3,
        cooldown_seconds=60,
    )

    result = asyncio.run(provider.rerank("query", ["document"], top_n=1))

    assert result[0]["index"] == 0
    assert primary.calls == 1
    assert fallback.calls == 1


def test_rerank_circuit_breaker_skips_failed_primary():
    primary = _FailingProvider()
    fallback = _WorkingProvider()
    provider = FailoverRerankProvider(
        [("remote", primary), ("local", fallback)],
        failure_threshold=2,
        cooldown_seconds=60,
    )

    for _ in range(3):
        result = asyncio.run(provider.rerank("query", ["document"], top_n=1))
        assert result

    assert primary.calls == 2
    assert fallback.calls == 3


def test_local_fallback_workload_is_bounded_without_affecting_primary():
    primary = _FailingProvider()
    fallback = _CapturingProvider()
    provider = FailoverRerankProvider(
        [("remote", primary), ("local", fallback)],
        fallback_max_documents=2,
        fallback_max_document_chars=20,
    )
    documents = ["a" * 40, "b" * 10, "c" * 10]

    result = asyncio.run(provider.rerank("query", documents, top_n=3))

    assert result
    assert len(fallback.documents) == 2
    assert len(fallback.documents[0]) == 20
    assert fallback.documents[0].startswith("a" * 11)
    assert fallback.documents[0].endswith("a" * 4)
    assert fallback.top_n == 2


def test_local_batch_overflow_retries_with_shorter_documents():
    primary = _FailingProvider()
    fallback = _OverflowThenWorkingProvider()
    provider = FailoverRerankProvider(
        [("remote", primary), ("local", fallback)],
        fallback_max_document_chars=520,
        fallback_retry_document_chars=220,
    )

    result = asyncio.run(provider.rerank("query", ["文" * 500], top_n=1))

    assert result
    assert fallback.calls == 2
    assert fallback.document_lengths == [[500], [220]]


def test_provider_without_top_n_parameter_remains_compatible():
    underlying = _ProviderWithoutTopN()
    provider = FailoverRerankProvider([("legacy", underlying)])

    result = asyncio.run(provider.rerank("query", ["one", "two"], top_n=1))

    assert underlying.calls == 1
    assert [item["index"] for item in result] == [0]
