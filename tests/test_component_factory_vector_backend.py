from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")

    event = types.ModuleType("astrbot.api.event")

    class AstrMessageEvent:
        pass

    event.AstrMessageEvent = AstrMessageEvent

    provider = types.ModuleType("astrbot.api.provider")

    class ProviderRequest:
        pass

    provider.ProviderRequest = ProviderRequest

    core_pkg = types.ModuleType("astrbot.core")
    agent_pkg = types.ModuleType("astrbot.core.agent")
    message = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text: str = ""):
            self.text = text

    message.TextPart = TextPart

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.provider"] = provider
    sys.modules["astrbot.core"] = core_pkg
    sys.modules["astrbot.core.agent"] = agent_pkg
    sys.modules["astrbot.core.agent.message"] = message


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.core import component_factory
from astrbot_plugin_angel_memory.core.component_factory import ComponentFactory


FAISS_MODULE = "astrbot_plugin_angel_memory.llm_memory.components.faiss_memory_index"


def _factory_without_init(tmp_path: Path) -> ComponentFactory:
    factory = ComponentFactory.__new__(ComponentFactory)
    factory.logger = logging.getLogger("component-factory-test")
    factory._faiss_probe_result = None
    factory.plugin_context = SimpleNamespace(
        get_faiss_index_dir=lambda: tmp_path / "index" / "faiss",
        get_sqlite_vector_index_dir=lambda: tmp_path / "index" / "sqlite",
    )
    return factory


def test_faiss_probe_failure_selects_sqlite_without_importing_faiss_index(monkeypatch, tmp_path):
    sys.modules.pop(FAISS_MODULE, None)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=132, stderr="Illegal instruction", stdout="")

    monkeypatch.setattr(component_factory.subprocess, "run", fake_run)
    factory = _factory_without_init(tmp_path)

    backend_name, store_class, index_dir, label = factory._select_vector_store_backend()

    assert backend_name == "sqlite"
    assert store_class.__name__ == "SqliteVectorStore"
    assert index_dir == tmp_path / "index" / "sqlite"
    assert label == "SQLite"
    assert FAISS_MODULE not in sys.modules


def test_faiss_probe_result_is_cached(monkeypatch, tmp_path):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stderr="No module named faiss", stdout="")

    monkeypatch.setattr(component_factory.subprocess, "run", fake_run)
    factory = _factory_without_init(tmp_path)

    assert factory._probe_faiss_available()[0] is False
    assert factory._probe_faiss_available()[0] is False
    assert len(calls) == 1


def test_faiss_probe_success_selects_faiss(monkeypatch, tmp_path):
    fake_module = types.ModuleType(FAISS_MODULE)

    class FakeFaissVectorStore:
        pass

    fake_module.FaissVectorStore = FakeFaissVectorStore
    monkeypatch.setitem(sys.modules, FAISS_MODULE, fake_module)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(component_factory.subprocess, "run", fake_run)
    factory = _factory_without_init(tmp_path)

    backend_name, store_class, index_dir, label = factory._select_vector_store_backend()

    assert backend_name == "faiss"
    assert store_class is FakeFaissVectorStore
    assert index_dir == tmp_path / "index" / "faiss"
    assert label == "FAISS"
