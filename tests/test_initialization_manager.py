import asyncio
import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "initialization_manager.py"
_SPEC = importlib.util.spec_from_file_location("initialization_manager", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

InitializationManager = _MODULE.InitializationManager
InitializationState = _MODULE.InitializationState


class _DummyContext:
    def get_all_providers(self):
        return []

    def get_all_embedding_providers(self):
        return []


def test_mark_failed_state_and_info():
    manager = InitializationManager(_DummyContext())
    manager.mark_failed("unit-test failure")

    assert manager.get_current_state() == InitializationState.FAILED
    info = manager.get_failed_info()
    assert info["failed_reason"] == "unit-test failure"
    assert info["failed_at"] is not None


def test_wait_until_ready_returns_false_when_failed():
    manager = InitializationManager(_DummyContext())
    manager.mark_failed("boom")
    assert manager.wait_until_ready(timeout=0.01) is False


def test_async_wait_until_ready_timeout():
    manager = InitializationManager(_DummyContext())
    result = asyncio.run(manager.wait_until_ready_async(timeout=0.01))
    assert result is False
