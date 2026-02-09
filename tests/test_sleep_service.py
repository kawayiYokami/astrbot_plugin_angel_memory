import asyncio
import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "services" / "sleep_service.py"
_SPEC = importlib.util.spec_from_file_location("sleep_service", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

DeepMindSleepService = _MODULE.DeepMindSleepService


class _DummyMemorySystem:
    def __init__(self):
        self.called = 0

    async def consolidate_memories(self):
        self.called += 1


class _DummyLogger:
    def error(self, *_args, **_kwargs):
        return None


class _DummyDeepMind:
    def __init__(self):
        self.memory_system = _DummyMemorySystem()
        self.logger = _DummyLogger()
        self.last_sleep_time = None

    def is_enabled(self):
        return True


def test_check_and_sleep_first_time():
    deepmind = _DummyDeepMind()
    service = DeepMindSleepService(deepmind)

    result = asyncio.run(service.check_and_sleep_if_needed(1))
    assert result is True
    assert deepmind.memory_system.called == 1


def test_check_and_sleep_disabled():
    deepmind = _DummyDeepMind()
    service = DeepMindSleepService(deepmind)

    result = asyncio.run(service.check_and_sleep_if_needed(0))
    assert result is False
    assert deepmind.memory_system.called == 0

