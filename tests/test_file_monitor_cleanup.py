import importlib.util
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

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "astrbot_plugin_angel_memory"
CORE_PACKAGE = f"{PACKAGE_NAME}.core"
LLM_PACKAGE = f"{PACKAGE_NAME}.llm_memory"
COMPONENTS_PACKAGE = f"{LLM_PACKAGE}.components"
SERVICE_PACKAGE = f"{LLM_PACKAGE}.service"

for name, path in (
    (PACKAGE_NAME, ROOT),
    (CORE_PACKAGE, ROOT / "core"),
    (LLM_PACKAGE, ROOT / "llm_memory"),
    (COMPONENTS_PACKAGE, ROOT / "llm_memory" / "components"),
    (SERVICE_PACKAGE, ROOT / "llm_memory" / "service"),
):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

file_index_manager_module_name = f"{COMPONENTS_PACKAGE}.file_index_manager"
if file_index_manager_module_name not in sys.modules:
    file_index_manager_module = types.ModuleType(file_index_manager_module_name)

    class FileIndexManager:
        def __init__(self, *args, **kwargs):
            pass

    file_index_manager_module.FileIndexManager = FileIndexManager
    sys.modules[file_index_manager_module_name] = file_index_manager_module

note_service_module_name = f"{SERVICE_PACKAGE}.note_service"
if note_service_module_name not in sys.modules:
    note_service_module = types.ModuleType(note_service_module_name)

    class NoteService:
        pass

    note_service_module.NoteService = NoteService
    sys.modules[note_service_module_name] = note_service_module

file_monitor_spec = importlib.util.spec_from_file_location(
    f"{CORE_PACKAGE}.file_monitor",
    ROOT / "core" / "file_monitor.py",
)
file_monitor_module = importlib.util.module_from_spec(file_monitor_spec)
sys.modules[file_monitor_spec.name] = file_monitor_module
assert file_monitor_spec.loader is not None
file_monitor_spec.loader.exec_module(file_monitor_module)
FileMonitorService = file_monitor_module.FileMonitorService


class _FakePathManager:
    def __init__(self, root: Path):
        self._root = root

    def get_raw_dir(self) -> Path:
        return self._root / "raw"

    def get_memory_center_index_dir(self) -> Path:
        return self._root / "index"


class _FakeClosable:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class _FakeIDService:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.close_count = 0

    def close(self):
        self.close_count += 1


def _make_note_service(root: Path, *, id_service=None):
    plugin_context = SimpleNamespace(get_path_manager=lambda: _FakePathManager(root))
    note_service = SimpleNamespace(plugin_context=plugin_context)
    if id_service is not None:
        note_service.id_service = id_service
    return note_service


def test_cleanup_does_not_close_shared_file_manager(tmp_path):
    shared_file_manager = _FakeClosable()
    id_service = _FakeIDService(shared_file_manager)
    note_service = _make_note_service(tmp_path, id_service=id_service)

    service = FileMonitorService(str(tmp_path), note_service)
    service._cleanup_all_resources()

    assert shared_file_manager.close_count == 0
    assert id_service.close_count == 0


def test_cleanup_closes_fallback_file_manager(monkeypatch, tmp_path):
    owned_file_manager = _FakeClosable()
    monkeypatch.setattr(file_monitor_module, "FileIndexManager", lambda *args, **kwargs: owned_file_manager)
    note_service = _make_note_service(tmp_path)

    service = FileMonitorService(str(tmp_path), note_service)
    service._cleanup_all_resources()

    assert owned_file_manager.close_count == 1
