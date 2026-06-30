import importlib.util
import logging
import sqlite3
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace


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
PARSER_PACKAGE = f"{LLM_PACKAGE}.parser"
UTILS_PACKAGE = f"{LLM_PACKAGE}.utils"
CONFIG_PACKAGE = f"{LLM_PACKAGE}.config"
MODELS_PACKAGE = f"{LLM_PACKAGE}.models"

for name, path in (
    (PACKAGE_NAME, ROOT),
    (CORE_PACKAGE, ROOT / "core"),
    (LLM_PACKAGE, ROOT / "llm_memory"),
    (COMPONENTS_PACKAGE, ROOT / "llm_memory" / "components"),
    (SERVICE_PACKAGE, ROOT / "llm_memory" / "service"),
    (PARSER_PACKAGE, ROOT / "llm_memory" / "parser"),
    (UTILS_PACKAGE, ROOT / "llm_memory" / "utils"),
    (CONFIG_PACKAGE, ROOT / "llm_memory" / "config"),
    (MODELS_PACKAGE, ROOT / "llm_memory" / "models"),
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


id_service_module_name = f"{SERVICE_PACKAGE}.id_service"
if id_service_module_name not in sys.modules:
    id_service_module = types.ModuleType(id_service_module_name)

    class IDService:
        @classmethod
        def from_plugin_context(cls, plugin_context):
            return SimpleNamespace(file_to_id=lambda relative_path, ts: 1)

    id_service_module.IDService = IDService
    sys.modules[id_service_module_name] = id_service_module


note_chunk_search_module_name = f"{COMPONENTS_PACKAGE}.note_chunk_search"
if note_chunk_search_module_name not in sys.modules:
    note_chunk_search_module = types.ModuleType(note_chunk_search_module_name)

    class NoteChunkSearchEngine:
        pass

    note_chunk_search_module.NoteChunkSearchEngine = NoteChunkSearchEngine
    sys.modules[note_chunk_search_module_name] = note_chunk_search_module


system_config_module_name = f"{CONFIG_PACKAGE}.system_config"
if system_config_module_name not in sys.modules:
    system_config_module = types.ModuleType(system_config_module_name)
    system_config_module.system_config = SimpleNamespace()
    sys.modules[system_config_module_name] = system_config_module


data_models_module_name = f"{MODELS_PACKAGE}.data_models"
if data_models_module_name not in sys.modules:
    data_models_module = types.ModuleType(data_models_module_name)

    class BaseMemory:
        pass

    class MemoryType:
        pass

    class ValidationError(Exception):
        pass

    data_models_module.BaseMemory = BaseMemory
    data_models_module.MemoryType = MemoryType
    data_models_module.ValidationError = ValidationError
    sys.modules[data_models_module_name] = data_models_module


user_profile_module_name = f"{UTILS_PACKAGE}.user_profile"
if user_profile_module_name not in sys.modules:
    user_profile_module = types.ModuleType(user_profile_module_name)
    user_profile_module.PROFILE_ATTRIBUTE_TAGS = []
    user_profile_module.is_user_profile_tags = lambda tags: False
    sys.modules[user_profile_module_name] = user_profile_module


memory_decay_module_name = f"{SERVICE_PACKAGE}.memory_decay_policy"
if memory_decay_module_name not in sys.modules:
    memory_decay_module = types.ModuleType(memory_decay_module_name)

    class MemoryDecayConfig:
        pass

    class MemoryDecayPolicy:
        def __init__(self, config=None):
            self.config = config

    memory_decay_module.MemoryDecayConfig = MemoryDecayConfig
    memory_decay_module.MemoryDecayPolicy = MemoryDecayPolicy
    sys.modules[memory_decay_module_name] = memory_decay_module


bm25_module_name = f"{COMPONENTS_PACKAGE}.bm25_retriever"
if bm25_module_name not in sys.modules:
    bm25_module = types.ModuleType(bm25_module_name)

    class TantivyBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

    bm25_module.TantivyBM25Retriever = TantivyBM25Retriever
    sys.modules[bm25_module_name] = bm25_module


hybrid_module_name = f"{COMPONENTS_PACKAGE}.hybrid_retrieval_engine"
if hybrid_module_name not in sys.modules:
    hybrid_module = types.ModuleType(hybrid_module_name)

    class HybridRetrievalEngine:
        def __init__(self, *args, **kwargs):
            pass

    hybrid_module.HybridRetrievalEngine = HybridRetrievalEngine
    sys.modules[hybrid_module_name] = hybrid_module


note_service_spec = importlib.util.spec_from_file_location(
    f"{SERVICE_PACKAGE}.note_service",
    ROOT / "llm_memory" / "service" / "note_service.py",
)
note_service_module = importlib.util.module_from_spec(note_service_spec)
sys.modules[note_service_spec.name] = note_service_module
assert note_service_spec.loader is not None
note_service_spec.loader.exec_module(note_service_module)
NoteService = note_service_module.NoteService

memory_sql_manager_spec = importlib.util.spec_from_file_location(
    f"{COMPONENTS_PACKAGE}.memory_sql_manager",
    ROOT / "llm_memory" / "components" / "memory_sql_manager.py",
)
memory_sql_manager_module = importlib.util.module_from_spec(memory_sql_manager_spec)
sys.modules[memory_sql_manager_spec.name] = memory_sql_manager_module
assert memory_sql_manager_spec.loader is not None
memory_sql_manager_spec.loader.exec_module(memory_sql_manager_module)
MemorySqlManager = memory_sql_manager_module.MemorySqlManager

file_monitor_spec = importlib.util.spec_from_file_location(
    f"{CORE_PACKAGE}.file_monitor",
    ROOT / "core" / "file_monitor.py",
)
file_monitor_module = importlib.util.module_from_spec(file_monitor_spec)
sys.modules[file_monitor_spec.name] = file_monitor_module
assert file_monitor_spec.loader is not None
file_monitor_spec.loader.exec_module(file_monitor_module)
FileMonitorService = file_monitor_module.FileMonitorService


class _FakePluginContext:
    def __init__(self, root: Path, components: dict):
        self._root = root
        self._components = components

    def get_component(self, name: str):
        return self._components.get(name)

    def get_path_manager(self):
        root = self._root

        class _PathManager:
            def get_raw_dir(self):
                return root / "raw"

            def get_memory_center_index_dir(self):
                return root / "index"

        return _PathManager()


def _make_note_service_for_test(plugin_context, file_id=42):
    service = NoteService.__new__(NoteService)
    service.logger = logging.getLogger("note-service-test")
    service.plugin_context = plugin_context
    service.id_service = SimpleNamespace(file_to_id=lambda relative_path, timestamp: file_id)
    return service


def test_parse_and_store_extracts_heading_and_uses_sync_registry_write(tmp_path):
    note_file = tmp_path / "sample.md"
    note_file.write_text("# 第一标题\n\n正文\n", encoding="utf-8")

    class _MemorySqlManager:
        def __init__(self):
            self.async_called = 0
            self.sync_calls = []

        async def upsert_note_file_entry(self, **kwargs):
            self.async_called += 1
            raise AssertionError("不应走异步 upsert_note_file_entry")

        def _upsert_note_file_entry_sync(self, **kwargs):
            self.sync_calls.append(kwargs)
            return {"scanned": 1, "upserted": 1, "failed": 0}

    memory_sql_manager = _MemorySqlManager()
    plugin_context = _FakePluginContext(tmp_path, {"memory_sql_manager": memory_sql_manager})
    service = _make_note_service_for_test(plugin_context)

    chunk_count, _ = service.parse_and_store_file_sync(str(note_file), "sample.md", update_search_index=False)

    assert chunk_count > 0
    assert memory_sql_manager.async_called == 0
    assert memory_sql_manager.sync_calls[0]["heading_h1"] == "第一标题"
    assert memory_sql_manager.sync_calls[0]["source_file_path"] == "sample.md"


def test_remove_file_data_by_file_id_returns_false_when_chunk_delete_fails(tmp_path):
    class _MemorySqlManager:
        def __init__(self):
            self.deleted = []

        def _delete_note_index_by_file_id_sync(self, file_id):
            self.deleted.append(file_id)
            return [f"note_file_{file_id}"]

    class _ChunkStore:
        def delete_by_file_id(self, file_id):
            raise RuntimeError("chunk delete failed")

    class _SearchEngine:
        def __init__(self):
            self.called = 0

        def delete_by_file_id(self, file_id):
            self.called += 1

    memory_sql_manager = _MemorySqlManager()
    search_engine = _SearchEngine()
    plugin_context = _FakePluginContext(
        tmp_path,
        {
            "memory_sql_manager": memory_sql_manager,
            "note_chunk_store": _ChunkStore(),
            "note_chunk_search": search_engine,
        },
    )
    service = _make_note_service_for_test(plugin_context)

    ok = service.remove_file_data_by_file_id(7)

    assert ok is False
    assert memory_sql_manager.deleted == ["7"]
    assert search_engine.called == 0


def test_upsert_note_file_entry_sync_persists_heading_h1(tmp_path):
    db_path = tmp_path / "memory.db"
    manager = MemorySqlManager.__new__(MemorySqlManager)
    manager.db_path = db_path
    manager.logger = logging.getLogger("memory-sql-test")

    with MemorySqlManager._connect(manager) as conn:
        conn.executescript(
            """
            CREATE TABLE note_index_records (
                source_id TEXT PRIMARY KEY,
                note_short_id INTEGER UNIQUE,
                file_id TEXT NOT NULL,
                source_file_path TEXT NOT NULL,
                heading_h1 TEXT,
                heading_h2 TEXT,
                heading_h3 TEXT,
                heading_h4 TEXT,
                heading_h5 TEXT,
                heading_h6 TEXT,
                total_lines INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE note_short_id_seq (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_id INTEGER NOT NULL
            );
            INSERT INTO note_short_id_seq (id, next_id) VALUES (1, 0);
            """
        )
        conn.commit()

    MemorySqlManager._upsert_note_file_entry_sync(
        manager,
        file_id="f-1",
        source_file_path="docs/sample.md",
        heading_h1="写入标题",
        total_lines=12,
        updated_at=123.0,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT heading_h1, total_lines FROM note_index_records WHERE file_id = ?",
            ("f-1",),
        ).fetchone()

    assert row == ("写入标题", 12)


def test_rebuild_chunk_search_syncs_disk_before_rebuilding(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    live_file = raw_dir / "live.md"
    live_file.write_text("# 活着\n内容\n", encoding="utf-8")

    class _FileIndexManager:
        def __init__(self):
            self.was_version_reset = False

        def get_all_files(self):
            return [{"id": 1, "relative_path": "gone.md", "file_timestamp": 1}]

    class _ChunkStore:
        def list_all_chunks(self):
            return [{"file_id": "2", "content": "chunk"}]

    class _ChunkSearch:
        def __init__(self):
            self.rebuilt = None

        def rebuild_all(self, chunks):
            self.rebuilt = list(chunks)
            return len(chunks)

    chunk_search = _ChunkSearch()
    plugin_context = _FakePluginContext(
        tmp_path,
        {
            "note_chunk_store": _ChunkStore(),
            "note_chunk_search": chunk_search,
        },
    )
    note_service = SimpleNamespace(
        plugin_context=plugin_context,
        id_service=SimpleNamespace(file_manager=_FileIndexManager()),
    )
    service = FileMonitorService(str(tmp_path), note_service)

    deleted = []
    added = []

    service._delete_file_data_by_file_id = MethodType(
        lambda self, file_ids: deleted.append(list(file_ids)) or True,
        service,
    )
    service._process_file_change = MethodType(
        lambda self, relative_path, timestamp, update_search_index=False: added.append(
            (relative_path, update_search_index)
        )
        or (1, {}),
        service,
    )

    service._rebuild_chunk_search_index_from_store()

    assert deleted == [[1]]
    assert added == [("live.md", False)]
    assert chunk_search.rebuilt == [{"file_id": "2", "content": "chunk"}]


def test_clear_note_indexes_for_file_index_rebuild_uses_sync_registry_clear(tmp_path):
    class _MemorySqlManager:
        def __init__(self):
            self.sync_called = 0
            self.async_called = 0

        async def clear_note_file_registry(self):
            self.async_called += 1
            raise AssertionError("不应走异步 clear_note_file_registry")

        def _clear_note_file_registry_sync(self):
            self.sync_called += 1
            return {"deleted": 2}

    class _ChunkStore:
        def clear_all(self):
            return 3

    class _ChunkSearch:
        def __init__(self):
            self.rebuilt = None

        def rebuild_all(self, chunks):
            self.rebuilt = list(chunks)
            return 0

    memory_sql_manager = _MemorySqlManager()
    chunk_search = _ChunkSearch()
    plugin_context = _FakePluginContext(
        tmp_path,
        {
            "memory_sql_manager": memory_sql_manager,
            "note_chunk_store": _ChunkStore(),
            "note_chunk_search": chunk_search,
        },
    )
    note_service = SimpleNamespace(
        plugin_context=plugin_context,
        id_service=SimpleNamespace(file_manager=SimpleNamespace()),
    )
    service = FileMonitorService(str(tmp_path), note_service)

    service._clear_note_indexes_for_file_index_rebuild()

    assert memory_sql_manager.sync_called == 1
    assert memory_sql_manager.async_called == 0
    assert chunk_search.rebuilt == []
