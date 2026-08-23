from __future__ import annotations

import logging
import sys
import tempfile
import types
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

from astrbot_plugin_angel_memory.llm_memory.components.memory_sql_manager import MemorySqlManager


def _make_manager() -> MemorySqlManager:
    tmp = tempfile.mkdtemp()
    return MemorySqlManager(db_path=Path(tmp) / "test.db")


def test_user_ledger_upsert_merge_names_and_dedupe():
    m = _make_manager()
    m.upsert_user("aiocqhttp", "123456", "遥酱")
    m.upsert_user("aiocqhttp", "123456", "小遥")
    m.upsert_user("aiocqhttp", "123456", "遥酱")  # 去重
    assert m.get_known_user_ids("aiocqhttp") == ["123456"]
    assert m.get_user_names("aiocqhttp", "123456") == ["遥酱", "小遥"]


def test_user_ledger_same_id_different_platform_independent():
    m = _make_manager()
    m.upsert_user("aiocqhttp", "123456", "QQ遥酱")
    m.upsert_user("discord", "123456", "DC遥酱")
    assert m.get_known_user_ids("aiocqhttp") == ["123456"]
    assert m.get_known_user_ids("discord") == ["123456"]
    assert m.get_user_names("aiocqhttp", "123456") == ["QQ遥酱"]
    assert m.get_user_names("discord", "123456") == ["DC遥酱"]
    # 不带平台时跨平台去重
    assert m.get_known_user_ids() == ["123456"]


def test_user_ledger_ignores_placeholders():
    m = _make_manager()
    m.upsert_user("aiocqhttp", "assistant", "助理")
    m.upsert_user("aiocqhttp", "", "空")
    m.upsert_user("", "123456", "无平台")
    assert m.get_known_user_ids("aiocqhttp") == []
    assert m.get_known_user_ids() == []


def test_group_ledger_upsert_and_name_update():
    m = _make_manager()
    m.upsert_group("111111", "测试群")
    m.upsert_group("111111", "改名群")
    m.upsert_group("", "空群")
    m.upsert_group("unknown", "未知")
    assert m.get_known_group_ids() == ["111111"]
    assert m.get_group_name("111111") == "改名群"
    assert m.get_group_name("999") == ""
