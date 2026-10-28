"""日志规范守卫：插件运行时只允许使用 astrbot 提供的 logger。

市场审核规则要求 logger 必须且只能 ``from astrbot.api import logger``，
禁止使用内置 logging 模块（``import logging``、``logging.getLogger``），
包括 ``try/except ImportError`` 的回退写法。本用例防止回退写法再次混入。
"""

import ast
import os
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIR_NAMES = {"tests", "test", "__pycache__", ".git", ".pai"}


def _runtime_files():
    """插件运行时源码文件，排除测试与本地资产目录。"""
    for dirpath, dirnames, filenames in os.walk(PLUGIN_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def test_no_builtin_logging_module():
    """运行时代码不得导入内置 logging 模块。"""
    offenders = []
    for path in _runtime_files():
        tree = _parse(path)
        rel = path.relative_to(PLUGIN_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logging" or alias.name.startswith("logging."):
                        offenders.append(f"{rel}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "logging" or module.startswith("logging."):
                    offenders.append(f"{rel}:{node.lineno} from {module} import ...")
    assert offenders == [], "发现内置 logging 引用：" + "；".join(offenders)


def test_module_logger_comes_from_astrbot():
    """模块顶层 logger 只能来自 astrbot.api。"""
    offenders = []
    for path in _runtime_files():
        tree = _parse(path)
        rel = path.relative_to(PLUGIN_ROOT)

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "logger":
                        offenders.append(f"{rel}:{node.lineno} 顶层 logger 赋值")

        parameters = {
            arg.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.arguments)
            for arg in list(node.args) + list(node.kwonlyargs)
        }
        if "logger" in parameters:
            # 允许把 logger 作为依赖参数传入
            continue

        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if "logger" not in used:
            continue

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "astrbot.api":
                for alias in node.names:
                    if alias.name == "logger":
                        imported.add(alias.asname or "logger")
        if not imported:
            offenders.append(f"{rel} 使用 logger 但没有 from astrbot.api import logger")
    assert offenders == [], "logger 来源不合规：" + "；".join(offenders)
