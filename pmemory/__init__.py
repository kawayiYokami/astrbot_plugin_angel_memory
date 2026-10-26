"""Angel Memory 的 p-memory 封装层。

对外只暴露四样东西：
- `pmemory`：知识库单例（配置数据目录与提供商回调）
- `MemoryStore`：记忆读写与生命周期
- `NotesStore`：笔记导入与三块检索
- `config`：命名、域映射、scope 口径
"""

from .config import (
    DOMAIN_DIRS,
    SCOPE_PUBLIC,
    knowledge_namespace,
    memory_namespace,
    normalize_domain,
    read_scopes,
    resolve_domain_for_group,
)
from .kb import PMemory, pmemory
from .memory import MemoryStore, to_seconds, to_us
from .notes import NotesStore

__all__ = [
    "DOMAIN_DIRS",
    "SCOPE_PUBLIC",
    "PMemory",
    "MemoryStore",
    "NotesStore",
    "knowledge_namespace",
    "memory_namespace",
    "normalize_domain",
    "pmemory",
    "read_scopes",
    "resolve_domain_for_group",
    "to_seconds",
    "to_us",
]
