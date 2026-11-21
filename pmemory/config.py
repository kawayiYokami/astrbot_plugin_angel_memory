"""p-memory 命名与域映射。

两套域不要混用：

- 记忆域固定 `angel/memory`。记忆的可见范围走 scope，直接沿用现有的
  `memory_scope` 解析结果（人格键优先、会话 ID 次之、默认 public）。
- 知识域按 raw/ 下的目录划分：`angel/knowledge/{domain}`。未绑定 domain 的
  会话不做知识自动召回，只能靠工具主动搜索。
"""

from __future__ import annotations

from collections.abc import Iterable

SCOPE_PUBLIC = "public"

# 知识可见性三态
MODE_AUTO = "auto"
MODE_MANUAL = "manual"
MODE_HIDDEN = "hidden"
KNOWLEDGE_MODES = (MODE_AUTO, MODE_MANUAL, MODE_HIDDEN)
DEFAULT_KNOWLEDGE_MODE = MODE_MANUAL

NAMESPACE_MEMORY = "angel/memory"
NAMESPACE_KNOWLEDGE_PREFIX = "angel/knowledge"

# 域 → raw/ 下的相对目录。顺序即导入顺序。
DOMAIN_DIRS: dict[str, str] = {
    "gi": "原神",
    "hsr": "崩坏星穹铁道",
    "zzz": "绝区零",
    "personal": "海豹的个人信息",
    "angel": ".angel/note",
}

_DIR_TO_DOMAIN = {value: key for key, value in DOMAIN_DIRS.items()}


def memory_namespace() -> str:
    """记忆域固定，不随人格或群聊变化。"""
    return NAMESPACE_MEMORY


def knowledge_namespace(domain: str | None) -> str:
    """知识域 namespace。空 domain 返回知识根。"""
    name = str(domain or "").strip().lower()
    if not name:
        return NAMESPACE_KNOWLEDGE_PREFIX
    return f"{NAMESPACE_KNOWLEDGE_PREFIX}/{name}"


def read_scopes(scope: str | None) -> list[str]:
    """读过滤口径：自己的 scope 加上 public。

    与旧 `MemorySqlManager._scope_sql` 一致——非 public 的会话既能看到自己的
    记忆，也能看到公共记忆。
    """
    name = str(scope or SCOPE_PUBLIC).strip() or SCOPE_PUBLIC
    if name == SCOPE_PUBLIC:
        return [SCOPE_PUBLIC]
    return [name, SCOPE_PUBLIC]


def normalize_domain(name: object) -> str | None:
    """把域标识或中文目录名规整成域键，非法返回 None。"""
    text = str(name or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in DOMAIN_DIRS:
        return lowered
    return _DIR_TO_DOMAIN.get(text)


def normalize_mode(value: object) -> str | None:
    """把三态值规整成 auto / manual / hidden，非法返回 None。"""
    text = str(value or "").strip().lower()
    return text if text in KNOWLEDGE_MODES else None


def iter_domains() -> Iterable[str]:
    return DOMAIN_DIRS.keys()


def build_knowledge_matrix(
    knowledge_map: dict | None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """把配置里的可见性表规整成「每个群 × 全部域」的完整矩阵。

    规则：
    - 没写的群与域一律 manual；
    - 非法项不落进矩阵，作为错误信息返回给调用方打日志。
    """
    errors: list[str] = []
    groups: dict[str, dict[str, str]] = {}

    def _ensure(group_id: str) -> dict[str, str]:
        if group_id not in groups:
            groups[group_id] = {domain: DEFAULT_KNOWLEDGE_MODE for domain in DOMAIN_DIRS}
        return groups[group_id]

    if isinstance(knowledge_map, dict):
        for raw_key, raw_modes in knowledge_map.items():
            group_id = str(raw_key).strip()
            if not group_id:
                errors.append("group_knowledge_map 存在空群号键")
                continue
            if not isinstance(raw_modes, dict):
                errors.append(f"group_knowledge_map[{group_id}] 应为对象")
                continue
            modes = _ensure(group_id)
            for raw_domain, raw_mode in raw_modes.items():
                domain = normalize_domain(raw_domain)
                mode = normalize_mode(raw_mode)
                if not domain or mode is None:
                    errors.append(f"group_knowledge_map[{group_id}] 非法项: {raw_domain}={raw_mode}")
                    continue
                modes[domain] = mode

    return groups, errors


def auto_domains_from(matrix: dict[str, dict[str, str]], group_id: object) -> list[str]:
    """矩阵里标记为 auto 的域。"""
    key = str(group_id or "").strip()
    modes = matrix.get(key) or {}
    return [domain for domain, mode in modes.items() if mode == MODE_AUTO]


def visible_domains_from(matrix: dict[str, dict[str, str]], group_id: object) -> list[str]:
    """可主动搜索的域（auto + manual）。私聊没有群号，按全部域可见。"""
    key = str(group_id or "").strip()
    if not key:
        return list(DOMAIN_DIRS)
    modes = matrix.get(key) or {}
    if not modes:
        # 表里没有这个群：默认全部 manual
        return list(DOMAIN_DIRS)
    return [domain for domain, mode in modes.items() if mode != MODE_HIDDEN]
