"""
画像注入器 — recall 时将 user_profile scope 的记忆合并到检索结果
支持版本兼容、原子写入、画像衰减、被遗忘权。
"""
import os
import time
import logging

logger = logging.getLogger(__name__)

PROFILE_SCOPE_PREFIX = "user_profile"
PROFILE_VERSION = 1
TAG_STALE_SECONDS = 86400 * 90
VERSION_HEADER = f"<!-- profile_v{PROFILE_VERSION} -->"


class ProfileInjector:
    """画像注入器：从 raw/user_profile/{user_id}/ 读取画像，合并到 recall 结果"""

    def __init__(self, raw_data_dir: str, allowed_tags: set | None = None):
        self.raw_dir = raw_data_dir
        self.allowed_tags = allowed_tags  # 当前有效的标签类型白名单

    def get_profile_dir(self, user_id: str) -> str:
        return os.path.join(self.raw_dir, PROFILE_SCOPE_PREFIX, user_id)

    # ---- 原子写入 ----

    @staticmethod
    def write_tag(filepath: str, value: str) -> None:
        """原子写入（先 .tmp 再 rename，防崩溃损坏）；空值跳过"""
        if not value.strip():
            return
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{VERSION_HEADER}\n{value.strip()}")
        os.replace(tmp, filepath)

    # ---- 读取 ----

    def read_profile_tags(self, user_id: str) -> list[dict]:
        """
        读取画像标签。
        自动处理 .opt_out、版本头、衰减、配置变更清理、同类型去重。
        """
        profile_dir = self.get_profile_dir(user_id)
        if not os.path.isdir(profile_dir):
            return []

        if os.path.isfile(os.path.join(profile_dir, ".opt_out")):
            return []

        now = time.time()
        tags = []
        try:
            for fname in os.listdir(profile_dir):
                if not fname.endswith(".md") or fname.startswith("."):
                    continue
                tag_type = fname.replace(".md", "")
                # 配置变更清理：标签类型不在当前白名单中则跳过
                if self.allowed_tags and tag_type not in self.allowed_tags:
                    continue
                filepath = os.path.join(profile_dir, fname)
                try:
                    mtime = os.path.getmtime(filepath)
                    with open(filepath, "r", encoding="utf-8") as f:
                        raw = f.read()
                    value = ProfileInjector._strip_header(raw).strip()
                    if not value:
                        continue
                    age_days = (now - mtime) / 86400
                    tags.append({
                        "type": tag_type,
                        "value": value,
                        "source": PROFILE_SCOPE_PREFIX,
                        "mtime": mtime,
                        "stale": age_days > 90,
                    })
                except (OSError, UnicodeDecodeError):
                    logger.warning("[画像注入] 读取失败: %s", filepath, exc_info=True)
        except OSError:
            logger.warning("[画像注入] 列出目录失败: %s", profile_dir, exc_info=True)

        # 去重：非 stale 优先，同 mtime 保留后出现的
        deduped: dict[str, dict] = {}
        for tag in sorted(tags, key=lambda t: (t["stale"], -t["mtime"])):
            if tag["type"] not in deduped or not tag["stale"]:
                deduped[tag["type"]] = tag

        return list(deduped.values())

    @staticmethod
    def _strip_header(raw: str) -> str:
        lines = raw.split("\n")
        if lines and lines[0].startswith("<!-- profile_v"):
            return "\n".join(lines[1:])
        return raw

    def inject(self, user_id: str, current_memories: list[dict], max_inject: int = 5) -> list[dict]:
        tags = self.read_profile_tags(user_id)
        if not tags:
            return current_memories

        injected = []
        for tag in tags[:max_inject]:
            extra = " [过期]" if tag.get("stale") else ""
            injected.append({
                "scope": PROFILE_SCOPE_PREFIX,
                "content": f"{tag['type']}: {tag['value']}{extra}",
                "tag_type": tag["type"],
                "user_id": user_id,
            })

        logger.info("[画像注入] user=%s 注入%d条", user_id, len(injected))
        return current_memories + injected
