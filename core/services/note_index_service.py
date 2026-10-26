"""笔记索引服务：扫描 raw/、写入 p-memory、对外提供检索。

真相源仍是文件本身，p-memory 只保存切片与索引。因此这里做三件事：
1. 启动时全量扫描各域目录，按修改时间增量同步；
2. 写入 / 删除后立即让库追平索引；
3. 检索统一走 `NotesStore` 的三块检索，返回旧调用方熟悉的形状。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...pmemory import DOMAIN_DIRS, NotesStore

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 60


class NoteIndexService:
    """笔记导入与检索。"""

    def __init__(
        self,
        raw_dir: str | Path,
        notes: Optional[NotesStore] = None,
        scan_interval: int = SCAN_INTERVAL_SECONDS,
    ) -> None:
        self._raw_dir = Path(raw_dir)
        self._notes = notes or NotesStore(self._raw_dir)
        self._scan_interval = max(10, int(scan_interval))
        self._state: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # === 域与路径 ===

    @staticmethod
    def domain_for_path(path: Path, raw_dir: Path) -> Optional[str]:
        try:
            relative = path.resolve().relative_to(raw_dir.resolve())
        except ValueError:
            return None
        head = relative.parts[0] if relative.parts else ""
        for domain, directory in DOMAIN_DIRS.items():
            if directory.split("/")[0] == head:
                return domain
        return None

    def _domain_dirs(self) -> List[Tuple[str, Path]]:
        output: List[Tuple[str, Path]] = []
        for domain, relative in DOMAIN_DIRS.items():
            directory = self._raw_dir / relative
            if directory.is_dir():
                output.append((domain, directory))
        return output

    # === 扫描 ===

    def scan_all(self) -> Dict[str, int]:
        """全量对比：新增/变更写入，消失的删除。返回各域处理条数。"""
        result: Dict[str, int] = {}
        for domain, directory in self._domain_dirs():
            changed = 0
            seen: set[str] = set()
            for path in sorted(directory.rglob("*.md")):
                key = str(path)
                seen.add(key)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                signature = (stat.st_mtime, stat.st_size)
                if self._state.get(key) == signature:
                    continue
                if self._notes.upsert_file(domain, path):
                    self._state[key] = signature
                    changed += 1
            for key in [item for item in self._state if item.startswith(str(directory)) and item not in seen]:
                if self._notes.delete_file(domain, key):
                    self._state.pop(key, None)
            result[domain] = changed
            if changed:
                logger.debug("[笔记导入] 域=%s 更新=%s", domain, changed)
        return result

    def start(self) -> None:
        """启动后台扫描线程，并立即做一次全量导入。"""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="pmemory-notes")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._thread is None:
                return
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        started = time.time()
        try:
            counts = self.scan_all()
            total = sum(counts.values())
            logger.info(
                "[笔记导入] 首次扫描完成 总计=%s 耗时毫秒=%s",
                total,
                int((time.time() - started) * 1000),
            )
        except Exception:
            logger.exception("[笔记导入] 首次扫描失败")
        while not self._stop.wait(self._scan_interval):
            try:
                counts = self.scan_all()
                if any(counts.values()):
                    logger.info("[笔记导入] 增量扫描 更新=%s", counts)
            except Exception:
                logger.exception("[笔记导入] 增量扫描失败")

    # === 写 ===

    def upsert_file(self, path: str | Path) -> bool:
        file_path = Path(path)
        domain = self.domain_for_path(file_path, self._raw_dir)
        if domain is None:
            logger.warning("[笔记导入] 路径不在任何知识域内，跳过 path=%s", file_path)
            return False
        ok = self._notes.upsert_file(domain, file_path)
        if ok:
            try:
                stat = file_path.stat()
                self._state[str(file_path)] = (stat.st_mtime, stat.st_size)
            except OSError:
                pass
        return ok

    async def upsert_file_async(self, path: str | Path) -> bool:
        return await asyncio.to_thread(self.upsert_file, path)

    def delete_file(self, path: str | Path) -> bool:
        file_path = Path(path)
        domain = self.domain_for_path(file_path, self._raw_dir)
        if domain is None:
            return False
        ok = self._notes.delete_file(domain, file_path)
        if ok:
            self._state.pop(str(file_path), None)
        return ok

    # === 读 ===

    def find_note(
        self, note_id: int, domains: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Optional[dict]]:
        """按 id 定位笔记。给 domains 时只查白名单内的域，hidden 域碰都不碰。"""
        if domains is not None:
            wanted = [str(name).strip().lower() for name in domains if str(name).strip()]
            if not wanted:
                return None, None
            known = dict(self._domain_dirs())
            for domain in wanted:
                if domain not in known:
                    continue
                note = self._notes.get_note(int(note_id), domain)
                if note:
                    return domain, note
            return None, None
        for domain, _ in self._domain_dirs():
            note = self._notes.get_note(int(note_id), domain)
            if note:
                return domain, note
        return None, None

    @staticmethod
    def file_for_note(note: dict) -> Optional[Path]:
        """从已查到的 note 直接解析文件路径，不再回查库。"""
        if not isinstance(note, dict):
            return None
        source = str(note.get("source") or "").strip()
        if not source:
            return None
        path = Path(source)
        return path if path.is_file() else None

    def path_for_note(self, domain: Optional[str], note: dict) -> str:
        """从已查到的 note 直接算相对路径，不再回查库。"""
        if not isinstance(note, dict):
            return ""
        source = str(note.get("source") or "")
        root = self._notes.root_dir(domain or "")
        if root is None:
            return source
        try:
            return str(Path(source).relative_to(root)).replace("\\", "/")
        except ValueError:
            return Path(source).name

    def note_file(self, note_id: int) -> Optional[Path]:
        _, note = self.find_note(note_id)
        return self.file_for_note(note or {})

    def relative_path(self, note_id: int) -> str:
        domain, note = self.find_note(note_id)
        if not note:
            return ""
        return self.path_for_note(domain, note)

    def search_notes(
        self,
        query: str,
        max_results: int = 50,
        domains: Optional[List[str]] = None,
    ) -> List[dict]:
        """按笔记聚合的检索结果，形状对齐旧 `search_notes`。

        `domains` 是允许检索的域白名单；不给表示全部已知域。不可见域由调用方
        从白名单里剔除。
        """
        text = str(query or "").strip()
        if not text:
            return []
        allowed = list(domains) if domains is not None else [name for name, _ in self._domain_dirs()]
        items: List[dict] = []
        for name in allowed:
            if not name:
                continue
            blocks = self._notes.search_blocks(
                name,
                text,
                title_limit=5,
                content_chars=3000,
                candidate_limit=1000,
            )
            for item in blocks["contents"] + blocks["titles"]:
                items.append(
                    {
                        "content": item.get("snippet") or item.get("name") or "",
                        "similarity": float(item.get("score") or 0.0),
                        "metadata": {
                            "source_file_path": item.get("path") or "",
                            "note_id": int(item.get("note_id") or 0),
                            "line_start": int(item.get("line") or 0),
                            "line_end": int(item.get("line") or 0),
                            "domain": name,
                        },
                    }
                )
        items.sort(key=lambda row: row.get("similarity") or 0.0, reverse=True)
        return items[: max(1, int(max_results))]

    async def search_notes_async(
        self,
        query: str,
        max_results: int = 50,
        domains: Optional[List[str]] = None,
    ) -> List[dict]:
        return await asyncio.to_thread(self.search_notes, query, max_results, domains)

    async def search_notes_by_top_k(
        self,
        query: str,
        recall_count: int = 100,
        top_k: int = 20,
        domain: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> List[dict]:
        del recall_count
        allowed = domains if domains is not None else ([domain] if domain else None)
        return await asyncio.to_thread(self.search_notes, query, top_k, allowed)

    def get_status(self) -> dict:
        return {
            "raw_dir": str(self._raw_dir),
            "tracked_files": len(self._state),
            "domains": [name for name, _ in self._domain_dirs()],
        }
