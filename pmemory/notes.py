"""p-memory 笔记适配层，对齐 story 的 KnowledgeBase 用法。

写入：`set_root` 登记域根目录，`upsert_file` 让库自己读文件、自己切片。
检索：`search_preset("notes")` 出书名 / 内容 / 路径三块；单篇检索用
`filter.note_ids` 限定；路径定位走名字列搜索再逐条校验。
读取：正文仍以文件为准，库只负责定位笔记。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from p_memory.errors import PMemoryError

from .config import DOMAIN_DIRS, SCOPE_PUBLIC, knowledge_namespace
from .kb import pmemory

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 120
DEFAULT_TITLE_LIMIT = 5
DEFAULT_CONTENT_CHARS = 3000
DEFAULT_CANDIDATE_LIMIT = 1000


def _text(value: object) -> str:
    return str(value or "").strip()


class NotesStore:
    """笔记读写。所有方法同步执行。"""

    def __init__(self, raw_dir: str | Path | None = None, store=None) -> None:
        self._raw_dir = Path(raw_dir) if raw_dir else None
        self._store = store or pmemory

    # === 目录与路径 ===

    def root_dir(self, domain: str) -> Optional[Path]:
        relative = DOMAIN_DIRS.get(str(domain or "").strip().lower())
        if not relative or self._raw_dir is None:
            return None
        return self._raw_dir / relative

    def _relative_path(self, domain: str, source: object) -> str:
        text = _text(source).replace("\\", "/")
        root = self.root_dir(domain)
        if not text or root is None:
            return text
        prefix = str(root).replace("\\", "/").rstrip("/") + "/"
        return text[len(prefix):] if text.startswith(prefix) else text.rsplit("/", 1)[-1]

    def _filter(self, domain: str, note_ids: Optional[List[int]] = None) -> dict:
        flt: Dict[str, Any] = {
            "namespace": knowledge_namespace(domain),
            "scopes": [SCOPE_PUBLIC],
        }
        if note_ids:
            flt["note_ids"] = list(note_ids)
        return flt

    def _register_root(self, domain: str) -> bool:
        directory = self.root_dir(domain)
        if directory is None or not directory.is_dir():
            return False
        try:
            self._store.kb.notes.set_root(knowledge_namespace(domain), str(directory))
            return True
        except PMemoryError:
            logger.exception("[PMEMORY] 登记笔记根目录失败 域=%s", domain)
            return False

    # === 写 ===

    def upsert_file(self, domain: str, path: str | Path) -> bool:
        file_path = Path(path)
        if not file_path.is_file():
            logger.warning("[PMEMORY] 笔记文件不存在，跳过 path=%s", file_path)
            return False
        if not self._register_root(domain):
            logger.warning("[PMEMORY] 笔记域根目录不可用，跳过写入 域=%s", domain)
            return False
        try:
            self._store.kb.notes.upsert_file(
                path=str(file_path),
                namespace=knowledge_namespace(domain),
                scope=SCOPE_PUBLIC,
            )
            self._store.kb.update_index()
            return True
        except PMemoryError:
            logger.exception("[PMEMORY] 笔记写入失败 域=%s path=%s", domain, file_path)
            return False

    def delete_file(self, domain: str, path: str | Path) -> bool:
        file_path = Path(path)
        note_id = self.resolve_note_id(domain, file_path)
        if note_id is None:
            return False
        try:
            self._store.kb.notes.delete([note_id])
            self._store.kb.update_index()
            return True
        except PMemoryError:
            logger.exception("[PMEMORY] 笔记删除失败 域=%s path=%s", domain, file_path)
            return False

    # === 读 ===

    def get_note(self, note_id: int, domain: Optional[str] = None) -> Optional[dict]:
        flt = self._filter(domain) if domain else {"scopes": [SCOPE_PUBLIC]}
        try:
            return self._store.kb.notes.get(int(note_id), filter=flt)
        except PMemoryError:
            return None

    def get_chunk(self, chunk_id: int, domain: Optional[str] = None) -> Optional[dict]:
        flt = self._filter(domain) if domain else {"scopes": [SCOPE_PUBLIC]}
        try:
            return self._store.kb.notes.get_chunk(int(chunk_id), filter=flt)
        except PMemoryError:
            return None

    def chunks(self, note_id: int, domain: Optional[str] = None) -> List[dict]:
        flt = self._filter(domain) if domain else {"scopes": [SCOPE_PUBLIC]}
        try:
            return self._store.kb.notes.chunks(int(note_id), filter=flt) or []
        except PMemoryError:
            return []

    def resolve_note_id(self, domain: str, path: str | Path) -> Optional[int]:
        """按域内相对路径定位笔记 id。

        文件名只挂在切片的名字列上，所以先用 stem 命中候选，再逐个校验
        完整路径——同名文件落在不同目录时不会拿错。
        """
        relative = self._relative_path(domain, Path(path))
        if not relative:
            return None
        stem = relative.rsplit("/", 1)[-1]
        if stem.lower().endswith(".md"):
            stem = stem[:-3]
        if not stem:
            return None
        try:
            result = self._store.kb.search(
                stem,
                filter=self._filter(domain),
                kinds=["chunk"],
                limit=50,
                text=True,
                vector=False,
                rerank=False,
                match_field="name",
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 路径定位笔记失败 域=%s path=%s", domain, relative)
            return None
        seen: List[int] = []
        for hit in result.get("hits") or []:
            record = hit.get("record") or {}
            note_id = record.get("note_id")
            if isinstance(note_id, int) and note_id not in seen:
                seen.append(note_id)
        for note_id in seen:
            note = self.get_note(note_id, domain)
            if note and self._relative_path(domain, note.get("source")) == relative:
                return note_id
        return None

    # === 检索 ===

    def search_blocks(
        self,
        domain: str,
        query: str,
        *,
        title_limit: int = DEFAULT_TITLE_LIMIT,
        content_chars: int = DEFAULT_CONTENT_CHARS,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> Dict[str, List[dict]]:
        text = _text(query)
        empty = {"titles": [], "contents": [], "paths": []}
        if not text:
            return empty
        try:
            result = self._store.kb.search_preset(
                "notes",
                text,
                filter=self._filter(domain),
                text=True,
                vector=False,
                rerank=True,
                candidate_limit=max(1, int(candidate_limit)),
                budget={
                    "note_titles": max(1, int(title_limit)),
                    "notes_chars": max(1, int(content_chars)),
                },
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 笔记三块检索失败 域=%s", domain)
            return empty
        section = (result or {}).get("notes") or {}
        return {
            "titles": self._block_items(domain, section.get("titles"), with_snippets=False),
            "contents": self._block_items(domain, section.get("contents"), with_snippets=True),
            "paths": self._block_items(domain, section.get("paths"), with_snippets=False),
        }

    def search_docs(self, domain: str, query: str, *, limit: int = 20) -> List[dict]:
        text = _text(query)
        if not text:
            return []
        try:
            result = self._store.kb.search(
                text,
                filter=self._filter(domain),
                kinds=["note", "chunk"],
                limit=max(1, int(limit)) * 5,
                text=True,
                vector=False,
                rerank=True,
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 笔记检索失败 域=%s", domain)
            return []
        groups: Dict[int, List[dict]] = {}
        for hit in result.get("hits") or []:
            record = hit.get("record") or {}
            note_id = record.get("note_id") if record.get("kind") == "chunk" else record.get("id")
            if not isinstance(note_id, int):
                continue
            score = hit.get("text_score")
            if score is None:
                score = hit.get("score") or 0.0
            groups.setdefault(note_id, []).append(
                {"chunk_id": record.get("id"), "line": record.get("offset"), "score": score}
            )
        items: List[dict] = []
        for note_id, hits in list(groups.items())[: max(1, int(limit))]:
            note = self.get_note(note_id, domain)
            if not note:
                continue
            items.append(
                {
                    "note_id": note_id,
                    "path": self._relative_path(domain, note.get("source")),
                    "name": _text(note.get("title")),
                    "score": max((float(hit.get("score") or 0.0) for hit in hits), default=0.0),
                    "hits": hits,
                }
            )
        return items

    def search_doc_chunks(self, domain: str, note_id: int, query: str, *, limit: int = 50) -> List[dict]:
        text = _text(query)
        if not text or not isinstance(note_id, int):
            return []
        try:
            result = self._store.kb.search(
                text,
                filter=self._filter(domain, [note_id]),
                kinds=["chunk"],
                limit=max(1, int(limit)),
                text=True,
                vector=False,
                rerank=False,
                top_chunks_per_note=max(1, int(limit)) * 2,
            )
        except PMemoryError:
            logger.exception("[PMEMORY] 单篇笔记检索失败 域=%s note=%s", domain, note_id)
            return []
        refs: List[dict] = []
        seen: set[int] = set()
        for hit in result.get("hits") or []:
            record = hit.get("record") or {}
            score = hit.get("text_score")
            if score is None:
                score = hit.get("score") or 0.0
            candidates = [{"id": record.get("id"), "offset": record.get("offset")}]
            candidates.extend(hit.get("top_chunks") or [])
            for ref in candidates:
                chunk_id = ref.get("id")
                if not isinstance(chunk_id, int) or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                refs.append({"chunk_id": chunk_id, "offset": ref.get("offset"), "score": score})
        chunks_map = {chunk.get("id"): chunk for chunk in self.chunks(note_id, domain)}
        output: List[dict] = []
        for ref in refs:
            chunk = chunks_map.get(ref["chunk_id"]) or {}
            output.append(
                {
                    "chunk_id": ref["chunk_id"],
                    "line": int(ref.get("offset") or 0),
                    "content": _text(chunk.get("content")),
                    "score": float(ref.get("score") or 0.0),
                }
            )
        return output

    # === 内部 ===

    def _block_items(self, domain: str, hits: object, *, with_snippets: bool) -> List[dict]:
        rows: List[tuple] = []
        seen: set[int] = set()
        for hit in hits or []:
            record = hit.get("record") or {}
            note_id = record.get("note_id") if record.get("kind") == "chunk" else record.get("id")
            if not isinstance(note_id, int) or note_id in seen:
                continue
            seen.add(note_id)
            rows.append((note_id, record, hit))
        items: List[dict] = []
        for note_id, record, hit in rows:
            note = self.get_note(note_id, domain)
            if not note:
                continue
            score = hit.get("text_score")
            if score is None:
                score = hit.get("score") or 0.0
            item = {
                "note_id": note_id,
                "path": self._relative_path(domain, note.get("source")),
                "name": _text(note.get("title")),
                "score": float(score or 0.0),
                "line": 0,
                "snippet": "",
            }
            if with_snippets:
                refs = hit.get("top_chunks") or []
                chunk_id = refs[0].get("id") if refs else record.get("id")
                line = refs[0].get("offset") if refs else record.get("offset")
                item["line"] = int(line or 0)
                chunk = self.get_chunk(chunk_id, domain) if isinstance(chunk_id, int) else None
                item["snippet"] = _text((chunk or {}).get("content"))[:SNIPPET_MAX_CHARS]
            items.append(item)
        return items
