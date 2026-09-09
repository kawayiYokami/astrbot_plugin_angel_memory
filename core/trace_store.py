"""反思与睡眠的过程追踪持久化。

两类记录共用一份 JSONL 文件，用 `type` 字段区分。每轮执行追加一行，行内
长文本字段（提示词、模型返回）先用标准库 lzma 压缩再 base64 编码，整行仍是
合法 JSON。文件永不自动删除、不轮转，读取从尾部倒序扫描，清理由用户手动发起。
"""

from __future__ import annotations

import asyncio
import base64
import json
import lzma
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

COMPRESSED_FIELDS = ("prompt", "system_prompt", "response")
COMPRESSED_MARKER = "__lzma__"


def _encode_field(value: Any) -> Any:
    text = str(value or "")
    if not text:
        return ""
    compressed = lzma.compress(text.encode("utf-8"))
    return {COMPRESSED_MARKER: base64.b64encode(compressed).decode("ascii")}


def _decode_field(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        payload = value.get(COMPRESSED_MARKER)
        if isinstance(payload, str):
            try:
                return lzma.decompress(base64.b64decode(payload)).decode("utf-8")
            except Exception:
                return ""
    return ""


class TraceStore:
    """进程追踪记录的 JSONL 持久化仓库。"""

    def __init__(self, base_dir: Path | str, filename: str = "process_trace.jsonl"):
        self._path = Path(base_dir) / "traces" / filename
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # === 写入 ===

    def append_sync(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """追加一条记录（同步），返回补齐 id/时间后的记录。"""
        entry = self._normalize_record(record)
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    async def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.append_sync, record)

    # === 读取 ===

    def read_recent_sync(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """从文件尾部倒序取最近若干条，返回按时间正序（旧 -> 新）的列表。"""
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        if not self._path.exists():
            return []

        collected: List[Dict[str, Any]] = []
        skipped = 0
        try:
            with self._lock:
                for raw_line in self._iter_lines_reverse():
                    if skipped < offset:
                        skipped += 1
                        continue
                    parsed = self._parse_line(raw_line)
                    if parsed is None:
                        continue
                    collected.append(parsed)
                    if len(collected) >= limit:
                        break
        except Exception:
            return collected[::-1]
        return collected[::-1]

    async def read_recent(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.read_recent_sync, limit, offset)

    # === 清理 ===

    def delete_records_sync(self, record_ids: Iterable[str]) -> int:
        """删除指定 id 的记录，重写文件；返回删除条数。"""
        targets = {str(rid).strip() for rid in (record_ids or []) if str(rid).strip()}
        if not targets or not self._path.exists():
            return 0

        with self._lock:
            kept: List[str] = []
            removed = 0
            with self._path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        kept.append(line)
                        continue
                    if str(parsed.get("id", "")) in targets:
                        removed += 1
                        continue
                    kept.append(line)

            if removed:
                tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    for line in kept:
                        f.write(line + "\n")
                os.replace(tmp_path, self._path)
        return removed

    async def delete_records(self, record_ids: Iterable[str]) -> int:
        return await asyncio.to_thread(self.delete_records_sync, record_ids)

    # === 内部 ===

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        source = dict(record or {})
        entry: Dict[str, Any] = {
            "id": str(source.pop("id", "") or self._new_id()),
            "timestamp": float(source.pop("timestamp", 0.0) or time.time()),
            "type": str(source.pop("type", "") or "reflection"),
            "status": str(source.pop("status", "") or "success"),
        }
        entry["time"] = str(
            source.pop("time", "")
            or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
        )
        for field_name in COMPRESSED_FIELDS:
            entry[field_name] = _encode_field(source.pop(field_name, ""))
        for key, value in source.items():
            entry[str(key)] = value
        return entry

    @staticmethod
    def _new_id() -> str:
        return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    def _parse_line(self, raw_line: bytes) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        for field_name in COMPRESSED_FIELDS:
            parsed[field_name] = _decode_field(parsed.get(field_name, ""))
        return parsed

    def _iter_lines_reverse(self, chunk_size: int = 65536):
        """从文件尾部按块反向产出完整行（bytes）。"""
        with self._path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            tail = b""
            while position > 0:
                read_size = min(chunk_size, position)
                position -= read_size
                f.seek(position)
                tail = f.read(read_size) + tail
                lines = tail.split(b"\n")
                tail = lines[0]
                for line in reversed(lines[1:]):
                    if line.strip():
                        yield line
            if tail.strip():
                yield tail
