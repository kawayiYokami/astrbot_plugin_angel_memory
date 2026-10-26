"""旧 SQL 记忆自动迁移到 p-memory。

用户不做任何操作：插件启动时在后台先导出再导入，对得上就跳过，对不上就补齐。
两步走，中间一定落一个 JSON 文件，方便打开看：
1. 导出：旧 SQL（优先）+ 最新 JSON 冷备份（补缺）合并，写到
   `memory_center/legacy_memory_export.json`；
2. 导入：只从这份导出文件读，逐条写进 p-memory。

判重键是 `metadata.legacy.id`（旧 UUID），重复执行只写缺失条。
失败的条目不记完成，下次启动继续补，不会永久丢失。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EXPORT_FILE = "legacy_memory_export.json"
BACKUP_GLOB = "memory_backup_*.json"
LEGACY_FORMAT = "angel_memory"
LEGACY_TABLE = "memory_records"


class LegacyMemoryMigration:
    """把旧库导出成 JSON，再从 JSON 补齐进 p-memory。"""

    def __init__(
        self,
        memory_store,
        memory_sql_manager,
        state_dir,
        backup_dir=None,
        logger=None,
    ):
        self._store = memory_store
        self._sql = memory_sql_manager
        self._state_dir = Path(state_dir)
        self._export_file = self._state_dir / EXPORT_FILE
        self._backup_dir = Path(backup_dir) if backup_dir else None
        self.logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    # === 入口 ===

    def start(self) -> None:
        """后台执行，不阻塞插件启动。"""
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.logger.warning("[记忆迁移] 当前没有事件循环，迁移未启动")
            return
        self._task = loop.create_task(self.run_if_needed())

    async def run_if_needed(self) -> Dict[str, Any]:
        async with self._lock:
            started = time.time()
            sql_records, sql_tags, sql_error = await self._read_sql()
            backup_records, backup_tags, backup_file = await asyncio.to_thread(
                self._read_latest_backup
            )

            records = self._merge_records(sql_records, backup_records)
            tags = {**backup_tags, **sql_tags}
            sources = []
            if sql_records:
                sources.append("旧SQL")
            if backup_records:
                sources.append(f"JSON备份({backup_file})")
            source_text = "+".join(sources) if sources else "无"

            if not records:
                self.logger.info("[记忆迁移] 跳过 旧库与备份都没有记忆")
                return {"status": "skipped", "written": 0, "source": source_text}

            if sql_error and backup_records:
                self.logger.warning(
                    f"[记忆迁移] 旧SQL不可用，改用 JSON 备份 错误={sql_error}"
                )

            # 第一步：导出成 JSON，落盘才算数
            try:
                export_path = await asyncio.to_thread(
                    self._export_to_json, records, tags
                )
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 导出失败 来源={source_text} 候选={len(records)} "
                    f"耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            # 第二步：只从这份导出文件导入
            try:
                import_records, import_tags = await asyncio.to_thread(
                    self._read_export_file, export_path
                )
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 读取导出文件失败 文件={export_path.name} "
                    f"耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            self.logger.debug(
                f"[记忆迁移] 开始 来源={source_text} 导出={export_path.name} "
                f"待迁移候选={len(import_records)}"
            )
            try:
                result = await asyncio.to_thread(
                    self._migrate, import_records, import_tags
                )
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 失败 来源={source_text} 候选={len(import_records)} "
                    f"耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            cost_ms = int((time.time() - started) * 1000)
            written = int(result["written"])
            failed = int(result["failed"])
            skipped = int(result["skipped"])
            if failed:
                self.logger.warning(
                    f"[记忆迁移] 部分失败 来源={source_text} 本次写入={written} "
                    f"失败={failed} 已存在={skipped} 耗时毫秒={cost_ms} "
                    f"下次启动会继续补"
                )
                status = "partial" if written else "failed"
                return {
                    "status": status,
                    "written": written,
                    "failed": failed,
                    "skipped": skipped,
                    "source": source_text,
                    "export_file": str(export_path),
                }
            if not written:
                self.logger.info(
                    f"[记忆迁移] 跳过 导出与新库已对齐 来源={source_text} "
                    f"已存在={skipped}"
                )
                return {
                    "status": "skipped",
                    "written": 0,
                    "failed": 0,
                    "skipped": skipped,
                    "source": source_text,
                    "export_file": str(export_path),
                }
            self.logger.info(
                f"[记忆迁移] 完成 来源={source_text} 导出={export_path.name} "
                f"本次写入={written} 已存在={skipped} 耗时毫秒={cost_ms}"
            )
            return {
                "status": "success",
                "written": written,
                "failed": 0,
                "skipped": skipped,
                "source": source_text,
                "export_file": str(export_path),
            }

    # === 导出 / 导入 ===

    def _export_to_json(
        self, records: List[dict], tag_index: Dict[str, List[str]]
    ) -> Path:
        """合并后的记录连同标签一起落盘，返回导出文件路径。"""
        export_records: List[dict] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            legacy_id = str(item.get("id") or "").strip()
            if legacy_id and "tags" not in item:
                inline = tag_index.get(legacy_id, [])
                if inline:
                    item["tags"] = list(inline)
            export_records.append(item)
        payload = {
            "schema_version": 1,
            "exported_at": int(time.time()),
            "records": export_records,
            "global_tags": [],
            "memory_tag_rel": [],
        }
        self._export_file.parent.mkdir(parents=True, exist_ok=True)
        with self._export_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return self._export_file

    def _read_export_file(self, path: Path) -> Tuple[List[dict], Dict[str, List[str]]]:
        with Path(path).open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("records"), list
        ):
            raise ValueError("导出文件结构不符")
        records = [row for row in snapshot["records"] if isinstance(row, dict)]
        return records, self._tag_index(snapshot)

    # === 数据源读取 ===

    async def _read_sql(self) -> Tuple[List[dict], Dict[str, List[str]], str]:
        try:
            snapshot = await self._sql.export_backup_snapshot()
        except Exception as e:
            self.logger.warning(f"[记忆迁移] 读取旧SQL失败: {e}")
            return [], {}, str(e)
        return (
            list(snapshot.get("records") or []),
            self._tag_index(snapshot),
            "",
        )

    def _read_latest_backup(self) -> Tuple[List[dict], Dict[str, List[str]], str]:
        if self._backup_dir is None or not self._backup_dir.is_dir():
            return [], {}, ""
        files = sorted(self._backup_dir.glob(BACKUP_GLOB))
        if not files:
            return [], {}, ""
        latest = files[-1]
        try:
            with latest.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as e:
            self.logger.warning(f"[记忆迁移] 读取 JSON 备份失败 文件={latest.name} 错误={e}")
            return [], {}, ""

        records = snapshot.get("records") if isinstance(snapshot, dict) else None
        if not isinstance(records, list):
            self.logger.warning(f"[记忆迁移] JSON 备份结构不符，已忽略 文件={latest.name}")
            return [], {}, ""
        return list(records), self._tag_index(snapshot), latest.name

    @staticmethod
    def _merge_records(primary: List[dict], fallback: List[dict]) -> List[dict]:
        """旧 SQL 优先，备份只补 SQL 里缺失的条目。"""
        merged: List[dict] = []
        seen: Set[str] = set()
        for row in list(primary) + list(fallback):
            if not isinstance(row, dict):
                continue
            legacy_id = str(row.get("id") or "").strip()
            if not legacy_id or legacy_id in seen:
                continue
            seen.add(legacy_id)
            merged.append(row)
        return merged

    # === 实际写入（同步，跑在线程里） ===

    def _migrate(self, records: List[dict], tag_index: Dict[str, List[str]]) -> Dict[str, int]:
        existing = self._existing_legacy_ids()
        written = 0
        skipped = 0
        failed = 0

        for row in records:
            legacy_id = str(row.get("id") or "").strip()
            judgment = str(row.get("judgment") or "").strip()
            if not legacy_id or not judgment:
                skipped += 1
                continue
            if legacy_id in existing:
                skipped += 1
                continue
            try:
                self._store.remember(
                    judgment,
                    str(row.get("reasoning") or ""),
                    tag_index.get(legacy_id, []),
                    str(row.get("memory_type") or "知识记忆"),
                    str(row.get("memory_scope") or "public"),
                    pinned=bool(row.get("is_active")),
                    strength=int(row.get("strength") or 1),
                    useful_count=int(row.get("useful_count") or 0),
                    useful_score=float(row.get("useful_score") or 0.0),
                    last_recalled_at=row.get("last_recalled_at") or None,
                    last_decay_at=row.get("last_decay_at") or None,
                    metadata=self._legacy_key(legacy_id),
                    created_at=row.get("created_at") or None,
                    updated_at=row.get("updated_at") or None,
                )
            except Exception as e:
                failed += 1
                self.logger.warning(
                    f"[记忆迁移] 单条写入失败 legacy_id={legacy_id} 错误={e}"
                )
                continue
            existing.add(legacy_id)
            written += 1

        return {"written": written, "skipped": skipped, "failed": failed}

    @staticmethod
    def _legacy_key(legacy_id: str) -> Dict[str, Any]:
        # 与 p-memory 自带导入器的溯源结构保持一致，排障时能直接对回旧表
        return {
            "legacy": {
                "format": LEGACY_FORMAT,
                "table": LEGACY_TABLE,
                "id": legacy_id,
            }
        }

    @staticmethod
    def _tag_index(snapshot: Dict[str, Any]) -> Dict[str, List[str]]:
        """标签来源有两种：旧库的 global_tags + 关系表，或备份内嵌在记录里。"""
        names = {
            str(tag.get("id")): str(tag.get("name") or "")
            for tag in (snapshot.get("global_tags") or [])
            if isinstance(tag, dict)
        }
        index: Dict[str, List[str]] = {}
        for rel in snapshot.get("memory_tag_rel") or []:
            if not isinstance(rel, dict):
                continue
            memory_id = str(rel.get("memory_id") or "")
            name = names.get(str(rel.get("tag_id") or ""), "")
            if memory_id and name:
                index.setdefault(memory_id, []).append(name)
        for row in snapshot.get("records") or []:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("id") or "").strip()
            inline = row.get("tags")
            if memory_id and isinstance(inline, list) and inline:
                index.setdefault(memory_id, [str(tag) for tag in inline if str(tag).strip()])
        return index

    def _existing_legacy_ids(self) -> Set[str]:
        found: Set[str] = set()
        for record in self._store.list_records(all_scopes=True):
            metadata = record.get("metadata") or {}
            legacy = metadata.get("legacy") or {}
            legacy_id = str(legacy.get("id") or "").strip()
            if legacy_id and str(legacy.get("format") or "") == LEGACY_FORMAT:
                found.add(legacy_id)
        return found

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
