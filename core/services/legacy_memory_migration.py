"""旧 SQL 记忆自动迁移到 p-memory。

一次性动作：插件启动时在后台跑一次，完成后写 `legacy_migration_state.json`，
之后无论升级多少次都不再读旧库与备份。

数据源只有两处，且旧 SQL 优先：
1. 旧 SQL（`MemorySqlManager`）有数据就用它；
2. 旧 SQL 完全读不到时才看 `memory_center/backups/` 里的冷备份兜一次。
   备份平时是给用户手动导入用的，不参与每次启动。

两步走，中间落一个 JSON 文件，方便打开看：
1. 导出：合并后的记录写到 `memory_center/legacy_memory_export.json`；
2. 导入：只从这份导出文件读，逐条写进 p-memory。

判重键是 `metadata.legacy.id`（旧 UUID），另外按「正文 + 柜子」判重，
避免备份里混进新库已有的记忆时被复制一份。
失败按性质分开处理：字段非法是数据本身坏，单条跳过、照常收尾，下次不再尝试；
库调用失败（磁盘满、IO 错误）说明库此刻写不进去，整批不收尾、下次重试，
免得一次环境故障把整批旧记忆永久丢掉。两种都只在日志留痕。
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from astrbot.api import logger as plugin_logger

EXPORT_FILE = "legacy_memory_export.json"
STATE_FILE = "legacy_migration_state.json"
BACKUP_GLOB = "memory_backup_*.json"
LEGACY_FORMAT = "angel_memory"
LEGACY_TABLE = "memory_records"


class LegacyMemoryMigration:
    """把旧库导出成 JSON，再从 JSON 补齐进 p-memory。

    这是一次性动作：迁完就在 `legacy_migration_state.json` 写下完成标记，
    之后无论插件升级多少次都不再读旧库与备份。备份是给用户手动导入用的，
    这里只在旧 SQL 完全读不到数据时兜一次。
    """

    def __init__(
        self,
        memory_store,
        memory_sql_manager,
        state_dir,
        backup_dir=None,
        version="",
        logger=None,
    ):
        self._store = memory_store
        self._sql = memory_sql_manager
        self._state_dir = Path(state_dir)
        self._export_file = self._state_dir / EXPORT_FILE
        self._state_file = self._state_dir / STATE_FILE
        self._backup_dir = Path(backup_dir) if backup_dir else None
        self._version = str(version or "")
        self.logger = logger or plugin_logger
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        # 自管单线程 executor：asyncio.to_thread 走默认池，拿不到底层线程，
        # 取消协程时写库线程仍在跑。自管才能保证 shutdown 等到它真正结束。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legacy-migration")
        self._migrate_future = None

    def _already_done(self) -> bool:
        """迁过一次就永久跳过：不读旧 SQL，也不读备份。"""
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(isinstance(state, dict) and state.get("done"))

    def _mark_done(
        self,
        written: int,
        source: str,
        failed_data: int = 0,
        failed_env: int = 0,
    ) -> None:
        payload = {
            "done": True,
            "version": self._version,
            "written": int(written),
            "failed_data": int(failed_data),
            "failed_env": int(failed_env),
            "source": source,
            "done_at": int(time.time()),
        }
        try:
            self._state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"[记忆迁移] 写入完成标记失败 错误={e}")

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

    async def run_if_needed(self) -> dict[str, Any]:
        async with self._lock:
            if self._already_done():
                return {"status": "skipped", "written": 0, "source": "已迁移过"}

            started = time.time()
            sql_records, sql_tags, sql_error = await self._read_sql()

            # 旧 SQL 优先；只有 SQL 完全没有数据时才看冷备份。
            # 备份是给用户手动导入用的，不能每次启动都当迁移源读一遍。
            if sql_records:
                records, tags, source_text = list(sql_records), sql_tags, "旧SQL"
            else:
                backup_records, backup_tags, backup_file = await asyncio.to_thread(self._read_latest_backup)
                records = list(backup_records)
                tags = backup_tags
                source_text = f"JSON备份({backup_file})" if backup_records else "无"
                if sql_error and backup_records:
                    self.logger.warning(f"[记忆迁移] 旧SQL不可用，改用 JSON 备份 错误={sql_error}")

            if not records:
                if sql_error:
                    # 读失败不等于没数据：这时候写完成标记会让旧记忆永久迁不进来，
                    # 所以只记失败、不写标记，下次启动继续尝试。
                    self.logger.warning(f"[记忆迁移] 失败 旧SQL读取失败且没有可用备份，本次不迁移 错误={sql_error}")
                    return {"status": "failed", "written": 0, "source": source_text}
                self.logger.info("[记忆迁移] 跳过 旧库与备份都没有记忆")
                self._mark_done(0, source_text)
                return {"status": "skipped", "written": 0, "source": source_text}

            # 第一步：导出成 JSON，落盘才算数
            try:
                export_path = await asyncio.to_thread(self._export_to_json, records, tags)
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 导出失败 来源={source_text} 候选={len(records)} 耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            # 第二步：只从这份导出文件导入
            try:
                import_records, import_tags = await asyncio.to_thread(self._read_export_file, export_path)
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 读取导出文件失败 文件={export_path.name} 耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            self.logger.debug(
                f"[记忆迁移] 开始 来源={source_text} 导出={export_path.name} 待迁移候选={len(import_records)}"
            )
            try:
                future = self._executor.submit(self._migrate, import_records, import_tags)
                self._migrate_future = future
                result = await asyncio.wrap_future(future)
            except Exception as e:
                cost_ms = int((time.time() - started) * 1000)
                self.logger.error(
                    f"[记忆迁移] 失败 来源={source_text} 候选={len(import_records)} 耗时毫秒={cost_ms} 错误={e}",
                    exc_info=True,
                )
                return {"status": "failed", "written": 0, "source": source_text}

            cost_ms = int((time.time() - started) * 1000)
            written = int(result["written"])
            skipped = int(result["skipped"])
            failed_data = int(result.get("failed_data", 0))
            failed_env = int(result.get("failed_env", 0))
            failed = failed_data + failed_env
            if failed:
                # 失败分性质再决定收不收尾：
                # failed_data = 数据本身坏（字段转换失败），跳过即收、不再试。
                # failed_env = 库调用失败（磁盘满、IO 错误），说明库此刻不可写，
                #   这时收尾等于整批永久丢，所以不写标记、下次重试。
                # 注意不能用 skipped 判库可写——判重是内存比对、不写库。
                if failed_env:
                    self.logger.error(
                        f"[记忆迁移] 失败 来源={source_text} 库调用失败={failed_env} "
                        f"数据损坏={failed_data} 已存在={skipped} 不收尾、下次重试"
                    )
                    return {
                        "status": "failed",
                        "written": written,
                        "failed": failed,
                        "failed_data": failed_data,
                        "failed_env": failed_env,
                        "skipped": skipped,
                        "source": source_text,
                        "export_file": str(export_path),
                    }
                self.logger.warning(
                    f"[记忆迁移] 部分跳过 来源={source_text} 本次写入={written} "
                    f"数据损坏跳过={failed_data} 已存在={skipped} 耗时毫秒={cost_ms}"
                )
                self._mark_done(written, source_text, failed_data, failed_env)
                return {
                    "status": "partial",
                    "written": written,
                    "failed": failed,
                    "failed_data": failed_data,
                    "failed_env": failed_env,
                    "skipped": skipped,
                    "source": source_text,
                    "export_file": str(export_path),
                }
            if not written:
                self._mark_done(0, source_text)
                self.logger.info(f"[记忆迁移] 跳过 导出与新库已对齐 来源={source_text} 已存在={skipped}")
                return {
                    "status": "skipped",
                    "written": 0,
                    "failed": 0,
                    "skipped": skipped,
                    "source": source_text,
                    "export_file": str(export_path),
                }
            self._mark_done(written, source_text)
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

    def _export_to_json(self, records: list[dict], tag_index: dict[str, list[str]]) -> Path:
        """合并后的记录连同标签一起落盘，返回导出文件路径。"""
        export_records: list[dict] = []
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

    def _read_export_file(self, path: Path) -> tuple[list[dict], dict[str, list[str]]]:
        with Path(path).open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("records"), list):
            raise ValueError("导出文件结构不符")
        records = [row for row in snapshot["records"] if isinstance(row, dict)]
        return records, self._tag_index(snapshot)

    # === 数据源读取 ===

    async def _read_sql(self) -> tuple[list[dict], dict[str, list[str]], str]:
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

    def _read_latest_backup(self) -> tuple[list[dict], dict[str, list[str]], str]:
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

    # === 实际写入（同步，跑在线程里） ===

    def _migrate(self, records: list[dict], tag_index: dict[str, list[str]]) -> dict[str, int]:
        existing = self._existing_legacy_ids()
        # 除了按旧 ID 判重，还要按「正文 + 柜子」判重：
        # 备份文件是新库自己导出的快照，把新库已有的记忆当成旧数据再写一遍，
        # 就会凭空多出一条一模一样的。这一层挡住它。
        existing_content = {
            (str(record.get("scope") or "public"), str(record.get("judgment") or "").strip())
            for record in self._store.list_records(all_scopes=True)
        }
        written = 0
        skipped = 0
        failed_data = 0
        failed_env = 0

        for row in records:
            legacy_id = str(row.get("id") or "").strip()
            judgment = str(row.get("judgment") or "").strip()
            if not legacy_id or not judgment:
                skipped += 1
                continue
            content_key = (str(row.get("memory_scope") or "public"), judgment)
            if legacy_id in existing or content_key in existing_content:
                self.logger.debug(f"[记忆迁移] 跳过重复 legacy_id={legacy_id} judgment=「{judgment[:20]}」")
                skipped += 1
                continue

            # 字段转换与库调用分开：转换抛的是数据错（旧库里的脏值，
            # 如 strength="3.5"），跟库能不能写无关，跳过即收、下次不试；
            # remember 抛的才是库调用失败，那才需要下次重试。
            try:
                strength = int(row.get("strength") or 1)
                useful_count = int(row.get("useful_count") or 0)
                useful_score = float(row.get("useful_score") or 0.0)
            except (ValueError, TypeError, OverflowError) as e:
                failed_data += 1
                self.logger.warning(f"[记忆迁移] 字段格式非法，跳过 legacy_id={legacy_id} 错误={e}")
                continue

            try:
                self._store.remember(
                    judgment,
                    str(row.get("reasoning") or ""),
                    tag_index.get(legacy_id, []),
                    str(row.get("memory_type") or "知识记忆"),
                    str(row.get("memory_scope") or "public"),
                    pinned=bool(row.get("is_active")),
                    strength=strength,
                    useful_count=useful_count,
                    useful_score=useful_score,
                    last_recalled_at=row.get("last_recalled_at") or None,
                    last_decay_at=row.get("last_decay_at") or None,
                    metadata=self._legacy_key(legacy_id),
                    created_at=row.get("created_at") or None,
                    updated_at=row.get("updated_at") or None,
                )
            except Exception as e:
                failed_env += 1
                self.logger.warning(f"[记忆迁移] 写入失败 legacy_id={legacy_id} 错误={e}")
                continue
            existing.add(legacy_id)
            existing_content.add(content_key)
            written += 1

        return {
            "written": written,
            "skipped": skipped,
            "failed_data": failed_data,
            "failed_env": failed_env,
        }

    @staticmethod
    def _legacy_key(legacy_id: str) -> dict[str, Any]:
        # 与 p-memory 自带导入器的溯源结构保持一致，排障时能直接对回旧表
        return {
            "legacy": {
                "format": LEGACY_FORMAT,
                "table": LEGACY_TABLE,
                "id": legacy_id,
            }
        }

    @staticmethod
    def _tag_index(snapshot: dict[str, Any]) -> dict[str, list[str]]:
        """标签来源有两种：旧库的 global_tags + 关系表，或备份内嵌在记录里。"""
        names = {
            str(tag.get("id")): str(tag.get("name") or "")
            for tag in (snapshot.get("global_tags") or [])
            if isinstance(tag, dict)
        }
        index: dict[str, list[str]] = {}
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

    def _existing_legacy_ids(self) -> set[str]:
        found: set[str] = set()
        for record in self._store.list_records(all_scopes=True):
            metadata = record.get("metadata") or {}
            legacy = metadata.get("legacy") or {}
            legacy_id = str(legacy.get("id") or "").strip()
            if legacy_id and str(legacy.get("format") or "") == LEGACY_FORMAT:
                found.add(legacy_id)
        return found

    async def shutdown(self) -> None:
        """取消迁移任务并等它收束。

        只 cancel 不 await 是不够的：任务可能正停在 await asyncio.to_thread(...)
        里，取消不会立刻停掉底层写库线程，随后 pmemory.close() 会和它抢同一份
        native 句柄。
        """
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if task.cancelled():
                self.logger.debug("[旧记忆迁移] 任务已取消")
            elif task.exception() is not None:
                self.logger.warning(f"[旧记忆迁移] 任务异常退出: {task.exception()}")

        # 协程取消不等于线程停下：写库跑在自管 executor 的线程里，必须单独等。
        # 等待期间的任何结果都不能让 executor 逃过关闭，所以 shutdown 放 finally：
        # 迁移线程在 30 秒内抛错结束时会从 future.result() 原样重抛，
        # 那条路径既不是超时也不该吞掉，但绝不能因此跳过 executor.shutdown。
        future = self._migrate_future
        self._migrate_future = None
        try:
            if future is not None and not future.done():
                try:
                    await asyncio.to_thread(future.result, 30)
                except (TimeoutError, FuturesTimeoutError):
                    # 超时后必须报出来，否则后面的 shutdown(wait=True) 会等很久。
                    # 两个类型都接：Python 3.11 之前 concurrent.futures.TimeoutError
                    # 还不是内建 TimeoutError 的别名。
                    self.logger.warning("[旧记忆迁移] 写库线程 30 秒未结束，卸载将等待它退出（可能较久）")
                except Exception as e:
                    # 迁移线程已抛错结束，线程本身已经退了；记一条即可，
                    # 不要让它冒到 shutdown() 外面——那会跳过下面的 executor 关闭。
                    self.logger.warning(f"[旧记忆迁移] 写库线程异常结束: {e}")
        finally:
            # wait=True 无界等会把卸载挂死，给个上限：超时就放弃，线程随进程退出。
            try:
                await asyncio.wait_for(asyncio.to_thread(self._executor.shutdown, True), timeout=30)
            except TimeoutError:
                self.logger.error("[旧记忆迁移] 写库线程 30 秒仍未退出，放弃等待，随进程退出")

    def stop(self) -> None:
        """同步版：只发取消信号。关闭流程请用 shutdown()。"""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
