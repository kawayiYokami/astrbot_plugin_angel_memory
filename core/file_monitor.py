"""
文件扫描服务

扫描raw文件夹中的文件，并自动同步到数据库。
"""

import os
from pathlib import Path
from typing import Dict, List, Union

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
from ..llm_memory.components.file_index_manager import FileIndexManager
from ..llm_memory.service.note_service import NoteService


class FileMonitorService:
    """文件扫描服务类"""

    def __init__(
        self, data_directory: str, note_service: NoteService, config: dict = None
    ):
        """
        初始化文件扫描服务

        Args:
            data_directory: 插件数据目录
            note_service: 笔记服务实例
            config: 配置字典
        """
        self.logger = logger
        self.data_directory = Path(data_directory)
        self.note_service = note_service

        # 使用PathManager统一管理路径
        path_manager = note_service.plugin_context.get_path_manager()
        self.raw_directory = path_manager.get_raw_dir()

        self.logger.info("文件监控器初始化完成（顺序处理模式）")

        # 文件扫描状态统一使用中央真相源（global），并复用 IDService 的 file_manager
        if hasattr(note_service, "id_service") and hasattr(
            note_service.id_service, "file_manager"
        ):
            self.file_index_manager = note_service.id_service.file_manager
        else:
            # 兜底：避免因注入异常导致监控器不可用
            self.file_index_manager = FileIndexManager(
                str(path_manager.get_memory_center_index_dir()), "global"
            )

        # 确保raw目录存在
        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("文件扫描服务初始化完成（增量同步模式）")
        self.logger.info(f"数据目录: {self.data_directory}")
        self.logger.info(f"扫描目录: {self.raw_directory}")
        self.logger.info(f"扫描目录存在: {self.raw_directory.exists()}")

    def start_monitoring(self):
        """启动文件扫描服务（增量同步模式）"""
        try:
            self.logger.info("🔄 开始增量同步...")
            self._incremental_sync()
            self.logger.info("📂 文件扫描服务已完成")

        except Exception as e:
            self.logger.error(f"启动文件扫描服务失败: {e}")
        finally:
            # 关键修复：扫描完成后彻底清理所有资源
            self._cleanup_all_resources()

    def stop_monitoring(self):
        """停止文件扫描服务"""
        try:
            self._cleanup_all_resources()
            self.logger.info("文件扫描服务已停止。")
        except Exception as e:
            self.logger.error(f"停止文件扫描服务时发生错误: {e}")

    def _force_cleanup_connections(self):
        """强制清理所有连接（仅在必要时调用）"""
        try:
            # 只有在内存压力大或程序结束时才关闭所有连接
            if hasattr(self.file_index_manager, "close"):
                self.file_index_manager.close()

            if hasattr(self.note_service, "id_service"):
                if hasattr(self.note_service.id_service, "close"):
                    self.note_service.id_service.close()
        except Exception:
            pass

    def _cleanup_all_resources(self):
        """彻底清理所有资源（扫描完成后调用）"""
        try:
            # 1. 强制关闭SQLite连接（只在程序结束时）
            self._force_cleanup_connections()

            # 2. 关闭NoteService的线程池（如果有）
            if hasattr(self.note_service, "_thread_pool"):
                self.note_service._thread_pool.shutdown(wait=True)
                self.logger.debug("✅ NoteService线程池已关闭")

            # 3. 向量索引由睡眠维护管线统一同步。
            self.logger.debug("跳过向量索引维护操作（由睡眠维护统一管理）")

            self.logger.info("🔓 所有资源已释放，线程已回收")

        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")

    def _force_vector_index_vacuum(self):
        """已废弃：向量索引空间回收由FAISS重建流程处理。"""
        self.logger.debug("已跳过向量索引VACUUM操作（由FAISS重建流程处理）")
        return


    def _format_timing_log(self, timings: dict) -> str:
        """格式化计时信息为日志字符串（中央索引链路）"""
        parts = []

        # 1. 文件解析（切块 + ID查询）
        if "parse" in timings:
            parse_parts = [f"切块{timings['parse']:.0f}ms"]
            if "id_lookup" in timings and timings["id_lookup"] > 1:
                parse_parts.append(f"ID{timings['id_lookup']:.0f}ms")
            parts.append(f"文件解析：{' + '.join(parse_parts)}")

        # 2. 旧索引清理
        if "delete_old_index" in timings:
            parts.append(f"旧索引清理：{timings['delete_old_index']:.0f}ms")

        # 3. 中央索引写入
        if "store_total" in timings:
            parts.append(f"中央索引写入：{timings['store_total']:.0f}ms")

        # 4. 向量缓存写入（包含 embedding）
        if "vector_sync" in timings:
            parts.append(f"向量缓存写入：{timings['vector_sync']:.0f}ms")

        # 5. 线程等待（如果显著）
        if "_thread_wait" in timings and timings["_thread_wait"] > 100:
            parts.append(f"线程等待：{timings['_thread_wait']:.0f}ms")

        return " | ".join(parts)

    # ===== 增量同步功能 =====

    def _incremental_sync(self):
        """异步执行增量同步"""
        import time

        start_time = time.time()
        self.logger.info(f"开始增量同步: {self.raw_directory}")

        try:
            # 1. 获取数据库状态
            old_files = self.file_index_manager.get_all_files()
            self.logger.debug(f"数据库中有 {len(old_files)} 个文件记录")

            # 2. 扫描文件系统
            current_files = self._scan_directory_for_files(self.raw_directory)
            self.logger.debug(f"文件系统中有 {len(current_files)} 个文件")

            # 3. 对比分析变更
            changes = self._compare_file_states(old_files, current_files)
            self.logger.info(
                f"变更检测完成: 删除 {len(changes['to_delete'])} 个, 新增/更新 {len(changes['to_add'])} 个, 无变化 {len(changes['unchanged'])} 个"
            )

            # 4. 执行删除操作（先删除旧数据）
            delete_count = 0
            if changes["to_delete"]:
                # 收集所有需要删除的文件ID
                file_ids = [file_id for file_id, _ in changes["to_delete"]]

                # 批量删除所有文件数据
                if self._delete_file_data_by_file_id(file_ids):
                    delete_count = len(file_ids)
                    self.logger.info(f"批量删除完成: {delete_count} 个文件")
                else:
                    self.logger.error("批量删除文件数据失败")

            # 5. 执行新增/更新操作（顺序处理，降低文件索引抖动）
            add_count = 0
            if changes["to_add"]:
                self.logger.info(
                    f"开始顺序处理 {len(changes['to_add'])} 个新增/更新文件..."
                )

                # 顺序处理每个文件
                for idx, (relative_path, timestamp) in enumerate(changes["to_add"]):
                    try:
                        import time as time_module

                        file_start = time_module.time()

                        doc_count, timings = self._process_file_change(
                            relative_path, timestamp
                        )
                        if doc_count > 0:
                            add_count += 1

                        # 详细的处理日志
                        total_time = (time_module.time() - file_start) * 1000
                        file_name = Path(relative_path).name
                        timing_str = self._format_timing_log(timings)

                        self.logger.info(
                            f"[{idx + 1}/{len(changes['to_add'])}] ✅ {file_name} | "
                            f"块数:{doc_count} | 总耗时:{total_time:.0f}ms | {timing_str}"
                        )

                        # 每100个文件显示进度
                        if (idx + 1) % 100 == 0:
                            progress = (idx + 1) / len(changes["to_add"]) * 100
                            self.logger.info(
                                f"📊 进度: {progress:.1f}% ({idx + 1}/{len(changes['to_add'])})"
                            )

                    except Exception as e:
                        self.logger.error(f"处理文件失败: {relative_path}, 错误: {e}")
                        continue

            # 6. 计算执行时间
            execution_time = time.time() - start_time

            self.logger.info(
                f"增量同步完成: 耗时 {execution_time:.2f}s, 删除 {delete_count} 个文件, 新增 {add_count} 个文件"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"增量同步失败: {e}, 耗时 {execution_time:.2f}s")
            import traceback

            self.logger.error(f"错误详情: {traceback.format_exc()}")

    def _scan_directory_for_files(self, directory_path: Path) -> Dict[str, int]:
        """
        扫描目录，获取所有支持的文件及其时间戳（优化版，使用os.walk）

        Args:
            directory_path: 要扫描的目录路径

        Returns:
            字典格式：{相对路径: 时间戳}
        """
        import time

        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.warning(f"目录不存在或不是目录: {directory_path}")
            return {}

        current_files = {}
        supported_extensions = {
            ".md",
            ".txt",
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".html",
            ".csv",
            ".json",
            ".xml",
        }

        try:
            # 使用os.walk递归扫描（比Path.rglob快很多）
            t_start = time.time()
            file_count = 0
            base_path = str(directory_path)
            base_path_len = len(base_path) + 1  # +1 for trailing slash

            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    file_count += 1
                    # 快速检查扩展名（避免创建Path对象）
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in supported_extensions:
                        try:
                            # 构建完整路径
                            full_path = os.path.join(root, filename)

                            # 获取文件时间戳
                            timestamp = int(os.path.getmtime(full_path))

                            # 计算相对路径（字符串切片，比Path.relative_to快）
                            if full_path.startswith(base_path):
                                relative_path = full_path[base_path_len:].replace(
                                    "\\", "/"
                                )
                            else:
                                relative_path = os.path.relpath(
                                    full_path, base_path
                                ).replace("\\", "/")

                            current_files[relative_path] = timestamp
                        except (OSError, ValueError) as e:
                            self.logger.warning(
                                f"无法获取文件信息: {full_path}, 错误: {e}"
                            )
                            continue

            scan_time = time.time() - t_start
            self.logger.info(
                f"✅ 扫描完成，发现 {len(current_files)} 个支持的文件（共{file_count}个文件） | 耗时: {scan_time:.2f}秒"
            )
            return current_files

        except Exception as e:
            self.logger.error(f"扫描目录失败: {directory_path}, 错误: {e}")
            return {}

    def _compare_file_states(
        self, old_files: List[Dict], current_files: Dict[str, int]
    ) -> Dict:
        """
        比较文件状态，识别变更

        Args:
            old_files: 数据库中的文件列表，格式：[{id: int, relative_path: str, file_timestamp: int}]
            current_files: 当前文件系统状态，格式：{相对路径: 时间戳}

        Returns:
            变更分析结果，格式：
            {
                "to_delete": [(file_id, relative_path)],  # 已删除或需要重新处理的文件
                "to_add": [(relative_path, timestamp)],    # 新增或修改的文件
                "unchanged": [(file_id, relative_path)]    # 无变化的文件
            }
        """
        # 构建数据库文件的快速查找字典
        db_files = {}
        for file_info in old_files:
            db_files[file_info["relative_path"]] = file_info

        to_delete = []
        to_add = []
        unchanged = []

        # 检查数据库中的文件（查找已删除或时间戳变化的文件）
        for relative_path, file_info in db_files.items():
            if relative_path not in current_files:
                # 文件已删除
                to_delete.append((file_info["id"], relative_path))
            elif current_files[relative_path] > file_info["file_timestamp"]:
                # 文件时间戳更新，需要重新处理
                to_delete.append((file_info["id"], relative_path))
                to_add.append((relative_path, current_files[relative_path]))
            else:
                # 文件无变化
                unchanged.append((file_info["id"], relative_path))

        # 检查当前文件系统中的新文件
        for relative_path, timestamp in current_files.items():
            if relative_path not in db_files:
                # 新文件
                to_add.append((relative_path, timestamp))

        return {"to_delete": to_delete, "to_add": to_add, "unchanged": unchanged}

    def _process_file_change(self, relative_path: str, timestamp: int) -> tuple:
        """处理单个文件的变更，返回(文档数量, 计时字典)（同步版本）"""
        try:
            # 构建完整文件路径
            full_path = self.raw_directory / relative_path

            if not full_path.exists():
                self.logger.warning(f"文件不存在: {full_path}")
                return 0, {}

            # 小弟向领导申请file_id（领导串行分配，避免一次性创建5800个）
            file_id = self.file_index_manager.get_or_create_file_id(
                relative_path, timestamp
            )

            try:
                # 小弟处理文件，使用领导分配的file_id（同步调用）
                doc_count, timings = self.note_service.parse_and_store_file_sync(
                    str(full_path), relative_path
                )
                return doc_count, timings
            except Exception as e:
                # 失败了，回滚这个file_id
                self.logger.error(
                    f"文件处理失败，回滚file_id: {relative_path}, 错误: {e}"
                )
                try:
                    # 使用改造后的方法，支持单个文件删除
                    self._delete_file_data_by_file_id(file_id)
                    self.logger.debug(
                        f"已回滚文件索引: {relative_path} (ID: {file_id})"
                    )
                except Exception as rollback_error:
                    self.logger.error(
                        f"回滚文件索引失败: {relative_path}, 错误: {rollback_error}"
                    )
                raise

        except Exception as e:
            self.logger.error(f"文件处理失败: {relative_path}, 错误: {e}")
            return 0, {}

    def _delete_file_data_by_file_id(self, file_ids: Union[int, List[int]]) -> bool:
        """
        删除文件相关的所有数据（支持单个和批量删除）

        Args:
            file_ids: 单个文件ID或文件ID列表

        Returns:
            是否删除成功
        """
        # 统一处理输入参数
        if isinstance(file_ids, int):
            file_ids = [file_ids]
        elif not file_ids:
            return True

        try:
            self.logger.info(f"开始删除 {len(file_ids)} 个文件的数据")
            ok = True
            for file_id in file_ids:
                if not self.note_service.remove_file_data_by_file_id(file_id):
                    ok = False

            # 批量删除 file_index SQLite 记录
            self._batch_delete_sqlite_records(file_ids)
            return ok
        except Exception as e:
            self.logger.error(f"删除文件数据失败 (IDs: {file_ids}): {e}")
            return False

    def _batch_delete_sqlite_records(self, file_ids: List[int]) -> bool:
        """批量删除SQLite记录和内存缓存"""
        try:
            table_name = self.file_index_manager._get_table_name()
            placeholders = ",".join(["?" for _ in file_ids])

            # 批量查询文件路径（用于缓存清理）
            select_query = f"SELECT id, relative_path FROM {table_name} WHERE id IN ({placeholders})"
            cursor = self.file_index_manager._execute_query(
                select_query, tuple(file_ids)
            )
            file_mappings = cursor.fetchall()  # [(id, path), (id, path), ...]

            # 批量删除SQLite记录（使用新的可靠方法）
            delete_query = f"DELETE FROM {table_name} WHERE id = ?"
            params_list = [(file_id,) for file_id in file_ids]
            deleted_count = self.file_index_manager._execute_batch_delete(
                delete_query, params_list, caller="batch_delete_sqlite"
            )
            self.logger.debug(
                f"请求删除 {len(file_ids)} 个SQLite记录，通过新的批量方法实际删除了 {deleted_count} 个。"
            )

            # 批量清理内存缓存
            with self.file_index_manager._cache_lock:
                for file_id, relative_path in file_mappings:
                    self.file_index_manager._id_cache.pop(file_id, None)
                    self.file_index_manager._path_cache.pop(relative_path, None)

            self.logger.debug(f"已清理 {len(file_mappings)} 个文件的内存缓存")
            return True
        except Exception as e:
            self.logger.error(f"删除SQLite记录失败: {e}")
            return False
