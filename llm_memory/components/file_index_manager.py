"""
文件索引管理器 - 将文件路径映射为整数ID

这个模块提供文件路径到整数ID的双向映射功能，
用于优化存储空间和提升查询性能。
"""

import threading
from typing import Dict, List, Optional, Tuple
from .sqlite_database_manager import SQLiteDatabaseManager


class FileIndexManager(SQLiteDatabaseManager):
    """文件索引管理器 - 将文件路径映射为整数ID"""

    def __init__(self, data_directory: str, provider_id: str = "default"):
        """
        初始化文件索引管理器

        Args:
            data_directory: 数据存储目录
            provider_id: 提供商ID，用于支持多个独立的数据集
        """
        super().__init__(data_directory, "file_index", provider_id)

        # 内存缓存
        self._path_cache = {}      # {relative_path: (id, timestamp)}
        self._id_cache = {}        # {id: relative_path}
        self._cache_lock = threading.Lock()

        # 启动时加载所有文件索引到内存
        self._load_all_files()

    def _get_table_name(self) -> str:
        """获取文件索引表名"""
        return f"file_index_{self.provider_id}"

    def _init_database(self) -> None:
        """初始化文件索引数据库表结构"""
        table_name = self._get_table_name()
        query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT UNIQUE NOT NULL,
                file_timestamp INTEGER NOT NULL
            )
        '''
        self._execute_single(query, caller="初始化文件索引表")
        self.logger.debug(f"文件索引表初始化完成: {table_name}")

    def _load_all_files(self) -> None:
        """
        加载所有文件索引到内存缓存
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(f'SELECT id, relative_path, file_timestamp FROM {table_name}', caller="启动时加载文件索引")

            file_count = 0
            for row in cursor.fetchall():
                file_id, relative_path, file_timestamp = row
                # 双向缓存
                self._path_cache[relative_path] = (file_id, file_timestamp)
                self._id_cache[file_id] = relative_path
                file_count += 1

            self.logger.info(f"文件索引缓存加载完成: {file_count} 个文件")

        except Exception as e:
            self.logger.error(f"加载文件索引缓存失败: {e}")
            # 即使加载失败，也不应该影响初始化

    def get_or_create_file_id(self, relative_path: str, timestamp: int = 0) -> int:
        """
        获取或创建文件ID（优先使用内存缓存）

        Args:
            relative_path: 相对文件路径
            timestamp: 文件时间戳

        Returns:
            文件ID
        """
        with self._cache_lock:
            # 第一步：检查内存缓存
            if relative_path in self._path_cache:
                file_id, current_timestamp = self._path_cache[relative_path]

                # 如果提供了新的时间戳且比当前时间戳新，需要更新
                if timestamp > 0 and timestamp > current_timestamp:
                    # 更新数据库
                    try:
                        table_name = self._get_table_name()
                        self._execute_single(
                            f'UPDATE {table_name} SET file_timestamp = ? WHERE id = ?',
                            (timestamp, file_id),
                            caller="get_or_create_file_id->update_timestamp"
                        )
                        # 更新内存缓存
                        self._path_cache[relative_path] = (file_id, timestamp)
                        self.logger.debug(f"文件时间戳已更新: {relative_path} -> {timestamp}")
                    except Exception as e:
                        self.logger.error(f"更新文件时间戳失败: {relative_path}, 错误: {e}")

                return file_id

            # 第二步：缓存中没有，直接插入（不查询，省掉一次SELECT）
        table_name = self._get_table_name()

        try:
            # 直接尝试插入新记录
            try:
                cursor = self._execute_single(
                    f'INSERT INTO {table_name} (relative_path, file_timestamp) VALUES (?, ?)',
                    (relative_path, timestamp),
                    caller="小弟insert_file_id"
                )
                file_id = cursor.lastrowid
                
                # 加载到内存缓存
                with self._cache_lock:
                    self._path_cache[relative_path] = (file_id, timestamp)
                    self._id_cache[file_id] = relative_path
                
                return file_id
                
            except Exception as insert_error:
                # 插入失败（UNIQUE冲突），说明其他地方已创建，查询获取
                if "UNIQUE constraint failed" in str(insert_error):
                    cursor = self._execute_query(
                        f'SELECT id, file_timestamp FROM {table_name} WHERE relative_path = ?',
                        (relative_path,),
                        caller="get_or_create_file_id->查询冲突"
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        file_id, current_timestamp = result
                        
                        # 加载到内存缓存
                        with self._cache_lock:
                            self._path_cache[relative_path] = (file_id, current_timestamp)
                            self._id_cache[file_id] = relative_path
                        
                        # 检查是否需要更新时间戳
                        if timestamp > 0 and timestamp > current_timestamp:
                            self._execute_single(
                                f'UPDATE {table_name} SET file_timestamp = ? WHERE id = ?',
                                (timestamp, file_id),
                                caller="get_or_create_file_id->update_timestamp"
                            )
                            with self._cache_lock:
                                self._path_cache[relative_path] = (file_id, timestamp)
                        
                        return file_id
                else:
                    # 其他错误，重新抛出
                    raise

        except Exception as e:
            self.logger.error(f"获取或创建文件ID失败: {relative_path}, 错误: {e}")
            raise

    def get_file_path(self, file_id: int) -> Optional[str]:
        """
        根据文件ID获取文件路径（优先使用内存缓存）

        Args:
            file_id: 文件ID

        Returns:
            文件路径，如果不存在则返回None
        """
        with self._cache_lock:
            # 第一步：检查内存缓存
            if file_id in self._id_cache:
                return self._id_cache[file_id]

        # 第二步：缓存中没有，需要查询数据库
        table_name = self._get_table_name()

        try:
            cursor = self._execute_query(
                f'SELECT relative_path FROM {table_name} WHERE id = ?',
                (file_id,)
            )
            result = cursor.fetchone()
            
            if result:
                relative_path = result[0]
                # 加载到内存缓存
                with self._cache_lock:
                    self._id_cache[file_id] = relative_path
                    self._path_cache[relative_path] = (file_id, 0)  # 时间戳暂时为0
                return relative_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"获取文件路径失败 (ID: {file_id}): {e}")
            return None

    def update_timestamp(self, file_id: int, timestamp: int) -> bool:
        """
        更新文件时间戳

        Args:
            file_id: 文件ID
            timestamp: 新的时间戳

        Returns:
            是否更新成功
        """
        table_name = self._get_table_name()

        try:
            self._execute_single(
                f'UPDATE {table_name} SET file_timestamp = ? WHERE id = ?',
                (timestamp, file_id),
                caller="update_timestamp"
            )
            return True
        except Exception as e:
            self.logger.error(f"更新文件时间戳失败 (ID: {file_id}): {e}")
            return False

    def get_all_files(self) -> List[Dict]:
        """
        获取所有文件索引

        Returns:
            文件索引列表，每个元素包含id, relative_path, file_timestamp
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(
                f'SELECT id, relative_path, file_timestamp FROM {table_name} ORDER BY relative_path'
            )

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'relative_path': row[1],
                    'file_timestamp': row[2]
                })
            return results
        except Exception as e:
            self.logger.error(f"获取所有文件索引失败: {e}")
            return []

    def delete_file(self, file_id: int) -> bool:
        """
        删除文件索引（同时清理内存缓存）

        Args:
            file_id: 文件ID

        Returns:
            是否删除成功
        """
        try:
            # 在删除前先获取文件路径用于清理缓存
            relative_path = None
            with self._cache_lock:
                if file_id in self._id_cache:
                    relative_path = self._id_cache[file_id]
                else:
                    # 缓存中没有，从数据库查询
                    table_name = self._get_table_name()
                    cursor = self._execute_query(
                        f'SELECT relative_path FROM {table_name} WHERE id = ?',
                        (file_id,)
                    )
                    result = cursor.fetchone()
                    if result:
                        relative_path = result[0]

            # 删除数据库记录
            success = self.delete_by_id(file_id)
            
            if success and relative_path:
                # 清理内存缓存
                with self._cache_lock:
                    self._id_cache.pop(file_id, None)
                    self._path_cache.pop(relative_path, None)
                    self.logger.debug(f"已清理缓存: file_id={file_id}, path={relative_path}")

            return success
            
        except Exception as e:
            self.logger.error(f"删除文件索引失败 (ID: {file_id}): {e}")
            return False

    def get_changed_files(self, current_files: Dict[str, int]) -> Tuple[List[int], List[str]]:
        """
        比较当前文件列表与数据库中的文件，找出变更的文件

        Args:
            current_files: 当前文件列表，格式为 {相对路径: 时间戳}

        Returns:
            tuple: (需要删除的文件ID列表, 需要更新/新增的文件路径列表)
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(f'SELECT id, relative_path, file_timestamp FROM {table_name}')

            db_files = {}
            to_delete = []
            to_update = []

            # 收集数据库中的文件
            for row in cursor.fetchall():
                file_id, rel_path, timestamp = row
                db_files[rel_path] = {'id': file_id, 'timestamp': timestamp}

            # 找出需要删除的文件（数据库中有但当前没有）
            for rel_path, file_info in db_files.items():
                if rel_path not in current_files:
                    to_delete.append(file_info['id'])

            # 找出需要更新或新增的文件
            for rel_path, timestamp in current_files.items():
                if rel_path not in db_files or db_files[rel_path]['timestamp'] < timestamp:
                    to_update.append(rel_path)

            self.logger.debug(f"变更检测完成: 删除 {len(to_delete)} 个, 更新 {len(to_update)} 个")
            return to_delete, to_update

        except Exception as e:
            self.logger.error(f"获取变更文件失败: {e}")
            return [], []

    def batch_get_or_create_file_ids(self, file_paths: List[str], timestamps: List[int] = None) -> List[int]:
        """
        批量获取或创建文件ID（领导专用）

        Args:
            file_paths: 文件路径列表
            timestamps: 对应的时间戳列表（可选，默认为0）

        Returns:
            文件ID列表，顺序与输入文件路径对应
        """
        if not file_paths:
            return []

        if timestamps is None:
            timestamps = [0] * len(file_paths)
        elif len(timestamps) != len(file_paths):
            raise ValueError("时间戳列表长度必须与文件路径列表长度一致")

        try:
            # 使用基础类的批量方法
            file_ids = self._batch_get_or_create_ids(file_paths, 'relative_path', caller="领导批量分配file_id")

            # 如果有非零时间戳，需要更新
            if any(ts > 0 for ts in timestamps):
                table_name = self._get_table_name()
                update_params = [
                    (ts, file_id)
                    for ts, file_id in zip(timestamps, file_ids)
                    if ts > 0
                ]

                if update_params:
                    self._batch_execute(f'UPDATE {table_name} SET file_timestamp = ? WHERE id = ?', update_params)

            return file_ids

        except Exception as e:
            self.logger.error(f"批量获取或创建文件ID失败: {e}")
            raise

    def batch_get_file_paths(self, file_ids: List[int]) -> List[str]:
        """
        批量根据文件ID获取文件路径

        Args:
            file_ids: 文件ID列表

        Returns:
            文件路径列表，顺序与输入ID对应
        """
        return self._batch_get_names_by_ids(file_ids, 'relative_path')

    def get_file_statistics(self) -> Dict[str, any]:
        """
        获取文件统计信息（包含缓存统计）

        Returns:
            包含文件统计信息的字典
        """
        try:
            with self._cache_lock:
                cached_files = len(self._id_cache)
                
            table_name = self._get_table_name()
            cursor = self._execute_query(f'SELECT COUNT(*) FROM {table_name}')
            total_count = cursor.fetchone()[0]

            return {
                'total_files': total_count,
                'cached_files': cached_files,
                'cache_hit_ratio': cached_files / total_count if total_count > 0 else 0.0,
                'table_name': table_name
            }
        except Exception as e:
            self.logger.error(f"获取文件统计失败: {e}")
            return {'total_files': 0, 'cached_files': 0, 'cache_hit_ratio': 0.0, 'table_name': self._get_table_name()}

    def _batch_execute(self, query: str, params_list: List[Tuple]) -> None:
        """
        批量执行SQL操作（带提交）

        Args:
            query: SQL语句
            params_list: 参数列表
        """
        if not params_list:
            return

        try:
            # 直接批量执行，依赖SQLite内置锁机制
            self._execute_batch(query, params_list)
        except Exception as e:
            self.logger.error(f"批量执行失败: {e}")
            raise