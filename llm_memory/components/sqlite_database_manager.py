"""
SQLite数据库管理器 - 提供基础的连接管理和CRUD操作

这个模块提供SQLite数据库的通用管理功能，包括连接池、
批量操作、安全验证等，为FileIndexManager和TagManager提供
统一的技术基础设施。
"""

import re
import sqlite3
import threading
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue, Empty

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """数据库操作异常"""
    pass


class SQLiteDatabaseManager(ABC):
    """SQLite数据库管理器 - 提供基础的连接管理和CRUD操作"""

    def __init__(self, data_directory: str, db_name: str, provider_id: str = "default"):
        """
        初始化SQLite数据库管理器

        Args:
            data_directory: 数据存储目录
            db_name: 数据库名称（不含provider_id后缀）
            provider_id: 提供商ID，用于支持多个独立的数据集
        """
        self.logger = logger
        self.data_directory = Path(data_directory)
        self.provider_id = provider_id
        self.db_name = db_name

        # 在生成数据库文件名时确保安全
        safe_provider_id = re.sub(r'[^\w\-]', '_', str(provider_id))
        self.db_path = self.data_directory / f"{db_name}_{safe_provider_id}.db"

        self._lock = threading.Lock()
        # 优化：使用连接池替代线程本地存储
        self._connection_pool = Queue(maxsize=8)  # 限制连接池大小为8
        self._pool_lock = threading.Lock()
        self._created_connections = 0
        self._max_pool_size = 8

        # 确保数据目录存在
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # 预先创建2个连接以提升性能
        self._initialize_pool(2)

        # 初始化数据库
        self._init_database()
        self.logger.info(f"{self.__class__.__name__}初始化完成 (提供商: {provider_id}, 数据库: {self.db_path}, 连接池: {self._created_connections}/{self._max_pool_size})")


    def _initialize_pool(self, initial_size: int = 2):
        """
        初始化连接池，预先创建指定数量的连接
        
        Args:
            initial_size: 初始连接数量
        """
        for _ in range(initial_size):
            if self._created_connections < self._max_pool_size:
                conn = self._create_optimized_connection()
                with self._pool_lock:
                    self._connection_pool.put(conn)
                    self._created_connections += 1

    def _create_optimized_connection(self) -> sqlite3.Connection:
        """
        创建优化的SQLite连接
        
        Returns:
            配置好的SQLite连接对象
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0
        )
        # 设置优化参数
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        return conn

    def _get_connection(self, caller: str = "unknown") -> sqlite3.Connection:
        """
        从连接池获取数据库连接

        Args:
            caller: 调用者说明，用于追踪连接来源

        Returns:
            SQLite连接对象
        """
        try:
            # 尝试从连接池获取连接
            conn = self._connection_pool.get_nowait()
            # self.logger.debug(f"[{caller}] 从池中获取连接")  # 注释掉，减少日志噪音
            return conn
        except Empty:
            # 连接池为空，尝试创建新连接
            with self._pool_lock:
                if self._created_connections < self._max_pool_size:
                    conn = self._create_optimized_connection()
                    self._created_connections += 1
                    self.logger.debug(f"[{caller}] 创建新连接，当前连接数: {self._created_connections}/{self._max_pool_size}")
                    return conn
                else:
                    # 连接池已满，等待可用连接
                    try:
                        # self.logger.debug(f"[{caller}] 连接池已满，等待可用连接...")  # 注释掉
                        conn = self._connection_pool.get(timeout=5.0)  # 等待5秒
                        # self.logger.debug(f"[{caller}] 等待成功，获得连接")  # 注释掉
                        return conn
                    except Empty:
                        # 等待超时，创建临时连接（会被立即关闭）
                        self.logger.warning(f"[{caller}] 连接池等待超时，创建临时连接")
                        return self._create_optimized_connection()

    def _init_database(self) -> None:
        """
        初始化数据库表结构（子类实现）

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现_init_database方法")

    def _get_table_name(self) -> str:
        """
        获取表名（子类实现）

        Returns:
            表名字符串

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现_get_table_name方法")

    def _execute_with_connection(self, query_func, caller: str = "unknown"):
        """
        包装数据库操作，自动管理连接
        
        Args:
            query_func: 接收连接对象作为参数的函数
            caller: 调用者说明
        
        Returns:
            query_func的返回值
        """
        conn = None
        try:
            conn = self._get_connection(caller)
            return query_func(conn)
        finally:
            if conn:
                self._return_connection(conn)

    def _execute_query(self, query: str, params: Optional[Tuple] = None, caller: str = "unknown") -> sqlite3.Cursor:
        """
        执行SQL查询

        Args:
            query: SQL查询语句
            params: 查询参数
            caller: 调用者说明

        Returns:
            游标对象

        Raises:
            DatabaseError: 数据库操作失败时
        """
        def query_func(conn):
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
        
        try:
            return self._execute_with_connection(query_func, caller)
        except sqlite3.Error as e:
            self.logger.error(f"SQL查询执行失败: {query}, 参数: {params}, 错误: {e}")
            raise DatabaseError(f"SQL查询执行失败: {e}") from e

    def _execute_batch(self, query: str, params_list: List[Tuple], caller: str = "unknown") -> None:
        """
        批量执行SQL操作

        Args:
            query: SQL语句
            params_list: 参数列表
            caller: 调用者说明

        Raises:
            DatabaseError: 批量操作失败时
        """
        if not params_list:
            return

        try:
            conn = self._get_connection(caller)
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            self._return_connection(conn)
        except sqlite3.Error as e:
            self.logger.error(f"批量操作失败: {query}, 错误: {e}")
            raise DatabaseError(f"批量操作失败: {e}") from e

    def _execute_single(self, query: str, params: Optional[Tuple] = None, caller: str = "unknown", auto_commit: bool = True) -> sqlite3.Cursor:
        """
        执行单个SQL操作（依赖SQLite内置锁机制）

        Args:
            query: SQL查询语句
            params: 查询参数
            caller: 调用者说明
            auto_commit: 是否自动提交

        Returns:
            游标对象
        """
        def query_func(conn):
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if auto_commit:
                conn.commit()
            return cursor
        
        try:
            return self._execute_with_connection(query_func, caller)
        except sqlite3.Error as e:
            self.logger.error(f"SQL操作执行失败: {query}, 参数: {params}, 错误: {e}")
            raise DatabaseError(f"SQL操作执行失败: {e}") from e

    def _batch_get_or_create_ids(self, values: List[str], value_field: str, caller: str = "unknown") -> List[int]:
        """
        批量获取或创建ID的通用方法（依赖SQLite内置锁机制）

        Args:
            values: 值列表（文件路径或标签名）
            value_field: 字段名（'relative_path' 或 'name'）
            caller: 调用者说明

        Returns:
            ID列表，顺序与输入值对应

        Raises:
            DatabaseError: 操作失败时
        """
        if not values:
            return []

        table_name = self._get_table_name()
        result_ids = []

        # 依赖SQLite内置锁机制，不使用手动锁
        conn = None
        try:
            conn = self._get_connection(f"{caller}->batch_get_or_create_ids")
            cursor = conn.cursor()

            # 第一步：批量查询已存在的记录
            placeholders = ','.join(['?' for _ in values])
            query = f'SELECT id, {value_field} FROM {table_name} WHERE {value_field} IN ({placeholders})'
            cursor.execute(query, values)
            existing_records = {row[1]: row[0] for row in cursor.fetchall()}

            # 第二步：找出需要插入的新记录
            new_values = [value for value in values if value not in existing_records]
            if new_values:
                # 批量插入新记录
                if value_field == 'relative_path':
                    # 文件表需要时间戳字段
                    insert_query = f'INSERT INTO {table_name} ({value_field}, file_timestamp) VALUES (?, ?)'
                    params_list = [(value, 0) for value in new_values]  # 默认时间戳为0
                else:
                    # 标签表只需要name字段
                    insert_query = f'INSERT OR IGNORE INTO {table_name} ({value_field}) VALUES (?)'
                    params_list = [(value,) for value in new_values]

                cursor.executemany(insert_query, params_list)
                conn.commit()

                # 获取新插入记录的ID
                if new_values:
                    new_placeholders = ','.join(['?' for _ in new_values])
                    cursor.execute(f'SELECT id, {value_field} FROM {table_name} WHERE {value_field} IN ({new_placeholders})', new_values)
                    new_records = {row[1]: row[0] for row in cursor.fetchall()}
                    existing_records.update(new_records)

            # 第三步：按输入顺序返回对应的ID
            for value in values:
                result_ids.append(existing_records[value])

            return result_ids

        except sqlite3.Error as e:
            self.logger.error(f"批量获取或创建ID失败: {e}")
            raise DatabaseError(f"批量获取或创建ID失败: {e}") from e
        finally:
            # 确保连接总是被返还
            if conn:
                self._return_connection(conn)

    def _batch_get_names_by_ids(self, ids: List[int], name_field: str, caller: str = "unknown") -> List[str]:
        """
        批量根据ID获取名称的通用方法

        Args:
            ids: ID列表
            name_field: 名称字段名（'relative_path' 或 'name'）
            caller: 调用者说明

        Returns:
            名称列表，顺序与输入ID对应
        """
        if not ids:
            return []

        table_name = self._get_table_name()

        try:
            placeholders = ','.join(['?' for _ in ids])
            query = f'SELECT id, {name_field} FROM {table_name} WHERE id IN ({placeholders})'
            cursor = self._execute_query(query, tuple(ids), caller=caller)

            id_to_name = {row[0]: row[1] for row in cursor.fetchall()}

            # 按输入顺序返回对应的名称
            result_names = []
            for id_val in ids:
                name = id_to_name.get(id_val)
                if name:
                    result_names.append(name)

            return result_names

        except sqlite3.Error as e:
            self.logger.error(f"批量获取名称失败: {e}")
            return []

    def get_all_records(self) -> List[Dict[str, Any]]:
        """
        获取所有记录（子类可重写以提供特定字段）

        Returns:
            记录列表，每个记录包含id和其他字段
        """
        table_name = self._get_table_name()

        try:
            cursor = self._execute_query(f'SELECT * FROM {table_name} ORDER BY id')
            columns = [description[0] for description in cursor.description]

            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except sqlite3.Error as e:
            self.logger.error(f"获取所有记录失败: {e}")
            return []

    def delete_by_id(self, record_id: int) -> bool:
        """
        根据ID删除记录（依赖SQLite内置锁机制）

        Args:
            record_id: 记录ID

        Returns:
            是否删除成功
        """
        table_name = self._get_table_name()

        try:
            cursor = self._execute_single(f'DELETE FROM {table_name} WHERE id = ?', (record_id,), caller="delete_by_id")
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            self.logger.error(f"删除记录失败 (ID: {record_id}): {e}")
            return False

    def _return_connection(self, conn: sqlite3.Connection):
        """
        将连接返回到连接池
        
        Args:
            conn: SQLite连接对象
        """
        try:
            # 检查连接是否仍然有效
            try:
                conn.execute("SELECT 1")
                # 连接有效，返回到池中
                self._connection_pool.put_nowait(conn)
            except sqlite3.Error:
                # 连接无效，关闭并减少计数
                conn.close()
                with self._pool_lock:
                    if self._created_connections > 0:
                        self._created_connections -= 1
                self.logger.debug("无效连接已关闭，连接池大小减少")
        except Exception:
            # 连接池已满或其它错误，直接关闭连接
            try:
                conn.close()
            except Exception:
                pass
            with self._pool_lock:
                if self._created_connections > 0:
                    self._created_connections -= 1

    def close(self):
        """关闭所有数据库连接"""
        with self._pool_lock:
            # 关闭连接池中的所有连接
            while not self._connection_pool.empty():
                try:
                    conn = self._connection_pool.get_nowait()
                    conn.close()
                except (Empty, Exception):
                    break
            self._created_connections = 0
            self.logger.debug("所有连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(provider={self.provider_id}, db_path={self.db_path})"