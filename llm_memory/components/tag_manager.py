"""
标签管理器

用于管理标签去重，将标签字符串映射为整数ID，减少存储冗余。
"""

import threading
from typing import Dict, List, Optional
from .sqlite_database_manager import SQLiteDatabaseManager


class TagManager(SQLiteDatabaseManager):
    """标签管理器 - 将标签字符串映射为整数ID"""

    def __init__(self, data_directory: str, provider_id: str = "default"):
        """
        初始化标签管理器

        Args:
            data_directory: 插件数据目录
            provider_id: 提供商ID，用于分表
        """
        super().__init__(data_directory, "tag_index", provider_id)

        # 内存缓存（双向）
        self._tag_cache = {}  # {tag_name: tag_id}
        self._id_cache = {}  # {tag_id: tag_name} 反向缓存，避免查数据库
        self._cache_lock = threading.Lock()

        # 启动时加载所有标签到内存
        self._load_all_tags()

    def _get_table_name(self) -> str:
        """获取标签表名"""
        return f"tag_{self.safe_table_provider_id}"

    def _init_database(self) -> None:
        """初始化标签数据库表"""
        table_name = self._get_table_name()
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """
        self._execute_single(query, caller="初始化标签表")
        self.logger.debug(f"标签表初始化完成: {table_name}")

    def _load_all_tags(self) -> None:
        """
        加载所有标签到内存缓存（双向）
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(
                f"SELECT id, name FROM {table_name}", caller="启动时加载标签"
            )

            tag_count = 0
            for row in cursor.fetchall():
                tag_id, tag_name = row
                self._tag_cache[tag_name] = tag_id
                self._id_cache[tag_id] = tag_name  # 反向缓存
                tag_count += 1

            self.logger.info(f"标签缓存加载完成: {tag_count} 个标签")

        except Exception as e:
            self.logger.error(f"加载标签缓存失败: {e}")
            # 即使加载失败，也不应该影响初始化

    def get_or_create_tag_id(self, tag_name: str) -> int:
        """
        获取或创建标签ID

        Args:
            tag_name: 标签名称

        Returns:
            标签ID
        """
        with self._cache_lock:
            # 第一步：检查内存缓存
            if tag_name in self._tag_cache:
                return self._tag_cache[tag_name]

            # 第二步：缓存中没有，需要创建新标签
            try:
                table_name = self._get_table_name()
                cursor = self._execute_single(
                    f"INSERT INTO {table_name} (name) VALUES (?)",
                    (tag_name,),
                    caller="get_or_create_tag_id->insert",
                )
                tag_id = cursor.lastrowid

                # 第三步：更新内存缓存（双向）
                self._tag_cache[tag_name] = tag_id
                self._id_cache[tag_id] = tag_name
                self.logger.debug(f"新标签已创建: {tag_name} -> ID:{tag_id}")
                return tag_id

            except Exception as e:
                self.logger.error(f"创建标签失败: {tag_name}, 错误: {e}")
                # 如果创建失败，尝试重新查询一次（可能是其他线程已经创建）
                try:
                    cursor = self._execute_query(
                        f"SELECT id FROM {self._get_table_name()} WHERE name = ?",
                        (tag_name,),
                    )
                    result = cursor.fetchone()
                    if result:
                        tag_id = result[0]
                        self._tag_cache[tag_name] = tag_id
                        return tag_id
                except Exception as retry_error:
                    self.logger.error(
                        f"重新查询标签失败: {tag_name}, 错误: {retry_error}"
                    )
                raise

    def get_tag_name(self, tag_id: int) -> Optional[str]:
        """
        根据标签ID获取标签名

        Args:
            tag_id: 标签ID

        Returns:
            标签名称，如果不存在返回None
        """
        try:
            tag_names = self._batch_get_names_by_ids([tag_id], "name")
            return tag_names[0] if tag_names else None
        except Exception as e:
            self.logger.error(f"获取标签名称失败 (ID: {tag_id}): {e}")
            return None

    def get_or_create_tag_ids(self, tag_names: List[str]) -> List[int]:
        """
        批量获取或创建标签ID

        Args:
            tag_names: 标签名称列表

        Returns:
            标签ID列表，顺序与输入标签名称对应
        """
        if not tag_names:
            return []

        with self._cache_lock:
            result_ids = []
            new_tag_names = []
            new_tag_indices = []

            # 第一步：在内存缓存中查找已存在的标签
            for i, tag_name in enumerate(tag_names):
                if tag_name in self._tag_cache:
                    result_ids.append(self._tag_cache[tag_name])
                else:
                    result_ids.append(None)  # 占位符
                    new_tag_names.append(tag_name)
                    new_tag_indices.append(i)

            # 第二步：批量创建新标签（使用INSERT OR IGNORE避免冲突）
            if new_tag_names:
                try:
                    table_name = self._get_table_name()

                    # 在一个事务中完成插入和查询
                    def batch_create_and_query(conn):
                        cursor = conn.cursor()

                        # 使用INSERT OR IGNORE批量插入，忽略重复标签
                        insert_query = (
                            f"INSERT OR IGNORE INTO {table_name} (name) VALUES (?)"
                        )
                        params_list = [(tag_name,) for tag_name in new_tag_names]
                        cursor.executemany(insert_query, params_list)
                        conn.commit()

                        # 重新查询所有新标签的ID（包括已存在的和新插入的）
                        placeholders = ",".join(["?" for _ in new_tag_names])
                        cursor.execute(
                            f"SELECT id, name FROM {table_name} WHERE name IN ({placeholders})",
                            new_tag_names,
                        )
                        results = cursor.fetchall()
                        return results

                    # 执行整个事务
                    results = self._execute_with_connection(
                        batch_create_and_query, caller="小弟批量创建和查询tags"
                    )

                    # 构建名称到ID的映射
                    tag_id_map = {row[1]: row[0] for row in results}

                    # 更新缓存（双向）和返回结果
                    new_tag_ids = []
                    for tag_name in new_tag_names:
                        tag_id = tag_id_map[tag_name]
                        self._tag_cache[tag_name] = tag_id
                        self._id_cache[tag_id] = tag_name
                        new_tag_ids.append(tag_id)

                    # 第三步：填充结果中的空位
                    for i, tag_id in zip(new_tag_indices, new_tag_ids):
                        result_ids[i] = tag_id

                except Exception as e:
                    self.logger.error(f"批量创建标签失败: {e}")
                    raise

            return result_ids

    def get_tag_names(self, tag_ids: List[int]) -> List[str]:
        """
        批量获取标签名称（优先使用内存缓存，避免查数据库）

        Args:
            tag_ids: 标签ID列表

        Returns:
            标签名称列表，顺序与输入ID对应
        """
        if not tag_ids:
            return []

        result = []
        missing_ids = []

        # 先从缓存中查找
        with self._cache_lock:
            for tag_id in tag_ids:
                if tag_id in self._id_cache:
                    result.append(self._id_cache[tag_id])
                else:
                    # 缓存没有，记录下来去数据库查
                    missing_ids.append(tag_id)
                    result.append(None)  # 占位

        # 如果有缺失的，去数据库查询
        if missing_ids:
            db_names = self._batch_get_names_by_ids(
                missing_ids, "name", caller="get_tag_names->查询缺失"
            )
            # 填充结果并更新缓存
            missing_idx = 0
            with self._cache_lock:
                for i, tag_id in enumerate(tag_ids):
                    if result[i] is None:
                        tag_name = (
                            db_names[missing_idx]
                            if missing_idx < len(db_names)
                            else None
                        )
                        result[i] = tag_name
                        if tag_name:
                            self._id_cache[tag_id] = tag_name
                        missing_idx += 1

        return [name for name in result if name is not None]

    def get_all_tags(self) -> List[Dict]:
        """
        获取所有标签

        Returns:
            标签列表，每个字典包含id和name
        """
        try:
            return self.get_all_records()
        except Exception as e:
            self.logger.error(f"获取所有标签失败: {e}")
            return []

    def delete_tag(self, tag_id: int) -> bool:
        """
        删除标签

        Args:
            tag_id: 标签ID

        Returns:
            是否删除成功
        """
        try:
            with self._cache_lock:
                # 先获取标签名称，用于更新缓存
                tag_name = None
                for name, id_ in self._tag_cache.items():
                    if id_ == tag_id:
                        tag_name = name
                        break

                # 删除数据库记录
                success = self.delete_by_id(tag_id)

                # 如果删除成功，更新内存缓存
                if success and tag_name:
                    del self._tag_cache[tag_name]
                elif success:
                    # 如果找不到标签名称，可能缓存不一致，重新加载
                    self.logger.warning(f"删除标签成功但缓存中未找到: ID {tag_id}")
                    self._load_all_tags()

                return success

        except Exception as e:
            self.logger.error(f"删除标签失败 (ID: {tag_id}): {e}")
            return False

    def cleanup_unused_tags(self, used_tag_ids: List[int]) -> int:
        """
        清理未被使用的标签

        Args:
            used_tag_ids: 正在使用的标签ID列表

        Returns:
            删除的标签数量
        """
        if not used_tag_ids:
            self.logger.warning("提供的已使用标签ID列表为空，跳过清理")
            return 0

        try:
            with self._cache_lock:
                table_name = self._get_table_name()

                # 获取所有标签ID
                cursor = self._execute_query(f"SELECT id FROM {table_name}")
                all_tag_ids = [row[0] for row in cursor.fetchall()]

                # 找出未使用的标签ID
                unused_tag_ids = [
                    tag_id for tag_id in all_tag_ids if tag_id not in used_tag_ids
                ]

                if not unused_tag_ids:
                    self.logger.info("没有发现未使用的标签")
                    return 0

                # 批量删除未使用的标签
                placeholders = ",".join(["?" for _ in unused_tag_ids])
                delete_query = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"

                cursor = self._execute_single(
                    delete_query, tuple(unused_tag_ids), caller="cleanup_unused_tags"
                )

                deleted_count = cursor.rowcount

                # 从内存缓存中删除被清理的标签
                tags_to_remove = []
                for tag_name, tag_id in self._tag_cache.items():
                    if tag_id in unused_tag_ids:
                        tags_to_remove.append(tag_name)

                for tag_name in tags_to_remove:
                    del self._tag_cache[tag_name]

                self.logger.info(
                    f"清理了 {deleted_count} 个未使用的标签（从缓存中删除 {len(tags_to_remove)} 个）"
                )
                return deleted_count

        except Exception as e:
            self.logger.error(f"清理未使用标签失败: {e}")
            return 0

    def get_tag_statistics(self) -> Dict[str, int]:
        """
        获取标签统计信息

        Returns:
            包含标签统计信息的字典
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]

            return {"total_tags": total_count, "table_name": table_name}
        except Exception as e:
            self.logger.error(f"获取标签统计失败: {e}")
            return {"total_tags": 0, "table_name": self._get_table_name()}

    def search_tags_by_prefix(self, prefix: str, limit: int = 10) -> List[Dict]:
        """
        根据前缀搜索标签

        Args:
            prefix: 标签前缀
            limit: 返回结果数量限制

        Returns:
            匹配的标签列表
        """
        try:
            table_name = self._get_table_name()
            cursor = self._execute_query(
                f"SELECT id, name FROM {table_name} WHERE name LIKE ? ORDER BY name LIMIT ?",
                (f"{prefix}%", limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append({"id": row[0], "name": row[1]})

            return results
        except Exception as e:
            self.logger.error(f"搜索标签失败: {e}")
            return []
