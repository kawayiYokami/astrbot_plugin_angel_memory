"""
记忆ID解析器

负责处理记忆ID的转换和解析相关逻辑，避免代码重复。
"""

from typing import List, Dict, Any


class MemoryIDResolver:
    """记忆ID解析器"""

    @staticmethod
    def generate_id_mapping(
        items: List[Dict[str, Any]], id_field: str = "id"
    ) -> Dict[str, str]:
        """
        通用的ID映射生成方法

        Args:
            items: 包含ID字段的对象列表
            id_field: ID字段名，默认为'id'

        Returns:
            短ID到完整ID的映射字典
        """
        mapping = {}
        for item in items:
            full_id = item.get(id_field)
            if full_id:
                short_id = MemoryIDResolver.generate_short_id(full_id)
                mapping[short_id] = full_id
        return mapping

    @staticmethod
    def resolve_memory_ids(
        short_ids: List[str], memories: List, logger=None
    ) -> List[str]:
        """
        将短ID转换为完整ID

        Args:
            short_ids: 短ID列表（如 ["a1b2c3", "d4e5f6"]）
            memories: 记忆对象列表
            logger: 日志记录器（可选）

        Returns:
            完整ID列表
        """
        resolved_ids = []

        for short_id in short_ids:
            # 在记忆中查找匹配的完整ID
            for memory in memories:
                if memory.id.startswith(short_id):
                    resolved_ids.append(memory.id)
                    break
            else:
                # 如果没有找到匹配的ID，记录警告但继续处理
                if logger:
                    logger.warning(f"未找到匹配的完整ID: {short_id}")

        return resolved_ids

    @staticmethod
    def normalize_new_memories_format(
        new_memories_raw: Dict[str, Any] | List[Dict[str, Any]], logger=None
    ) -> List[Dict[str, Any]]:
        """
        统一化新记忆格式，从字典（按类型分组）转换为列表

        Args:
            new_memories_raw: 原始的新记忆数据
            logger: 日志记录器（可选）

        Returns:
            统一格式后的记忆列表
        """
        new_memories = []

        if isinstance(new_memories_raw, dict):
            for memory_type, memories in new_memories_raw.items():
                if isinstance(memories, list):
                    for memory in memories:
                        # 检查 memory 是否是字典
                        if not isinstance(memory, dict):
                            if logger:
                                logger.warning(
                                    f"Skipping non-dict memory in {memory_type}: {type(memory)} - {memory}"
                                )
                            continue
                        # 添加类型字段
                        memory["type"] = memory_type
                        new_memories.append(memory)
        elif isinstance(new_memories_raw, list):
            # 如果已经是列表，直接使用
            new_memories = new_memories_raw

        if logger:
            logger.debug(f"Converted new_memories: {new_memories}")

        return new_memories

    @staticmethod
    def normalize_merge_groups_format(
        merge_groups_raw: List[Dict[str, Any] | List[str]],
    ) -> List[List[str]]:
        """
        统一化合并组格式：从对象列表提取ids字段或直接使用列表

        Args:
            merge_groups_raw: 原始的合并组数据

        Returns:
            统一格式后的合并组列表
        """
        merge_groups = []

        if isinstance(merge_groups_raw, list):
            for group in merge_groups_raw:
                if isinstance(group, dict) and "ids" in group:
                    merge_groups.append(group["ids"])
                elif isinstance(group, list):
                    # 如果已经是列表格式，直接使用
                    merge_groups.append(group)

        return merge_groups

    @staticmethod
    def generate_short_id(memory_id: str, length: int = 8) -> str:
        """
        生成记忆的短ID（使用哈希算法确保唯一性）

        Args:
            memory_id: 完整记忆ID
            length: 短ID长度（默认8位，更可靠）

        Returns:
            短ID（基于哈希的无冲突标识符）
        """
        import hashlib

        if not memory_id:
            return ""

        # 使用MD5哈希算法生成唯一短ID，避免截取可能产生的冲突
        hash_obj = hashlib.md5(memory_id.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return hash_hex[:length]
