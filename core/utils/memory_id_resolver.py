"""
记忆ID解析器

负责处理记忆ID的转换和解析相关逻辑，避免代码重复。
"""

from typing import List, Dict, Any


class MemoryIDResolver:
    """记忆ID解析器"""

    ALLOWED_NEW_MEMORY_TYPES = {"knowledge", "skill", "emotional", "event"}
    ALLOWED_MEMORY_ACTIONS = {"create", "merge", "updata"}

    @staticmethod
    def generate_id_mapping(
        items: List[Dict[str, Any]], id_field: str = "id"
    ) -> Dict[str, str]:
        """
        生成内存态自增短 ID 映射（每次调用从 m0 开始递增）。

        Args:
            items: 包含ID字段的对象列表
            id_field: ID字段名，默认为'id'

        Returns:
            短ID到完整ID的映射字典，如 {"m0": "uuid-...", "m1": "uuid-..."}
        """
        mapping = {}
        counter = 0
        for item in items:
            full_id = item.get(id_field)
            if full_id:
                short_id = f"m{counter}"
                mapping[short_id] = full_id
                counter += 1
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
            for memory in memories:
                if memory.id.startswith(short_id):
                    resolved_ids.append(memory.id)
                    break
            else:
                if logger:
                    logger.warning(f"未找到匹配的完整ID: {short_id}")

        return resolved_ids

    @staticmethod
    def normalize_memory_actions_format(
        memory_actions_raw: List[Dict[str, Any]], logger=None
    ) -> List[Dict[str, Any]]:
        """
        统一化 memory_actions 结构。

        每条动作必须符合：
        - action=create 时：仅允许携带 memory，不应携带 source_memory_ids
        - action=merge 时：必须携带 source_memory_ids 和 memory
        - action=updata 时：必须携带且只能携带 1 个 source_memory_id，并携带 memory
        """
        normalized_actions: List[Dict[str, Any]] = []

        if not isinstance(memory_actions_raw, list):
            if logger:
                logger.warning(
                    f"memory_actions 必须是列表，实际类型: {type(memory_actions_raw)}"
                )
            return normalized_actions

        for raw_action in memory_actions_raw:
            if not isinstance(raw_action, dict):
                if logger:
                    logger.warning(
                        f"Skipping non-dict memory action: {type(raw_action)} - {raw_action}"
                    )
                continue

            action = str(raw_action.get("action", "") or "").strip().lower()
            if action not in MemoryIDResolver.ALLOWED_MEMORY_ACTIONS:
                if logger:
                    logger.warning(f"Skipping unsupported memory action: {action}")
                continue

            raw_memory = raw_action.get("memory")
            if not isinstance(raw_memory, dict):
                if logger:
                    logger.warning(f"Skipping action without memory object: {raw_action}")
                continue

            memory_type = str(raw_memory.get("type", "") or "").strip().lower()
            if memory_type not in MemoryIDResolver.ALLOWED_NEW_MEMORY_TYPES:
                if logger:
                    logger.warning(
                        f"Skipping action with unsupported memory type: {memory_type}"
                    )
                continue

            normalized_action: Dict[str, Any] = {
                "action": action,
                "memory": {
                    "type": memory_type,
                    "judgment": raw_memory.get("judgment"),
                    "reasoning": raw_memory.get("reasoning", ""),
                    "tags": raw_memory.get("tags", []),
                },
            }

            if action == "create" and "source_memory_ids" in raw_action and logger:
                logger.warning(
                    f"create 动作不应携带 source_memory_ids，已忽略: {raw_action}"
                )

            if action in {"merge", "updata"}:
                source_memory_ids = raw_action.get("source_memory_ids", [])
                if not isinstance(source_memory_ids, list) or not source_memory_ids:
                    if logger:
                        logger.warning(
                            f"{action} 动作缺少有效的 source_memory_ids，已跳过: {raw_action}"
                        )
                    continue
                normalized_source_ids = list(
                    dict.fromkeys(
                        str(memory_id).strip()
                        for memory_id in source_memory_ids
                        if str(memory_id).strip()
                    )
                )
                if not normalized_source_ids:
                    if logger:
                        logger.warning(
                            f"{action} 动作的 source_memory_ids 为空，已跳过: {raw_action}"
                        )
                    continue
                if action == "updata" and len(normalized_source_ids) != 1:
                    if logger:
                        logger.warning(
                            f"updata 动作必须且只能包含 1 个 source_memory_id，已跳过: {raw_action}"
                        )
                    continue
                normalized_action["source_memory_ids"] = normalized_source_ids

            normalized_actions.append(normalized_action)

        if logger:
            logger.debug(f"Converted memory_actions: {normalized_actions}")

        return normalized_actions

    @staticmethod
    def generate_short_id(memory_id: str, length: int = 8) -> str:
        """
        生成记忆的短ID（基于 MD5 哈希，仅用于无映射表的展示场景）。

        优先使用 generate_id_mapping 的自增短 ID；本方法仅作为兼容后备。

        Args:
            memory_id: 完整记忆ID
            length: 短ID长度（默认8位）

        Returns:
            短ID字符串
        """
        import hashlib

        if not memory_id:
            return ""

        hash_obj = hashlib.md5(memory_id.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return hash_hex[:length]
