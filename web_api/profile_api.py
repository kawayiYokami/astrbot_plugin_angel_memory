"""
用户画像 API

直接复用 llm_memory.utils.user_profile 的识别逻辑，
确保 WebUI 展示与运行时行为一致。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..llm_memory.utils.user_profile import (
    PROFILE_ATTRIBUTE_TAGS,
    extract_profile_attribute_from_tags,
    extract_user_ids_from_tags,
    is_user_id_tag,
    is_user_profile_tags,
    normalize_tags,
)


class ProfileAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    def _get_central_conn(self) -> Optional[sqlite3.Connection]:
        path_manager = self.plugin_context.get_path_manager()
        db_path = str(path_manager.get_simple_memory_db_path())
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    async def list_users(self):
        """列出所有可识别用户

        扫描所有记忆，用 is_user_profile_tags 判定哪些是画像记忆，
        再用 extract_user_id_from_tags 提取用户 ID。
        """
        conn = self._get_central_conn()
        if not conn:
            return jsonify({"users": []})

        try:
            cur = conn.cursor()
            # 获取所有记忆及其标签
            rows = cur.execute(
                """
                SELECT
                    mr.id, mr.judgment, mr.reasoning, mr.strength, mr.is_active,
                    mr.created_at, mr.updated_at,
                    IFNULL(tags.tags_text, '') AS tags_text
                FROM memory_records mr
                LEFT JOIN (
                    SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                WHERE mr.id IN (
                    SELECT DISTINCT mtr2.memory_id
                    FROM memory_tag_rel mtr2
                    JOIN global_tags gt2 ON gt2.id = mtr2.tag_id
                    WHERE gt2.name IN (?, ?, ?, ?, ?)
                )
                ORDER BY mr.is_active DESC, mr.updated_at DESC
                """,
                tuple(sorted(PROFILE_ATTRIBUTE_TAGS)),
            ).fetchall()

            # 账本为主数据源：平台 + 用户 ID + 昵称列表 + 群信息
            ledger_rows = cur.execute(
                """
                SELECT platform, user_id, user_names,
                       first_seen_at, last_seen_at
                FROM user_ledger
                ORDER BY last_seen_at DESC
                """
            ).fetchall()
            group_rows = cur.execute(
                """
                SELECT group_id, group_name, first_seen_at, last_seen_at
                FROM group_ledger
                ORDER BY last_seen_at DESC
                """
            ).fetchall()

            ledger_users = []
            for row in ledger_rows:
                platform = str(row["platform"] or "").strip()
                user_id = str(row["user_id"] or "").strip()
                if not platform or not user_id:
                    continue
                try:
                    names = json.loads(str(row["user_names"] or "[]"))
                except (json.JSONDecodeError, ValueError):
                    names = []
                if not isinstance(names, list):
                    names = []
                ledger_users.append({
                    "platform": platform,
                    "user_id": user_id,
                    "nickname": names[-1] if names else "",
                    "nicknames": names,
                    "first_seen_at": row["first_seen_at"],
                    "last_seen_at": row["last_seen_at"],
                })

            groups = [
                {
                    "group_id": str(g["group_id"] or "").strip(),
                    "group_name": str(g["group_name"] or "").strip(),
                    "first_seen_at": g["first_seen_at"],
                    "last_seen_at": g["last_seen_at"],
                }
                for g in group_rows
                if str(g["group_id"] or "").strip()
            ]

            # 账本 ID 作为画像判定的权威锚点（跨平台同号不串）
            known_user_ids = [u["user_id"] for u in ledger_users]

            # 用统一的识别函数分组
            user_data: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                tags_text = str(row["tags_text"] or "")
                tag_list = [t.strip() for t in tags_text.split(",") if t.strip()]

                if not is_user_profile_tags(tag_list, known_user_ids=known_user_ids):
                    continue

                user_ids = extract_user_ids_from_tags(tag_list, known_user_ids=known_user_ids)
                if not user_ids:
                    continue

                for user_id in user_ids:
                    if user_id not in user_data:
                        user_data[user_id] = {
                            "user_id": user_id,
                            "platforms": [],
                            "nickname": "",
                            "memory_count": 0,
                            "attributes": {},
                        }

                    user_data[user_id]["memory_count"] += 1

                    # 属性统计
                    attr = extract_profile_attribute_from_tags(tag_list)
                    if attr:
                        user_data[user_id]["attributes"][attr] = (
                            user_data[user_id]["attributes"].get(attr, 0) + 1
                        )

                    # 提取昵称（非用户ID、非属性标签的第一个标签）
                    if not user_data[user_id]["nickname"]:
                        for tag in tag_list:
                            if tag != user_id and tag not in PROFILE_ATTRIBUTE_TAGS and not is_user_id_tag(tag):
                                user_data[user_id]["nickname"] = tag
                                break

            # 合并账本平台/昵称信息到分组结果
            for ledger_user in ledger_users:
                uid = ledger_user["user_id"]
                if uid in user_data:
                    user_data[uid]["platforms"].append(ledger_user["platform"])
                    if not user_data[uid]["nickname"] and ledger_user["nickname"]:
                        user_data[uid]["nickname"] = ledger_user["nickname"]

            users = sorted(user_data.values(), key=lambda u: u["memory_count"], reverse=True)
            return jsonify({"users": users, "groups": groups})
        except Exception as e:
            logger.error(f"[WebUI] 用户画像列表查询失败: {e}", exc_info=True)
            return jsonify({"users": [], "error": str(e)}), 500
        finally:
            conn.close()

    async def get_user_profile(self):
        """获取指定用户的完整画像"""
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"error": "缺少 user_id 参数"}), 400

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"error": "数据库不可用"}), 500

        try:
            cur = conn.cursor()
            # 获取该用户标签 ID
            user_tag_row = cur.execute(
                "SELECT id FROM global_tags WHERE name = ?", (user_id,)
            ).fetchone()
            if not user_tag_row:
                return jsonify({"user_id": user_id, "memories": []})

            user_tag_id = int(user_tag_row["id"])

            # 获取该用户关联的所有记忆
            rows = cur.execute(
                """
                SELECT DISTINCT mr.id, mr.judgment, mr.reasoning, mr.strength,
                       mr.is_active, mr.memory_scope, mr.created_at, mr.updated_at,
                       IFNULL(tags.tags_text, '') AS tags_text
                FROM memory_records mr
                JOIN memory_tag_rel utr ON utr.memory_id = mr.id AND utr.tag_id = ?
                LEFT JOIN (
                    SELECT mtr.memory_id, GROUP_CONCAT(gt.name, ', ') AS tags_text
                    FROM memory_tag_rel mtr JOIN global_tags gt ON gt.id = mtr.tag_id
                    GROUP BY mtr.memory_id
                ) tags ON tags.memory_id = mr.id
                ORDER BY mr.is_active DESC, mr.updated_at DESC
                """,
                (user_tag_id,),
            ).fetchall()

            memories = []
            for row in rows:
                tags_text = str(row["tags_text"] or "")
                tag_list = [t.strip() for t in tags_text.split(",") if t.strip()]

                # 用统一函数判定是否为画像记忆（以当前用户 ID 为账本锚点）
                if not is_user_profile_tags(tag_list, known_user_ids=[user_id]):
                    continue

                attr_type = extract_profile_attribute_from_tags(tag_list)

                memories.append({
                    "id": row["id"],
                    "attribute": attr_type,
                    "judgment": row["judgment"],
                    "reasoning": row["reasoning"],
                    "strength": row["strength"],
                    "is_active": bool(row["is_active"]),
                    "memory_scope": row["memory_scope"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "tags": tags_text,
                })

            return jsonify({"user_id": user_id, "memories": memories})
        except Exception as e:
            logger.error(f"[WebUI] 用户画像详情查询失败: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
