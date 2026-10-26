"""
维护与导入导出 API（p-memory 版：快照导出/导入走新库，旧 SQL 只留作迁移源）
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.services.legacy_memory_migration import LegacyMemoryMigration


class MaintenanceAPI:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context

    async def get_maintenance_state(self):
        """获取维护状态"""
        path_manager = self.plugin_context.get_path_manager()
        memory_center_dir = path_manager.get_memory_center_dir()
        state_path = memory_center_dir / "maintenance_state.json"
        result: Dict[str, Any] = {"state": None, "backups": []}

        # 读取 maintenance_state.json
        if os.path.exists(str(state_path)):
            try:
                with open(str(state_path), "r", encoding="utf-8") as f:
                    result["state"] = json.load(f)
            except Exception as e:
                result["state_error"] = str(e)

        # 列出备份文件
        backup_dir = memory_center_dir / "backups"
        if os.path.isdir(str(backup_dir)):
            try:
                backups = []
                for name in sorted(os.listdir(str(backup_dir)), reverse=True):
                    full_path = os.path.join(str(backup_dir), name)
                    if os.path.isfile(full_path):
                        stat = os.stat(full_path)
                        backups.append({
                            "name": name,
                            "size": stat.st_size,
                            "modified_at": stat.st_mtime,
                        })
                result["backups"] = backups[:10]
            except Exception as e:
                result["backups_error"] = str(e)

        return jsonify(result)

    async def download_backup(self):
        """下载备份文件"""
        filename = request.args.get("filename", "").strip()
        if not filename:
            return jsonify({"error": "缺少 filename 参数"}), 400

        # 安全检查：防止路径穿越
        if "/" in filename or "\\" in filename or ".." in filename:
            return jsonify({"error": "非法文件名"}), 400

        path_manager = self.plugin_context.get_path_manager()
        memory_center_dir = path_manager.get_memory_center_dir()
        backup_dir = memory_center_dir / "backups"
        target = (backup_dir / filename).resolve()
        backup_root = backup_dir.resolve()

        if backup_root not in target.parents and target != backup_root:
            return jsonify({"error": "路径越界"}), 400

        if not target.exists():
            return jsonify({"error": f"文件不存在: {filename}"}), 404

        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
            return jsonify({"filename": filename, "content": json.loads(content)})
        except json.JSONDecodeError:
            # 非 JSON 文件，返回原始文本
            content = target.read_text(encoding="utf-8", errors="ignore")
            return jsonify({"filename": filename, "raw_content": content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _memory_store(self):
        try:
            return self.plugin_context.get_component("memory_store")
        except Exception:
            return None

    @staticmethod
    def _is_pmemory_id(value: Any) -> bool:
        try:
            int(str(value or "").strip())
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _existing_legacy_ids(store) -> Set[str]:
        """新库里已有的旧 ID（与自动迁移同一口径，避免导入后迁移又写一遍）。"""
        found: Set[str] = set()
        try:
            records = store.list_records(all_scopes=True)
        except Exception:
            return found
        for record in records or []:
            if not isinstance(record, dict):
                continue
            legacy = (record.get("metadata") or {}).get("legacy") or {}
            legacy_id = str(legacy.get("id") or "").strip()
            if legacy_id:
                found.add(legacy_id)
        return found

    @staticmethod
    def _normalize_snapshot_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """旧快照（关系表）与新快照（内嵌标签）统一成带 tags 列表的记录。"""
        tag_index = LegacyMemoryMigration._tag_index(data)
        normalized = []
        for row in data.get("records") or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            legacy_id = str(item.get("id") or "").strip()
            tags = item.get("tags")
            if not isinstance(tags, list) or not tags:
                tags = tag_index.get(legacy_id, [])
            item["tags"] = [str(t).strip() for t in (tags or []) if str(t).strip()]
            normalized.append(item)
        return normalized

    async def export_snapshot(self):
        """导出记忆快照（从 p-memory，与旧版同结构，标签内嵌在记录里）"""
        store = self._memory_store()
        if store is None:
            return jsonify({"error": "记忆组件不可用"}), 500
        try:
            snapshot = await asyncio.to_thread(store.export_snapshot)
            return jsonify(snapshot)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    async def import_snapshot(self):
        """导入记忆快照（写进 p-memory）

        新旧快照都收：旧 ID 写回去时带上溯源标记，自动迁移认得，不会重复搬。
        """
        data = await request.get_json()
        if not data:
            return jsonify({"error": "请求体为空"}), 400

        store = self._memory_store()
        if store is None:
            return jsonify({"error": "记忆组件不可用"}), 500

        try:
            records = self._normalize_snapshot_records(data)
            existing_legacy = await asyncio.to_thread(self._existing_legacy_ids, store)
            existing_ids: Set[str] = set()
            try:
                for record in await asyncio.to_thread(store.list_records, all_scopes=True):
                    rid = (record or {}).get("id")
                    if rid is not None:
                        existing_ids.add(str(rid))
            except Exception:
                pass

            inserted = 0
            skipped = 0
            failed = 0
            for item in records:
                legacy_id = str(item.get("id") or "").strip()
                judgment = str(item.get("judgment") or "").strip()
                if not legacy_id or not judgment:
                    skipped += 1
                    continue
                if legacy_id in existing_legacy or legacy_id in existing_ids:
                    skipped += 1
                    continue
                try:
                    metadata = None
                    if not self._is_pmemory_id(legacy_id):
                        metadata = LegacyMemoryMigration._legacy_key(legacy_id)
                    await asyncio.to_thread(
                        store.remember,
                        judgment,
                        str(item.get("reasoning") or ""),
                        item.get("tags") or [],
                        str(item.get("memory_type") or "知识记忆"),
                        str(item.get("memory_scope") or "public"),
                        pinned=bool(item.get("is_active")),
                        strength=item.get("strength"),
                        useful_count=item.get("useful_count"),
                        useful_score=item.get("useful_score"),
                        last_recalled_at=item.get("last_recalled_at"),
                        last_decay_at=item.get("last_decay_at"),
                        metadata=metadata,
                        created_at=item.get("created_at"),
                        updated_at=item.get("updated_at"),
                    )
                    if metadata:
                        existing_legacy.add(legacy_id)
                    inserted += 1
                except Exception:
                    failed += 1

            return jsonify({
                "success": True,
                "inserted": inserted,
                "skipped": skipped,
                "failed": failed,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
