"""
维护与导入导出 API
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from quart import jsonify, request

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MaintenanceAPI:
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

    async def export_snapshot(self):
        """导出中央记忆快照"""
        conn = self._get_central_conn()
        if not conn:
            return jsonify({"error": "数据库不可用"}), 500

        try:
            cur = conn.cursor()
            records = [dict(row) for row in cur.execute(
                """
                SELECT id, memory_type, judgment, reasoning, strength, is_active,
                       useful_count, useful_score, last_recalled_at, last_decay_at,
                       memory_scope, created_at, updated_at
                FROM memory_records ORDER BY created_at DESC
                """
            ).fetchall()]

            tags = [dict(row) for row in cur.execute(
                "SELECT id, name FROM global_tags ORDER BY id ASC"
            ).fetchall()]

            rel = [dict(row) for row in cur.execute(
                "SELECT memory_id, tag_id FROM memory_tag_rel"
            ).fetchall()]

            snapshot = {
                "schema_version": 1,
                "exported_at": int(time.time()),
                "records": records,
                "global_tags": tags,
                "memory_tag_rel": rel,
            }
            return jsonify(snapshot)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    async def import_snapshot(self):
        """导入中央记忆快照"""
        data = await request.get_json()
        if not data:
            return jsonify({"error": "请求体为空"}), 400

        records = data.get("records", [])
        tags = data.get("global_tags", [])
        rel = data.get("memory_tag_rel", [])

        conn = self._get_central_conn()
        if not conn:
            return jsonify({"error": "数据库不可用"}), 500

        try:
            cur = conn.cursor()
            inserted = 0
            skipped = 0
            failed = 0

            # 导入标签
            for tag in tags:
                try:
                    cur.execute(
                        "INSERT OR IGNORE INTO global_tags(id, name) VALUES (?, ?)",
                        (int(tag["id"]), str(tag["name"])),
                    )
                except Exception:
                    pass

            # 导入记忆
            for record in records:
                try:
                    existing = cur.execute(
                        "SELECT id FROM memory_records WHERE id = ?",
                        (record["id"],),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue

                    cur.execute(
                        """
                        INSERT INTO memory_records(
                            id, memory_type, judgment, reasoning, strength, is_active,
                            useful_count, useful_score, last_recalled_at, last_decay_at,
                            memory_scope, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["id"], record.get("memory_type", ""),
                            record.get("judgment", ""), record.get("reasoning", ""),
                            record.get("strength", 50), record.get("is_active", 0),
                            record.get("useful_count", 0), record.get("useful_score", 0.0),
                            record.get("last_recalled_at"), record.get("last_decay_at"),
                            record.get("memory_scope", "public"),
                            record.get("created_at", time.time()),
                            record.get("updated_at", time.time()),
                        ),
                    )
                    inserted += 1
                except Exception:
                    failed += 1

            # 导入标签关联
            for r in rel:
                try:
                    cur.execute(
                        "INSERT OR IGNORE INTO memory_tag_rel(memory_id, tag_id) VALUES (?, ?)",
                        (str(r["memory_id"]), int(r["tag_id"])),
                    )
                except Exception:
                    pass

            conn.commit()
            return jsonify({
                "success": True,
                "inserted": inserted,
                "skipped": skipped,
                "failed": failed,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
