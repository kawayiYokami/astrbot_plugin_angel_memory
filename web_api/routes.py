"""
路由注册入口

将所有 WebUI API 路由注册到 AstrBot context。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.star import Context

PLUGIN_NAME = "astrbot_plugin_angel_memory"


def register_all_routes(context: "Context", plugin_context) -> None:
    """注册所有 WebUI API 路由。"""
    from .memory_api import MemoryAPI
    from .notes_api import NotesAPI
    from .tags_api import TagsAPI
    from .maintenance_api import MaintenanceAPI
    from .profile_api import ProfileAPI

    memory_api = MemoryAPI(plugin_context)
    notes_api = NotesAPI(plugin_context)
    tags_api = TagsAPI(plugin_context)
    maintenance_api = MaintenanceAPI(plugin_context)
    profile_api = ProfileAPI(plugin_context)

    routes = [
        # 总览
        (f"/{PLUGIN_NAME}/overview", memory_api.get_overview, ["GET"], "记忆系统总览"),
        # 记忆
        (f"/{PLUGIN_NAME}/memories", memory_api.browse_memories, ["GET"], "浏览记忆"),
        (f"/{PLUGIN_NAME}/memories/delete", memory_api.delete_memory, ["POST"], "删除记忆"),
        # 标签
        (f"/{PLUGIN_NAME}/tags", tags_api.get_tags, ["GET"], "全局标签列表"),
        (f"/{PLUGIN_NAME}/tags/hit-search", tags_api.hit_search, ["POST"], "标签命中搜索"),
        # 向量
        (f"/{PLUGIN_NAME}/vector/search", memory_api.vector_search, ["GET"], "向量检索"),
        (f"/{PLUGIN_NAME}/vector/browse", memory_api.vector_browse, ["GET"], "向量浏览"),
        (f"/{PLUGIN_NAME}/vector/collections", memory_api.vector_collections, ["GET"], "向量集合列表"),
        # 笔记
        (f"/{PLUGIN_NAME}/notes", notes_api.browse_notes, ["GET"], "笔记索引浏览"),
        (f"/{PLUGIN_NAME}/notes/recall", notes_api.recall_note, ["POST"], "笔记内容读取"),
        (f"/{PLUGIN_NAME}/notes/files", notes_api.list_note_files, ["GET"], "笔记文件列表"),
        (f"/{PLUGIN_NAME}/notes/file-content", notes_api.get_file_content, ["GET"], "笔记文件内容"),
        # 导入导出
        (f"/{PLUGIN_NAME}/export", maintenance_api.export_snapshot, ["GET"], "导出快照"),
        (f"/{PLUGIN_NAME}/import", maintenance_api.import_snapshot, ["POST"], "导入快照"),
        # 维护
        (f"/{PLUGIN_NAME}/maintenance", maintenance_api.get_maintenance_state, ["GET"], "维护状态"),
        (f"/{PLUGIN_NAME}/maintenance/download-backup", maintenance_api.download_backup, ["GET"], "下载备份"),
        # 用户画像
        (f"/{PLUGIN_NAME}/profiles", profile_api.list_users, ["GET"], "用户画像列表"),
        (f"/{PLUGIN_NAME}/profiles/detail", profile_api.get_user_profile, ["GET"], "用户画像详情"),
    ]

    for path, handler, methods, description in routes:
        context.register_web_api(path, handler, methods, description)
