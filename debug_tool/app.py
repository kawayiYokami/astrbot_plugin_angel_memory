import datetime
import json
import math
import os
import re

import streamlit as st

from utils.config_loader import ConfigLoader
from utils.db import DBManager

st.set_page_config(
    page_title="Angel Memory Debug Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stTextArea textarea { font-family: Consolas, Monaco, monospace; }
    .main .block-container { padding-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_managers():
    loader = ConfigLoader()
    provider = loader.get_embedding_provider()
    db_mgr = DBManager(loader, provider)
    return loader, db_mgr, loader.get_raw_notes_dir()


def render_item(item, show_hit_count: bool = False):
    meta = item.get("metadata", {}) or {}
    content = item.get("document") or meta.get("judgment") or "*[无内容]*"
    header = []
    if meta.get("memory_type"):
        header.append(f"🏷️ `{meta['memory_type']}`")
    if show_hit_count and item.get("hit_count") is not None:
        header.append(f"🎯 命中标签数 `{item.get('hit_count')}`")
    if meta.get("memory_scope"):
        header.append(f"🔒 scope `{meta.get('memory_scope')}`")
    if meta.get("created_at"):
        try:
            ts = float(meta.get("created_at"))
            if ts > 1e11:
                ts /= 1000
            header.append(
                f"🕒 {datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception:
            pass
    if meta.get("strength") is not None:
        header.append(f"💪 `{meta.get('strength')}`")
    if meta.get("tags"):
        header.append(f"🔖 {meta.get('tags')}")
    if header:
        st.markdown(" | ".join(header))
    st.markdown(content)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _scan_md_headings(lines):
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
    rows = []
    for idx, line in enumerate(lines):
        m = heading_re.match(line)
        if not m:
            continue
        rows.append((len(m.group(1)), _normalize_text(m.group(2)), idx))
    return rows


def _extract_markdown_segment(text: str, h_values, heading_level: int, heading_text: str):
    lines = text.split("\n")
    headings = _scan_md_headings(lines)
    if not headings:
        return text, []

    available = [(lvl, title) for lvl, title, _ in headings]
    target_start = -1
    target_level = 0
    if heading_level and heading_text:
        target = _normalize_text(heading_text)
        for level, title, line_no in headings:
            if level == heading_level and _normalize_text(title) == target:
                target_start = line_no
                target_level = level
                break
    else:
        expected = [_normalize_text(v) for v in h_values]
        current = ["", "", "", "", "", ""]
        for level, title, line_no in headings:
            current[level - 1] = _normalize_text(title)
            for i in range(level, 6):
                current[i] = ""
            matched = True
            for i, val in enumerate(expected):
                if val and current[i] != val:
                    matched = False
                    break
            if matched and any(expected):
                target_start = line_no
                target_level = max([i + 1 for i, v in enumerate(expected) if v], default=0)
                break

    if target_start < 0:
        return "", available

    target_end = len(lines)
    for level, _, line_no in headings:
        if line_no <= target_start:
            continue
        if level <= target_level:
            target_end = line_no
            break
    return "\n".join(lines[target_start:target_end]).strip(), available


loader, db_mgr, raw_dir = get_managers()

with st.sidebar:
    st.title("🧠 Angel Memory")
    mode = st.radio(
        "调试模式",
        [
            "📌 总览",
            "🧾 中央记忆浏览",
            "🔖 全局Tags调试",
            "🧭 memory_index 检索",
            "🗂️ 中央笔记索引",
            "🧠 notes_index 检索",
            "📝 note_recall 模拟",
            "🔄 中央库导入导出",
            "🛠️ 维护状态",
            "📂 浏览笔记文件",
        ],
        index=0,
    )
    st.divider()
    overview = db_mgr.get_overview()
    st.caption(f"Provider: {overview.get('provider_id')}")
    st.caption(f"中央记忆: {overview.get('memory_count', 0)}")
    st.caption(f"全局Tags: {overview.get('global_tag_count', 0)}")
    st.caption(f"memory_index: {overview.get('memory_index_count', 0)}")

if mode == "📌 总览":
    st.subheader("📌 当前调试环境")
    ov = db_mgr.get_overview()
    st.json(ov)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**路径状态**")
        st.write(f"- central db: `{ov.get('simple_db_path')}`")
        st.write(f"- maintenance: `{ov.get('maintenance_state_path')}`")
        st.write(f"- backups: `{ov.get('backup_dir')}`")
        st.write(f"- chromadb: `{ov.get('chromadb_path')}`")
    with c2:
        st.markdown("**scope 列表**")
        scopes = ov.get("scopes") or []
        if scopes:
            for s in scopes:
                st.write(f"- `{s}`")
        else:
            st.caption("暂无")

elif mode == "🧾 中央记忆浏览":
    st.subheader("🧾 中央记忆浏览（memory_center/index/simple_memory.db）")
    if not db_mgr.has_central_db():
        st.error("未找到中央记忆库。")
        st.stop()

    stats = db_mgr.get_central_stats()
    scopes = stats.get("scopes", [])
    c1, c2 = st.columns([2, 3])
    with c1:
        selected_scope = st.selectbox("scope 过滤", ["(全部)"] + scopes, index=0)
    with c2:
        keyword = st.text_input("关键词（judgment/reasoning/tags）", value="")

    page_size = 20
    current_page = int(st.session_state.get("central_page", 1) or 1)
    scope_filter = "" if selected_scope == "(全部)" else selected_scope
    offset = max(0, (current_page - 1) * page_size)
    items, total = db_mgr.browse_central_memories(
        limit=page_size,
        offset=offset,
        scope=scope_filter,
        keyword=keyword,
        return_total=True,
    )
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page = st.number_input(
            f"页码 (共 {total_pages} 页)",
            min_value=1,
            max_value=max(1, total_pages),
            value=min(current_page, total_pages),
            key="central_page",
        )
    if page != current_page:
        offset = max(0, (page - 1) * page_size)
        items, total = db_mgr.browse_central_memories(
            limit=page_size,
            offset=offset,
            scope=scope_filter,
            keyword=keyword,
            return_total=True,
        )
    st.caption(f"总记录 {total}，当前页 {len(items)} 条")
    for item in items:
        with st.container(border=True):
            render_item(item)
            with st.expander("元数据"):
                st.json(item.get("metadata", {}))

elif mode == "🔖 全局Tags调试":
    st.subheader("🔖 全局 Tags 调试（global_tags + memory_tag_rel + note_tag_rel）")
    if not db_mgr.has_central_db():
        st.error("未找到中央记忆库。")
        st.stop()

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        query_text = st.text_input("输入原始问题（用于 tags 命中）", value="")
    with c2:
        scope_for_hit = st.text_input("scope（可空）", value="")
    with c3:
        hit_limit = st.number_input("命中记忆上限", min_value=1, max_value=200, value=30)

    if query_text:
        result = db_mgr.unified_tag_hit_search(query_text, limit=int(hit_limit), scope=scope_for_hit)
        st.markdown("**命中 tags**")
        st.write(result.get("matched_tags", []))
        st.caption(f"matched_tag_ids: {result.get('matched_tag_ids', [])}")

        st.markdown("**命中记忆**")
        hits = result.get("memory_hits", [])
        if hits:
            for item in hits:
                with st.container(border=True):
                    render_item(item, show_hit_count=True)
                    with st.expander("元数据"):
                        st.json(item.get("metadata", {}))
        else:
            st.info("无命中记忆。")

    st.divider()
    st.markdown("**全局 tags 列表**")
    tag_keyword = st.text_input("筛选 tag name", value="", key="tag_keyword_filter")
    tags = db_mgr.get_global_tags(limit=300, offset=0, keyword=tag_keyword)
    st.caption(f"共返回 {len(tags)} 条")
    if tags:
        st.dataframe(tags, use_container_width=True, hide_index=True)

elif mode == "🧭 memory_index 检索":
    st.subheader("🧭 向量轻量索引调试（collection = memory_index）")
    if not db_mgr.has_vector_db():
        st.error("未连接向量库。")
        st.stop()

    collections = db_mgr.get_collections()
    if "memory_index" not in collections:
        st.warning("未找到 `memory_index` 集合。")
    else:
        st.caption(f"memory_index count: {db_mgr.get_collection_stats('memory_index').get('count', 0)}")
        query = st.text_input("向量查询", value="", placeholder="输入一句话测试向量召回")
        topk = st.slider("Top K", min_value=1, max_value=30, value=10)
        if query:
            results = db_mgr.query_collection(
                collection_name="memory_index",
                query_text=query,
                n_results=int(topk),
            )
            for item in results:
                if item.get("error"):
                    st.error(item.get("error"))
                    continue
                with st.container(border=True):
                    st.markdown(f"**score:** `{item.get('score', 0):.4f}` | **id:** `{item.get('id')}`")
                    st.code(item.get("document") or "", language="text")
                    if item.get("metadata"):
                        st.caption(f"metadata: {item.get('metadata')}")

        st.divider()
        st.markdown("**原始浏览**")
        page_size = 20
        page = st.number_input("页码", min_value=1, value=1, key="memory_index_page")
        offset = (int(page) - 1) * page_size
        items = db_mgr.browse_collection("memory_index", limit=page_size, offset=offset)
        for item in items:
            with st.container(border=True):
                st.markdown(f"**id:** `{item.get('id')}`")
                st.code(item.get("document") or "", language="text")
                if item.get("metadata"):
                    st.caption(f"metadata: {item.get('metadata')}")

elif mode == "🗂️ 中央笔记索引":
    st.subheader("🗂️ 中央笔记索引（note_index_records + note_tag_rel）")
    if not db_mgr.has_central_db():
        st.error("未找到中央记忆库。")
        st.stop()

    stats = db_mgr.get_note_index_stats()
    st.caption(
        f"note_index_records: {stats.get('note_index_count', 0)} | "
        f"note_tag_rel: {stats.get('note_tag_rel_count', 0)}"
    )
    keyword = st.text_input("关键词（路径/标题/tags）", value="")
    page_size = 20
    current_page = int(st.session_state.get("note_index_page", 1) or 1)
    offset = max(0, (current_page - 1) * page_size)
    items, total = db_mgr.browse_note_index_records(
        limit=page_size,
        offset=offset,
        keyword=keyword,
        return_total=True,
    )
    total_pages = math.ceil(total / page_size) if total > 0 else 1
    page = st.number_input(
        f"页码 (共 {total_pages} 页)",
        min_value=1,
        max_value=max(1, total_pages),
        value=min(current_page, total_pages),
        key="note_index_page",
    )
    if page != current_page:
        offset = max(0, (page - 1) * page_size)
        items, total = db_mgr.browse_note_index_records(
            limit=page_size,
            offset=offset,
            keyword=keyword,
            return_total=True,
        )
    st.caption(f"总记录 {total}，当前页 {len(items)} 条")
    for row in items:
        with st.container(border=True):
            title = " / ".join(
                [str(row.get(f"heading_h{i}") or "").strip() for i in range(1, 7) if str(row.get(f"heading_h{i}") or "").strip()]
            ) or "(无标题)"
            st.markdown(f"**{row.get('source_file_path', '')}**")
            st.caption(f"title: {title}")
            st.caption(
                f"note_short_id: {row.get('note_short_id', -1)} | total_lines: {row.get('total_lines', 0)} | "
                f"tags: {row.get('tags_text', '')}"
            )
            with st.expander("详情"):
                st.json(row)

elif mode == "🧠 notes_index 检索":
    st.subheader("🧠 向量轻量索引调试（collection = notes_index）")
    if not db_mgr.has_vector_db():
        st.error("未连接向量库。")
        st.stop()

    collections = db_mgr.get_collections()
    if "notes_index" not in collections:
        st.warning("未找到 `notes_index` 集合。")
    else:
        st.caption(f"notes_index count: {db_mgr.get_collection_stats('notes_index').get('count', 0)}")
        query = st.text_input("向量查询", value="", placeholder="输入一句话测试笔记向量召回")
        topk = st.slider("Top K", min_value=1, max_value=30, value=10, key="notes_index_topk")
        if query:
            results = db_mgr.query_collection(
                collection_name="notes_index",
                query_text=query,
                n_results=int(topk),
            )
            for item in results:
                if item.get("error"):
                    st.error(item.get("error"))
                    continue
                with st.container(border=True):
                    st.markdown(f"**score:** `{item.get('score', 0):.4f}` | **id:** `{item.get('id')}`")
                    st.code(item.get("document") or "", language="text")
                    if item.get("metadata"):
                        st.caption(f"metadata: {item.get('metadata')}")

elif mode == "📝 note_recall 模拟":
    st.subheader("📝 note_recall 模拟（按 note_short_id + 行范围）")
    if not os.path.exists(raw_dir):
        st.error(f"笔记目录不存在: {raw_dir}")
        st.stop()
    c1, c2, c3 = st.columns(3)
    with c1:
        note_short_id = st.number_input("note_short_id", min_value=0, value=0, step=1)
    with c2:
        start_line = st.number_input("start_line", min_value=1, value=1, step=1)
    with c3:
        end_line = st.number_input("end_line", min_value=1, value=200, step=1)

    if st.button("模拟读取", key="simulate_note_recall"):
        row = db_mgr.get_note_index_by_short_id(int(note_short_id))
        if not row:
            st.error(f"未找到 note_short_id={int(note_short_id)}")
            st.stop()
        rel = str(row.get("source_file_path") or "").replace("\\", "/").strip().lstrip("/")
        target = os.path.abspath(os.path.join(raw_dir, rel.replace("/", os.sep)))
        raw_root = os.path.abspath(raw_dir)
        if not target.startswith(raw_root):
            st.error("source_file_path 非法（越界路径）")
        elif not os.path.exists(target):
            st.error(f"文件不存在：{rel}")
        else:
            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            lines = text.splitlines()
            total_lines = len(lines)
            if start_line > end_line:
                st.error("start_line 不能大于 end_line")
                st.stop()
            if total_lines <= 0:
                actual_start = 0
                actual_end = 0
                seg = ""
            else:
                actual_start = min(max(1, int(start_line)), total_lines)
                actual_end = min(max(1, int(end_line)), total_lines)
                if actual_start > actual_end:
                    actual_start = actual_end
                seg = "\n".join(lines[actual_start - 1: actual_end])
            st.caption(
                f"note_short_id={int(note_short_id)} | total_lines={total_lines} | "
                f"actual_start_line={actual_start} | actual_end_line={actual_end}"
            )
            st.code(seg, language="text")

elif mode == "🔄 中央库导入导出":
    st.subheader("🔄 中央记忆库 JSON 导入导出")
    if not db_mgr.has_central_db():
        st.error("未找到中央记忆库。")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**导出中央快照**")
        if st.button("生成快照", key="btn_export_central"):
            st.session_state["central_snapshot"] = db_mgr.export_central_snapshot()
            st.success("快照已生成。")
        snapshot = st.session_state.get("central_snapshot")
        if snapshot:
            t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "下载 JSON",
                data=json.dumps(snapshot, ensure_ascii=False, indent=2),
                file_name=f"memory_center_snapshot_{t}.json",
                mime="application/json",
            )

    with c2:
        st.markdown("**导入 JSON**")
        uploaded = st.file_uploader("选择 JSON 文件", type=["json"])
        if uploaded is not None and st.button("执行导入", key="btn_import_central"):
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                stats = db_mgr.import_central_payload(payload)
                st.success(
                    f"导入完成：新增 {stats.get('inserted', 0)} / 更新 {stats.get('upserted', 0)} / "
                    f"跳过 {stats.get('skipped', 0)} / 失败 {stats.get('failed', 0)}"
                )
            except Exception as e:
                st.error(f"导入失败：{e}")

elif mode == "🛠️ 维护状态":
    st.subheader("🛠️ 维护状态与备份")
    state = db_mgr.get_maintenance_state()
    if state:
        st.markdown("**maintenance_state.json**")
        st.json(state)
    else:
        st.warning("未找到 maintenance_state.json 或文件为空。")

    st.divider()
    st.markdown("**备份文件（最多保留3个）**")
    backups = db_mgr.list_backups()
    if backups:
        st.dataframe(backups, use_container_width=True, hide_index=True)
        selected = st.selectbox(
            "选择备份预览",
            [b["name"] for b in backups],
            index=0,
        )
        selected_path = next((b["path"] for b in backups if b["name"] == selected), "")
        if selected_path:
            st.json(db_mgr.load_backup_preview(selected_path))
    else:
        st.info("暂无备份文件。")

elif mode == "📂 浏览笔记文件":
    st.subheader("📂 raw 目录笔记预览（仅文件视图）")
    if not os.path.exists(raw_dir):
        st.error(f"笔记目录不存在: {raw_dir}")
        st.stop()

    md_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".md"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, raw_dir).replace("\\", "/")
                md_files.append(rel_path)
    md_files.sort()

    if not md_files:
        st.info("未发现 markdown 文件。")
        st.stop()

    selected = st.selectbox("选择文件", md_files, index=0)
    full_path = os.path.join(raw_dir, selected.replace("/", os.sep))
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.caption(selected)
        st.markdown(content)
    except Exception as e:
        st.error(f"读取失败: {e}")
