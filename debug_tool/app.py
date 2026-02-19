import streamlit as st
import os
import math
import datetime
import json
from utils.config_loader import ConfigLoader
from utils.db import DBManager

st.set_page_config(
    page_title="天使记忆可视化",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: monospace;
    }
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 初始化 ---
@st.cache_resource
def get_managers():
    loader = ConfigLoader()
    provider = loader.get_embedding_provider()

    if not provider:
        st.error("未在 cmd_config.json 中找到启用的 'openai_embedding' 提供商！")
        return None, None, None

    provider_id = provider.get("id")
    db_path = loader.get_data_dir(provider_id)
    raw_dir = loader.get_raw_notes_dir()

    try:
        db_mgr = DBManager(db_path, provider)
    except Exception as e:
        st.error(f"无法连接到位于 {db_path} 的 ChromaDB: {e}")
        return None, None, None

    return loader, db_mgr, raw_dir

loader, db_mgr, raw_dir = get_managers()
if not db_mgr:
    st.stop()

collections = db_mgr.get_collections()
mem_cols = [c for c in collections if "memory" in c.lower()]
note_cols = [c for c in collections if "note" in c.lower()]

# --- 2. 侧边栏 ---
with st.sidebar:
    st.title("🧠 天使记忆")

    # 模式选择
    mode = st.radio(
        "选择模式",
        ["🔍 混合检索", "📖 浏览记忆", "🧾 浏览Simple记忆", "🔄 导入导出", "📂 浏览笔记"],
        index=0
    )

    st.divider()

    # 状态信息
    with st.expander("📊 数据库状态", expanded=False):
        st.caption(f"Provider: {db_mgr.provider_config.get('id')}")
        st.caption("集合统计:")
        for c in collections:
            count = db_mgr.get_collection_stats(c)["count"]
            st.write(f"- {c}: {count}")
        simple_stats = db_mgr.get_simple_memory_stats() if db_mgr.has_simple_memory_db() else {"count": 0}
        st.write(f"- simple_memory.db: {simple_stats.get('count', 0)}")

# --- 3. 主界面逻辑 ---

def build_scope_where_filter(scope_mode: str, scope_name: str):
    """构建 memory_scope 过滤条件。"""
    mode = (scope_mode or "").strip()
    scope = (scope_name or "").strip()

    if mode == "不筛选":
        return None
    if mode == "仅 public":
        return {"memory_scope": "public"}
    if not scope:
        return {"memory_scope": "public"}
    if mode == "scope + public":
        return {"$or": [{"memory_scope": scope}, {"memory_scope": "public"}]}
    if mode == "仅 scope":
        return {"memory_scope": scope}
    return None

def render_item(item, type="memory"):
    """智能渲染单个条目"""
    meta = item.get('metadata', {}) or {}
    doc = item.get('document')

    # 1. 尝试从 metadata 获取更丰富的内容
    content = doc

    # 针对笔记集合：内容通常在 'content' 字段
    if type == "note":
        if meta.get('content'):
            content = meta.get('content')

    # 针对记忆集合：优先显示 judgment，如果 doc 为空
    elif type == "memory":
        if not content and meta.get('judgment'):
            content = meta.get('judgment')

    # 处理 None 内容
    if content is None or content == "None":
        content = "*[无内容]*"

    # 2. 渲染头部信息 (Tags, Type, Time)
    header_parts = []

    # 记忆类型
    if meta.get('memory_type'):
        header_parts.append(f"🏷️ **{meta['memory_type']}**")

    # 时间戳 (转为可读格式)
    if meta.get('created_at'):
        try:
            ts = float(meta['created_at'])
            # 可能是秒或毫秒，通常是秒
            if ts > 1e11:
                ts /= 1000  # 毫秒修正
            time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            header_parts.append(f"🕒 {time_str}")
        except Exception:
            pass

    # 标签 (文本格式)
    if meta.get('tags'):
        header_parts.append(f"🔖 {meta['tags']}")

    # 标签 (ID格式)
    tag_ids = meta.get('tag_ids')
    if tag_ids:
        if isinstance(tag_ids, list):
             tag_ids_str = str(tag_ids)
        else:
             tag_ids_str = tag_ids

        header_parts.append(f"🔖 {tag_ids_str}")

    if header_parts:
        st.markdown(" | ".join(header_parts))

    # 3. 渲染主体内容
    st.markdown(content)

    # 4. 底部补充信息
    footer_parts = []
    if meta.get('relative_path'):
        footer_parts.append(f"📄 {meta['relative_path']}")
    elif meta.get('source'):
        footer_parts.append(f"📄 {meta['source']}")

    if footer_parts:
        st.caption(" | ".join(footer_parts))

# === 模式 1: 混合检索 ===
if mode == "🔍 混合检索":
    st.subheader("🔍 语义与关键词检索")

    with st.sidebar:
        st.divider()
        st.caption("记忆 scope 过滤（仅作用于记忆集合）")
        scope_mode = st.selectbox(
            "scope 过滤模式",
            ["不筛选", "仅 public", "scope + public", "仅 scope"],
            index=0
        )
        scope_name = st.text_input("scope 名称", value="", placeholder="例如：家人")

    memory_scope_filter = build_scope_where_filter(scope_mode, scope_name)

    query = st.text_input("输入查询内容", placeholder="例如：海豹的性格、关于绝区零的笔记...")

    if query:
        col1, col2 = st.columns(2)

        # 记忆检索结果
        with col1:
            st.info("🧠 记忆库匹配")
            if mem_cols:
                results = db_mgr.query_collections(
                    query,
                    mem_cols,
                    n_results=5,
                    where_filter=memory_scope_filter,
                )
                found = False
                for c_name, items in results.items():
                    if items:
                        found = True
                        st.caption(f"来源: {c_name}")
                        for item in items:
                            score = item['score']
                            color = "green" if score > 0.7 else "orange"
                            with st.container(border=True):
                                st.markdown(f"**最终得分:** :{color}[{score:.3f}]")
                                render_item(item, type="memory")
                                with st.expander("元数据"):
                                    st.json(item['metadata'])
                if not found:
                    st.caption("未找到相关记忆")
            else:
                st.warning("无记忆集合")

        # 笔记检索结果
        with col2:
            st.success("📝 笔记库匹配")
            if note_cols:
                results = db_mgr.query_collections(query, note_cols, n_results=5)
                found = False
                for c_name, items in results.items():
                    if items:
                        found = True
                        st.caption(f"来源: {c_name}")
                        for item in items:
                            score = item['score']
                            with st.container(border=True):
                                st.markdown(f"**最终得分:** {score:.3f}")
                                render_item(item, type="note")
                                with st.expander("元数据"):
                                    st.json(item.get('metadata'))
                if not found:
                    st.caption("未找到相关笔记")
            else:
                st.warning("无笔记集合")

    else:
        st.info("请输入关键词开始检索。支持自然语言。")

# === 模式 2: 浏览记忆 ===
elif mode == "📖 浏览记忆":
    st.subheader("📖 全量记忆浏览")

    if not mem_cols:
        st.warning("未找到记忆集合 (personal_memory_v1 等)")
        st.stop()

    selected_col = st.selectbox("选择集合", mem_cols)
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        browse_scope_mode = st.selectbox(
            "scope 过滤模式",
            ["不筛选", "仅 public", "scope + public", "仅 scope"],
            index=0,
            key="browse_scope_mode"
        )
    with col_f2:
        browse_scope_name = st.text_input(
            "scope 名称",
            value="",
            placeholder="例如：家人",
            key="browse_scope_name"
        )
    browse_scope_filter = build_scope_where_filter(browse_scope_mode, browse_scope_name)

    # 分页逻辑
    stats = db_mgr.get_collection_stats(selected_col)
    total_count = stats['count']
    page_size = 10
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page = st.number_input(f"页码 (共 {total_pages} 页)", min_value=1, max_value=max(1, total_pages), value=1)

    offset = (page - 1) * page_size
    items = db_mgr.browse_collection(
        selected_col,
        limit=page_size,
        offset=offset,
        where_filter=browse_scope_filter,
    )

    st.caption(f"显示第 {offset+1} - {min(offset+page_size, total_count)} 条，共 {total_count} 条")

    for item in items:
        with st.container(border=True):
            render_item(item, type="memory")
            with st.expander("详细信息 (Metadata)"):
                st.json(item['metadata'])

# === 模式 3: 浏览笔记 ===
elif mode == "🧾 浏览Simple记忆":
    st.subheader("🧾 Simple 记忆浏览（simple_memory.db）")

    if not db_mgr.has_simple_memory_db():
        st.warning("未找到 simple_memory.db，请先运行插件并完成至少一次备份。")
        st.stop()

    simple_stats = db_mgr.get_simple_memory_stats()
    scopes = simple_stats.get("scopes", [])

    c1, c2 = st.columns([2, 3])
    with c1:
        selected_scope = st.selectbox("scope 过滤", ["(全部)"] + scopes, index=0)
    with c2:
        keyword = st.text_input("关键词（匹配 judgment/reasoning/tags）", value="")

    page_size = 20
    scope_filter = "" if selected_scope == "(全部)" else selected_scope
    current_page = int(st.session_state.get("simple_page", 1) or 1)
    offset = max(0, (current_page - 1) * page_size)
    items, filtered_total = db_mgr.browse_simple_memories(
        limit=page_size,
        offset=offset,
        scope=scope_filter,
        keyword=keyword,
        return_total=True,
    )
    total_pages = math.ceil(filtered_total / page_size) if filtered_total > 0 else 1
    if current_page > total_pages:
        current_page = total_pages
        offset = max(0, (current_page - 1) * page_size)
        items, filtered_total = db_mgr.browse_simple_memories(
            limit=page_size,
            offset=offset,
            scope=scope_filter,
            keyword=keyword,
            return_total=True,
        )

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page = st.number_input(
            f"页码 (共 {total_pages} 页)",
            min_value=1,
            max_value=max(1, total_pages),
            value=current_page,
            key="simple_page",
        )
    if page != current_page:
        offset = max(0, (page - 1) * page_size)
        items, filtered_total = db_mgr.browse_simple_memories(
            limit=page_size,
            offset=offset,
            scope=scope_filter,
            keyword=keyword,
            return_total=True,
        )

    st.caption(f"总记录 {filtered_total}，当前页返回 {len(items)} 条")
    for item in items:
        with st.container(border=True):
            render_item(item, type="memory")
            with st.expander("详细信息 (Metadata)"):
                st.json(item.get("metadata", {}))

# === 模式 4: 导入导出 ===
elif mode == "🔄 导入导出":
    st.subheader("🔄 Simple 记忆 JSON 导入导出")

    if not db_mgr.has_simple_memory_db():
        st.warning("未找到 simple_memory.db，请先运行插件并完成至少一次备份。")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**导出 JSON**")
        export_scope = st.text_input(
            "导出 scope（可选）",
            value="",
            placeholder="留空导出全部",
            key="export_scope",
        )
        if st.button("生成导出文件", key="btn_export_json"):
            payload = db_mgr.export_simple_memories_payload(scope=export_scope)
            st.session_state["simple_export_payload"] = payload
            st.success(f"已生成导出数据，共 {payload.get('count', 0)} 条。")

        payload = st.session_state.get("simple_export_payload")
        if payload:
            exported_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"simple_memory_export_{exported_at}.json"
            st.download_button(
                label="下载 JSON",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=default_name,
                mime="application/json",
                key="download_export_json",
            )

    with c2:
        st.markdown("**导入 JSON**")
        uploaded = st.file_uploader(
            "选择 JSON 文件",
            type=["json"],
            key="import_json_file",
        )
        if uploaded is not None and st.button("执行导入", key="btn_import_json"):
            try:
                content = uploaded.read().decode("utf-8")
                payload = json.loads(content)
                stats = db_mgr.import_simple_memories_payload(payload)
                st.success(
                    "导入完成："
                    f"新增 {stats.get('inserted', 0)}，"
                    f"更新 {stats.get('upserted', 0)}，"
                    f"跳过 {stats.get('skipped', 0)}，"
                    f"失败 {stats.get('failed', 0)}"
                )
            except Exception as e:
                st.error(f"导入失败：{e}")

# === 模式 5: 浏览笔记 ===
elif mode == "📂 浏览笔记":
    st.subheader("📂 笔记文件浏览")

    if not os.path.exists(raw_dir):
        st.error(f"笔记目录不存在: {raw_dir}")
        st.stop()

    # 获取所有文件夹列表
    folders = set()
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".md"):
                rel_dir = os.path.relpath(root, raw_dir)
                if rel_dir == ".":
                    rel_dir = "(根目录)"
                folders.add(rel_dir)

    sorted_folders = sorted(list(folders))

    # 顶部选择文件夹
    selected_folder = st.selectbox("📂 选择文件夹", sorted_folders)

    # 获取该文件夹下的文件
    target_dir = raw_dir if selected_folder == "(根目录)" else os.path.join(raw_dir, selected_folder)
    files = [f for f in os.listdir(target_dir) if f.endswith(".md")]

    if not files:
        st.info("该文件夹下没有 Markdown 笔记")
    else:
        # 左右布局：左侧文件列表，右侧预览
        col_list, col_view = st.columns([1, 2])

        with col_list:
            st.caption(f"文件列表 ({len(files)})")
            # 使用 radio 来选择文件，模拟列表
            selected_file_name = st.radio("选择文件", files, label_visibility="collapsed")

        with col_view:
            if selected_file_name:
                file_path = os.path.join(target_dir, selected_file_name)
                with st.container(border=True):
                    st.markdown(f"**📄 {selected_file_name}**")
                    st.divider()
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.markdown(content)
                    except Exception as e:
                        st.error(f"读取失败: {e}")
