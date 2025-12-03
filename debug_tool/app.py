import streamlit as st
import os
import math
import datetime
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
        ["🔍 混合检索", "📖 浏览记忆", "📂 浏览笔记"],
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

# --- 3. 主界面逻辑 ---

def render_item(item, type="memory", use_flashrank=False):
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
            if ts > 1e11: ts /= 1000 # 毫秒修正
            time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            header_parts.append(f"🕒 {time_str}")
        except:
            pass

    # 标签 (文本格式)
    if meta.get('tags'):
        header_parts.append(f"🔖 {meta['tags']}")

    # 标签 (ID格式，需要解析) - FlashRank 可选
    tag_ids = meta.get('tag_ids')
    if tag_ids:
        # If FlashRank is used, it might be a list, otherwise might be a string.
        if isinstance(tag_ids, list):
             tag_ids_str = str(tag_ids)
        else:
             tag_ids_str = tag_ids

        header_parts.append(f"🔖 {tag_ids_str}")

    # 标签 (ID格式，需要解析) - FlashRank 可选
    tag_names = []
    if use_flashrank and item.get('final_ranked_score'):
        # 显示原始分数和重排后分数
        original_score = item.get('original_score', 0.0)
        final_score = item.get('final_ranked_score', 0.0)
        header_parts.append(f"⚖️ 向量分: {original_score:.3f} | 重排: {final_score:.3f}")
    elif tag_ids and not use_flashrank:
        header_parts.append(f"🔖 {tag_ids_str}")

    # 标签 (文本格式)
    if meta.get('tags'):
        header_parts.append(f"🔖 {meta['tags']}")

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

    # FlashRank 控制
    with st.sidebar:
        use_flashrank = st.checkbox("启用 FlashRank 重排", value=False, help="对向量检索结果进行重排序")
        flashrank_weight = st.slider("重排权重", min_value=0.0, max_value=1.0, value=0.5, help="越高越信赖FlashRank结果，越低越信赖向量相似度")

    query = st.text_input("输入查询内容", placeholder="例如：海豹的性格、关于绝区零的笔记...")

    if query:
        col1, col2 = st.columns(2)

        # 记忆检索结果
        with col1:
            st.info("🧠 记忆库匹配")
            if mem_cols:
                results = db_mgr.query_collections(query, mem_cols, n_results=5, use_flashrank=use_flashrank, flashrank_ratio=flashrank_weight)
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
                                render_item(item, type="memory", use_flashrank=use_flashrank)
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
                results = db_mgr.query_collections(query, note_cols, n_results=5, use_flashrank=use_flashrank, flashrank_ratio=flashrank_weight)
                found = False
                for c_name, items in results.items():
                    if items:
                        found = True
                        st.caption(f"来源: {c_name}")
                        for item in items:
                            score = item['score']
                            with st.container(border=True):
                                st.markdown(f"**最终得分:** {score:.3f}")
                                render_item(item, type="note", use_flashrank=use_flashrank)
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

    # 分页逻辑
    stats = db_mgr.get_collection_stats(selected_col)
    total_count = stats['count']
    page_size = 10
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page = st.number_input(f"页码 (共 {total_pages} 页)", min_value=1, max_value=max(1, total_pages), value=1)

    offset = (page - 1) * page_size
    items = db_mgr.browse_collection(selected_col, limit=page_size, offset=offset)

    st.caption(f"显示第 {offset+1} - {min(offset+page_size, total_count)} 条，共 {total_count} 条")

    for item in items:
        with st.container(border=True):
            render_item(item, type="memory")
            with st.expander("详细信息 (Metadata)"):
                st.json(item['metadata'])

# === 模式 3: 浏览笔记 ===
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
