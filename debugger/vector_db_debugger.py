#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向量数据库调试工具 - 独立的WebUI界面
用于调试生产环境的向量数据库，支持查询、浏览和分析功能
"""

import os
import sys
import json
import argparse
import traceback

import chromadb
from sentence_transformers import SentenceTransformer
from pywebio import start_server, config
from pywebio.output import clear, put_markdown, put_text, put_buttons, put_table, put_error, put_button, put_row, put_collapse, put_code, put_warning, put_success
from pywebio.input import input, textarea, input_group

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 默认配置
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_DB_PATH = r"E:\github\ai-qq\astrbot\data\plugin_data\astrbot_plugin_angel_memory\index\chromadb"
DEFAULT_COLLECTIONS = ["personal_memory_v1", "notes_main", "notes_sub"]

class VectorDBDebugger:
    def __init__(self, db_path: str = None, model_name: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.model = None
        self.client = None
        self.current_collection = None
        self.collections = []

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化调试器"""
        try:
            # 加载嵌入模型
            self.model = SentenceTransformer(self.model_name)

            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(path=self.db_path)

            # 获取所有集合
            self.collections = [c.name for c in self.client.list_collections()]

        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def list_collections(self):
        """列出所有集合"""
        try:
            clear()
            put_markdown("# 向量数据库调试工具")

            collections = self.client.list_collections()
            put_markdown("## 集合列表")

            if not collections:
                put_text("没有找到任何集合")
                return

            # 创建表格数据
            table_data = [["名称", "记录数", "操作"]]
            for collection in collections:
                count = collection.count()
                table_data.append([
                    collection.name,
                    str(count),
                    put_buttons([
                        {'label': '浏览', 'value': f"browse_{collection.name}", 'color': 'primary'},
                        {'label': '查询', 'value': f"query_{collection.name}", 'color': 'success'}
                    ], onclick=self._handle_collection_action)
                ])

            put_table(table_data)
        except Exception as e:
            put_error(f"列出集合失败: {str(e)}")
            put_text(traceback.format_exc())

    def _handle_collection_action(self, action):
        """处理集合操作"""
        action_type, collection_name = action.split('_', 1)

        if action_type == "browse":
            self.browse_collection(collection_name)
        elif action_type == "query":
            self.query_collection(collection_name)

    def browse_collection(self, collection_name: str, limit: int = 20, offset: int = 0):
        """浏览集合内容"""
        try:
            clear()
            put_markdown("# 向量数据库调试工具")

            collection = self.client.get_collection(collection_name)
            self.current_collection = collection

            # 获取记录
            results = collection.get(limit=limit, offset=offset)

            put_markdown(f"## 浏览集合: {collection_name}")
            put_text(f"记录总数: {collection.count()}")

            if not results['ids']:
                put_text("该集合为空")
                put_button("返回", onclick=lambda: self.list_collections(), color='primary')
                return

            # 显示记录
            for i, (doc_id, document, metadata) in enumerate(zip(
                results['ids'],
                results['documents'] if results['documents'] else [''] * len(results['ids']),
                results['metadatas'] if results['metadatas'] else [{}] * len(results['ids'])
            )):
                with put_collapse(f"记录 {offset+i+1}: {doc_id[:20]}..."):
                    put_markdown(f"**ID:** {doc_id}")
                    put_markdown(f"**内容:** {document[:200]}{'...' if len(document) > 200 else ''}")
                    put_markdown("**元数据:**")
                    put_code(json.dumps(metadata, ensure_ascii=False, indent=2), 'json')

            # 分页控件
            if collection.count() > limit:
                put_markdown("### 分页")
                page_count = (collection.count() + limit - 1) // limit
                current_page = offset // limit + 1

                buttons = []
                for i in range(1, page_count + 1):
                    if i == current_page:
                        buttons.append(put_text(f"[{i}]", inline=True))
                    else:
                        start_offset = (i - 1) * limit
                        buttons.append(put_button(str(i), onclick=lambda name=collection_name, lim=limit, off=start_offset: self.browse_collection(name, lim, off), small=True))

                put_row(buttons)

            put_button("返回", onclick=lambda: self.list_collections(), color='primary')

        except Exception as e:
            put_error(f"浏览集合失败: {str(e)}")
            put_text(traceback.format_exc())

    def query_collection(self, collection_name: str):
        """查询集合"""
        try:
            clear()
            put_markdown("# 向量数据库调试工具")
            put_markdown(f"## 查询集合: {collection_name}")

            collection = self.client.get_collection(collection_name)
            self.current_collection = collection

            # 使用input_group获取查询参数
            query_params = input_group("查询参数", [
                textarea("请输入查询内容:", name='query_text', rows=3, placeholder="输入要查询的内容..."),
                input("返回结果数量:", name='limit', type='number', value=10)
            ])

            query_text = query_params['query_text']
            limit = int(query_params['limit'])

            if not query_text:
                put_warning("请输入查询内容")
                return

            # 生成查询向量
            query_embedding = self.model.encode(query_text).tolist()

            # 执行查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            if not results['ids'] or not results['ids'][0]:
                put_text("没有找到相关结果")
                put_button("返回", onclick=lambda: self.list_collections(), color='primary')
                return

            put_markdown(f"### 查询结果 (共{len(results['ids'][0])}条)")

            # 显示结果
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0] if results['documents'] else [''] * len(results['ids'][0]),
                results['metadatas'][0] if results['metadatas'] else [{}] * len(results['ids'][0]),
                results['distances'][0] if results['distances'] else [0] * len(results['ids'][0])
            ), 1):
                similarity = 1 - distance
                with put_collapse(f"{i}. [{doc_id[:20]}...] (相似度: {similarity:.3f})"):
                    put_markdown(f"**ID:** {doc_id}")
                    put_markdown(f"**内容:** {document[:300]}{'...' if len(document) > 300 else ''}")
                    put_markdown(f"**相似度:** {similarity:.3f}")
                    put_markdown("**元数据:**")
                    put_code(json.dumps(metadata, ensure_ascii=False, indent=2), 'json')

            put_button("返回", onclick=lambda: self.list_collections(), color='primary')

        except Exception as e:
            put_error(f"查询集合失败: {str(e)}")
            put_text(traceback.format_exc())
            put_button("返回", onclick=lambda: self.list_collections(), color='primary')

    def collection_stats(self):
        """显示集合统计信息"""
        try:
            clear()
            put_markdown("# 向量数据库调试工具")
            put_markdown("## 集合统计信息")

            collections = self.client.list_collections()
            if not collections:
                put_text("没有找到任何集合")
                return

            total_records = 0
            table_data = [["名称", "记录数", "维度检查"]]

            for collection in collections:
                count = collection.count()
                total_records += count

                # 检查维度
                dimension_check = self._check_collection_dimension(collection)

                table_data.append([
                    collection.name,
                    str(count),
                    dimension_check
                ])

            put_table(table_data)
            put_markdown(f"**总计记录数:** {total_records}")

            put_button("返回", onclick=lambda: self.list_collections(), color='primary')

        except Exception as e:
            put_error(f"获取统计信息失败: {str(e)}")
            put_text(traceback.format_exc())
            put_button("返回", onclick=lambda: self.list_collections(), color='primary')

    def _check_collection_dimension(self, collection) -> str:
        """检查集合维度"""
        try:
            model_dimension = self.model.get_sentence_embedding_dimension()
            dummy_vector = [0.0] * model_dimension
            collection.query(
                query_embeddings=[dummy_vector],
                n_results=1
            )
            return f"✓ 匹配 ({model_dimension})"
        except Exception as e:
            if "dimension" in str(e).lower():
                return "✗ 不匹配"
            else:
                return "? 未知"

# Web界面函数
def main_page():
    """主页面"""
    put_markdown("# 向量数据库调试工具")
    put_markdown("用于调试生产环境的向量数据库，支持查询、浏览和分析功能")

    # 初始化调试器
    try:
        put_text("正在初始化...")
        debugger = VectorDBDebugger()
        clear()
        put_markdown("# 向量数据库调试工具")
        put_success("初始化成功!")

        # 导航菜单
        put_markdown("## 导航")
        put_buttons([
            {'label': '集合列表', 'value': 'collections', 'color': 'primary'},
            {'label': '统计信息', 'value': 'stats', 'color': 'success'},
            {'label': '刷新', 'value': 'refresh', 'color': 'warning'}
        ], onclick=lambda action: handle_navigation(action, debugger))

        # 显示集合列表
        debugger.list_collections()
    except Exception as e:
        put_error(f"初始化失败: {str(e)}")
        put_text(traceback.format_exc())

def handle_navigation(action, debugger):
    """处理导航"""
    if action == "collections":
        debugger.list_collections()
    elif action == "stats":
        debugger.collection_stats()
    elif action == "refresh":
        main_page()

def start_web_ui(port: int = 8080, db_path: str = None, model_name: str = None):
    """启动WebUI"""
    # 更新默认配置
    global DEFAULT_DB_PATH, DEFAULT_MODEL_NAME
    if db_path:
        DEFAULT_DB_PATH = db_path
    if model_name:
        DEFAULT_MODEL_NAME = model_name

    # 配置页面
    config(title="向量数据库调试工具", theme="dark")

    # 启动服务器
    start_server(main_page, port=port, debug=True)

def main():
    parser = argparse.ArgumentParser(description='向量数据库调试工具')
    parser.add_argument('--port', type=int, default=8080, help='Web服务器端口 (默认: 8080)')
    parser.add_argument('--db-path', type=str, help='数据库路径')
    parser.add_argument('--model-name', type=str, help='嵌入模型名称')

    args = parser.parse_args()

    # 启动WebUI
    start_web_ui(args.port, args.db_path, args.model_name)

if __name__ == "__main__":
    main()