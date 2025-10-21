#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆数据库CLI工具 - 核心交互式调试器
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
import requests

# 添加主项目路径到 sys.path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent / "astrbot"
if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLUGIN_ROOT = SCRIPT_DIR
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    print("❌ 交互式模式需要安装 'rich' 库: pip install rich")
    sys.exit(1)


class DatabasePathResolver:
    """数据库路径自动推断器"""
    def __init__(self, script_path: Path):
        self.script_path = script_path
        self.plugin_root = script_path.parent

    def resolve_database_path(self) -> Optional[Path]:
        """自动推断 ChromaDB 数据库路径"""
        try:
            data_dir = self.plugin_root.parent.parent
            index_dir = data_dir / "plugin_data" / "astrbot_plugin_angel_memory"
            if not index_dir.exists():
                print(f"❌ 插件数据目录不存在: {index_dir}")
                return None

            config_path = data_dir / "config" / "astrbot_plugin_angel_memory_config.json"
            provider_id = ""
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        config = json.load(f)
                        provider_id = config.get("astrbot_embedding_provider_id", "")
                except Exception as e:
                    print(f"⚠️  读取配置文件失败: {e}")
            else:
                print(f"⚠️  配置文件不存在: {config_path}")

            import re
            safe_provider_id = re.sub(r'[<>:"/\\|?*]', '_', provider_id.strip()) if provider_id and provider_id.strip() else "local"
            db_dir = index_dir / f"memory_{safe_provider_id}" / "chromadb"

            if not db_dir.exists():
                print(f"⚠️  数据库目录不存在: {db_dir}")
                return None

            print(f"✅ 成功定位数据库: {db_dir}")
            if provider_id:
                print(f"   当前提供商: {provider_id}")
            return db_dir
        except Exception as e:
            print(f"❌ 路径推断失败: {e}")
            return None


class SimpleEmbeddingProvider:
    """简化的嵌入提供商 - 用于CLI工具的向量化"""
    def __init__(self, provider_id: str, api_key: str = None):
        self.provider_id = provider_id
        self.api_key = ""
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.model = "BAAI/bge-m3"

    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """同步方法：为文档列表生成向量嵌入"""
        if not texts:
            return []

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "input": texts, "encoding_format": "float"}

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            console = Console()
            console.print(f"[red]❌ 向量化失败: {e}[/red]")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {"model": self.model}


class MemoryDBDebugger:
    """核心调试器"""
    def __init__(self, db_path: Path, provider_id: str, api_key: str = None):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.provider_id = provider_id
        self.api_key = api_key
        self.embedding_provider = None
        self.bm25_retriever = None
        self.integrated_mode = False
        self._try_load_components()

    def _try_load_components(self):
        """尝试从项目中加载组件"""
        try:
            if self.provider_id:
                self.embedding_provider = SimpleEmbeddingProvider(self.provider_id, self.api_key)
            else:
                raise RuntimeError("需要指定提供商ID")

            import importlib.util
            bm25_path = PLUGIN_ROOT / "llm_memory" / "components" / "bm25_retriever.py"
            spec = importlib.util.spec_from_file_location("bm25_module", bm25_path)
            bm25_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bm25_module)
            # 使用新的无状态精排函数
            self.rerank_with_bm25 = bm25_module.rerank_with_bm25

            self.integrated_mode = True
            print("✅ 混合检索模式已启用 (使用无状态 BM25 精排)")
            model_info = self.embedding_provider.get_model_info()
            print(f"   向量化模型: {model_info.get('model')}")
        except Exception as e:
            print(f"⚠️  初始化失败: {e}")
            self.integrated_mode = False

    def _vector_search(self, collection_name: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """执行纯向量搜索的核心实现"""
        console = Console()
        try:
            query_embedding_list = self.embedding_provider.embed_documents_sync([query])
            if not query_embedding_list:
                return []
            query_embedding = query_embedding_list[0]

            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_embeddings=[query_embedding], n_results=limit)

            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])[0]

            output = []
            for i, doc_id in enumerate(ids):
                distance = distances[i]
                similarity = max(0.0, 1.0 - (distance / 2.0))
                output.append({'id': doc_id, 'similarity': similarity})
            return output
        except Exception as e:
            console.print(f"[red]❌ _vector_search 发生错误: {e}[/red]")
            return []

    def hybrid_search(self, collection_name: str, query: str, limit: int = 10, vector_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Dict[str, Any]]:
        """执行混合检索并返回详细结果"""
        console = Console()
        if not self.integrated_mode:
            return []

        with console.status("[bold green]执行混合检索中...") as status:
            status.update("🔍 (1/3) 向量搜索...")
            vector_results = self._vector_search(collection_name, query, limit=limit * 2)
            vector_scores = {res['id']: res['similarity'] for res in vector_results}

            # --- 调试功能：保存向量搜索结果到 JSON ---
            collection = self.client.get_or_create_collection(collection_name)
            try:
                top_30_ids = [res['id'] for res in vector_results[:30]]
                if top_30_ids:
                    retrieved_items = collection.get(ids=top_30_ids, include=['metadatas', 'documents'])
                    export_data = []
                    for i, doc_id in enumerate(retrieved_items['ids']):
                        export_data.append({
                            "id": doc_id,
                            "metadata": retrieved_items['metadatas'][i],
                            "document": retrieved_items['documents'][i]
                        })

                    export_path = Path(__file__).parent / "debug_vector_results.json"
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, ensure_ascii=False, indent=2)
                    console.print(f"✅ [green]已将前 30 个向量搜索结果保存到 {export_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  保存调试 JSON 失败: {e}[/yellow]")
            # --- 调试功能结束 ---

            status.update("📚 (2/3) BM25 精排...")
            collection = self.client.get_or_create_collection(collection_name)

            # --- 新的“召回-过滤-精排”混合检索策略 ---
            # 1. 向量召回
            RECALL_COUNT = 100 # 召回数量
            vector_results = self._vector_search(collection_name, query, limit=RECALL_COUNT)
            vector_scores = {res['id']: res['similarity'] for res in vector_results}

            # 2. 向量分数过滤
            VECTOR_THRESHOLD = 0.5
            filtered_by_score = [res for res in vector_results if res['similarity'] >= VECTOR_THRESHOLD]

            if not filtered_by_score:
                console.print(f"[yellow]⚠️  没有向量分数高于 {VECTOR_THRESHOLD} 的文档，跳过 BM25 精排。[/yellow]")
                bm25_results = []
            else:
                # 3. 数量限制
                MAX_CANDIDATES = 100 # 最多对100个文档进行精排
                candidates_for_rerank = filtered_by_score[:MAX_CANDIDATES]
                console.print(f"[green]✅ 向量召回 {len(vector_results)} 条，过滤后剩余 {len(filtered_by_score)} 条，取前 {len(candidates_for_rerank)} 条进行 BM25 精排。[/green]")

                # 4. 准备精排数据
                candidate_ids = [c['id'] for c in candidates_for_rerank]
                retrieved_items = collection.get(ids=candidate_ids, include=['metadatas'])
                id_to_content = {
                    item['id']: item['metadata'].get('content', '')
                    for item in zip(retrieved_items['ids'], retrieved_items['metadatas'])
                }

                rerank_input = [
                    {"id": c_id, "content": id_to_content.get(c_id, "")}
                    for c_id in candidate_ids
                ]

                # 5. 调用无状态 BM25 精排
                bm25_results = self.rerank_with_bm25(query, rerank_input, limit=limit)
            # --- 新策略结束 ---

            max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 1.0
            bm25_scores = {doc_id: score / max_bm25_score for doc_id, score in bm25_results}

            status.update("🔄 (3/3) 结果融合...")
            all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

            combined_results = []
            items_data = {}
            if all_ids:
                retrieved_items = collection.get(ids=list(all_ids))
                for i, doc_id in enumerate(retrieved_items['ids']):
                    items_data[doc_id] = {'metadata': retrieved_items['metadatas'][i], 'document': retrieved_items['documents'][i]}

            for doc_id in all_ids:
                vec_score = vector_scores.get(doc_id, 0.0)
                bm25_s = bm25_scores.get(doc_id, 0.0)
                content = (items_data.get(doc_id, {}).get('metadata', {}).get('content') or items_data.get(doc_id, {}).get('document') or "N/A")
                combined_results.append({
                    "id": doc_id,
                    "content_summary": content[:100].strip().replace('\n', ' ') + "...",
                    "vector_score": vec_score,
                    "bm25_score": bm25_s,
                    "combined_score": (vec_score * vector_weight) + (bm25_s * bm25_weight)
                })

            return sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)[:limit]


def main():
    """主入口函数 - 交互式调试模式"""
    console = Console()
    console.print(Panel("[bold cyan]欢迎使用 Angel Memory 交互式调试工具[/bold cyan]", expand=False))

    resolver = DatabasePathResolver(Path(__file__))
    db_path = resolver.resolve_database_path()
    if not db_path:
        sys.exit(1)

    provider_id = "local"
    data_dir = Path(__file__).parent.parent.parent
    config_path = data_dir / "config" / "astrbot_plugin_angel_memory_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                provider_id = json.load(f).get("astrbot_embedding_provider_id", "local")
        except Exception as e:
            console.print(f"[yellow]⚠️  读取配置文件失败: {e}[/yellow]")

    debugger = MemoryDBDebugger(db_path, provider_id=provider_id)
    if not debugger.integrated_mode:
        sys.exit(1)

    mode = Prompt.ask("请选择集合 (memory, note)", choices=["memory", "note"], default="note")
    collection_name = "personal_memory_v1" if mode == "memory" else "notes_main"

    try:
        count = debugger.client.get_or_create_collection(name=collection_name).count()
        console.print(f"✅ [bold]已锁定集合:[/bold] {collection_name} ({count}条文档)")
    except Exception as e:
        console.print(f"[red]❌ 获取集合状态失败: {e}[/red]")

    while True:
        try:
            query = Prompt.ask("\n[bold]请输入查询内容[/bold] (或输入 'exit' 退出)")
            if query.lower() == 'exit':
                break

            results = debugger.hybrid_search(collection_name, query, limit=10)
            if not results:
                console.print("[yellow]⚠️  没有找到匹配结果[/yellow]")
                continue

            table = Table(title="混合检索调试结果")
            table.add_column("Rank", style="magenta")
            table.add_column("ID", style="cyan")
            table.add_column("Content", style="white")
            table.add_column("Vec Score", style="green")
            table.add_column("BM25 Score", style="yellow")
            table.add_column("Combined", style="blue")

            for i, res in enumerate(results):
                table.add_row(
                    str(i + 1), res['id'], res['content_summary'],
                    f"{res['vector_score']:.4f}", f"{res['bm25_score']:.4f}", f"{res['combined_score']:.4f}"
                )
            console.print(table)
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]发生错误: {e}[/red]")

    console.print("[bold yellow]再见！[/bold yellow]")


if __name__ == "__main__":
    main()
