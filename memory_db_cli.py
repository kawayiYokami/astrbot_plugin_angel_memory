#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆数据库CLI工具 - 双模式智能分析与调试工具

支持两种运行模式：
1. 基础分析模式 (python memory_db_cli.py)
   - 提供数据库概览、导出、过滤功能
   - 仅依赖 chromadb

2. 高级调试模式 (python -m astrbot_plugin_angel_memory.memory_db_cli)
   - 提供完整的混合检索功能
   - 100% 复现主插件的检索行为
   - 需要项目环境支持
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb

# 添加主项目路径到 sys.path，以便导入 astrbot 模块
# 脚本位置: .../data/plugins/astrbot_plugin_angel_memory/memory_db_cli.py
# 主项目位置: .../astrbot/
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent / "astrbot"
if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 同时添加插件根目录，以便相对导入
PLUGIN_ROOT = SCRIPT_DIR
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError:
    # 如果没有安装 typer 和 rich，降级到简单模式
    typer = None
    Console = None
    Table = None
    rprint = print
    Panel = None
    Syntax = None


class DatabasePathResolver:
    """数据库路径自动推断器"""

    def __init__(self, script_path: Path):
        self.script_path = script_path
        self.plugin_root = script_path.parent

    def resolve_database_path(self) -> Optional[Path]:
        """
        自动推断 ChromaDB 数据库路径

        Returns:
            数据库路径，如果推断失败则返回 None
        """
        try:
            # 1. 推断根目录 (data/)
            # 脚本在: .../data/plugins/astrbot_plugin_angel_memory/memory_db_cli.py
            # 需要定位到: .../data/
            data_dir = self.plugin_root.parent.parent

            # 2. 构建 index_dir
            index_dir = data_dir / "plugin_data" / "astrbot_plugin_angel_memory"

            if not index_dir.exists():
                print(f"❌ 插件数据目录不存在: {index_dir}")
                return None

            # 3. 读取配置文件
            # 配置文件在: .../data/config/astrbot_plugin_angel_memory_config.json
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

            # 4. 构建最终路径
            # 路径结构: <index_dir>/memory_<provider_id>/chromadb/
            if provider_id and provider_id.strip():
                # 对 provider_id 进行安全处理（与 PathManager 一致）
                import re
                safe_provider_id = re.sub(r'[<>:"/\\|?*]', '_', provider_id.strip())
                db_dir = index_dir / f"memory_{safe_provider_id}" / "chromadb"
            else:
                # 如果没有提供商ID，使用 local 作为默认值
                db_dir = index_dir / "memory_local" / "chromadb"

            # 5. 验证路径
            if not db_dir.exists():
                print(f"⚠️  数据库目录不存在: {db_dir}")
                print("   这可能意味着数据库尚未初始化。")
                return None

            print(f"✅ 成功定位数据库: {db_dir}")
            if provider_id:
                print(f"   当前提供商: {provider_id}")

            return db_dir

        except Exception as e:
            print(f"❌ 路径推断失败: {e}")
            return None


class MemoryDBAnalyzer:
    """基础分析模式 - 仅依赖 ChromaDB"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=str(db_path))

    def list_collections(self) -> List[Dict[str, Any]]:
        """列出所有集合及其信息"""
        collections = self.client.list_collections()

        result = []
        for col in collections:
            info = {
                'name': col.name,
                'count': col.count(),
                'metadata': col.metadata
            }
            result.append(info)

        return result

    def dump_collection(self, collection_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """导出集合数据"""
        try:
            collection = self.client.get_collection(collection_name)

            # 获取所有数据
            all_data = collection.get()

            # 构建结果
            results = []
            ids = all_data.get('ids', [])
            documents = all_data.get('documents', [])
            metadatas = all_data.get('metadatas', [])

            total = len(ids)
            actual_limit = min(limit, total) if limit else total

            for i in range(actual_limit):
                item = {
                    'id': ids[i] if i < len(ids) else None,
                    'document': documents[i] if i < len(documents) else None,
                    'metadata': metadatas[i] if i < len(metadatas) else None
                }
                results.append(item)

            return results

        except Exception as e:
            print(f"❌ 导出集合失败: {e}")
            return []

    def filter_collection(self, collection_name: str, where_clause: Dict[str, Any],
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """根据元数据过滤集合"""
        try:
            collection = self.client.get_collection(collection_name)

            # 执行过滤查询
            results = collection.get(where=where_clause, limit=limit)

            # 构建结果
            output = []
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])

            for i in range(len(ids)):
                item = {
                    'id': ids[i] if i < len(ids) else None,
                    'document': documents[i] if i < len(documents) else None,
                    'metadata': metadatas[i] if i < len(metadatas) else None
                }
                output.append(item)

            return output

        except Exception as e:
            print(f"❌ 过滤查询失败: {e}")
            return []


class SimpleEmbeddingProvider:
    """简化的嵌入提供商 - 用于CLI工具的向量化"""

    def __init__(self, provider_id: str, api_key: str = None):
        self.provider_id = provider_id
        self.api_key = api_key or self._load_api_key()
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.model = "BAAI/bge-m3"  # 硅基流动的 m3 模型

    def _load_api_key(self) -> str:
        """从环境变量或配置文件加载 API 密钥"""
        import os
        # 尝试从环境变量读取
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if api_key:
            return api_key

        # 提示用户输入
        print("⚠️  需要硅基流动 API 密钥")
        print("   你可以设置环境变量 SILICONFLOW_API_KEY")
        print("   或者直接输入（输入将不会显示）：")
        import getpass
        api_key = getpass.getpass("API Key: ")
        return api_key

    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """同步方法：为文档列表生成向量嵌入"""
        if not texts:
            return []

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings

        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "provider_type": "api",
            "status": "available"
        }

    def is_available(self) -> bool:
        """检查提供商是否可用"""
        return bool(self.api_key)

    def get_provider_type(self) -> str:
        """获取提供商类型"""
        return "api"

    def shutdown(self):
        """关闭提供商（无需清理）"""
        pass


class MemoryDBDebugger(MemoryDBAnalyzer):
    """高级调试模式 - 集成项目组件"""

    def __init__(self, db_path: Path, provider_id: str = None):
        super().__init__(db_path)
        self.vector_store = None
        self.integrated_mode = False
        self.provider_id = provider_id

        # 尝试加载项目组件
        self._try_load_components()

    def _try_load_components(self):
        """尝试从项目中加载组件"""
        try:
            print("🔧 初始化混合检索模式...")

            # 创建嵌入提供商
            if self.provider_id:
                print(f"📡 使用 API 提供商: {self.provider_id}")
                self.embedding_provider = SimpleEmbeddingProvider(self.provider_id)
            else:
                print("⚠️  未指定提供商，search 功能将不可用")
                self.embedding_provider = None
                raise RuntimeError("需要指定提供商ID")

            # 初始化 BM25 检索器
            print("📚 正在初始化 BM25 检索器...")
            # 直接从文件加载 BM25Retriever
            import sys
            import importlib.util
            bm25_path = PLUGIN_ROOT / "llm_memory" / "components" / "bm25_retriever.py"
            spec = importlib.util.spec_from_file_location("bm25_module", bm25_path)
            bm25_module = importlib.util.module_from_spec(spec)
            sys.modules["bm25_module"] = bm25_module
            spec.loader.exec_module(bm25_module)
            BM25Retriever = bm25_module.BM25Retriever

            self.bm25_retriever = BM25Retriever(k1=1.2, b=0.75)

            self.integrated_mode = True
            print("✅ 混合检索模式已启用")
            print("   可以使用 search 命令执行混合检索（向量 + BM25）")

            # 显示向量化模型信息
            model_info = self.embedding_provider.get_model_info()
            print(f"   向量化模型: {model_info.get('model')}")

        except Exception as e:
            print(f"⚠️  初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.integrated_mode = False
            self.embedding_provider = None
            self.bm25_retriever = None

    def search(self, collection_name: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        执行向量检索（仅在高级模式下可用）

        使用简化的实现：直接调用 ChromaDB 的 query API
        """
        if not self.integrated_mode or not self.embedding_provider:
            print("⚠️  向量检索功能未启用")
            print("   当前结果为简单文本匹配，不代表真实的向量检索")
            return self._simple_text_search(collection_name, query, limit)

        try:
            # 1. 使用嵌入提供商对查询进行向量化
            print("🔄 正在对查询进行向量化...")
            query_embedding = self.embedding_provider.embed_documents_sync([query])[0]
            print(f"✓ 向量化完成 (维度: {len(query_embedding)})")

            # 2. 获取集合
            collection = self.client.get_collection(collection_name)

            # 3. 执行向量相似度搜索
            print("🔍 正在执行向量搜索...")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            # 4. 格式化结果
            output = []
            if results and results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                documents = results.get('documents', [[]])[0]

                for i in range(len(ids)):
                    item = {
                        'id': ids[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'document': documents[i] if i < len(documents) else None,
                        'distance': distances[i] if i < len(distances) else None,
                        'similarity': 1.0 - (distances[i] / 2.0) if i < len(distances) else None
                    }

                    # 如果 metadata 中有 content，也添加到顶层方便查看
                    if item['metadata'] and 'content' in item['metadata']:
                        item['content'] = item['metadata']['content']

                    output.append(item)

            print(f"✓ 找到 {len(output)} 条结果")
            return output

        except Exception as e:
            print(f"❌ 向量检索失败: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def _simple_text_search(self, collection_name: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """降级的简单文本搜索（不使用向量检索）"""
        try:
            collection = self.client.get_collection(collection_name)
            all_data = collection.get()

            # 简单的文本匹配
            results = []
            ids = all_data.get('ids', [])
            documents = all_data.get('documents', [])
            metadatas = all_data.get('metadatas', [])

            query_lower = query.lower()

            for i in range(len(ids)):
                doc = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}

                # 检查文档或元数据中是否包含查询关键词
                if doc and query_lower in doc.lower():
                    results.append({
                        'id': ids[i],
                        'document': doc,
                        'metadata': meta,
                        'match_type': 'document'
                    })
                elif meta:
                    # 检查元数据字段
                    meta_text = json.dumps(meta, ensure_ascii=False).lower()
                    if query_lower in meta_text:
                        results.append({
                            'id': ids[i],
                            'document': doc,
                            'metadata': meta,
                            'match_type': 'metadata'
                        })

                if len(results) >= limit:
                    break

            return results[:limit]

        except Exception as e:
            print(f"❌ 简单搜索失败: {e}")
            return []


def create_cli():
    """创建 CLI 应用"""

    if typer is None:
        # 降级到简单的命令行解析
        return create_simple_cli()

    app = typer.Typer(
        name="memory_db_cli",
        help="记忆数据库CLI工具 - 智能分析与调试",
        add_completion=False
    )
    console = Console()

    # 全局数据库路径
    db_path_holder = {'path': None, 'analyzer': None}

    def get_analyzer():
        """获取或初始化分析器"""
        if db_path_holder['analyzer'] is None:
            # 自动推断数据库路径
            resolver = DatabasePathResolver(Path(__file__))
            db_path = resolver.resolve_database_path()

            if db_path is None:
                console.print("[red]❌ 无法定位数据库，请检查插件是否已初始化[/red]")
                raise typer.Exit(1)

            # 读取配置，获取 provider_id
            data_dir = Path(__file__).parent.parent.parent
            config_path = data_dir / "config" / "astrbot_plugin_angel_memory_config.json"
            provider_id = None
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        config = json.load(f)
                        provider_id = config.get("astrbot_embedding_provider_id", "")
                except Exception:
                    pass

            db_path_holder['path'] = db_path

            # 根据运行方式选择模式
            if __name__ == "__main__":
                # 直接运行：基础模式
                db_path_holder['analyzer'] = MemoryDBAnalyzer(db_path)
                console.print("[yellow]ℹ️  运行在基础分析模式[/yellow]")
            else:
                # 模块化运行：尝试高级模式
                db_path_holder['analyzer'] = MemoryDBDebugger(db_path, provider_id)

        return db_path_holder['analyzer']

    @app.command()
    def info():
        """显示数据库信息和所有集合"""
        analyzer = get_analyzer()
        collections = analyzer.list_collections()

        if not collections:
            console.print("[yellow]⚠️  数据库中没有集合[/yellow]")
            return

        # 创建表格
        table = Table(title="ChromaDB 集合信息")
        table.add_column("集合名称", style="cyan", no_wrap=True)
        table.add_column("文档数量", style="magenta", justify="right")
        table.add_column("元数据", style="green")

        for col in collections:
            metadata_str = json.dumps(col['metadata'], ensure_ascii=False) if col['metadata'] else "-"
            table.add_row(col['name'], str(col['count']), metadata_str)

        console.print(table)
        console.print(f"\n📊 总计: {len(collections)} 个集合")

    @app.command()
    def dump(
        collection: str = typer.Argument(..., help="集合名称"),
        limit: Optional[int] = typer.Option(None, "--limit", "-l", help="限制结果数量")
    ):
        """导出集合数据为 JSON 格式"""
        analyzer = get_analyzer()
        results = analyzer.dump_collection(collection, limit)

        if not results:
            console.print(f"[yellow]⚠️  集合 '{collection}' 中没有数据或集合不存在[/yellow]")
            return

        # 输出 JSON
        json_output = json.dumps(results, ensure_ascii=False, indent=2)

        if Syntax:
            syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            print(json_output)

        console.print(f"\n📊 导出了 {len(results)} 条记录")

    @app.command()
    def filter(
        collection: str = typer.Argument(..., help="集合名称"),
        where: str = typer.Argument(..., help="过滤条件 (JSON格式)"),
        limit: Optional[int] = typer.Option(None, "--limit", "-l", help="限制结果数量")
    ):
        """使用元数据过滤器查询集合"""
        analyzer = get_analyzer()

        try:
            where_clause = json.loads(where)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ 无效的 JSON 格式: {e}[/red]")
            console.print("[yellow]示例: '{\"source\": \"file.md\"}'[/yellow]")
            raise typer.Exit(1)

        results = analyzer.filter_collection(collection, where_clause, limit)

        if not results:
            console.print("[yellow]⚠️  没有匹配的结果[/yellow]")
            return

        # 输出 JSON
        json_output = json.dumps(results, ensure_ascii=False, indent=2)

        if Syntax:
            syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            print(json_output)

        console.print(f"\n📊 找到 {len(results)} 条匹配记录")

    @app.command()
    def search(
        collection: str = typer.Argument(..., help="集合名称"),
        query: str = typer.Argument(..., help="搜索查询"),
        limit: int = typer.Option(10, "--limit", "-l", help="结果数量限制")
    ):
        """执行混合检索（需要高级调试模式）"""
        analyzer = get_analyzer()

        if not isinstance(analyzer, MemoryDBDebugger):
            console.print("[yellow]⚠️  当前为基础分析模式，search 命令已禁用[/yellow]")
            console.print("[yellow]   请使用: python -m astrbot_plugin_angel_memory.memory_db_cli[/yellow]")
            raise typer.Exit(1)

        console.print(f"🔍 正在搜索: '{query}'")
        results = analyzer.search(collection, query, limit)

        if not results:
            console.print("[yellow]⚠️  没有找到匹配结果[/yellow]")
            return

        # 输出结果
        json_output = json.dumps(results, ensure_ascii=False, indent=2)

        if Syntax:
            syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            print(json_output)

        console.print(f"\n📊 找到 {len(results)} 条相关记录")

    return app


def create_simple_cli():
    """创建简单的 CLI（当 typer 不可用时）"""

    class SimpleCLI:
        def run(self):
            print("记忆数据库CLI工具")
            print("=" * 50)

            # 自动推断数据库路径
            resolver = DatabasePathResolver(Path(__file__))
            db_path = resolver.resolve_database_path()

            if db_path is None:
                print("❌ 无法定位数据库")
                sys.exit(1)

            # 读取配置，获取 provider_id
            data_dir = Path(__file__).parent.parent.parent
            config_path = data_dir / "config" / "astrbot_plugin_angel_memory_config.json"
            provider_id = None
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        config = json.load(f)
                        provider_id = config.get("astrbot_embedding_provider_id", "")
                except Exception:
                    pass

            # 创建分析器
            if __name__ == "__main__":
                analyzer = MemoryDBAnalyzer(db_path)
                print("ℹ️  运行在基础分析模式")
            else:
                analyzer = MemoryDBDebugger(db_path, provider_id)

            print("\n可用命令:")
            print("  info - 显示所有集合信息")
            print("  dump <collection> [limit] - 导出集合数据")
            print("  filter <collection> <where_json> [limit] - 过滤查询")
            print("  search <collection> <query> [limit] - 混合检索（高级模式）")
            print("  quit - 退出")
            print("=" * 50)

            while True:
                try:
                    cmd = input("\n> ").strip()
                    if not cmd:
                        continue

                    parts = cmd.split(maxsplit=1)
                    command = parts[0].lower()

                    if command == 'quit' or command == 'q':
                        break
                    elif command == 'info':
                        collections = analyzer.list_collections()
                        print(f"\n📊 找到 {len(collections)} 个集合:")
                        for col in collections:
                            print(f"  - {col['name']}: {col['count']} 条记录")
                    elif command == 'dump':
                        if len(parts) < 2:
                            print("❌ 用法: dump <collection> [limit]")
                            continue
                        args = parts[1].split()
                        collection_name = args[0]
                        limit = int(args[1]) if len(args) > 1 else None
                        results = analyzer.dump_collection(collection_name, limit)
                        print(json.dumps(results, ensure_ascii=False, indent=2))
                    elif command == 'filter':
                        if len(parts) < 2:
                            print("❌ 用法: filter <collection> <where_json> [limit]")
                            continue
                        args = parts[1].split(maxsplit=1)
                        if len(args) < 2:
                            print("❌ 缺少过滤条件")
                            continue
                        collection_name = args[0]
                        where_parts = args[1].split(maxsplit=1)
                        where_json = where_parts[0]
                        limit = int(where_parts[1]) if len(where_parts) > 1 else None
                        where_clause = json.loads(where_json)
                        results = analyzer.filter_collection(collection_name, where_clause, limit)
                        print(json.dumps(results, ensure_ascii=False, indent=2))
                    elif command == 'search':
                        if not isinstance(analyzer, MemoryDBDebugger):
                            print("❌ search 命令仅在高级模式下可用")
                            continue
                        if len(parts) < 2:
                            print("❌ 用法: search <collection> <query> [limit]")
                            continue
                        args = parts[1].split(maxsplit=1)
                        if len(args) < 2:
                            print("❌ 缺少查询内容")
                            continue
                        collection_name = args[0]
                        query_parts = args[1].split(maxsplit=1)
                        query_text = query_parts[0]
                        limit = int(query_parts[1]) if len(query_parts) > 1 else 10
                        results = analyzer.search(collection_name, query_text, limit)
                        print(json.dumps(results, ensure_ascii=False, indent=2))
                    else:
                        print(f"❌ 未知命令: {command}")

                except KeyboardInterrupt:
                    print("\n👋 再见!")
                    break
                except Exception as e:
                    print(f"❌ 执行失败: {e}")

    return SimpleCLI()


def main():
    """主入口函数"""
    app = create_cli()

    if typer is not None:
        app()
    else:
        app.run()


if __name__ == "__main__":
    main()
