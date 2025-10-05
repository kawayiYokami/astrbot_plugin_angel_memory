# 向量数据库调试工具

这是一个独立的向量数据库调试工具，用于调试生产环境的ChromaDB向量数据库。

## 功能特性

- 浏览数据库中的所有集合
- 查询集合中的数据
- 查看集合统计信息
- 检查集合维度是否匹配
- 轻量级WebUI界面

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 默认启动（端口8080）
python main.py

# 指定端口启动
python main.py --port 8081

# 指定数据库路径和模型
python main.py --port 8081 --db-path "your/db/path" --model-name "your/model/name"
```

## 默认配置

- 数据库路径：`E:\github\ai-qq\astrbot\data\plugin_data\astrbot_plugin_angel_memory\index\chromadb`
- 嵌入模型：`BAAI/bge-small-zh-v1.5`
- 默认端口：`8080`

## 界面功能

1. **集合列表**：显示所有集合及其记录数
2. **浏览功能**：查看集合中的具体记录
3. **查询功能**：对集合进行语义查询
4. **统计信息**：显示所有集合的统计信息和维度检查结果