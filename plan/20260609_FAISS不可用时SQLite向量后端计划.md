# FAISS 不可用时 SQLite 向量后端计划

## 背景

当前记忆向量索引使用 FAISS `IndexFlatIP + IndexIDMap`，向量持久化在 `index/faiss/*.index`，配套的 `*.sqlite` 只保存 ID 映射和 `vector_text` 等元数据。部分用户环境在 `import faiss` 时会触发原生二进制异常，导致插件不可用。

本次目标不是降级为 BM25，而是在 FAISS 不可用时继续提供向量检索能力。

## 目标

1. 启动时先检测 FAISS 是否可导入。
2. FAISS 可用时继续使用现有 FAISS 向量后端。
3. FAISS 不可用时使用独立的 SQLite Flat Cosine 向量后端。
4. SQLite 向量后端存放在当前 provider 的 `index/sqlite/` 目录，不放入 `index/faiss/`。
5. 无论选择哪个后端，都从中央 SQL 真相层同步 `id + vector_text`，自然生成或修复向量索引。
6. 上层 `CognitiveService`、`MemoryRuntime`、反思和工具调用继续使用同一套 vector store 接口，不感知底层后端。

## 模块拆分

### FAISS 后端

- 保留 `llm_memory/components/faiss_memory_index.py`。
- 只在 FAISS 探测成功后导入。
- 继续使用 `index/faiss/`。

### SQLite 向量后端

- 新增 `llm_memory/components/sqlite_vector_index.py`。
- 不导入 FAISS。
- 使用 SQLite 保存：
  - `item_id`
  - `vector_text`
  - `embedding_blob`
  - `dimension`
  - `provider_id`
  - `model_key`
  - `updated_at`
- 查询时使用 numpy 执行归一化向量点积，得到余弦相似度。
- 使用 `index/sqlite/`。

### 后端选择

- `ComponentFactory` 负责启动时探测 FAISS。
- 探测通过：创建 `FaissVectorStore`。
- 探测失败：创建 `SqliteVectorStore`。
- 探测失败时记录清晰中文日志，说明继续使用 SQLite 向量模式。

## 同步策略

启动一致性检查仍以中央 SQL 真相层为准：

1. `MemorySqlManager.list_memory_index_rows()` 导出 `id + vector_text`。
2. 当前向量后端执行 `sync_rows()`。
3. 若之前使用 FAISS、现在 FAISS 不可用，不读取旧 `*.index`，由 SQLite 后端重新 embedding 并生成自己的向量库。

## 测试

1. SQLite 向量后端支持写入、检索、删除、重载。
2. SQLite 后端在模型变化时清空旧向量空间。
3. SQLite 后端 `sync_rows()` 能检测缺失、孤儿、文本变化。
4. ComponentFactory 在 FAISS 探测失败时不导入 `faiss_memory_index`，并选择 SQLite 后端。

## 验收

- FAISS 可用环境仍可走 FAISS。
- FAISS 不可用环境插件仍能启动，并使用 SQLite 向量检索。
- 不出现 BM25-only 作为 FAISS 失败的替代路径。
- `metadata.yaml` 版本号递增。
