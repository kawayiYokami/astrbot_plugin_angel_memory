# FAISS 轻量向量索引替换 Chroma 计划

## 1. 背景与问题
- 当前向量模式依赖 ChromaDB 作为记忆向量存储与轻量 `memory_index` 召回层。
- 项目真实检索能力已经主要沉淀在自身架构中：
  - SQL 中央记忆库。
  - Tantivy BM25 + jieba 预分词。
  - `HybridRetrievalEngine` 能力分档。
  - 可选上游 `rerank_provider`。
  - 时间衰减、实体优先、类型分组、scope 隔离、睡眠维护。
- 因此不需要继续引入完整向量数据库；只需要一个稳定、轻量、可控的向量索引组件。
- AstrBot 上游已将 `faiss-cpu`、`numpy` 等依赖纳入运行环境，但上游封装 API 不稳定，本项目不直接依赖其内部封装。

## 2. 目标
- 使用 `faiss-cpu` 自建轻量记忆向量索引，替换 ChromaDB 在记忆检索中的向量召回职责。
- 保留现有高级检索层：
  - Tantivy BM25。
  - `HybridRetrievalEngine`。
  - `rerank_provider`。
  - `MemoryRuntime` 统一接口。
  - DeepMind 的实体优先、类型分组、灵魂共鸣、睡眠反馈。
- 以 `simple_memory.db` 作为权威元数据源，FAISS 只保存向量索引。
- 支持 provider/model/dimension 变化时安全重建。
- 最终移除 ChromaDB 作为记忆系统运行依赖。

## 3. 非目标
- 本阶段不改写 DeepMind 认知工作流。
- 本阶段不替换 Tantivy BM25。
- 本阶段不引入 LanceDB、Qdrant、Milvus 等新向量数据库。
- 本阶段不直接使用 AstrBot 上游 `FaissVecDB`、`EmbeddingStorage`、`DocumentStorage` 封装。
- 本阶段不改变主动记忆/被动记忆、反馈、睡眠巩固语义。
- 本阶段不删除已存在的 `simple_memory.db` 数据。

## 4. 总体方案

### 4.1 新增组件
- 新增 `llm_memory/components/faiss_memory_index.py`。
- 组件职责：
  - 初始化/加载 FAISS index 文件。
  - 管理 `memory_id <-> vector_id` 映射。
  - 批量构建索引。
  - upsert 单条记忆向量。
  - 删除记忆向量。
  - 查询 topK 向量相似记忆 ID。
  - 检测 provider/model/dimension 变化并标记重建。

### 4.2 存储位置
- FAISS index 文件不能放在全局单一路径；必须沿用当前“嵌入供应商目录隔离”的设计。
- 当前路径体系：
  - `PathManager.set_provider(embedding_provider_id, base_data_dir)` 会生成供应商专用目录：`memory_<safe_provider_id>/`
  - Chroma 当前使用该目录下的 `chromadb/`，不同 embedding provider 天然物理隔离。
- FAISS 应使用当前供应商专用索引目录，例如：
  - `memory_<safe_provider_id>/index/faiss_memory.index`
  - 或 `memory_<safe_provider_id>/index/faiss/memory.index`
- 中央 `simple_memory.db` 仍保持 provider 无关：
  - `memory_center/index/simple_memory.db`
- 映射与元信息写入当前 provider 专用 FAISS sidecar SQLite：
  - `memory_<safe_provider_id>/index/faiss/memory_index.sqlite`
  - `memory_<safe_provider_id>/index/faiss/notes_index.sqlite`
- sidecar SQLite 记录 provider/model/dimension，用于判断当前供应商索引是否已同步。

### 4.3 SQLite 表结构
- 实际实现中，每个 FAISS index 使用独立 sidecar SQLite，而不是写入全局中央库。
- 映射表：

```sql
CREATE TABLE IF NOT EXISTS index_rows (
    item_id TEXT PRIMARY KEY,
    vector_id INTEGER NOT NULL UNIQUE,
    vector_text TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
```

- 元信息表：

```sql
CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

- vector id 序列表：

```sql
CREATE TABLE IF NOT EXISTS vector_id_seq (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    next_id INTEGER NOT NULL
);
```

- 关键 meta：
  - `provider_id`
  - `embedding_model`
  - `dimension`
  - `index_version`
  - `index_path`
  - `updated_at`
- 说明：
  - 不同 embedding provider/model 的向量空间不可混用。
  - 即使 dimension 相同，只要 provider/model 不同，也必须视为不同索引。
  - 当前项目已有“供应商变化时同步”的维护状态，本计划应复用该语义。

### 4.4 FAISS 索引类型
- 使用：
  - `faiss.IndexFlatIP`
  - `faiss.IndexIDMap`
- 插入与查询前都显式执行 L2 normalize。
- 相似度分数直接使用 inner product，语义为 cosine similarity，越大越相似。
- 避免 `IndexFlatL2` + `1 - distance / 2` 的隐式前提与分数换算。

## 5. 检索链路改造

### 5.1 当前链路

```text
query
-> Chroma memory_index
-> vector_scores
-> MemorySqlManager.recall_by_tags(...)
-> Tantivy BM25 + HybridRetrievalEngine
-> MemoryManager 时间衰减 / 实体优先 / 类型分组
```

### 5.2 目标链路

```text
query
-> existing embedding_provider 生成 query_vector
-> FAISS memory index
-> vector_scores
-> MemorySqlManager.recall_by_tags(...)
-> Tantivy BM25 + HybridRetrievalEngine
-> MemoryManager 时间衰减 / 实体优先 / 类型分组
```

### 5.3 保持不变
- `MemorySqlManager.recall_by_tags(...)` 接口保持 `vector_scores: Dict[str, float]`。
- `HybridRetrievalEngine` 的 BM25-only / 融合 / rerank 分档逻辑保持。
- 记忆对象仍从 SQL 回查，FAISS 不承担业务字段存储。

## 6. 写入与同步规则

### 6.1 向量文本
- 复用现有逻辑：
  - `MemorySqlManager.build_vector_text(judgment, tags)`
- 内容为：
  - `judgment + tags`

### 6.2 新增记忆
- `MemorySqlManager.remember(...)` 写入 SQL 成功后：
  - 构造 `vector_text`。
  - 通过现有 embedding provider 生成向量。
  - upsert 到 FAISS。
  - 写入/更新 `memory_vector_index` 映射。

### 6.3 删除/合并记忆
- 记忆删除时：
  - 删除 FAISS 中对应 `vector_id`。
  - 删除 `memory_vector_index` 映射。
- 记忆合并时：
  - 删除被合并旧记忆向量。
  - 新记忆写入后 upsert 新向量。

### 6.4 启动一致性检查
- 启动时检查 SQL 真相层与当前 provider 的 FAISS 向量层是否一一匹配：
  - FAISS index 文件是否存在。
  - sidecar SQLite 映射表数量是否与 SQL 可索引行一致。
  - SQL 中存在但 FAISS 不存在的缺失项。
  - FAISS 中存在但 SQL 不存在的孤儿项。
  - SQL 向量文本变化但 FAISS sidecar 仍是旧文本的变化项。
  - FAISS index 文件 `ntotal` 是否与 sidecar 映射数量一致。
  - 当前 `provider_id` / `embedding_model` / `dimension` 是否与 meta 一致。
- 不一致时执行启动同步：
  - 缺失项补写。
  - 孤儿项删除。
  - 文本变化项重写。
  - provider/model/dimension/meta 或 index 计数不可信时，按当前 SQL 真相层重建当前 provider 的 FAISS index。
- 当前项目已有 `memory_vector_sync_last_provider` 状态：
  - Chroma 路径中，供应商变化会触发 `MemoryVectorSyncService` 将 `simple_memory.db` 的 `memory_index` rows 同步到当前 provider 的向量库。
  - FAISS 路径应将该服务改造为“当前 provider 的 FAISS 索引同步/重建服务”，而不是引入全局单索引。

## 7. 初始化与组件工厂

### 7.1 ComponentFactory
- 创建 embedding provider 后创建 `FaissMemoryIndex`。
- 将其注入 `MemorySqlManager` 或 `MemoryManager`，替代 `memory_index_collection` 路径。
- `VectorMemoryRuntime` 可保留名称，但内部不再依赖 Chroma `CognitiveService` 作为主向量实现；后续可重命名为 `HybridMemoryRuntime`。

### 7.2 CognitiveService 去 Chroma 化
- 保留 `CognitiveService` 外壳以降低上层改动面。
- `VectorStore` 名称保留为兼容 shim，底层已指向 `FaissVectorStore`。
- `main_collection` 不再创建；中央 SQL 是真相源，FAISS 仅维护 `memory_index` / `notes_index` 轻量召回索引。

## 8. 配置与版本
- 本阶段不新增配置项，默认使用 FAISS。
- 由于行为与依赖变化，提交前必须更新：
  - `metadata.yaml` 的 `version`。
  - `requirements` 依赖声明：移除 `chromadb`，新增 `faiss-cpu` / `numpy`。

## 9. 日志与可观测
- 日志默认中文。
- 统一前缀：
  - `[FAISS向量索引] 开始`
  - `[FAISS向量索引] 完成`
  - `[FAISS向量索引] 跳过`
  - `[FAISS向量索引] 失败`
- 关键日志字段：
  - `任务名`
  - `触发条件`
  - `provider_id`
  - `embedding_model`
  - `dimension`
  - `记忆总数`
  - `写入条数`
  - `删除条数`
  - `失败条数`
  - `耗时毫秒`
- 高频单条同步明细放 DEBUG，INFO 只输出聚合结果。

## 10. 迁移步骤

### 10.1 阶段一：新增 FAISS 组件
- 新增 `FaissMemoryIndex`。
- 新增 SQLite 映射表迁移。
- 提供 `rebuild_from_sql(...)` 能力。
- 提供 `search_by_vector(...)` 能力。

### 10.2 阶段二：影子索引
- 启动后后台从 `simple_memory.db` 为当前 embedding provider 构建/同步 FAISS。
- 主检索仍走当前 Chroma memory_index。
- 对比 Chroma vector_scores 与 FAISS vector_scores，输出抽样日志。
- 切换 embedding provider 后：
  - 使用当前 provider 专用 FAISS index。
  - 若不存在或 meta 不一致，则从中央库重建。
  - 不覆盖其他 provider 的历史索引。

### 10.3 阶段三：切换向量召回
- 将 `MemoryManager.comprehensive_recall(...)` 的向量 ID 召回改为 FAISS。
- 保留 BM25-only 降级。
- Chroma 不再参与记忆检索主路径。

### 10.4 阶段四：清理 Chroma 依赖
- 移除记忆系统对 `VectorStore` / Chroma collection 的强依赖。
- 睡眠维护中的向量回灌改为 FAISS 重建/增量同步。
- 文件监控中 Chroma 专用维护说明清理或改写。

## 11. 回滚策略
- 本实现按用户决策移除 Chroma 运行依赖，不保留 Chroma 后端开关。
- FAISS index 可删除后从 `simple_memory.db` 重建，不影响中央库权威数据。
- 如果 FAISS 初始化失败：
  - 自动降级 BM25-only。
  - 输出清晰 INFO/WARNING 日志。

## 12. 风险与应对
- 风险1：FAISS index 文件与 SQLite 映射不一致。
  - 应对：启动校验数量、维度、meta；不一致则后台全量重建。
- 风险2：并发写入导致 index 文件损坏。
  - 应对：FAISS 写操作使用进程内 `RLock`；保存文件采用临时文件 + 原子替换策略。
- 风险3：embedding provider/model 切换后旧向量不可用。
  - 应对：沿用 provider 目录隔离；meta 检测到变化后只重建当前 provider 索引；重建完成前 BM25-only。
- 风险4：FAISS 依赖在个别环境不可用。
  - 生产应对：捕获 ImportError，自动降级 BM25-only，并提示安装 `faiss-cpu`。
  - 测试应对：不跳过、不静默降级；FAISS 是本阶段核心依赖，测试环境缺失 `faiss` 应直接失败。
- 风险5：分数分布与 Chroma 不一致影响融合阈值。
  - 应对：第一阶段影子对比；必要时调整 `memory_threshold` 或 vector score 归一策略。

## 13. 测试策略
- 使用当前机器可用的 Python 作为测试运行器，不在插件目录内创建独立虚拟环境。
- 默认测试入口：

```powershell
python -m pytest tests
```

- FAISS 相关测试不使用 `pytest.importorskip("faiss")`。
- 缺少 `faiss`、`numpy`、`pytest` 等必要测试依赖时，测试应直接失败，提醒提交者先配好环境。
- AstrBot 运行时对象使用 fake/stub 测试本插件适配契约，不要求启动完整 AstrBot。
- 核心测试覆盖：
  - provider/model/dimension 隔离。
  - index 文件与 SQLite 映射一致性。
  - upsert/search/delete/rebuild。
  - 维度不匹配保护。
  - FAISS 向量召回进入现有 `HybridRetrievalEngine` 的候选链路。
  - BM25-only、FAISS+BM25、rerank_provider 三档降级与排序行为。

## 14. 验收清单
- 启动后可从 `simple_memory.db` 构建 FAISS 记忆索引。
- 新增主动/被动记忆后，FAISS 索引增量同步成功。
- 删除、合并、睡眠遗忘后，FAISS 索引同步删除旧向量。
- 向量召回返回 `memory_id -> similarity`，并能进入现有 `recall_by_tags(...)`。
- BM25-only、FAISS+BM25 融合、rerank_provider 三种能力分档均可运行。
- provider/model/dimension 变化时能自动重建或明确降级。
- 生产试运行可回退到 Chroma 或 BM25-only。

## 15. 后续决策
- 若 FAISS 召回质量与 Chroma 接近，且依赖/启动/维护更稳定：
  - 保持当前 FAISS 主路径。
- 若 FAISS 召回质量不足：
  - 调整向量文本、阈值或重排候选策略；不回退 LanceDB。
- 若 FAISS 足够稳定：
  - LanceDB hybrid 计划降级为独立实验，不作为主线迁移前置条件。

## 16. 归档规则
- 本计划完成后不立即归档。
- 必须等待用户在实际生产数据上测试并明确反馈“通过”后，才能根据最终实现情况修订计划并归档到 `plan/done/`。
