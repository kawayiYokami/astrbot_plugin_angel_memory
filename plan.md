# 🚀 架构升级计划：移除 BM25S，全面拥抱 FlashRank

## 背景与目标
原有的 BM25S 重排方案在处理中文时依赖 Jieba 分词，导致严重的性能瓶颈（延迟高、CPU占用高）和日志干扰。
本次升级旨在**彻底移除 BM25S 和 Jieba 依赖**，引入轻量级、高性能的 **FlashRank** (基于 TinyBERT 的 Cross-Encoder) 作为唯一的重排引擎。

## 执行路线图

### 1. 配置层更新
- [ ] **`_conf_schema.json`**: 新增 `enable_flashrank` 开关 (Default: False)。虽然我们计划全面拥抱，但保留开关以便于用户感知和过渡（或者直接设为默认开启，如果用户安装了依赖）。
    - *决策*: 由于 FlashRank 是新引入的重依赖，建议在 `metadata.yaml` 或文档中明确提示用户安装 `pip install flashrank`。配置项可设为 `enable_reranking`，后端自动检测 FlashRank。

### 2. 核心组件开发
- [ ] **`llm_memory/components/flashrank_retriever.py`**:
    - 实现 `FlashRankRetriever` 类。
    - 封装 `ranker.rerank(request)` 调用。
    - 包含详细的异常处理（如未安装库时的友好提示）。

### 3. 业务逻辑重构
- [ ] **`llm_memory/components/vector_store.py`**:
    - 移除 `_rerank_with_bm25` 相关逻辑。
    - 替换为 `FlashRankRetriever` 的调用。
    - 优化 `search_notes` 和 `recall` 方法的混合检索流程。

### 4. 清理遗留资产 (De-Jieba)
- [ ] **`llm_memory/components/bm25_retriever.py`**: 彻底删除。
- [ ] **`llm_memory/parser/markdown_parser.py`**:
    - 这是一个关键点。MarkdownParser 目前使用 `jieba.posseg` 从标签中提取子标签。
    - *方案*: 暂时移除该特性，或者改用简单的基于规则的分词（如按空格/标点切分），或者完全移除“自动提取子标签”功能（通常这个功能比较鸡肋）。
- [ ] **依赖管理**: 从 `requirements.txt` 中移除 `jieba`, `bm25s`。

## 风险评估
- **环境兼容性**: FlashRank 依赖 `onnxruntime`，在某些极简环境（如无 C++ 运行时的 Alpine Linux）可能需要额外配置。
- **内存占用**: 虽然 TinyBERT 很小，但相比无状态的 BM25，它需要常驻内存（约 100MB）。这对于 1GB 内存的小鸡可能是个负担，但对于大多数用户可接受。

## 下一步
切换到 **Engineer Mode** 执行代码变更。