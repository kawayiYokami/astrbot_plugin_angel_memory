# Changelog

All notable changes to this plugin will be documented in this file.

## [1.2.0] - 2026-02-20

### Highlights
- 重启并落地 FTS5 混合检索链路，替代旧关键词子串匹配主路径。
- 记忆与笔记索引彻底分离（同库分表），统一融合算法并支持真实向量分接入。
- 检索稳定性与可观测性增强：启动预构建、索引体积日志、一致性巡检与自动修复。
- 清理历史遗留：移除废弃笔记向量接口与 FlashRank 残留文案/命名。

### Core Changes
- `feat(retrieval)`: 新增 `FTS5HybridRetriever`，采用 `jieba` 预分词 + FTS5 (`unicode61`) 检索。
- `feat(retrieval)`: 记忆检索支持 `tags > judgment` 权重（`bm25(memory_fts, 2.0, 1.0)`）。
- `feat(retrieval)`: 笔记检索切换为纯 tags 索引（`note_fts`），不索引正文。
- `feat(fusion)`: 融合公式统一为 `final = fts_weight * fts_score + vector_weight * vector_score`。
- `feat(fusion)`: 向量模式主链路接入真实 `vector_scores`（记忆/笔记）；向量缺失时保留占位分降级。
- `feat(stability)`: 增加 FTS 初始化线程安全保护（原子重建 + 失败日志 + 状态回滚）。
- `feat(observability)`: 新增 `[FTS5重建]` 索引规模日志（memory_fts/note_fts 行数与 DB 大小）。
- `feat(maintenance)`: 新增 FTS 与 SQL 一致性巡检及自动修复，并接入睡眠维护任务。
- `perf(sync)`: 批量场景改为增量同步优先，减少不必要的全量重建。
- `refactor(cleanup)`: 删除 `vector_store` 中废弃旧笔记接口（仅抛异常的历史方法）。
- `refactor(cleanup)`: 清理 `vector_store.py` 与 `README.md` 中 FlashRank 历史残留。
- `fix(safety)`: `note_service` 增加 `similarity` 安全转换，规避 `None`/非数值类型异常。
- `fix(safety)`: FTS MATCH 构造改进，token 采用字面量安全拼接；`clear_index` 增加 target 严格校验。

### Notes
- 本版本为检索架构升级版本（minor），语义与行为有明显变更，按 `1.2.0` 发布。
- 建议升级后重点关注：top-K 命中率、召回率、融合分分布与 FTS 巡检日志。

## [1.1.0] - 2026-02-20

### Highlights
- 实现三档自适应记忆衰减算法（TMD），让记忆系统具备真正的遗忘能力。
- 记忆按使用价值自动分为易逝档（T0）、待证档（T1）、核心档（T2），各档施加不同遗忘策略。
- 噪声记忆自然清退，经过验证的重要记忆永久保留，年驱散率稳定在 ~95%。
- 遗忘参数支持用户自定义覆盖，默认参数经仿真验证可直接使用。

### Core Changes
- `feat(decay)`: 新增 `MemoryDecayPolicy` 与 `MemoryDecayConfig`，实现三档分级遗忘策略。
- `feat(decay)`: 记忆模型新增 `useful_count`、`useful_score`、`last_recalled_at`、`last_decay_at` 字段，支持价值累积与衰减追踪。
- `feat(decay)`: 反馈链路新增 `recalled_memory_ids` 参数，区分"被召回且有用"与"被召回但无用"两种信号。
- `feat(decay)`: T0 自然衰减采用单条 CTE SQL 批量执行，避免逐行 IO。
- `refactor(feedback)`: 移除召回阶段的即时衰减逻辑，衰减统一收归反馈阶段与睡眠周期。
- `refactor(reinforce)`: 强化增量从 3 调整为 1，配合 `useful_score` 升档机制重新平衡。
- `fix(merge)`: 记忆合并时保留最高 `useful_score`，防止合并导致降档。
- `feat(config)`: 新增 `decay_policy_override` 配置项，支持用户覆盖遗忘参数。
- `feat(compat)`: 数据库在线迁移，自动补充新字段，兼容已有数据。

### Notes
- 已有记忆升级后 `useful_score` 初始为 0（T0 档），需通过后续召回反馈重新积累价值。
- 遗忘参数默认值经过多压力等级仿真验证（500~10,000 次/天），一般场景无需修改。
- 详见 `docs/tiered_memory_decay_paper.md` 获取完整的算法论文与仿真数据。

## [1.0.0] - 2026-02-20

### Highlights
- 重构整体数据结构，建立中央索引作为真相源。
- 向量库职责收敛为向量检索缓存，不再承载主数据语义。
- 支持无向量模式下的可用检索路径。
- 大规模笔记场景索引效率显著提升（数万条可在十几秒级完成索引）。
- 检索链路提速，整体稳定性提升。
- 存储体积明显下降。

### Core Changes
- `feat(core)`: 落地中央索引与记忆/笔记向量库同步链路。
- `refactor(memory)`: 统一记忆接口，根除关联字段与旧耦合。
- `feat(memory)`: 中央记忆迁移链路修复，补齐本地生成 ID 与中央目录管理。
- `refactor(config)`: 统一 `note_topk` 配置口径。
- `fix(notes-sync)`: 修正 notes 向量同步状态回写，并补充重建前日志。
- `perf(retrieval)`: 笔记向量召回阈值统一为 `0.5`。
- `chore(debug)`: 升级 debug tool，增强导入导出与可观测能力。

### Notes
- 本版本为重构里程碑版本，建议升级后执行一次初始化扫描与睡眠维护流程，确认索引状态一致。
