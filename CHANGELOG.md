# Changelog

All notable changes to this plugin will be documented in this file.

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

