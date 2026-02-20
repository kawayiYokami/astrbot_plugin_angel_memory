# Changelog

All notable changes to this plugin will be documented in this file.

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

