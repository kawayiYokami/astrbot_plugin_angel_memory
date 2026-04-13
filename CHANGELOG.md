# Changelog

All notable changes to this plugin will be documented in this file.

## [1.3.10] - 2026-04-13

### Highlights
- 移除 `markitdown[all]` 依赖，修复 onnxruntime 版本冲突导致的用户安装失败问题。
- 插件现仅支持 Markdown 和 TXT 格式文件解析，其他格式（PDF/DOCX/PPTX 等）不再支持。

### Core Changes
- `chore(deps)`: 从 `requirements.txt` 和 `debug_tool/requirements.txt` 中移除 `markitdown[all]>=0.1.0`。
- `refactor(parser)`: `universal_parser.py` 注释掉所有 markitdown 导入、初始化与文件转换代码，`SUPPORTED_EXTENSIONS` 仅保留 `.md`、`.txt`。
- `refactor(parser)`: `parser_manager.py` 更新相关注释。

### Notes
- 如需启用文档转换功能，请自行安装 `markitdown[all]`，并参考 `universal_parser.py` 中注释的代码手动恢复。

## [1.3.9] - 2026-03-22

### Highlights
- 修复 `system_context` 注入策略：仅在拿不到天使之心 `secretary_decision` 时，才使用 `_no_save` 注入，避免历史污染与行为回归。

### Bug Fixes
- `fix(injection)`: `DeepMind` 新增 `has_secretary_decision` 标志并传递到注入服务。
- `fix(injection)`: `DeepMindInjectionService` 按标志分流注入路径。
- `fix(injection)`: 有决策时走 `extra_user_content_parts(TextPart)`；无决策时走 `request.contexts` 且 `_no_save=True`。

## [1.3.8] - 2026-03-21

### Highlights
- 优化新记忆 `tags` 规则，要求按知识图谱节点拆分为简短实体词、关系词、属性词，并明确用户昵称与 `uid` 需分别入 tag。

### Core Changes
- `refactor(prompt)`: `memory_system_guide_async.md` 收紧 `tags` 生成规则，明确使用简短图谱节点，并以 `["小明", "123456", "喜好", "咖啡"]` 作为正确示例。

## [1.3.7] - 2026-03-21

### Highlights
- 优化反思整理提示词中的用户身份表达，聊天记录与论断生成统一强调“昵称（ID）”格式，降低同名用户混淆。

### Core Changes
- `refactor(prompt)`: 反思上下文中的用户展示格式由 `昵称/uid` 调整为 `昵称（uid）`，与记忆论断规范保持一致。
- `refactor(prompt)`: `memory_system_guide_async.md` 新增用户身份书写规则，要求 `judgment` 涉及具体用户时准确同时使用昵称和 ID，例如 `小明（123456）`。

## [1.3.4] - 2026-03-20

### Highlights
- 修复调试工具读取 embedding provider 与运行时配置不一致的问题，减少配置排查时的误导。
- 调试侧栏新增 provider 回退告警，便于直接识别“配置值”和“实际使用值”是否偏离。

### Bug Fixes
- `fix(debug-tool)`: `debug_tool` 现在优先读取插件配置 `retrieval.embedding_provider_id`，再回退到首个可用 embedding provider。
- `feat(observability)`: 新增 provider 状态信息与回退警告展示，帮助定位 provider 未启用或配置缺失问题。

## [1.3.3] - 2026-03-17

### Highlights
- 优化人格解析逻辑，完全委托给 `persona_manager.resolve_selected_persona`，与 AstrBot 主链路保持一致。
- 简化人格解析路径，移除旧的 `conversation_manager` 回读逻辑，直接从事件读取 `conversation.persona_id`。
- 增强 `provider_settings` 类型检查，缺失或非法时记录警告日志并使用空配置兜底。

### Core Changes
- `refactor(persona)`: `get_event_persona_name` 方法完全重构，统一使用 `resolve_selected_persona` 解析人格。
- `feat(config)`: `get_config` 方法新增 `umo` 参数支持，可获取会话级别的运行时配置。
- `fix(safety)`: 修复 `get_config` 中 `getter` 变量可能未定义的问题。

### Thanks
- 感谢 @lubuoren 提交的 PR #27，优化了 persona 解析逻辑。

## [1.3.2] - 2026-03-16

### Highlights
- 修复人格名称来源，改为基于当前消息事件解析生效人格，不再依赖 `angel_heart` 的 `secretary_decision` 字段。
- 记忆域解析与反思链路同步使用事件人格，降低多人格会话下误落入 `public` 的概率。

### Bug Fixes
- `fix(scope)`: 人格获取逻辑从外部决策字段迁移到事件侧解析，修复 `conversation_scope_map` 人格键命中不稳定问题。

## [1.2.9] - 2026-03-09

### Highlights
- 修复记忆合并后向量索引丢失的问题，合并记忆现在可被向量检索正确命中。
- 公开 `build_vector_text` 方法，消除跨层调用私有方法的代码异味。

### Bug Fixes
- `fix(vector)`: 合并记忆后新增向量 upsert，修复合并记忆无法被向量检索命中的问题。
- `fix(style)`: `memory_vector_sync_service.py` 末尾补充缺失的换行符。

### Refactor
- `refactor(api)`: `_build_vector_text` → `build_vector_text`，公开为正式 API，更新所有调用点。

## [1.2.6] - 2026-02-22

### Highlights
- 反思链路从 `AstrMessageEvent` 解耦为纯数据输入，便于测试与审查。
- 反思调度日志细化，覆盖入缓冲、触发判定、tick 扫描、执行完成与失败回滚。

### Core Changes
- `refactor(reflection)`: 新增 `ReflectionInput` 数据载体，反思执行入口改为 DTO 参数，不再依赖事件对象重建。
- `refactor(reflection)`: 会话反思状态缓存改为持有 `latest_input`，移除反思路径中的事件/响应序列化依赖。
- `feat(logging)`: 新增 `[反思调度]` / `[反思执行]` 关键日志，输出触发原因、会话、累计轮次、记录数、执行结果与回滚状态。

## [1.2.5] - 2026-02-22

### Highlights
- 检索策略进一步收敛为“能力分档”：重排能力与嵌入能力解耦，支持 BM25-only + 重排。
- 记忆反思提示词精简为强规则版本，并移除 task 记忆生成要求。
- 新增验收白名单，阻止 task 类型新记忆进入入库链路。

### Core Changes
- `refactor(retrieval)`: `HybridRetrievalEngine` 调整为先 BM25 候选，再按向量/重排能力组合策略。
- `fix(retrieval)`: 重排失败自动降级到无重排策略（有向量降级融合，无向量降级 BM25）。
- `feat(test)`: 增加混合检索模拟测试覆盖重排失败降级路径。
- `refactor(prompt)`: `memory_system_guide_async.md` 改为精简强规则版，并删除 task 记忆要求。
- `fix(validation)`: `MemoryIDResolver.normalize_new_memories_format` 增加类型白名单，仅允许 `knowledge/skill/emotional/event`。
- `refactor(config)`: 调整向量化检索配置文案与顺序，强调“向量非必须、重排推荐”。

## [1.2.4] - 2026-02-22

### Highlights
- 移除“简化记忆模式”配置开关，检索模式改为按能力自动分档。
- 新增“向量化检索”配置分组，集中管理嵌入与重排相关项。

### Core Changes
- `refactor(config)`: 删除 `enable_simple_memory` 配置入口与对应代码分支。
- `feat(config)`: 新增 `retrieval` 分组（`embedding_provider_id` / `enable_local_embedding` / `rerank_provider_id`）。
- `feat(runtime)`: 无向量能力时自动进入 BM25-only；有向量/重排能力时自动升级到融合或重排策略。
- `refactor(maintenance)`: 睡眠维护中与 simple 模式相关的判定改为能力判定（BM25-only / vector）。

## [1.2.3] - 2026-02-22

### Highlights
- 检索层重构为 Tantivy BM25，实现中文 `jieba` 预分词 + BM25 检索主链路。
- 新增三档检索策略：BM25 直出、向量融合、向量+BM25 候选混合重排。
- SimpleMemoryRuntime 路径接入 rerank_provider，不再与 Vector 路径能力割裂。

### Core Changes
- `refactor(retrieval)`: `llm_memory/components/bm25_retriever.py` 改为 Tantivy 检索实现，并保留模块引用路径。
- `feat(retrieval)`: BM25 分数按最大值归一到 `[0,1]`，融合公式固定为 `0.7*vector + 0.3*bm25`。
- `feat(retrieval)`: 在 `memory_sql_manager` 中实现三档策略调度与重排候选合并去重。
- `feat(simple-runtime)`: `ComponentFactory` 创建 `MemorySqlManager` 时注入 `rerank_provider`。
- `refactor(maintenance)`: 移除睡眠维护中的 FTS5 一致性巡检任务。
- `chore(deps)`: 新增依赖 `tantivy>=0.22.0`。

## [1.2.2] - 2026-02-20

### Highlights
- 统一笔记展开工具的代码命名，避免“文件/类名与工具名不一致”造成维护误导。

### Core Changes
- `refactor(tooling)`: `tools/expand_note_context.py` 重命名为 `tools/note_recall.py`。
- `refactor(tooling)`: 工具类 `ExpandNoteContextTool` 重命名为 `NoteRecallTool`。
- `refactor(tooling)`: `main.py` 同步更新导入与注册日志文案，保持对外工具名 `note_recall` 与代码命名一致。

## [1.2.1] - 2026-02-20

### Highlights
- 修复插件卸载阶段异步任务清理不完整问题，避免出现 pending task 销毁与协程未等待风险。

### Core Changes
- `fix(shutdown)`: `PluginManager.shutdown()` 与 `BackgroundInitializer.shutdown()` 调整为异步关闭链路，统一执行任务取消与等待收束。
- `fix(shutdown)`: `AngelMemoryPlugin` 追踪 `after_message_sent` 创建的后台任务，并在 `terminate()` 中先取消并等待，再关闭组件。
- `fix(stability)`: 增加关闭流程幂等保护，避免重复关闭导致状态混乱。

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
