# AstrBot Angel Memory Plugin - 项目上手简报

AstrBot 插件。给机器人长期记忆、本地知识库和灵魂状态。当前版本见 `metadata.yaml`。

记忆、笔记与其索引由 `p-memory` 承载。插件侧只做两件事：**提供商回调**（嵌入 / 重排）与**后处理**（结果转换、灵魂共鸣、画像、注入、退休审查、Web 展示）。

## 构建与开发命令

```bash
pip install -r requirements.txt
ruff format .
```

测试必须在 AstrBot 容器内执行，宿主机不代表运行环境：

```bash
docker exec astrbot bash -lc 'cd /AstrBot/data/plugins/astrbot_plugin_angel_memory && PYTHONPATH=/AstrBot/data/plugins:/AstrBot/data/plugins/astrbot_plugin_angel_memory python -m pytest -q'
```

必须用 `python -m pytest`，不要直接跑 `pytest`。可执行脚本不会把当前目录加入 `sys.path`。另外 `tests/` 内混用扁平导入与包导入两种写法，所以要同时把插件目录与 `plugins` 目录加进 `PYTHONPATH`。

WebUI 源码在 `webui/`，改页面后需重新构建，产物落到 `pages/memory-dashboard/`。

## 架构概览

三层认知：

- **灵魂层**（[`SoulState`](core/soul/soul_state.py)）：四对互斥情绪轴（喜/悲、怒/恐、意料中/意外、厌恶/认可），注入时换算成社交、认知、表达、情绪四种倾向。
- **潜意识层**（[`DeepMind`](core/deepmind.py)）：观察 → 回忆 → 注入 → 反馈 → 睡眠。
- **主意识层**（LLM）：通过 4 个工具主动铭记、回忆、读笔记、写笔记。

主线程创建全部组件。[`ComponentFactory`](core/component_factory.py) 组装，[`PluginContext`](core/plugin_context.py) 持有配置、路径与组件引用。[`InitializationManager`](core/initialization_manager.py) + [`BackgroundInitializer`](core/background_initializer.py) 负责极速启动与后台就绪。

### 存储边界

| 职责 | 落点 |
|------|------|
| 记忆本体、记忆检索、笔记切片与检索 | `p-memory`，封装在 [`pmemory/`](pmemory/) |
| 记忆协议 | [`MemoryRuntime`](core/memory_runtime/protocol.py)，唯一实现 [`PMemoryMemoryRuntime`](core/memory_runtime/pmemory_runtime.py) |
| 笔记导入与扫描 | [`NoteIndexService`](core/services/note_index_service.py) |
| 用户/群账本、标签、旧库迁移源 | [`MemorySqlManager`](llm_memory/components/memory_sql_manager.py)，不做检索 |
| 旧记忆一次性迁入 | [`LegacyMemoryMigration`](core/services/legacy_memory_migration.py) |

`pmemory` 对外四样：`pmemory` 单例、`MemoryStore`、`NotesStore`、`config`（命名与域映射）。

命名与可见范围：

- 记忆 namespace 固定 `angel/memory`。可见范围走 `memory_scope`（人格键优先，会话 ID 次之，默认 `public`）。读过滤为自己的 scope 加 `public`。
- 知识 namespace `angel/knowledge/{domain}`。域：`gi` / `hsr` / `zzz` / `personal` / `angel`，对应 `raw/` 下「原神」「崩坏星穹铁道」「绝区零」「海豹的个人信息」「`.angel/note`」。
- 群对每个域三态：`auto` 自动注入、`manual` 只能工具搜、`hidden` 完全不可见。私聊与未配群默认全部 `manual`。
- 记忆开向量，笔记关向量；两边都可重排。未注册回调时降级为纯全文。

## LLM 工具

四个 FunctionTool，启动时 [`context.add_llm_tools()`](main.py) 注入：

| 工具名 | 类 | 作用 |
|--------|----|------|
| `angel_remember` | [`CoreMemoryRememberTool`](tools/angel_remember.py) | create / update / merge |
| `angel_recall` | [`CoreMemoryRecallTool`](tools/angel_recall.py) | 同时搜记忆与当前群可见知识库 |
| `angel_note_read` | [`NoteRecallTool`](tools/angel_note_read.py) | 按整数 `note_id` 分页读正文 |
| `angel_note_create` | [`NoteCreateTool`](tools/angel_note_create.py) | 写 Markdown 到 `.angel/note/` |

`note_assistant.enable_create` 控制是否注册写笔记工具，配置保存时可热增删。提供商类改动需重载插件。

`run()` 一律从 `event.plugin_context` 取组件。笔记直读必须走与主动搜索同一套域白名单：`hidden` 域当「未找到」。

## 数据流

```
用户消息
  → on_llm_request
  → DeepMind.organize_and_inject_memories
      → RetrievalService：记忆 chained_recall + 笔记检索（仅 auto 域自动注入）
      → SoulState.resonate
      → InjectionService：记忆 / 笔记 / 灵魂仪表盘写入请求
  → LLM
  → on_llm_response 把回复写入 event.angelmemory_context
  → after_message_sent 提交异步反馈
      → 小模型分析 → memory_runtime.feedback
      → 强化 / 新增 / 合并 / 灵魂反思 / 短期仓更新
```

睡眠独立于对话：每天本地时间 04:00 一次，当时进程没跑则当天不补。

```
衰减（只降强度，不删）
→ 列出强度归零候选
→ LLM 退休审查
→ keep 的 pin 成永久、delete 的才删、无结论或异常则整批跳过
→ 每日 JSON 备份（最多 3 份）
```

天使之心不是硬依赖。存在 `event.angelheart_context` 时用秘书决策与聊天记录；没有则走原生当前轮。

## 关键目录

| 路径 | 职责 |
|------|------|
| `main.py` | 插件入口、工具注册、LLM 钩子、Web 路由 |
| `core/` | DeepMind、工厂、上下文、灵魂、睡眠调度 |
| `core/services/` | 检索、注入、睡眠、笔记索引、画像、迁移 |
| `core/memory_runtime/` | 记忆协议与 p-memory 实现 |
| `pmemory/` | p-memory 封装 |
| `tools/` | 四个 LLM 工具 |
| `web_api/` | Plugin Pages 后端 |
| `webui/` | Naive UI 源码 |
| `pages/memory-dashboard/` | 构建产物 |
| `llm_memory/` | 数据模型、账本、提示词、衰减策略、嵌入提供商 |
| `_conf_schema.json` | 插件配置 schema |
| `tests/` | pytest |
| `docs/` | 设计文档与上游能力参考 |
| `plan/` | 计划与归档 |

配置键以 `_conf_schema.json` 为准。群知识三态写在 `group_knowledge_map`；旧 `group_domain_map` 只作兼容，命中的域视为 `auto`。

Web 页面在 `webui/src/router/index.ts`：总览、记忆浏览、用户画像、群聊知识库、Tags 调试、笔记索引、笔记读取、导入导出、维护状态、工具测试、功能调试、插件设置。没有向量检索页。

工具测试页：自动回忆与主动回忆两条独立接口，结果都进同一个输出框，框内只显示接口原文。

## 开发约定

### 新增 LLM 工具

1. 在 `tools/` 建 `FunctionTool` 子类，定义 `name` / `description` / `parameters`。
2. `run()` 从 `event.plugin_context` 取服务。
3. 在 `main.py` 注册。

### 记忆结构

统一三元组：`judgment` + `reasoning` + `tags`。类型中文枚举存库，工具层用英文键（`knowledge` / `skill` / `emotional` / `event` / `task`）。主动记忆 `pinned` 不参与衰减。

上层只依赖 `MemoryRuntime`，不要绕过它直接打 `p-memory`，除非你在改封装层本身。

`update` 原地改写；`merge` 新建一条后删除全部源记忆，任一源不存在则整次作废，避免假成功。合并删除有意不看 scope，允许跨会话收拢。

### 知识库

真相源是 `raw/` 下的文件。库只存切片与索引。改目录划分必须同时改 `pmemory/config.py` 的 `DOMAIN_DIRS` 与群知识面板的域名。

### 衰减

插件把配置翻译成库接受的 `policy` 后随衰减调用下发，避免两边各算一套：

- `useful_score_boost` 取 `consolidate_speed`
- `strength_boost` 固定 1
- `tier0_cycle_days` 由 `cycle_tier0_days / (forget_speed * tier0_forget_speed)` 折算取整，最小 1 天
- `tier1_threshold` 强制不低于 `tier0_threshold`

### 文档口径

- README 只写功能。禁止出现 namespace、scope、词表、RRF、BM25、FTS5、Chroma、FAISS、Tantivy、jieba、p-memory。
- 配置项可以出现在 README 配置表，说明写功能。
- 文档不要内联 markdown 图片，展示素材写路径。
- PR 与文档只写最终实现与最终结果，不写过程尝试、版本对比、参考了什么。
- 文档文件名与首行标题一致，用中文说清讲什么。

## 测试与验证

容器需在跑（`docker ps` 可见 `astrbot`）。与本次改动相关的最小测试即可，但涉及检索、可见性、睡眠、Web 契约时不要跳过对应文件：

- `tests/test_pmemory_wiring.py`
- `tests/test_group_knowledge_modes.py`
- `tests/test_sleep_retirement_flow.py` / `tests/test_sleep_scheduler.py`
- `tests/test_web_convergence.py`
- `tests/test_tool_test_api.py`

## 提交与发布

- 影响运行行为、配置、依赖或接口语义的变更，提交前同步递增 `metadata.yaml` 的 `version`。
- 约定式提交，中文：`type(scope): 简要中文描述`。常用 `feat` / `fix` / `perf` / `refactor` / `docs` / `chore`。

## 计划与归档

- 新功能先写计划，放 `plan/`，用户明确确认后才实现。
- 常规小改、文档更新、单点修复不写计划。
- 功能完成后，等用户在生产环境明确说「通过」，再把计划归档到 `plan/done/`。归档以用户结论为准。
- 归档时按最终实现修正计划书，文件名带日期。

## 日志标准

文案默认中文。仅在必要时保留英文标识（集合名、配置键、异常类名）。

任务前缀：`[睡眠]`、`[睡眠维护]`、`[淘汰审查]`、`[笔记导入]`、`[记忆迁移]`、`[反思调度]`、`[过程追踪]`、`[PMEMORY]`、`[工具测试]`、`[WebUI]`。

状态用「开始 / 完成 / 跳过 / 失败」，不要 `status=success` 这种写法。

级别：

- **INFO**：启动结果、模式切换、低频任务的一行对账。
- **DEBUG**：每轮对话都会发生的状态。
- **WARNING / ERROR**：失败与降级必须带原因；需要堆栈时 `exc_info=True`。
- 不记录无参考价值的过程日志。高频循环只打聚合，明细放 DEBUG。

用户可见行为变化（切模式、禁用、跳过原因）必须有一条清晰 INFO。

## 灵魂参数

四对情绪轴：`joy`（喜/悲）、`anger`（怒/恐）、`surprise`（意料中/意外）、`disgust`（厌恶/认可），取值 0.0 ~ 1.0，0.5 为中性。注入上下文时直接使用这份读数，面板标签决定两端含义；橡皮筋保持惯性。状态只在内存，重载插件后回到中性。检索条数与记忆生成条数是工程参数，走固定配置，不受情绪驱动。

## 故障排查

1. 初始化失败：提供商配置、`p-memory` 是否装上、数据目录权限。
2. 检索为空：查询是否过短、群知识是否 `hidden`、笔记文件是否在五个域目录内。
3. 嵌入/重排超时：看 `[PMEMORY]` 与 `[WebUI-探针]` 降级日志；功能调试页可单次探测。
4. 灵魂异常：能量软限制 [-20, 20]，映射结果应落在配置区间。
5. 面板与对话不一致：工具测试页对账自动回忆与主动回忆的原文。
