# AstrBot Angel Memory Plugin - 项目上手简报

## 核心设计模式

### LLM工具注入机制
项目通过4个FunctionTool为LLM提供记忆能力：
- **core_memory_remember**（[`CoreMemoryRememberTool`](tools/core_memory_remember.py:18)）：主动记忆工具，永久保存重要信息
- **core_memory_recall**（[`CoreMemoryRecallTool`](tools/core_memory_recall.py:20)）：主动回忆工具，加权随机抽取核心记忆
- **note_recall**（[`ExpandNoteContextTool`](tools/expand_note_context.py:19)）：笔记展开工具，查看完整笔记内容
- **research_topic**（[`ResearchTool`](tools/research_tool.py:18)）：研究助手工具，启动独立研究Agent

这些工具在插件初始化时通过[`context.add_llm_tools()`](main.py:139)注入，成为LLM的主动能力。

## 核心设计模式

### 三层认知架构
项目采用「灵魂-潜意识-主意识」三层认知架构：
- **灵魂层**（[`SoulState`](core/soul/soul_state.py:13)）：AI的精神状态管理器，通过4维能量槽控制行为参数
- **潜意识层**（[`DeepMind`](core/deepmind.py:33)）：后台自动处理记忆的检索、整理和巩固
- **主意识层**（LLM）：通过工具接口主动记忆和回忆

这种设计模拟人类认知过程，实现了「观察→回忆→反馈→睡眠→灵魂共鸣」的完整认知工作流。

### 依赖注入与组件工厂
使用[`ComponentFactory`](core/component_factory.py:26)统一管理核心组件创建，确保：
- 主线程创建所有组件实例，避免跨线程问题
- 通过[`PluginContext`](core/plugin_context.py:23)统一资源管理
- 组件间依赖关系清晰，便于测试和维护

### 统一记忆运行时
通过[`MemoryRuntime`](core/memory_runtime/protocol.py)协议统一记忆接口：
- **VectorMemoryRuntime**：向量实现，内部委托`CognitiveService`
- **SimpleMemoryRuntime**：SQL+tags实现，基于`MemorySqlManager`
- `ComponentFactory`根据`enable_simple_memory`开关在初始化时选择实现
- 上层（DeepMind/tools）仅依赖统一接口，不感知底层实现

### 异步初始化架构
采用「极速启动+后台预初始化」模式：
- [`InitializationManager`](core/initialization_manager.py:29)管理状态转换
- [`BackgroundInitializer`](core/background_initializer.py:20)异步执行耗时操作
- 系统毫秒级启动，不阻塞主线程

## 数据流与控制流

### LLM工具调用流程
```
LLM决策调用工具 → FunctionTool.run()
→ 从event.plugin_context获取memory_runtime组件
→ 调用统一记忆接口（MemoryRuntime）
→ 返回结果给LLM
```

### 记忆存储流程
```
用户输入 → DeepMind.organize_and_inject_memories()
→ memory_runtime.remember() → 底层实现（向量/SQL）
```

### 记忆检索流程
```
查询输入 → 预处理(实体提取) → memory_runtime.chained_recall()
→ 向量模式（VectorMemoryRuntime）：
  向量相似度检索 → FlashRank语义重排 → 实体优先召回 → 类型分组 → 结果融合
→ Simple模式（SimpleMemoryRuntime）：
  实体提取结果会并入查询词（query + entities）后执行标签子串匹配
  当前不执行 FlashRank 语义重排；保留实体优先与类型分组阶段；最终返回合并结果
  排序依据为命中次数/日期桶/强度（SQL侧排序）
```

### 记忆反馈流程
```
LLM响应 → DeepMind.async_analyze_and_update_memory()
→ 小模型分析 → memory_runtime.feedback() → 记忆强化/新增/合并 → 睡眠巩固
```

### 灵魂状态流程
```
记忆检索 → SoulState.resonate() (旧记忆冲击当前状态)
→ 小模型分析 → SoulState.update_energy() (根据16态代码更新)
→ 动态参数注入 → LLM请求参数调整
→ 新记忆生成 → 注入灵魂快照
```

## 开发场景导航

### 新增LLM工具
1. 创建新的`FunctionTool`子类，继承自[`astrbot.api.FunctionTool`](tools/core_memory_remember.py:18)
2. 定义`name`、`description`、`parameters`（JSON Schema格式）
3. 实现`async def run()`方法，从`event.plugin_context`获取所需服务
4. 在[`main.py`](main.py:139)的`context.add_llm_tools()`中注册

### 新增记忆类型
1. 在[`MemoryType`](llm_memory/models/data_models.py:24)枚举中添加新类型
2. 在[`MemoryHandlerFactory`](llm_memory/service/memory_handlers.py:102)中注册处理器
3. 更新[`process_feedback()`](llm_memory/service/memory_manager.py:398)方法处理新类型

### 修改数据模型
- 记忆模型：[`BaseMemory`](llm_memory/models/data_models.py:61)
- 笔记模型：[`NoteData`](llm_memory/models/note_models.py)
- 配置模型：[`MemorySystemConfig`](llm_memory/config/system_config.py:16)

### 灵魂状态配置
灵魂系统4维能量槽配置：
- **RecallDepth**：回忆量倾向，控制RAG检索数量
- **ImpressionDepth**：记住量倾向，控制记忆生成数量
- **ExpressionDesire**：发言长度倾向，控制LLM输出长度
- **Creativity**：思维发散倾向，控制LLM温度参数

每个维度支持min/mid/max三级配置，通过橡皮筋算法(Tanh)映射到具体参数。

### 配置环境变量
关键配置项：
- `MEMORY_EMBEDDING_MODEL`：嵌入模型选择
- `MEMORY_COLLECTION_NAME`：记忆集合名称
- `MEMORY_STRENGTH_THRESHOLD`：记忆强度阈值
- `soul_*_min/mid/max`：灵魂参数范围配置

### 调试记忆系统
1. 检查插件状态：[`plugin.get_plugin_status()`](main.py:352)
2. 查看组件初始化：[`ComponentFactory.get_components()`](core/component_factory.py:276)
3. 监控记忆流程：DeepMind日志输出
4. 灵魂状态调试：[`soul.get_state_description()`](core/soul/soul_state.py:323)

## 核心技术栈

### 工具层
- **FunctionTool框架**：AstrBot官方工具协议，支持JSON Schema参数定义
- **PluginContext注入**：通过`event.plugin_context`访问所有核心服务
- **参数验证**：工具层执行严格的参数验证（长度、格式、范围）
- **错误处理**：友好的错误提示，指导LLM正确使用工具

### 存储层
- **ChromaDB**：向量数据库，存储记忆和笔记（向量模式）
- **SQLite**：标签管理、文件索引、SimpleMemory存储（`MemorySqlManager`）

### 检索层
- **向量模式**：语义检索（向量相似度+FlashRank重排）+ 链式召回（实体优先+类型分组）
- **Simple模式**：标签子串匹配 → 命中次数/日期/强度排序

### 嵌入层
- **本地模型**：SentenceTransformers（BAAI/bge-small-zh-v1.5）
- **API提供商**：支持多种第三方嵌入服务
- **缓存机制**：内存缓存提升重复查询性能

### 灵魂系统
- **状态管理**：4维能量槽实时状态维护
- **橡皮筋算法**：Tanh函数实现非线性映射
- **共鸣机制**：旧记忆状态快照冲击当前状态
- **16态编码**：二进制状态码表示AI精神状态

## 关键决策点

### 1. 工具即能力：双层记忆架构
项目采用「被动记忆+主动记忆」双层架构：
- **被动记忆**：DeepMind自动管理，LLM无感知，通过事件钩子自动注入
- **主动记忆**：通过工具暴露，LLM主动调用，显式的记忆操作

工具设计原则：
- **最小化参数**：judgment/reasoning/tags三元组，50字符限制
- **加权随机**：recall工具按strength加权抽样，避免确定性偏见
- **上下文传递**：通过`event.plugin_context`统一获取服务实例
- **错误友好**：详细的错误提示，帮助LLM自我纠正

### 2. 统一三元组记忆结构
所有记忆类型采用`(judgment, reasoning, tags)`三元组：
- **judgment**：核心论断，用于向量化
- **reasoning**：解释背景，提供上下文
- **tags**：分类标签，支持实体识别

这种设计简化了记忆模型，同时保持了表达力。

### 3. 主动记忆与被动记忆
- **主动记忆**(`is_active=True`)：永不衰减，用户重要偏好
- **被动记忆**：会随时间和使用衰减，一般性知识

这种区分模拟人类记忆，确保重要信息长期保存。

### 4. 灵魂状态动态参数调整
通过4维能量槽实现AI行为参数动态调整：
- **能量累积**：历史刺激累积影响当前行为
- **自然衰减**：能量值随时间自动回归中庸
- **橡皮筋映射**：非线性映射保证参数稳定
- **状态共鸣**：旧记忆状态影响当前决策

### 5. 异步反馈队列
使用[`FeedbackQueue`](core/utils/feedback_queue.py)处理记忆反馈：
- 不阻塞主对话流程
- 批量处理提升性能
- 失败重试保证可靠性

### 6. 向量↔Simple双向同步
- **备份（向量→SQL）**：每次睡眠巩固后，自动将向量库记忆备份到`simple_memory.db`
- **回灌（SQL→向量）**：在睡眠维护管线中按模式/供应商变更条件触发，将SQL库中缺失的记忆回灌到向量库并重新向量化
- 实现供应商无关的记忆持久化，切换嵌入模型不丢数据

### 7. 文件监控与笔记系统
- 自动监控文档变化，实时更新索引
- 支持Markdown、文本等多种格式
- 与记忆系统协同，提供结构化知识库

## 性能优化策略

### 1. 向量缓存
- 查询场景：3秒超时，避免阻塞
- 存储场景：自动重试，处理速率限制

### 2. 集合缓存
- ChromaDB集合实例缓存，避免重复初始化
- 基于提供商和模型的精确缓存键

### 3. 批量操作
- 标签ID批量获取
- 文件路径批量映射
- 记忆批量更新

### 4. 灵魂系统优化
- **线程安全**：使用RLock保证并发访问安全
- **状态快照**：轻量级状态复制，避免深度拷贝
- **参数预计算**：缓存映射结果，减少重复计算
- **软限制机制**：能量值范围软限制，避免数值溢出

### 5. 懒加载
- 组件按需创建
- 模型延迟加载
- 服务懒初始化

## 扩展点设计

### 1. 新增嵌入提供商
实现[`EmbeddingProvider`](llm_memory/components/embedding_provider.py:282)接口，注册到工厂。

### 2. 自定义记忆处理器
继承[`MemoryHandler`](llm_memory/service/memory_handlers.py:20)，实现特定业务逻辑。

### 3. 扩展检索策略
修改[`chained_recall()`](llm_memory/service/memory_manager.py:183)方法，添加新的检索阶段。

### 4. 灵魂系统扩展
- **新增能量维度**：扩展SoulState.energy字典添加新维度
- **自定义映射函数**：替换Tanh函数实现特定映射曲线
- **状态编码扩展**：扩展16态编码为更多状态表示
- **共鸣算法优化**：实现更复杂的状态融合算法

## 故障排查指南

### 常见问题
1. **初始化失败**：检查提供商配置和网络连接
2. **向量化错误**：确认嵌入模型可用性
3. **检索为空**：检查相似度阈值和查询预处理
4. **性能下降**：清理嵌入缓存，优化批量操作
5. **灵魂状态异常**：检查能量值范围和映射函数
6. **参数注入失败**：验证LLM请求对象的extension字段

### 日志级别
- **INFO**：关键流程和状态变化
- **DEBUG**：详细执行过程
- **WARNING**：非致命错误和降级处理
- **ERROR**：异常和失败情况

### 监控指标
- 初始化时间
- 检索延迟
- 记忆数量
- 缓存命中率
- 灵魂能量值变化
- 状态转换频率

---

*此简报面向资深开发者，聚焦核心架构和关键决策。完整API参考请查看各模块文档。*

## 开发准则

### 版本号维护
- 任何影响运行行为、配置、依赖或接口语义的变更，提交前必须同步更新 `metadata.yaml` 的 `version`。
- 版本号至少递增最末位，确保上游可识别到新版本并触发更新。

### 提交信息规范
- 采用约定式提交（Conventional Commits），推荐格式：`type(scope): 简要中文描述`。
- 提交信息默认使用中文，便于与现有项目历史保持一致。
- 常用类型：`feat`、`fix`、`perf`、`refactor`、`docs`、`chore`。

### 日志标准
- 日志文案默认使用中文，避免中英混杂；仅在必要时保留英文标识（如集合名、配置键名、异常类名）。
- 任务型日志统一前缀：`[睡眠维护]`、`[睡眠]`、`[简单记忆回灌]`，便于平台日志检索。
- 状态表达统一使用：`开始`、`完成`、`跳过`、`失败`，不要再使用 `status=success/failed/skipped` 风格。
- 日志内容应包含可排障字段：任务名、触发条件（如供应商变化/模式）、关键计数（如写入条数/失败条数）、耗时毫秒。
- 异常日志必须带异常信息；需要堆栈时使用 `exc_info=True`，避免只打印“失败”无上下文。
- 高频循环日志仅输出聚合信息，避免每条记录都打印 INFO；明细放 DEBUG。
- 用户可见行为变化（例如切换模式、禁用功能、跳过原因）必须有一条清晰 INFO 日志。
