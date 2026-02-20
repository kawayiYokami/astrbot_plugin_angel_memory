# Angel Memory - AI记忆与认知系统

> **为AstrBot赋予真正的记忆能力**：让AI不仅能记住，还能主动思考、自主进化

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-green.svg)](https://github.com/Soulter/AstrBot)

## 🌟 核心特性

### 🧠 三层认知架构
模拟人类认知结构，让AI拥有真正的"思维层次"：

- **灵魂层（Soul）**：4维情感能量槽，动态调整AI行为参数
- **潜意识层（DeepMind）**：后台自动处理记忆检索、整理和巩固
- **主意识层（LLM）**：通过4个专用工具主动管理核心记忆

### 🛠️ 4个LLM工具
赋予AI**主动的记忆能力**，不再被动等待：

1. **core_memory_remember** - 永久铭记重要信息（主动记忆，永不遗忘）
2. **core_memory_recall** - 加权随机回忆核心知识（避免确定性偏见）
3. **note_recall** - 展开笔记完整上下文（深度阅读能力）
4. **research_topic** - 启动独立研究Agent（自主学习能力）

### ⚡ 智能记忆系统
- **双轨记忆架构**：主动记忆（永久保存）+ 被动记忆（自然衰减）
- **双模式运行**：向量模式（语义检索）或 Simple模式（SQL+tags，无需向量依赖）
- **链式召回机制**：混合检索（向量+FlashRank重排）+ 实体优先 + 类型分组
- **睡眠巩固**：定期清理弱记忆，强化重要内容
- **中央索引维护（已实现）**：睡眠维护管线自动执行“向量→中央迁移、记忆向量库同步、笔记向量库同步、每日JSON备份”

### 🎨 灵魂状态系统
通过4维能量槽实现**情感化AI行为**：

- **RecallDepth**（回忆深度）：控制RAG检索数量
- **ImpressionDepth**（印象深度）：控制记忆生成数量
- **ExpressionDesire**（表达欲望）：控制LLM输出长度
- **Creativity**（创造力）：控制LLM温度参数

能量值通过**橡皮筋算法（Tanh）**映射到具体参数，旧记忆状态通过**共鸣机制**影响当前决策。

### 📚 知识库系统
- **多格式支持**：PDF、Word、PPT、Excel、Markdown、纯文本
- **FlashRank重排**：超快速高性能重排算法，精准提升检索质量
- **主动搜索优先**：通过research_tool主动学习，而非被动RAG注入
- **短条目优化**：推荐100字以内的结构化条目，避免长文档污染上下文

## 🚀 快速开始

### 前置要求

**必须安装前置插件：[astrbot_plugin_angel_heart](https://github.com/kawayiYokami/astrbot_plugin_angel_heart)**

> Angel Heart 提供增强的聊天记录管理能力，是记忆系统的必要依赖。

### 安装步骤

1. **安装Angel Heart插件**
   ```bash
   # 在AstrBot的plugins目录中安装
   cd plugins
   git clone https://github.com/kawayiYokami/astrbot_plugin_angel_heart.git
   ```

2. **安装Angel Memory插件**
   ```bash
   git clone https://github.com/kawayiYokami/astrbot_plugin_angel_memory.git
   ```

3. **安装依赖**
   ```bash
   cd astrbot_plugin_angel_memory
   pip install -r requirements.txt
   ```

4. **重启AstrBot**
   ```bash
   # 插件将自动初始化
   ```

### 基础配置

通过AstrBot配置界面设置以下参数：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `provider_id` | "" | 记忆整理LLM提供商ID |
| `astrbot_embedding_provider_id` | "local" | 嵌入模型提供商ID |
| `enable_local_embedding` | false | 是否启用本地嵌入模型 |
| `enable_simple_memory` | false | 启用简化记忆模式（SQL+tags，无需向量） |
| `min_message_length` | 5 | 触发记忆处理的最小消息长度 |
| `sleep_interval` | 3600 | 记忆巩固间隔（秒） |
| `short_term_memory_capacity` | 1.0 | 短期记忆容量倍数（0.1-10.0） |
| `soul_*_mid` | - | 灵魂系统各维度的默认值 |

## 🧰 Debug Tool（查看记忆 + 导入导出）

在插件根目录执行：

```bash
streamlit run debug_tool/app.py
```

打开页面（默认）：

- `http://localhost:8501`

左侧模式说明：

- `📖 浏览记忆`：查看向量记忆集合内容（支持 scope 过滤）
- `🧾 浏览Simple记忆`：查看 `simple_memory.db` 中的记忆（支持 scope/关键词）
- `🗂️ 中央笔记索引`：分页查看 `note_index_records + note_tag_rel`，含 `note_short_id/总行数`
- `🔎 note_recall 调试`：按 `note_short_id + 行范围` 模拟读取正文
- `🔄 导入导出`：对 Simple 记忆进行 JSON 导出/导入

导出步骤：

1. 进入 `🔄 导入导出`
2. 可选填写 `scope`（留空导出全部）
3. 点击“生成导出文件”
4. 点击“下载 JSON”

导入步骤：

1. 进入 `🔄 导入导出`
2. 上传 JSON 文件
3. 点击“执行导入”

导入规则：

- 支持两种格式：`{"memories":[...]}` 或 `[...]`
- 按 `judgment` 去重 upsert（较新 `created_at` 覆盖较旧）

## 📖 使用指南

### LLM工具使用

AI可以通过以下工具主动管理记忆：

**1. 铭记重要信息**
```
当用户明确表达重要偏好时，AI会调用：
core_memory_remember(
    judgment="用户偏好猫胜过狗",
    reasoning="用户明确提到更喜欢猫的性格",
    tags=["用户偏好", "宠物"],
    strength=80,
    memory_type="knowledge"
)
```

**2. 回忆核心记忆**
```
AI可主动反思核心原则：
core_memory_recall(
    limit=5,
    query="用户偏好"  # 可选的检索关键词
)
→ 返回按重要性加权随机抽取的5条记忆
```

**3. 展开笔记内容**
```
查看完整笔记上下文：
note_recall(
    note_short_id=12,
    start_line=1,    # 可选，默认1
    end_line=120,    # 可选，默认到文件末尾
    token_budget=2000
)
→ 返回按行范围读取后的正文，并附带实际起止行与总行数
```

**4. 启动研究任务**
```
深度研究某个论题：
research_topic(
    topic="详细的研究论题描述...",
    complexity="normal"  # 或 "complex"
)
→ 启动独立研究Agent，生成探索报告
```

### 知识库管理

⚠️ **重要：知识库的正确使用方式**

**知识库不是RAG系统！它是research_tool的主动学习源。**

#### ❌ 错误的使用方式

**千万不要把知识库当作RAG系统使用！**

以下做法会严重污染上下文，导致AI理解能力下降：

- ❌ 上传长篇PDF文档（如整本书、完整论文）
- ❌ 存放大段落的技术文档（超过100字的连续文本）
- ❌ 期待系统自动检索并注入到对话中
- ❌ 把知识库当作"背景知识"让AI被动吸收

**为什么长文档RAG是错误的？**

**用做菜来比喻**：

想象你要做一道"宫保鸡丁"：

- ❌ **长文档RAG**：给你一整本《中华烹饪大全》，里面有川菜、粤菜、鲁菜...你得翻遍整本书找宫保鸡丁的做法，还可能找到"宫保虾仁"、"鱼香肉丝"等干扰信息
- ✅ **短条目知识库**：给你一张"宫保鸡丁食谱卡"，100字写清配料和步骤，拿来就用

**具体问题**：

1. **上下文污染**：就像做菜时灶台上堆满了不相关的食材，找不到需要的那瓶酱油
2. **信息噪音**：检索到的长文档里，只有10%是你需要的，90%是"我知道番茄炒蛋、红烧肉的做法"等无用信息
3. **检索失准**：搜"宫保鸡丁"，可能给你返回整个"川菜章节"，里面有20道菜，你还得自己筛选
4. **Token浪费**：就像用一个大冰箱存一颗白菜，99%的空间都被浪费了

#### ✅ 正确的使用方式

**知识库应该存放结构化的短条目（100字以内），作为research_tool的主动学习源。**

**推荐的文件组织结构**

```
raw/
├── 美食食谱/
│   ├── 家常菜.md
│   │   # 每条不超过100字，就像食谱卡片
│   │   ## 番茄炒蛋
│   │   材料: 番茄2个、鸡蛋3个、盐、糖。步骤: 蛋液加盐打散炒熟盛出→番茄切块炒软加糖→倒入鸡蛋翻炒
│   │
│   │   ## 蒜蓉西兰花
│   │   西兰花焯水1分钟→蒜末爆香→倒入西兰花快炒→加盐出锅。保持翠绿脆嫩的秘诀: 焯水后过冷水
│   │
│   └── 川菜.md
│       ## 宫保鸡丁
│       鸡胸肉丁腌制→花生米炸香→鸡丁炒至变色→加干辣椒花椒爆香→倒入宫保汁收汁→加花生米
│       关键: 宫保汁=酱油2勺+醋1勺+糖1勺+淀粉水
│
├── 穿搭技巧/
│   └── 基础搭配.md
│       ## 冬季保暖公式
│       内层贴身打底衫+中层羊毛衫/卫衣+外层大衣/羽绒服。颜色: 上浅下深显高。避免: 全身黑膨胀
│
│       ## 腿粗怎么穿裤子
│       选择: 直筒裤、阔腿裤、烟管裤。避免: 紧身裤、哈伦裤。长度: 盖住脚踝显瘦
│
└── 生活妙招/
    └── 收纳技巧.md
        ## 衣柜整理法则
        当季衣服放中间→换季衣服放上层→过季衣服收纳箱。T恤竖着卷起来存放，既省空间又方便找

        ## 冰箱保鲜顺序
        熟食上层、生食下层。蔬菜独立保鲜盒。肉类冷冻前分小份。避免: 剩菜超过3天
```

**为什么这样组织效果好？**

**继续用做菜比喻**：

1. **精确检索**：你说"我要做宫保鸡丁"，系统直接给你那张食谱卡，而不是整本烹饪书
2. **即用即懂**：100字就像一张便签纸，扫一眼就记住了，不用翻阅厚重的说明书
3. **上下文清爽**：灶台上只放当前菜的食材，而不是把冰箱搬过来
4. **主动学习**：想学新菜时通过research_tool主动问"教我做红烧肉"，而不是被动接收"你可能需要知道500道菜"
5. **结构化记忆**：就像把调料按"咸/甜/辣"分格，衣服按"春夏秋冬"分区，拿取方便

**生活化类比总结**：

- 长文档RAG = 带着整个衣柜出门，找一件T恤都费劲
- 短条目知识库 = 整理好的胶囊衣橱，每件都是精选，随时搭配

**最佳实践**

- ✅ 每条知识控制在100字以内
- ✅ 使用Markdown的二级标题（##）分隔条目
- ✅ 文件夹名即分类标签（如"Python快速参考"）
- ✅ 一个主题一个文件，避免大杂烩
- ✅ 定期精简，删除过时或冗余内容

**索引同步**
- 添加/修改/删除文件后重启插件即可自动同步索引

## 🏗️ 系统架构

### 认知工作流

```
用户消息
    ↓
【潜意识层 - DeepMind】
    ├─ 观察：消息预处理 + 实体提取
    ├─ 回忆：链式召回（向量检索+FlashRank语义重排）
    │   ├─ 记忆轨道：已内化的知识
    │   └─ 笔记轨道：知识库短条目
    ├─ 灵魂共鸣：旧记忆状态影响当前决策
    └─ 注入：记忆+笔记+灵魂状态 → LLM
    ↓
【主意识层 - LLM】
    ├─ 思考：结合上下文生成回复
    └─ 工具调用：主动管理核心记忆
    ↓
【反馈与进化】
    ├─ 小模型分析：识别有用记忆和新认知
    ├─ 记忆强化：增强有用记忆的强度
    ├─ 记忆生成：创建新的理解
    ├─ 记忆合并：抽象化相似记忆
    └─ 睡眠巩固：定期清理和优化
```

## 🎯 使用场景

### 1. 个性化AI助手
- 记住用户偏好和历史对话
- 学习用户的沟通风格
- 提供个性化建议

### 2. 主动学习助手
- 通过research_tool主动研究论题
- 从知识库短条目中快速学习
- 自主整合多源知识生成报告

### 3. 技能速查系统
- 存储精简的命令速查卡片
- 快速检索关键语法和配置
- 避免长文档污染上下文

### 4. 持续学习系统
- AI在对话中不断进化
- 自主积累领域知识
- 形成稳定的认知体系

## 🔧 高级配置

### 睡眠维护同步（中央索引）

当前状态：`已实现`（无需额外功能开关，随睡眠维护管线自动执行）。

触发与模式：
- `向量 -> 中央迁移`：在向量模式（`enable_simple_memory=false`）下，按 provider 变化触发（同 provider 已迁移则跳过）。
- `记忆向量库同步`：在向量模式下，睡眠前维护阶段按 provider 变化触发；同步中央记忆索引到 `memory_index`。
- `笔记向量库同步`：在向量模式下，睡眠后维护阶段执行；provider 变化时全量重建 `notes_index`，否则按名单增量同步（新增补写、失效删除）。
- `Simple 模式`（`enable_simple_memory=true`）：跳过记忆/笔记向量库同步。
- `每日JSON备份`：中央库每天 1 次，最多保留 3 份。

是否需要用户配置：
- 常规使用无需额外配置同步开关。
- 频率由 `sleep_interval` 决定；当睡眠周期触发时，维护任务随之触发。
- 若需要记忆/笔记向量库同步，必须可用嵌入提供商（向量化写入需要 embedding）。

频率与性能影响：
- 默认频率：随 `sleep_interval`（默认 `3600` 秒）周期执行。
- 吞吐/延迟：迁移与同步会增加一次睡眠周期内的 I/O 与向量化耗时；规模越大，单次维护耗时越高。
- 失败与重试：单次失败会记录错误日志；后续睡眠周期会再次尝试，不阻塞主对话流程。

### 灵魂系统调优

```yaml
# 调整AI的行为倾向（通过配置mid值）
soul_recall_depth_mid: 7        # 回忆深度（1-20）
soul_impression_depth_mid: 3    # 记住倾向（1-10）
soul_expression_desire_mid: 0.5 # 表达欲望（0.0-1.0）
soul_creativity_mid: 0.7        # 创造力（0.0-1.0）
```

### 性能优化

```yaml
# Token预算控制
note_topk:
  top_k: 8                      # 注入笔记数量上限（候选固定为7倍）

# 记忆系统
short_term_memory_capacity: 1.0 # 短期记忆容量
sleep_interval: 3600            # 巩固间隔（秒）

```

### 嵌入模型选择

**推荐新手直接用 Simple 模式：**

- `enable_simple_memory: true`
- 不需要配置嵌入模型
- 可直接使用记忆查看、导入导出、睡眠维护

**进阶再用向量模式：**

- `enable_simple_memory: false`
- 需要配置可用的嵌入 provider（本地或 API）
- 适合追求语义检索效果的场景

补充：
- 只用 Simple 模式时，不需要关心向量回灌。
- 只有启用向量模式时，`SQL → 向量` 回灌才会执行（回灌时会重新向量化）。

## 🐛 故障排查

### 常见问题

**Q: 插件启动失败？**
- 检查是否已安装`astrbot_plugin_angel_heart`
- 查看日志确认具体错误信息

**Q: 向量化速度慢？**
- 考虑使用API提供商而非本地模型
- 调整Token预算减少处理量

**Q: 记忆系统不工作？**
- 确认`provider_id`已正确配置
- 检查消息长度是否达到`min_message_length`

**Q: 灵魂系统参数异常？**
- 检查能量值是否超出软限制（-20到+20）
- 验证配置中的min/mid/max值设置

### 日志调试

查看插件日志了解详细运行情况：
- 初始化状态
- 记忆检索过程
- 灵魂状态变化
- 工具调用记录

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境

```bash
# 克隆项目
git clone https://github.com/kawayiYokami/astrbot_plugin_angel_memory.git
cd astrbot_plugin_angel_memory

# 安装依赖
pip install -r requirements.txt

# 代码格式化
ruff format .
```

### 扩展开发

详见[AGENTS.md](AGENTS.md)开发文档。

## 📄 许可证

本项目采用 [GNU General Public License v3.0](LICENSE)。

## 🙏 致谢

感谢所有贡献者和用户的支持！

- [AstrBot](https://github.com/Soulter/AstrBot) - 优秀的QQ机器人框架
- [ChromaDB](https://www.trychroma.com/) - 强大的向量数据库
- [sentence-transformers](https://www.sbert.net/) - 高质量的嵌入模型

---

**赋予AI真正的记忆与认知 - Angel Memory** 🧠✨
