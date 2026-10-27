# Angel Memory - AI 记忆与认知系统

让 AI 记住和你相处过的事，也能在本地知识库里查资料。越用越懂你。

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-green.svg)](https://github.com/Soulter/AstrBot)

## 能做什么

- **记住人**：偏好、关系、正在做的事、和你一起经历过的约定。千人聊天室也不会把人记混。
- **分开记**：不同人格、不同群可以各记各的；没单独分类的进公共记忆，谁都能用到。
- **本地知识库**：把 Markdown 按目录放进去，AI 就能查。每个群对每个库都能选「自动想起 / 要搜才给 / 完全看不见」。
- **自己整理**：AI 能主动铭记、主动回忆、自己写笔记、按篇翻正文。
- **有状态**：想起多少、记住多少、话多话少、是否跳脱，会随对话轻轻晃，不会永远同一张脸。
- **自己打扫**：每天凌晨整理一次。不重要的会变淡，淡到尽头先审查再决定留下还是忘掉。

## 天使五件套

同一只粉毛天使的五个能力化身，合在一起就是一个完整的灵魂：

| 插件 | 能力 |
|------|------|
| [天使之心](https://github.com/kawayiYokami/astrbot_plugin_angel_heart) | 智能群聊/私聊交互，懂分寸、知进退 |
| [天使之眼](https://github.com/kawayiYokami/astrbot_plugin_angel_eye) | 聊天记录查询，随时翻旧账 |
| [天使之笑](https://github.com/kawayiYokami/astrbot_plugin_angel_smile) | 表情包管理，看见好看的就想收进库存 |
| [天使之魂](https://github.com/kawayiYokami/astrbot_plugin_angel_memory) | AI 记忆与认知，越用越懂你 |
| [天使的画笔](https://github.com/kawayiYokami/astrbot_plugin_angel_brush) | 说话出图、发图改图 |

## 快速开始

**推荐同时安装 [astrbot_plugin_angel_heart](https://github.com/kawayiYokami/astrbot_plugin_angel_heart)**。装它时检索词和聊天记录更完整；单独装本插件也能用。

1. 安装本插件

```bash
cd plugins
git clone https://github.com/kawayiYokami/astrbot_plugin_angel_memory.git
cd astrbot_plugin_angel_memory
pip install -r requirements.txt
```

2. 重启 AstrBot。插件会自己完成初始化。

### 基础配置

在 AstrBot 配置界面填写：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `provider_id` | 空 | 记忆整理用的模型，建议带思考能力。留空则只做基础记忆。 |
| `retrieval.embedding_provider_id` | 空 | 可选。语义检索增强，不是必须。 |
| `retrieval.rerank_provider_id` | 空 | 推荐。结果会再排一次序，提升明显。留空会尝试复用 `provider_id`。 |
| `conversation_scope_map` | `{}` | 记忆分类表。先按人格名匹配，再按会话 ID 匹配。 |
| `group_knowledge_map` | `{}` | 每个群对每个知识库的用法。建议在面板「群聊知识库」里改。 |
| `memory_behavior.min_message_length` | 5 | 太短的消息不进入记忆处理。 |
| `memory_behavior.short_term_memory_capacity` | 1 | 最近在意的记忆保留多少，2 就是两倍。 |
| `enable_soul_system.enabled` | 开 | 关掉后行为固定，上下文里也不再出现灵魂仪表盘。 |
| `note_assistant.enable_recall` | 开 | 回忆时要不要一并查知识库。 |
| `note_assistant.enable_create` | 开 | AI 能不能自己写笔记。 |
| `note_assistant.top_k` | 3 | 一次最多带几篇笔记进上下文。 |

记忆分类表这样写：

```json
{
  "女友": "恋爱",
  "aiocqhttp:group:12345": "家人"
}
```

匹配顺序：当前人格名优先，会话 ID 其次。没写到的进公共。

写错分类名、写成空值都不会崩，会回落到公共。

## 知识库

知识库是 AI 主动查阅的本地笔记，不是把整本书塞进对话。

目录在插件数据下的 `raw/`：

```
raw/
├── 原神/
├── 崩坏星穹铁道/
├── 绝区零/
├── 海豹的个人信息/
└── .angel/note/          # AI 自己写的笔记落在这里
```

只有这五个目录里的东西会被检索，其它文件夹不参与。放入或改过文件后大约一分钟内自动纳入，不必重启。只收 `.md` / `.txt`。

**怎么写才好用**

- 一条知识尽量短，用二级标题切开，像卡片而不是论文。
- 一个主题一个文件。
- 不要塞整本书、整篇长文。长文会把真正用得上的那几句淹没。

示例：

```
raw/原神/武器/法器.md

## 乌髓孑灯
法器。技能与获取方式写在这里，一条一张卡片。
```

### 群里怎么用这些库

每个群对每个库有三种用法，在面板「群聊知识库」里改：

| 用法 | 对话时会不会自动带上 | AI 能不能主动去搜 | AI 知不知道有这个库 |
|------|----------------------|--------------------|----------------------|
| 自动 | 会 | 能 | 知道 |
| 手动 | 不会 | 能 | 知道 |
| 不可见 | 不会 | 不能 | 不知道 |

没配过的群默认全部「手动」。私聊同样全部「手动」：能搜，不自动带。

群列表来自群账本，群里产生过对话才会出现。

旧配置 `group_domain_map` 仍可读：写中的那个库按「自动」处理。新表里出现同一个群时，以新表为准。

## 管理面板

插件自带管理页，不用另起服务。

1. 打开 AstrBot WebUI（默认 `http://localhost:6185`）
2. 左侧「插件管理」→「天使之魂」
3. 进入 `memory-dashboard`

| 页面 | 能做什么 |
|------|----------|
| 总览 | 系统状态、提供商、记忆/标签/笔记数量、过程追踪记录 |
| 记忆浏览 | 分页看、按分类和关键词筛、看详情、删除 |
| 用户画像 | 已识别用户，按属性分组看画像记忆 |
| 群聊知识库 | 每个群对五个知识库下拉选自动 / 手动 / 不可见 |
| Tags 调试 | 输入一段话，看命中了哪些标签和记忆 |
| 笔记索引 | 浏览笔记，按关键词搜摘要 |
| 笔记读取 | 按笔记 ID 分页读正文 |
| 导入导出 | 导出记忆快照 / 从 JSON 导入 |
| 维护状态 | 看维护记录、下载备份 |
| 工具测试 | 两个按钮：自动回忆、主动回忆。结果进同一个框，只显示原文 |
| 功能调试 | 探一探嵌入和重排通不通 |
| 插件设置 | 改本插件配置，保存后即时生效（提供商类改动需重载插件） |

导入导出：已存在的记忆按 ID 跳过，不会覆盖。

## AI 能调用的工具

**铭记** `angel_remember`

```
angel_remember(
    action="create",
    memory={
        "judgment": "用户偏好猫胜过狗",
        "reasoning": "用户明确提到更喜欢猫的性格",
        "tags": ["用户偏好", "宠物"],
        "memory_type": "knowledge"
    }
)
```

`action` 还可以是 `update` 改一条、`merge` 把几条收成一条。类型可选 `knowledge` / `skill` / `emotional` / `event` / `task`。

**回忆** `angel_recall`

```
angel_recall(query="用户偏好")
```

一次同时搜记忆和当前群能看见的知识库。

**读笔记** `angel_note_read`

```
angel_note_read(note_id=12, offset=1, limit=200)
```

`note_id` 是检索结果里的整数编号。单次最多返回 3 万字符，读不完会告诉还有没有。

**写笔记** `angel_note_create`

```
angel_note_create(
    title="Python异步编程",
    content="# Python异步编程\n\n## 协程基础\n..."
)
```

写成 Markdown 落到 `.angel/note/`，随后就能检索。

## 灵魂状态

四对情绪，每一对都是一条 0 到 1 的刻度，0.5 是平常心：

| 情绪对 | 偏向前面 | 偏向后面 |
|--------|---------|---------|
| 喜 / 悲 | 想亲近，话多起来 | 有点低落，收着说 |
| 怒 / 恐 | 被冒犯了，说得很细 | 有些不安，说得简短 |
| 意料中 / 意外 | 一切照旧，按经验来 | 不按常理，想追问 |
| 厌恶 / 认可 | 排斥，语气严肃 | 认可，语气活泼 |

对话里产生什么情绪，此后几轮的回答就会带一点那个味道。想起旧事时，当时记下这份记忆的心情也会轻轻牵一下现在；记得越牢的事，牵引越明显。

状态只存在内存里，重载插件会回到平常心。想在配置里关掉，就用 `enable_soul_system`。

## 睡眠整理

每天凌晨 4 点整理一次，按运行环境本地时间。进程当时没在跑，当天不补。

- 不常用的记忆会变淡。
- 淡到尽头的，先交给模型审查：该留下的升成永久记忆，此后不再参与变淡；该忘的才删。
- 审查失败则整批不动，等下一次睡眠重来，一条都不删。
- 每天留一份记忆 JSON 备份，最多保留 3 份。

## 故障排查

**启动失败**
看日志。依赖没装全时先执行 `pip install -r requirements.txt`。

**检索不准**
优先配 `retrieval.rerank_provider_id`。知识库条目过长也会干扰，尽量写成短卡片。

**完全不记**
确认 `provider_id` 有值。消息是否短于 `min_message_length`。

**群里不该出现的资料冒出来了**
去「群聊知识库」把对应库改成手动或不可见。

**面板和对话对不上**
用「工具测试」看自动回忆和主动回忆各自吐出的原文。框里只显示接口返回，便于对账。

**记混了两个人**
检查记忆分类表是否把该会话分到了同一类。

## 开发

开发约定见 [AGENTS.md](AGENTS.md)。

```bash
ruff format .
```

测试在 AstrBot 容器内执行：

```bash
docker exec astrbot bash -lc 'cd /AstrBot/data/plugins/astrbot_plugin_angel_memory && PYTHONPATH=/AstrBot/data/plugins:/AstrBot/data/plugins/astrbot_plugin_angel_memory python -m pytest -q'
```

必须用 `python -m pytest`，不要直接跑 `pytest`。

## 许可证

[GNU General Public License v3.0](LICENSE)

## 致谢

- [AstrBot](https://github.com/Soulter/AstrBot)
- 所有一起把这套记忆用出温度的人

---

赋予 AI 真正的记忆与认知。Angel Memory
