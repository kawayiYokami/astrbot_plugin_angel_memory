# Angel Memory · 天使之魂

给 AI 装上真正的记忆。它记得你这个人，也翻得到你放进去的资料，越用越懂你。

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-green.svg)](https://github.com/Soulter/AstrBot)

## 它会做什么

- **记得住人**：你的口味、你在意的人、手头在忙的事、随口许下的约定。千人聊天室也不会把人记混。
- **分得清场合**：不同人格、不同群各记各的；没单独归类的进公共记忆，谁都用得上。
- **翻得到资料**：把 Markdown 按目录放好，它就能查。每个群对每个资料库可以选「自动想起 / 要搜才给 / 装作没有」。
- **会自己收拾**：聊着聊着把要紧的挑出来记住，也会主动翻旧账、自己写笔记、按篇读正文。
- **有情绪**：喜怒惊厌随对话轻轻晃，话多话少、语气冷暖跟着走，不会永远同一张脸。
- **会睡觉**：每天凌晨自己整理一遍记忆，不常想起的慢慢变淡，淡到尽头的先审再忘。

## 天使五件套

同一只粉毛天使的五个能力化身，合在一起就是一个完整的灵魂：

| 插件 | 能力 |
|------|------|
| [天使之心](https://github.com/kawayiYokami/astrbot_plugin_angel_heart) | 智能群聊/私聊交互，懂分寸、知进退 |
| [天使之眼](https://github.com/kawayiYokami/astrbot_plugin_angel_eye) | 聊天记录查询，随时翻旧账 |
| [天使之笑](https://github.com/kawayiYokami/astrbot_plugin_angel_smile) | 表情包管理，看见好看的就想收进库存 |
| [天使之魂](https://github.com/kawayiYokami/astrbot_plugin_angel_memory) | AI 记忆与认知，越用越懂你 |
| [天使的画笔](https://github.com/kawayiYokami/astrbot_plugin_angel_brush) | 说话出图、发图改图 |

## 装上它

推荐同时装 [天使之心](https://github.com/kawayiYokami/astrbot_plugin_angel_heart)，检索词和聊天记录会更完整；只装本插件也能跑。

```bash
cd plugins
git clone https://github.com/kawayiYokami/astrbot_plugin_angel_memory.git
cd astrbot_plugin_angel_memory
pip install -r requirements.txt
```

重启 AstrBot，插件会自己完成初始化。

## 调一调它

在 AstrBot 配置界面填。最省事的起步：只填一个 `provider_id`，其余全部留默认。

| 配置项 | 默认 | 它管什么 |
|--------|------|----------|
| `provider_id` | 空 | 整理记忆用的模型，建议带思考能力。留空也能照常聊天、照常想起旧事，只是不再自动记新的。 |
| `retrieval.embedding_provider_id` | 空 | 可选。配上后换个说法也找得到；不填就按文字匹配，多数情况够用。 |
| `retrieval.rerank_provider_id` | 空 | 推荐。下拉已按模型能力自动筛选。把捞出来的记忆再排一遍序，最贴话题的排前面，提升明显。留空会复用 `provider_id`。 |
| `conversation_scope_mode` | 按人格 | 记忆分柜方式：按人格，或按群聊。 |
| `conversation_scope_map` | `{}` | 分柜对照表，见下。 |
| `group_knowledge_map` | `{}` | 每个群对每个资料库的用法，建议在面板「群聊知识库」里改。 |
| `memory_behavior.min_message_length` | 5 | 太短的消息不值得记，直接跳过。 |
| `memory_behavior.reflection_turn_threshold` | 6 | 每聊几轮整理一次记忆。调小记得勤，模型调用也多。 |
| `memory_behavior.short_term_memory_capacity` | 1 | 最近这段对话在脑子里占多大地方，设 2 就是翻倍。 |
| `enable_soul_system.enabled` | 开 | 关掉后情绪不再随对话浮动，语气稳定下来。 |
| `note_assistant.enable_recall` | 开 | 回答时要不要一并翻知识库。 |
| `note_assistant.enable_create` | 开 | 它能不能自己写笔记。 |
| `note_assistant.top_k` | 3 | 一次最多带几篇笔记进对话。 |

**记忆分柜**。按人格分：

```json
{
  "女友": "恋爱",
  "旅人": "须弥"
}
```

按群聊分：

```json
{
  "default:GroupMessage:12345": "家人"
}
```

没写到的进公共柜子，谁都能用到。柜子名写错、写成空值都不会崩，会自动回落到公共。在设置页从下拉里选，不用手写。

## 两间屋子

**记忆**是它从对话里挑出来的事，聊着聊着自己长出来，不用你管。

**知识库**是你放进去的资料，住在插件数据下的 `raw/`：

```
raw/
├── 原神/
├── 崩坏星穹铁道/
├── 绝区零/
├── 海豹的个人信息/
└── .angel/note/          # 它自己写的笔记落在这里
```

只有这五个目录会被检索，其它文件夹不参与。只收 `.md` 文件；放进去大约一分钟内生效，删掉文件它也会跟着忘掉，都不用重启。

### 资料怎么写才好用

- 一条知识尽量短，用二级标题切开，像卡片而不是论文。
- 一个主题一个文件。
- 不要塞整本书、整篇长文。长文会把真正用得上的那几句淹没。

```
raw/原神/武器/法器.md

## 乌髓孑灯
法器。技能与获取方式写在这里，一条一张卡片。
```

### 群里怎么用这些库

每个群对每个库三种用法，在面板「群聊知识库」里改：

| 用法 | 对话时会不会自动带上 | 能不能主动去搜 | 知不知道有这个库 |
|------|----------------------|----------------|------------------|
| 自动 | 会 | 能 | 知道 |
| 手动 | 不会 | 能 | 知道 |
| 不可见 | 不会 | 不能 | 不知道 |

没配过的群默认全部「手动」，私聊同样：能搜，不自动带。群列表来自群账本，群里聊过天才会出现。

## 管理面板

自带管理页，不用另起服务：打开 AstrBot WebUI（默认 `http://localhost:6185`），左侧「插件管理」→「天使的记忆」。

| 页面 | 能做什么 |
|------|----------|
| 总览 | 系统状态、提供商、记忆/标签/笔记数量、情绪波动与过程追踪记录 |
| 记忆浏览 | 分页看、按分类和关键词筛、看详情、删除 |
| 用户画像 | 已识别用户，按属性分组看画像记忆 |
| 群聊知识库 | 每个群对五个知识库下拉选自动 / 手动 / 不可见 |
| Tags 调试 | 输入一段话，看命中了哪些标签和记忆 |
| 笔记索引 | 浏览笔记，按关键词搜摘要 |
| 笔记读取 | 按笔记 ID 分页读正文 |
| 导入导出 | 导出记忆快照 / 从 JSON 导入，已存在的记忆跳过不覆盖 |
| 维护状态 | 看维护记录、下载备份 |
| 工具测试 | 两个按钮：自动回忆、主动回忆。结果进同一个框，只显示原文 |
| 功能调试 | 探一探嵌入和重排通不通 |
| 插件设置 | 改本插件配置，保存后即时生效（提供商类改动需重载插件） |

## 它是怎么记住的

聊到要紧的事，它会整理成一条条记忆存起来。每条记忆就三样东西：

| 部分 | 是什么 | 例子 |
|------|--------|------|
| 判断 | 记下来的那件事 | 红豆不吃香菜 |
| 依据 | 凭什么这么记 | 点菜那次他特意说了不要香菜 |
| 线索 | 以后靠什么想起来 | 红豆、口味、香菜 |

要回答问题时，它按线索把相关的记忆捞回来，一并翻知识库，然后才开口。自动回忆自带重排与质量门槛过滤，只保留关联度最高的前几条，宁缺毋滥。这些动作模型都能主动做：记住新的（`angel_remember`）、想起来（`angel_recall`）、读笔记（`angel_note_read`）、写笔记（`angel_note_create`）。写下的笔记落在 `.angel/note/`，跟知识库一起参与检索。

## 灵魂

四对情绪，每对是一条 0 到 1 的刻度，0.5 是平常心：

| 情绪对 | 偏向前面 | 偏向后面 |
|--------|---------|---------|
| 喜 / 悲 | 想亲近，话多起来 | 有点低落，收着说 |
| 怒 / 恐 | 被冒犯了，说得很细 | 有些不安，说得简短 |
| 意料中 / 意外 | 一切照旧，按经验来 | 不按常理，想追问 |
| 厌恶 / 认可 | 排斥，语气严肃 | 认可，语气活泼 |

对话里起了什么情绪，此后几轮的回答就带一点那个味道。想起旧事时，当时记下这份记忆的心情也会轻轻牵一下现在，记得越牢的事牵引越明显。

情绪只影响语气和表达，不影响它记得多少、想起多少。状态只存在内存里，重载插件回到平常心。想让它情绪稳定，配置里关掉 `enable_soul_system`。

## 睡眠整理

每天凌晨 4 点（本地时间）自己收拾一遍记忆。进程当时没在跑，当天就不补了。

- 不常用的记忆慢慢变淡。
- 淡到尽头的先交给模型审一遍：该留的升成永久记忆，从此不再变淡；该忘的才真删。
- 审查出了岔子就整批不动，等下次睡眠重来，一条都不误删。
- 每天留一份记忆备份，最多存 3 份，面板「维护状态」里能下载。

## 遇到问题

**启动失败**：看日志，依赖没装全就先 `pip install -r requirements.txt`。

**检索不准**：优先配上 `retrieval.rerank_provider_id`。自动回忆自带质量过滤，只带出最契合的记忆；知识库条目别写太长，短卡片最灵。

**完全不记**：确认 `provider_id` 有值；消息是不是短于 `min_message_length`。

**群里冒出了不该有的资料**：去「群聊知识库」把那个库改成手动或不可见。

**面板和对话对不上**：「工具测试」里两个按钮各吐一份原文，正好拿来对账。

**记混了两个人**：检查记忆分柜是不是把两人的会话分进了同一个柜子。

## 开发

开发约定见 [AGENTS.md](AGENTS.md)。测试在 AstrBot 容器内跑：

```bash
docker exec astrbot bash -lc 'cd /AstrBot/data/plugins/astrbot_plugin_angel_memory && PYTHONPATH=/AstrBot/data/plugins:/AstrBot/data/plugins/astrbot_plugin_angel_memory python -m pytest -q'
```

必须用 `python -m pytest`，直接跑 `pytest` 会因为导入路径少一截而失败。

## 许可证

[GNU General Public License v3.0](LICENSE)

## 致谢

- [AstrBot](https://github.com/Soulter/AstrBot)
- 所有一起把这套记忆用出温度的人

---

赋予 AI 真正的记忆与认知。Angel Memory
