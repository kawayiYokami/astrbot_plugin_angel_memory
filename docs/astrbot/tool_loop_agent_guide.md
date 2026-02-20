# `self.context.tool_loop_agent` 使用指南

## 概述
`tool_loop_agent` 是 AstrBot v4.5.7+ 引入的 Agent 执行方法，支持多轮工具调用循环，直到 LLM 产生最终答案。适用于需要工具协作的复杂任务。

**位置**: `astrbot/core/star/context.py:124`

## 函数签名

```python
async def tool_loop_agent(
    self,
    *,
    event: AstrMessageEvent,
    chat_provider_id: str,
    prompt: str | None = None,
    image_urls: list[str] | None = None,
    tools: ToolSet | None = None,
    system_prompt: str | None = None,
    contexts: list[Message] | None = None,
    max_steps: int = 30,
    tool_call_timeout: int = 60,
    **kwargs: Any,
) -> LLMResponse
```

## 参数详解

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `event` | `AstrMessageEvent` | 是 | - | 消息事件对象，包含会话信息 |
| `chat_provider_id` | `str` | 是 | - | 提供商 ID（如 `openai_default`） |
| `prompt` | `str \| None` | 否 | `None` | 用户输入提示词，若与 `contexts` 同时提供，则追加为最后一条用户消息 |
| `image_urls` | `list[str] \| None` | 否 | `None` | 图片 URL 列表，支持本地路径或网络 URL |
| `tools` | `ToolSet \| None` | 否 | `None` | 可用的工具集合，为空则禁用工具调用 |
| `system_prompt` | `str \| None` | 否 | `None` | 系统提示词，若提供则**始终插入为第一条系统消息**（覆盖已有系统消息） |
| `contexts` | `list[Message] \| None` | 否 | `None` | 历史上下文消息（OpenAI 格式） |
| `max_steps` | `int` | 否 | `30` | Agent 最大执行步骤（工具调用循环次数） |
| `tool_call_timeout` | `int` | 否 | `60` | 单个工具调用超时时间（秒） |
| `**kwargs` | `Any` | 否 | - | 扩展参数，可包含：<br>- `agent_hooks: BaseAgentRunHooks[AstrAgentContext]` - Agent 执行钩子<br>- `agent_context: AstrAgentContext` - 自定义 Agent 上下文<br>- `model: str` - 指定模型名称（透传给提供商）<br>- `stream: bool` - 是否流式响应 |

## 返回值

返回 `LLMResponse` 对象，主要属性：

```python
class LLMResponse:
    role: str  # "assistant" | "tool" | "err"
    completion_text: str  # 文本回复（优先从 result_chain 提取）
    result_chain: MessageChain | None  # 消息链（包含富文本内容）
    tools_call_name: list[str]  # 调用的工具名称列表
    tools_call_args: list[dict[str, Any]]  # 工具调用参数列表
    tools_call_ids: list[str]  # 工具调用 ID 列表
    raw_completion: ChatCompletion | GenerateContentResponse | AnthropicMessage | None  # 原始响应
    is_chunk: bool  # 是否为流式分块
```

## 使用示例

### 基础调用（官方示例）
```python
from astrbot.core.agent.tool import ToolSet

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="搜索一下 bilibili 上关于 AstrBot 的相关视频。",
    tools=ToolSet([BilibiliTool()]),
    max_steps=30,
    tool_call_timeout=60,
)
print(llm_resp.completion_text)  # 获取返回的文本
```

### 添加系统提示词
```python
llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="查询北京天气",
    system_prompt="你是一个天气助手，请用中文简洁回复。",
    tools=ToolSet([WeatherTool()]),
)
```

### 带图片输入
```python
llm_resp = await self.context.tool_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="描述这张图片的内容",
    image_urls=["/path/to/image.jpg", "https://example.com/image.png"],
    system_prompt="你是一个图像描述专家。",
)
```

### 携带历史上下文
```python
from astrbot.core.agent.message import Message

contexts = [
    Message(role="user", content="你好"),
    Message(role="assistant", content="你好！有什么可以帮助你？"),
    Message(role="user", content="继续刚才的话题"),
]

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    contexts=contexts,
    prompt="继续解释",
)
```

### 使用全局工具
```python
# 获取所有已注册的全局工具（继承供应商预设工具）
tool_manager = self.context.get_llm_tool_manager()
all_tools = tool_manager.get_full_tool_set()

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="使用可用工具帮我处理任务",
    tools=all_tools,  # 传入所有全局工具
)
```
> **提示**：更多使用全局工具的方法详见 [使用全局工具](#使用全局工具) 章节。

### 指定模型和流式响应
```python
llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="请用 GPT-4 回答",
    system_prompt="你是一个专业顾问。",
    model="gpt-4",  # 指定模型
    stream=True,    # 启用流式响应（需平台支持）
)
```

### 使用 Agent Hooks
```python
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.astr_agent_context import AstrAgentContext

class MyHooks(BaseAgentRunHooks[AstrAgentContext]):
    async def on_agent_begin(self, run_context):
        print(f"Agent 开始执行，当前消息数: {len(run_context.messages)}")

    async def on_tool_start(self, run_context, tool, params):
        print(f"工具 {tool.name} 开始执行，参数: {params}")

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="测试工具调用",
    tools=ToolSet([MyTool()]),
    agent_hooks=MyHooks(),  # 传入自定义钩子
)
```

## 注意事项

### 1. **人格（Persona）不会自动注入**
`tool_loop_agent` **不会**自动添加当前人格的 system_prompt。如需使用人格，需显式获取并传入：

```python
umo = event.unified_msg_origin
persona = await self.context.persona_mgr.get_default_persona_v3(umo)
llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="用户问题",
    system_prompt=persona["prompt"],  # 手动传入人格
)
```

### 2. **人格工具配置不会自动继承**
`tool_loop_agent` **不会**自动继承当前人格配置的工具列表。如需使用人格配置的工具，需手动获取并传入（详见 [自动继承人格工具配置](#自动继承人格工具配置) 章节）。

### 3. **`system_prompt` 会覆盖已有系统消息**
若 `contexts` 已包含系统消息，传入的 `system_prompt` 将**替换**它（插入到索引 0 位置）。

### 4. **工具调用循环机制**
- Agent 会循环执行：LLM 生成 → 工具调用 → 结果反馈 → LLM 再生成
- 循环直到：LLM 不调用工具、达到 `max_steps`、工具返回最终答案
- 工具可返回 `MessageEventResult` 直接发送消息给用户（此时 Agent 循环终止）

### 5. **与 `llm_generate` 的区别**
| 特性 | `tool_loop_agent` | `llm_generate` |
|------|-------------------|----------------|
| 工具调用循环 | ✅ 支持多轮 | ❌ 不支持 |
| 自动工具执行 | ✅ 自动执行返回结果 | ❌ 只返回工具调用信息 |
| 适用场景 | 复杂 Agent 任务 | 单次 LLM 生成 |
| 返回值 | 最终答案文本 | 可能包含待执行工具 |

### 6. **Agent 上下文隔离**
每次调用创建独立的 Agent 上下文，不会影响主会话历史。如需持久化，需手动保存 `contexts`。

### 7. **工具过滤**
通过 `event.plugins_name` 可限制只使用特定插件的工具（在 `InternalAgentSubStage` 中实现，直接调用时无此限制）。

## 供应商能力限制 (Modalities)

在 AstrBot 中，每个供应商（Provider）都有一个 `modalities` 配置，控制该供应商支持的能力：

- `text`: 支持文本生成
- `image`: 支持图像输入（多模态）
- `tool_use`: 支持工具调用

**关键影响**：
- 如果供应商的 `modalities` 不包含 `tool_use`，即使传入了工具集，LLM 也无法调用工具。
- 如果供应商的 `modalities` 不包含 `image`，图像输入将被自动过滤。

### 检查供应商是否支持工具调用
```python
provider = await self.context.provider_manager.get_provider_by_id(prov_id)
if provider:
    modalities = provider.provider_config.get("modalities", [])
    supports_tool_use = "tool_use" in modalities
    if not supports_tool_use:
        print("该供应商不支持工具调用")
```

### 自动处理
在内部流水线中（如常规消息处理），`InternalAgentSubStage` 会自动检查 `modalities` 并过滤不支持的请求内容。但直接调用 `tool_loop_agent` 时，**不会自动检查**，需要开发者自行确保供应商支持工具调用。

## 自动继承人格工具配置

在 AstrBot 的 WebUI 中，用户可以为每个人格（Persona）配置可用的工具列表。当使用常规消息处理时，系统会自动继承该人格的工具配置。但 `tool_loop_agent` 不会自动继承，需要手动获取并传入。

### 获取当前人格的工具配置
```python
async def get_persona_toolset(context, event, persona_id=None):
    """获取指定人格的工具集，如果 persona_id 为 None 则获取当前会话的人格"""
    if persona_id is None:
        # 获取当前会话的人格 ID
        umo = event.unified_msg_origin
        persona_id = await context.persona_mgr.get_default_persona_v3(umo)
        if not persona_id:
            # 使用默认人格
            default_persona = context.persona_manager.selected_default_persona_v3
            persona_id = default_persona["name"] if default_persona else None

    if persona_id:
        # 获取人格信息
        personas = context.persona_manager.personas_v3
        persona = next((p for p in personas if p["name"] == persona_id), None)
        if persona:
            tool_manager = context.get_llm_tool_manager()
            if persona.get("tools") is None:
                # None 表示使用所有激活的工具
                toolset = tool_manager.get_full_tool_set()
                # 过滤未激活的工具
                for tool in toolset.tools[:]:
                    if not tool.active:
                        toolset.remove_tool(tool.name)
            elif isinstance(persona["tools"], list) and len(persona["tools"]) > 0:
                # 使用人格配置的特定工具列表
                toolset = ToolSet()
                for tool_name in persona["tools"]:
                    tool = tool_manager.get_func(tool_name)
                    if tool and tool.active:
                        toolset.add_tool(tool)
            else:
                # 空列表表示不使用任何工具
                toolset = ToolSet()
            return toolset

    # 默认返回所有激活的全局工具
    tool_manager = context.get_llm_tool_manager()
    toolset = tool_manager.get_full_tool_set()
    for tool in toolset.tools[:]:
        if not tool.active:
            toolset.remove_tool(tool.name)
    return toolset
```

### 在 tool_loop_agent 中自动继承人格工具
```python
# 自动获取当前人格的工具集
persona_toolset = await get_persona_toolset(self.context, event)

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="使用人格配置的工具处理任务",
    tools=persona_toolset,  # 自动继承人格工具配置
)
```

### 重要说明
1. **人格工具配置的优先级**：
   - `None`: 使用所有激活的全局工具（默认）
   - `[]`: 空列表，不使用任何工具
   - `["tool1", "tool2"]`: 只使用指定的工具

2. **与供应商 modalities 的关系**：
   - 人格工具配置仅控制**哪些工具可用**
   - 供应商的 `modalities` 控制**是否支持工具调用**
   - 两者需同时满足：供应商支持 `tool_use` + 人格配置包含该工具

3. **直接调用 tool_loop_agent 的场景**：
   - 插件开发、自定义 Agent 逻辑
   - 需要手动处理工具继承
   - 常规消息处理已自动处理，无需额外配置

## 使用全局工具

### Agent Tool 与全局 Tool 的关系

1. **全局工具**：通过 `self.context.add_llm_tools()` 注册，存储在 `provider_manager.llm_tools`（`FunctionToolManager` 实例）中，**全局可用**。

2. **Agent 局部工具**：调用 `tool_loop_agent()` 时通过 `tools` 参数传入的 `ToolSet`，**仅本次调用有效**。

3. **关系独立性**：Agent 的 `tools` 参数是独立的 `ToolSet`，**不会自动继承**全局工具，需要显式传入。

### 使用方法

#### 方法一：使用所有全局工具
```python
# 获取所有已注册的全局工具
tool_manager = self.context.get_llm_tool_manager()
all_global_tools = tool_manager.get_full_tool_set()

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="使用所有可用工具处理任务",
    tools=all_global_tools,  # 传入全部全局工具
)
```

#### 方法二：筛选特定工具
```python
tool_manager = self.context.get_llm_tool_manager()
global_tools = tool_manager.get_full_tool_set()

# 筛选需要的工具
selected_tools = [tool for tool in global_tools.tools if "search" in tool.name]
custom_toolset = ToolSet(selected_tools)

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="使用搜索相关工具",
    tools=custom_toolset,  # 传入筛选后的工具
)
```

#### 方法三：合并自定义工具与全局工具
```python
from astrbot.core.agent.tool import ToolSet

# 获取全局工具
global_tools = self.context.get_llm_tool_manager().get_full_tool_set()

# 创建合并的工具集
merged_tools = ToolSet()
# 添加全局工具
for tool in global_tools.tools:
    merged_tools.add_tool(tool)
# 添加自定义工具
merged_tools.add_tool(MyCustomTool())

llm_resp = await self.context.tool_loop_agent(
    event=event,
    chat_provider_id=prov_id,
    prompt="使用混合工具集",
    tools=merged_tools,
)
```

#### 方法四：按工具名称动态选择
```python
# 通过工具名获取特定工具
tool_manager = self.context.get_llm_tool_manager()
specific_tool = tool_manager.get_func("weather_tool")

if specific_tool:
    llm_resp = await self.context.tool_loop_agent(
        event=event,
        chat_provider_id=prov_id,
        prompt="查询天气",
        tools=ToolSet([specific_tool]),  # 只使用特定工具
    )
```

### 重要提示
- **工具激活状态**：全局工具默认激活，可通过 `deactivate_llm_tool()`/`activate_llm_tool()` 控制。
- **MCP 工具**：供应商可能通过 MCP 服务连接外部工具，这些工具也会出现在全局工具列表中。
- **工具冲突**：如果 Agent 传入的工具与全局工具同名，Agent 工具优先级更高（仅在该次调用中）。

## 常见问题

### Q: 如何获取当前会话的提供商 ID？
```python
umo = event.unified_msg_origin
prov_id = await self.context.get_current_chat_provider_id(umo=umo)
```

### Q: 工具调用失败如何处理？
Agent 会捕获工具异常，在响应中标记错误，并继续尝试其他工具或结束循环。

### Q: 如何控制工具调用顺序？
工具调用顺序由 LLM 决定，可通过 `system_prompt` 引导，如：“请先使用工具 A，再使用工具 B”。

### Q: 支持哪些图片格式？
- 本地路径：`/path/to/image.jpg`
- 网络 URL：`https://example.com/image.png`
- Base64：`data:image/jpeg;base64,...`
- 文件协议：`file:///path/to/image.jpg`

### Q: `max_steps` 设置多大合适？
根据任务复杂度：
- 简单任务：5-10
- 中等任务：10-20
- 复杂任务：20-30（默认）
- 无限循环：设为较大值，但需注意超时风险

## 相关链接
- [`llm_generate` 方法](astrbot/core/star/context.py:83) - 单次 LLM 生成
- [`ProviderRequest` 结构](astrbot/core/provider/entities.py:86) - 请求参数封装
- [`ToolLoopAgentRunner`](astrbot/core/agent/runners/tool_loop_agent_runner.py) - 底层执行器

---

**版本**: v4.5.7+
**最后更新**: 2025-12-15
**代码位置**: `astrbot/core/star/context.py:124`
