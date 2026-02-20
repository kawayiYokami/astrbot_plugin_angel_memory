# AstrBot 事件触发时机指南

本文档详细介绍了 AstrBot 事件流中的各类触发时机，这些时机通过装饰器的方式暴露给开发者，用以在事件流的不同阶段注入自定义逻辑。

所有事件注册的核心逻辑都定义在 [`astrbot/core/star/register/star_handler.py`](astrbot/core/star/register/star_handler.py) 文件中。

## 核心事件触发时机

以下是 AstrBot 事件流中的主要触发时机，按照其在生命周期中的典型顺序排列：

### 1. 应用与平台加载事件

- **`@filter.on_astrbot_loaded()`**
  - **触发时机**: 在 AstrBot 核心应用加载完成时
  - **用途**: 执行全局初始化、加载配置、注册全局服务等
  - **定义位置**: [`star_handler.py:260`](astrbot/core/star/register/star_handler.py:260)

- **`@filter.on_platform_loaded()`**
  - **触发时机**: 在每个平台适配器（如 QQ、Telegram、Slack 等）加载完成时
  - **用途**: 平台特定的初始化、注册平台相关服务
  - **定义位置**: [`star_handler.py:270`](astrbot/core/star/register/star_handler.py:270)

### 2. 消息与指令处理事件

- **`@command.command()`**
  - **触发时机**: 当用户发送的消息匹配注册的指令时
  - **用途**: 处理用户命令，执行特定功能
  - **定义位置**: [`star_handler.py:61`](astrbot/core/star/register/star_handler.py:61)

- **`@filter.regex()`**
  - **触发时机**: 通过正则表达式匹配消息内容时
  - **用途**: 处理符合特定模式的消息，如关键词触发、模式匹配等
  - **定义位置**: [`star_handler.py:229`](astrbot/core/star/register/star_handler.py:229)

- **`@filter.on_event_message_type()`**
  - **触发时机**: 根据消息类型（私聊、群聊等）触发
  - **用途**: 针对不同消息类型执行不同的处理逻辑
  - **定义位置**: [`star_handler.py:201`](astrbot/core/star/register/star_handler.py:201)

### 3. LLM 调用生命周期事件

- **`@filter.on_llm_request()`**
  - **触发时机**: 在准备向大语言模型（LLM）发送请求之前
  - **用途**: 修改请求参数、添加系统提示词、记录日志等
  - **参数**: `event: AstrMessageEvent`, `request: ProviderRequest`
  - **定义位置**: [`star_handler.py:282`](astrbot/core/star/register/star_handler.py:282)

- **`@filter.on_llm_response()`**
  - **触发时机**: 在接收到 LLM 的响应之后
  - **用途**: 分析响应内容、修改响应结果、记录日志、执行后处理等
  - **参数**: `event: AstrMessageEvent`, `response: LLMResponse`
  - **定义位置**: [`star_handler.py:304`](astrbot/core/star/register/star_handler.py:304)

- **`@filter.llm_tool()`**
  - **触发时机**: 当 LLM 决定调用注册的工具（Function Calling）时
  - **用途**: 实现具体的功能工具，供 LLM 调用
  - **定义位置**: [`star_handler.py:326`](astrbot/core/star/register/star_handler.py:326)

### 4. 消息发送生命周期事件

- **`@filter.on_decorating_result()`**
  - **触发时机**: 在消息发送前，对最终要发送的消息链进行处理时
  - **用途**: 格式化消息、添加额外信息、安全检查等
  - **定义位置**: [`star_handler.py:445`](astrbot/core/star/register/star_handler.py:445)

- **`@filter.after_message_sent()`**
  - **触发时机**: 在消息成功发送到平台之后
  - **用途**: 记录发送状态、执行清理操作、更新统计信息等
  - **定义位置**: [`star_handler.py:457`](astrbot/core/star/register/star_handler.py:457)

## 使用示例

```python
from astrbot.api.event.filter import on_llm_response, on_llm_request
from astrbot.api.provider import ProviderRequest, LLMResponse

class MyPlugin:
    @on_llm_request()
    async def modify_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """在LLM请求前添加系统提示"""
        request.system_prompt += "你是一个专业的助手，请用中文回答。"

    @on_llm_response()
    async def analyze_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """分析LLM响应并记录日志"""
        if "错误" in response.content:
            logger.warning(f"LLM响应包含错误内容: {response.content}")

    @after_message_sent()
    async def track_message_delivery(self, event: AstrMessageEvent):
        """消息发送后更新统计"""
        self.message_count += 1
```

## 最佳实践

1. **保持处理函数轻量**: 事件处理函数应该快速执行，避免阻塞事件流
2. **错误处理**: 在事件处理中妥善处理异常，避免影响其他事件处理器
3. **幂等性**: 确保事件处理器可以安全地多次执行
4. **性能监控**: 对于耗时操作，建议添加性能监控和日志记录

## 注意事项

- 事件处理器的执行顺序可能影响最终结果
- 多个插件可能注册相同事件的处理函数，需要注意兼容性
- 某些事件可能会被其他处理器中断（通过 `event.stop_event()`）

通过合理利用这些事件触发时机，可以构建出功能强大且灵活的机器人应用。