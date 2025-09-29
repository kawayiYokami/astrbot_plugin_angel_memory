# AngelHeart 插件上下文注入指南

## 1. 概述

AngelHeart 插件通过向 `AstrMessageEvent` 事件对象中注入一个名为 `angelheart_context` 的 JSON 字符串，来向事件流中的其他插件共享其分析结果。这使得其他插件能够获取到丰富的上下文信息，包括最近的聊天记录、秘书AI的决策以及是否需要进行网络搜索的建议。

注入操作发生在 `@filter.on_llm_request()` 钩子触发时，确保了这些上下文信息仅用于本次请求，不会污染数据库中的永久聊天记录。

## 2. 如何使用注入的上下文

其他插件可以在任何能够访问 `AstrMessageEvent` 对象的地方（例如，通过 `@filter.on_llm_request()` 或 `@filter.on_llm_response()` 钩子）读取和使用这些信息。

### 2.1 读取步骤

1.  **检查属性是否存在**：首先检查 `event` 对象上是否存在 `angelheart_context` 属性。
2.  **解析 JSON**：如果存在，读取该属性的值（一个字符串），并使用 `json.loads()` 将其解析为 Python 字典。
3.  **访问数据**：从解析后的字典中提取所需的数据。

### 2.2 代码示例

```python
import json
from astrbot.api.event import AstrMessageEvent
import astrbot.api.event.filter as filter
from astrbot.api.provider import ProviderRequest

class MyCustomPlugin:

    @filter.on_llm_request(priority=50) # 优先级应低于 AngelHeart 的注入钩子
    async def use_injected_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        一个演示如何读取和使用 AngelHeart 插件注入上下文的示例。
        """
        if hasattr(event, 'angelheart_context') and event.angelheart_context:
            try:
                # 1. 解析 JSON 字符串
                context_data = json.loads(event.angelheart_context)

                # 2. 提取所需信息
                chat_records = context_data.get('chat_records', [])
                secretary_decision = context_data.get('secretary_decision', {})
                needs_search = context_data.get('needs_search', False)

                # 3. 根据上下文信息执行你的逻辑
                if secretary_decision.get('should_reply'):
                    print(f"AngelHeart 建议回复，策略为：{secretary_decision.get('reply_strategy')}")

                if needs_search:
                    print("AngelHeart 认为需要进行网络搜索。")

                if chat_records:
                    latest_user_message = next((msg for msg in reversed(chat_records) if msg.get('role') == 'user'), None)
                    if latest_user_message:
                        print(f"最近的用户消息是：{latest_user_message.get('content')}")

            except json.JSONDecodeError:
                print("解析 angelheart_context 失败，可能不是有效的 JSON。")
            except Exception as e:
                print(f"处理注入的上下文时发生错误: {e}")

```

## 3. 数据结构详解

`angelheart_context` 解析后是一个包含三个顶级键的字典：

| 顶级键 | 类型 | 描述 |
| :--- | :--- | :--- |
| `chat_records` | `List[Dict]` | 聊天记录列表，包含了用于分析的上下文消息。 |
| `secretary_decision` | `Dict` | 秘书AI的决策结果，包含了对当前对话的分析和建议。 |
| `needs_search` | `bool` | 一个布尔标志，指示是否建议进行网络搜索来补充信息。 |

---

### 3.1 `chat_records` 内部结构

列表中的每个元素都是一个代表单条消息的字典，其典型结构如下：

```json
{
  "role": "user",
  "content": [{"type": "text", "text": "你好"}],
  "sender_id": "12345",
  "sender_name": "张三",
  "timestamp": 1696110000.0
}
```

| 键 | 类型 | 描述 |
| :--- | :--- | :--- |
| `role` | `str` | 消息发送者的角色，通常是 `user` 或 `assistant`。 |
| `content` | `List[Dict]` 或 `str` | 消息内容。可以是标准的多模态列表，也可以是纯文本字符串。 |
| `sender_id` | `str` | 发送者的唯一标识符。 |
| `sender_name` | `str` | 发送者的昵称。 |
| `timestamp` | `float` | 消息发送的Unix时间戳。 |

---

### 3.2 `secretary_decision` 内部结构

这个字典的结构由 `SecretaryDecision` Pydantic 模型定义（详见 [`models/analysis_result.py`](models/analysis_result.py)），包含了秘书AI对当前对话的详细分析和行动建议。

| 键 | 类型 | 描述 |
| :--- | :--- | :--- |
| `should_reply` | `bool` | **核心字段**。指示 AngelHeart 是否认为应该介入并回复当前对话。 |
| `reply_strategy` | `str` | 建议的回复策略，例如：“缓和气氛”、“技术指导”、“表示共情”等。 |
| `topic` | `str` | 对当前对话核心主题的概括。 |
| `persona_name` | `str` | 建议本次回复使用的 AI 人格名称。 |
| `alias` | `str` | AI 在对话中使用的别名或昵称。 |
| `reply_target` | `str` | 建议回复的目标用户昵称或ID。 |
| `created_at` | `str` | 决策创建的UTC时间戳 (ISO 8601 格式)。 |
| `boundary_timestamp` | `float` | 本次分析所用消息快照的边界时间戳，用于内部状态推进。 |
| `needs_search` | `bool` | 是否建议在生成回复前，先进行网络搜索以获取事实信息或背景知识。 |
