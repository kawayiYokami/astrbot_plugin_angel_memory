# extra_user_content_parts 使用说明

## 概述

`extra_user_content_parts` 是 AstrBot 中用于在用户消息后添加额外内容块的参数。它允许插件在发送给 LLM 的用户消息中插入额外的文本、图片等内容，常用于添加系统提醒、引用消息、图片描述等。

## 数据结构

```python
extra_user_content_parts: list[ContentPart]
```

`ContentPart` 是一个基类，支持以下类型：

- **TextPart**: 文本内容块
- **ImageURLPart**: 图片 URL 内容块
- **ThinkPart**: 思考内容块（用于思维链）

## ContentPart 类型

### 1. TextPart（文本块）

```python
from astrbot.core.agent.message import TextPart

text_part = TextPart(text="这是额外的文本内容")
# 序列化后: {'type': 'text', 'text': '这是额外的文本内容'}
```

### 2. ImageURLPart（图片块）

```python
from astrbot.core.agent.message import ImageURLPart

image_part = ImageURLPart(image_url={"url": "http://example.com/image.jpg"})
# 序列化后: {'type': 'image_url', 'image_url': {'url': 'http://example.com/image.jpg', 'id': None}}
```

### 3. ThinkPart（思考块）

```python
from astrbot.core.agent.message import ThinkPart

think_part = ThinkPart(think="这是思考内容")
# 序列化后: {'type': 'think', 'think': '这是思考内容', 'encrypted': None}
```

## 主要使用场景

### 1. 添加系统提醒

在用户消息后添加系统级别的提醒信息：

```python
req.extra_user_content_parts.append(
    TextPart(text="<system_reminder>当前时间: 2024-01-01 12:00:00</system_reminder>")
)
```

### 2. 添加引用消息

在用户消息后添加被引用的消息内容：

```python
quoted_text = f"<Quoted Message>\n引用的消息内容\n</Quoted Message>"
req.extra_user_content_parts.append(TextPart(text=quoted_text))
```

### 3. 添加图片描述

当用户发送图片时，自动添加图片描述：

```python
req.extra_user_content_parts.append(
    TextPart(text=f"<image_caption>{caption}</image_caption>")
)
```

## 在插件中使用

### 示例 1：在 LLM 请求前添加自定义内容

```python
from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.agent.message import TextPart

class MyPlugin(star.Star):
    async def process_llm_request(self, event: AstrMessageEvent, req):
        """在 LLM 请求前添加自定义内容"""
        # 添加系统提醒
        req.extra_user_content_parts.append(
            TextPart(text="<system_reminder>这是一个测试提醒</system_reminder>")
        )

        # 添加自定义指令
        req.extra_user_content_parts.append(
            TextPart(text="<instruction>请用简洁的语言回答</instruction>")
        )
```

### 示例 2：添加图片内容

```python
from astrbot.api import star
from astrbot.core.agent.message import ImageURLPart

class MyPlugin(star.Star):
    async def process_llm_request(self, event: AstrMessageEvent, req):
        """在 LLM 请求前添加图片"""
        req.extra_user_content_parts.append(
            ImageURLPart(image_url={"url": "http://example.com/image.jpg"})
        )
```

### 示例 3：组合多个内容块

```python
from astrbot.api import star
from astrbot.core.agent.message import TextPart, ImageURLPart

class MyPlugin(star.Star):
    async def process_llm_request(self, event: AstrMessageEvent, req):
        """在 LLM 请求前添加多个内容块"""
        # 添加文本
        req.extra_user_content_parts.append(
            TextPart(text="<system_reminder>系统提醒</system_reminder>")
        )

        # 添加图片
        req.extra_user_content_parts.append(
            ImageURLPart(image_url={"url": "http://example.com/image.jpg"})
        )

        # 添加更多文本
        req.extra_user_content_parts.append(
            TextPart(text="<instruction>请分析这张图片</instruction>")
        )
```

## 注意事项

1. **添加顺序**：`extra_user_content_parts` 中的内容块会按照添加顺序追加到用户消息之后
2. **内容格式**：建议使用 XML 标签包裹内容（如 `<system_reminder>`），便于 LLM 理解
3. **图片处理**：当添加图片 URL 时，系统会自动解析并转换为适合 LLM 的格式
4. **空内容处理**：如果只有 `extra_user_content_parts` 而没有主文本和图片，系统会自动添加占位文本
5. **向后兼容**：如果没有 `extra_user_content_parts`，系统会使用简单的文本格式

## 相关文件

- `astrbot/core/provider/provider.py` - Provider 接口定义
- `astrbot/core/provider/entities.py` - ProviderRequest 定义
- `astrbot/core/agent/message.py` - ContentPart 及其子类定义
- `astrbot/builtin_stars/astrbot/process_llm_request.py` - 实际使用示例
- `astrbot/core/provider/sources/openai_source.py` - OpenAI 实现
- `astrbot/core/provider/sources/gemini_source.py` - Gemini 实现
- `astrbot/core/provider/sources/anthropic_source.py` - Anthropic 实现