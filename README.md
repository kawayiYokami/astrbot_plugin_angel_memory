# Angel Memory - AstrBot 记忆系统插件

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-Plugin-green)](https://github.com/kawayiYokami/astrbot)

> 为 AI 装上"记忆大脑"，让对话更连贯、更智能

## 📖 概述

Angel Memory 是一个为 AstrBot 设计的记忆系统插件，实现了完整的认知工作流：观察→回忆→反馈→睡眠。通过双层认知架构（意识层/潜意识层）和短期记忆管理，让 AI 能够记住对话历史、学习新知识，并在后续对话中灵活运用这些记忆。

## ✨ 核心特性

- 🧠 **双层认知架构**: 意识层负责实时处理，潜意识层负责长期记忆管理
- 💭 **智能记忆召回**: 基于关联网络的链式回忆机制，精准提取相关记忆
- 🔄 **短期记忆管理**: 会话级别的短期记忆，支持FIFO队列和容量限制
- 😴 **定期睡眠**: 自动记忆巩固和优化，保持记忆系统健康
- 📊 **记忆反馈**: 智能分析记忆有用性，动态优化记忆网络

## 🏗️ 架构设计

### 核心工作流

1. **观察阶段** (`@filter.event_message_type`)
   - 事件到达时立即进行链式回忆
   - 将召回的记忆注入到 `event.angelmemory_context`

2. **回忆阶段** (`@filter.on_llm_request`)
   - 小模型分析记忆，筛选有用内容
   - 将有用记忆注入到系统提示词
   - 反馈记忆系统，更新记忆网络

3. **睡眠阶段** (定期执行)
   - 调用 `consolidate_memories()` 巩固记忆
   - 优化记忆关联网络

### 模块结构

```
core/
├── deepmind.py              # 核心记忆处理逻辑
├── config.py                # 配置管理
├── logger.py                # 日志管理
├── session_memory.py        # 短期记忆管理
└── utils/                   # 工具模块
    ├── memory_formatter.py   # 记忆格式化
    ├── memory_injector.py    # 记忆注入器
    └── small_model_prompt_builder.py  # 小模型提示构建
```

## 🚀 快速开始

### 前置要求

- Python 3.13+
- AstrBot 框架
- 可用的 LLM Provider

### 安装

1. 将插件目录放置到 AstrBot 的 `plugins` 目录中
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 在 AstrBot 配置中启用插件

### 配置

主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `min_message_length` | 5 | 触发记忆处理的最小消息长度 |
| `short_term_memory_capacity` | 1.0 | 短期记忆容量倍数 |
| `sleep_interval` | 3600 | 睡眠间隔（秒） |

```json
{
    "min_message_length": 5,
    "short_term_memory_capacity": 1.0,
    "sleep_interval": 3600
}
```

## 📝 记忆格式

### 显示格式（给小模型看）
```
=== 知识 ===
1. [id:a1b2c3]《三体》是刘慈欣写的科幻小说
   ——因为昨天用户询问科幻小说推荐时我推荐的

=== 事件 ===
2. [id:d4e5f6]昨天用户想要科幻小说推荐
   ——因为用户说想找一些硬科幻作品
```

### 引用格式（小模型输出）
```json
{
    "useful_memory_ids": ["a1b2c3", "d4e5f6"],
    "new_memories": [],
    "merge_groups": []
}
```

### 注入格式（给大模型看）
```
[相关记忆]

[知识]
《三体》是刘慈欣写的科幻小说
——因为昨天用户询问科幻小说推荐时我推荐的

[事件]
昨天用户想要科幻小说推荐
——因为用户说想找一些硬科幻作品
```

## 🔧 开发指南

### 添加新的记忆类型

1. 在 `llm_memory/models/data_models.py` 中定义新的记忆类型
2. 更新 `MemoryFormatter.MEMORY_TYPE_NAMES` 映射
3. 在 `session_memory.py` 中添加相应的容量配置

### 自定义记忆格式化

修改 `MemoryFormatter` 类中的格式化方法：

- `format_single_memory()` - 单条记忆格式化
- `format_memories_for_display()` - 显示格式化
- `format_memories_for_prompt()` - 提示词格式化

## 📊 性能特性

- **线程安全**: 短期记忆管理使用锁机制保证并发安全
- **容量限制**: 不同类型记忆有独立的容量限制
- **FIFO队列**: 短期记忆采用先进先出策略
- **智能缓存**: 避免重复的记忆处理
- **容错机制**: 记忆处理失败不影响主流程

## 🧪 测试

运行测试套件：

```bash
python -m pytest tests/ -v
```

测试覆盖：
- 短期记忆管理功能
- 记忆格式化功能
- 小模型提示构建
- 短ID系统
- 事件处理逻辑

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

1. 克隆项目
2. 安装开发依赖：`pip install -r requirements.txt`
3. 运行测试确保环境正常

## 📄 许可证

本项目采用 GPL-3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [AstrBot](https://github.com/kawayiYokami/astrbot) - 提供优秀的插件框架
- llm_memory 记忆系统 - 核心记忆处理能力
- 所有贡献者和用户

---

**让AI不再遗忘，让对话更有温度** 🧠✨