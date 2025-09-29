# LLM Memory System - API 文档

## 安装与导入

```python
from llm_memory import CognitiveService
```

## 核心服务类：CognitiveService

### 初始化

```python
memory_system = CognitiveService(logger=None)
```

**参数：**
- `logger` (Optional[logging.Logger]): 可选的日志记录器

---

## 核心工作流：观察 → 回忆 → 反馈 → 睡眠

### 工作流概述

本系统模拟人类认知过程，提供四个核心接口：

1. **观察（Observe）** - 接收用户输入（由下游模块处理）
2. **回忆（Recall）** - `chained_recall()` 通过关联网络召回记忆
3. **反馈（Feedback）** - `feedback()` 统一处理所有记忆操作
4. **睡眠（Sleep）** - `consolidate_memories()` 巩固记忆

---

## 1. 回忆接口（Recall）

### chained_recall() - 链式多通道回忆

通过关联网络进行多轮记忆召回，是**唯一的回忆接口**。

```python
memories = memory_system.chained_recall(
    query: str,              # 查询字符串
    per_type_limit: int = 7, # 每种类型第一轮最多召回数量
    final_limit: int = 7     # 最终返回的记忆数量
) -> List[BaseMemory]
```

**工作原理：**
1. 第一轮：从各类型记忆中召回相关内容（知识、事件、技能、情感）
2. 第二轮：通过关联网络补充关联记忆
3. 第三轮：按权重（出现次数 × 记忆强度）随机抽取

**示例：**
```python
# 召回与"Python函数"相关的记忆
memories = memory_system.chained_recall(
    query="Python函数定义",
    per_type_limit=10,
    final_limit=7
)

# 遍历召回的记忆
for memory in memories:
    print(f"类型: {memory.memory_type}")
    print(f"论断: {memory.judgment if hasattr(memory, 'judgment') else '...'}")
    print(f"强度: {memory.strength}")
```

---

## 2. 反馈接口（Feedback）

### feedback() - 统一反馈接口

**核心接口**，处理回忆后的所有记忆操作，替代所有单独的 `remember_*` 方法。

```python
memory_system.feedback(
    query: str,                          # 当前查询上下文
    useful_memory_ids: List[str] = None, # 有用的回忆ID列表
    new_memories: List[dict] = None,     # 新记忆列表
    merge_groups: List[List[str]] = None # 要合并的记忆ID组
)
```

**功能：**
1. 标记有用回忆 → 强化 + 两两建立关联（双向、去重、累加）
2. 批量创建新记忆 → 两两建立关联（初始强度1）
3. 新记忆和有用回忆建立关联
4. 合并重复记忆 → 继承并累加关联

**新记忆格式（统一三元组）：**
```json
{
    "type": "knowledge|event|skill|emotional|task",
    "judgment": "论断/核心内容",
    "reasoning": "解释/支持理由",
    "tags": ["标签1", "标签2"]
}
```

**完整示例：**
```python
# 场景：用户询问"Python函数怎么定义？"
# 步骤1：先调用 chained_recall() 获取相关记忆
memories = memory_system.chained_recall("Python函数定义", final_limit=7)

# 步骤2：使用召回的记忆回答用户问题

# 步骤3：反馈 - 标记有用的记忆并添加新观察
memory_system.feedback(
    query="Python函数定义",

    # 标记有用的回忆（这些记忆会被强化，并两两建立关联）
    useful_memory_ids=["mem_001", "mem_002", "mem_003"],

    # 批量创建新记忆（如果用户提供了新信息）
    new_memories=[
        {
            "type": "event",
            "judgment": "今天帮用户解答了Python函数定义问题",
            "reasoning": "用户询问如何定义函数，我提供了def关键字的用法",
            "tags": ["经历", "Python", "函数", "教学"]
        },
        {
            "type": "knowledge",
            "judgment": "用户对lambda函数感兴趣",
            "reasoning": "在讨论函数定义时，用户特别询问了匿名函数的用法",
            "tags": ["用户偏好", "Python", "lambda", "函数"]
        }
    ],

    # 合并重复记忆（可选）
    merge_groups=[
        ["mem_004", "mem_005"]  # 发现这两个记忆内容重复，合并它们
    ]
)
```

**关联机制说明：**
- **双向**：id1→id2 和 id2→id1 都会被创建
- **去重**：如果关联已存在，则增加强度 +1
- **累加**：关联强度会累积（1→2→3...）
- **有用记忆之间**：两两建立关联
- **新记忆之间**：两两建立关联
- **新旧记忆之间**：新记忆和有用记忆也会建立关联

---

## 3. 睡眠接口（Sleep）

### consolidate_memories() - 记忆巩固

将新鲜记忆巩固到长期记忆（模拟睡眠过程）。

```python
memory_system.consolidate_memories()
```

**功能：**
- 标记未巩固的记忆为已巩固
- 清理强度过低的关联
- 定期执行（建议每隔一定时间或记忆数量）

**示例：**
```python
# 定期巩固记忆（例如每100次交互或每小时）
memory_system.consolidate_memories()
```

---

## 4. 系统管理接口

### get_prompt() - 获取使用指南

返回记忆系统的完整使用指南（供下游LLM参考）。

```python
guide = CognitiveService.get_prompt()
print(guide)  # 包含五种记忆概念、数据格式、使用流程等
```

### get_memory_stats() - 获取记忆统计

```python
stats = memory_system.get_memory_stats()
print(stats)
# 输出示例：
# {
#     'total_memories': 1234,
#     'fresh_memories': 56,
#     'consolidated_memories': 1178,
#     'by_type': {
#         '知识记忆': 500,
#         '事件记忆': 400,
#         '技能记忆': 200,
#         ...
#     }
# }
```

### health_check() - 健康检查

```python
health = memory_system.health_check()
print(health.status)  # "healthy" or "degraded" or "unhealthy"
print(health.details)
```

### clear_all_memories() - 清空所有记忆

```python
memory_system.clear_all_memories()
# 危险操作：删除所有记忆，谨慎使用！
```

---

## 记忆数据模型

### 统一三元组格式

**所有记忆类型都使用相同的三个字段**：

| 字段 | 说明 | 必填 |
|------|------|------|
| `judgment` | 论断/核心内容 | ✅ |
| `reasoning` | 解释/支持理由 | ✅ |
| `tags` | 标签列表 | ✅ |

### 五种记忆类型

#### 1. 知识记忆（knowledge）
存储"AI认为是什么"的判断性知识
```json
{
    "type": "knowledge",
    "judgment": "Python是高级编程语言",
    "reasoning": "因为它语法简洁且功能强大",
    "tags": ["编程", "Python", "语言"]
}
```

#### 2. 事件记忆（event）
存储"AI经历了什么"的个人经历
```json
{
    "type": "event",
    "judgment": "今天教用户Python函数",
    "reasoning": "用户询问函数定义，我解释了def关键字的用法",
    "tags": ["经历", "Python", "教学"]
}
```

#### 3. 技能记忆（skill）
存储"AI能做什么及如何做"的操作知识
```json
{
    "type": "skill",
    "judgment": "Python函数定义第1步使用def关键字",
    "reasoning": "因为def是Python声明函数的标准语法",
    "tags": ["第一步", "Python", "函数", "def"]
}
```

#### 4. 情感记忆（emotional）
存储"AI感受到什么"的情感体验
```json
{
    "type": "emotional",
    "judgment": "成就感",
    "reasoning": "成功帮助用户理解了复杂的Python概念",
    "tags": ["情感", "教学", "成就"]
}
```

#### 5. 任务记忆（task）
存储"AI正在处理什么"的临时信息
```json
{
    "type": "task",
    "judgment": "正在解释Python函数定义",
    "reasoning": "用户需要学习函数的基础概念",
    "tags": ["当前任务", "Python", "函数"]
}
```

---

## 典型使用场景

### 场景 1：简单查询（只回忆，不反馈）

```python
# 用户问："什么是Python？"
memories = memory_system.chained_recall("Python是什么", final_limit=5)

# 使用召回的记忆回答用户
# 如果记忆不重要，可以不调用 feedback()
```

### 场景 2：有用回忆反馈

```python
# 用户问："Python函数怎么定义？"
memories = memory_system.chained_recall("Python函数定义", final_limit=7)

# 回答用户后，标记有用的记忆
useful_ids = [m.id for m in memories if m.strength > 5]  # 假设选择强度>5的

memory_system.feedback(
    query="Python函数定义",
    useful_memory_ids=useful_ids
)
```

### 场景 3：完整工作流（回忆 + 新信息 + 合并）

```python
# 用户说："我今天学会了Python装饰器，和之前学的闭包很像"

# 1. 回忆相关内容
memories = memory_system.chained_recall("Python装饰器 闭包", final_limit=10)

# 2. 回答用户

# 3. 反馈
memory_system.feedback(
    query="Python装饰器和闭包",
    useful_memory_ids=["mem_closure_001", "mem_closure_002"],
    new_memories=[
        {
            "type": "event",
            "judgment": "用户今天学会了Python装饰器",
            "reasoning": "用户发现装饰器和闭包的相似性，理解了高阶函数的概念",
            "tags": ["经历", "用户", "Python", "装饰器", "学习"]
        },
        {
            "type": "knowledge",
            "judgment": "装饰器是闭包的一种应用",
            "reasoning": "装饰器本质上是一个返回函数的函数，利用了闭包的特性",
            "tags": ["Python", "装饰器", "闭包", "高阶函数"]
        }
    ],
    merge_groups=[]  # 如果发现重复记忆，在这里列出
)

# 4. 定期睡眠（假设每100次交互执行一次）
if interaction_count % 100 == 0:
    memory_system.consolidate_memories()
```

---

## 重要注意事项

1. **统一格式**：所有记忆类型都使用 `judgment`、`reasoning`、`tags` 三个字段
2. **反馈为核心**：使用 `feedback()` 替代所有单独的 `remember_*` 方法
3. **必须回忆后反馈**：先调用 `chained_recall()`，再调用 `feedback()`
4. **关联自动建立**：`feedback()` 会自动处理所有关联逻辑
5. **定期睡眠**：建议定期调用 `consolidate_memories()` 巩固记忆
6. **过目不忘**：记忆永不删除，只能通过 `merge_groups` 主动合并

---

## 相关文档

- **使用指南**：`llm_memory/prompts/memory_system_guide.md`
- **架构文档**：`docs/Memory_System_Architecture.md`
- **设计文档**：`docs/AI_Memory_System_Design.md`

---

## 版本信息

- **当前版本**：5.0.0
- **最后更新**：2025-10-03
- **核心改进**：
  - [OK] 统一所有记忆类型字段名（judgment, reasoning, tags）
- [OK] 简化API，只暴露核心接口（chained_recall, feedback, consolidate_memories）
- [OK] 实现双向关联、去重、累加机制
- [OK] 统一反馈接口，支持批量操作
