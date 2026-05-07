## 唯一完整示例

整份提示词只保留这一份完整示例。中途章节不再重复局部示例、正反例或字段小样例。

```json
{
  "soul_state_code": "0110",
  "feedback_data": {
    "useful_memory_ids": ["id1", "id2"],
    "memory_actions": [
      {
        "action": "create",
        "memory": {
          "type": "knowledge",
          "judgment": "小明（123456）正在持续开发 Angel Memory 插件的人物画像记忆能力。",
          "reasoning": "小明围绕短期记忆引用化、用户画像提取和关系图谱连续提出设计要求。",
          "tags": ["小明", "123456", "活跃项目"]
        }
      },
      {
        "action": "merge",
        "source_memory_ids": ["31", "44"],
        "memory": {
          "type": "skill",
          "judgment": "用户偏好把复杂任务拆成可验证的小步骤执行。",
          "reasoning": "多轮对话都体现出用户偏好阶段性确认与逐步推进。",
          "tags": ["任务偏好", "执行方式"]
        }
      },
      {
        "action": "updata",
        "source_memory_ids": ["52"],
        "memory": {
          "type": "knowledge",
          "judgment": "用户要求所有记忆反馈统一采用 memory_actions 协议。",
          "reasoning": "用户本轮明确否定旧结构，并确认只保留新的动作式协议。",
          "tags": ["memory_actions", "反馈协议", "结构约束"]
        }
      }
    ]
  }
}
```
