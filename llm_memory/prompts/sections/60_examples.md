## 唯一完整示例

整份提示词只保留这一份完整示例。中途章节不再重复局部示例、正反例或字段小样例。

```json
{
  "soul_state_code": "0110",
  "feedback_data": {
    "useful_memory_ids": ["m0", "m2"],
    "memory_actions": [
      {
        "action": "create",
        "memory": {
          "type": "knowledge",
          "judgment": "小明（123456）是原神深度玩家，关注角色配队和深渊攻略。",
          "reasoning": "小明在多轮对话中反复讨论原神角色机制、圣遗物搭配和深渊阵容。",
          "tags": ["小明", "123456", "事实属性"]
        }
      },
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
        "source_memory_ids": ["m1", "m3"],
        "memory": {
          "type": "event",
          "judgment": "群里讨论过用「分步确认」方式推进复杂任务，大家普遍认同这种做法。",
          "reasoning": "多轮对话中多位群友都体现出偏好阶段性确认与逐步推进。",
          "tags": ["任务偏好", "执行方式", "群共识"]
        }
      },
      {
        "action": "updata",
        "source_memory_ids": ["m4"],
        "memory": {
          "type": "knowledge",
          "judgment": "小明（123456）要求所有记忆反馈统一采用 memory_actions 协议。",
          "reasoning": "小明本轮明确否定旧结构，并确认只保留新的动作式协议。",
          "tags": ["小明", "123456", "事实属性"]
        }
      }
    ]
  }
}
```
