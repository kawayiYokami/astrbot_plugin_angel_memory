## 动作语义

### `create`
- 用于新增一条全新记忆。
- 必须包含：
  - `action`
  - `memory`
- `create` 不允许携带 `source_memory_ids`。

### `merge`
- 用于更新、纠正、去重、合并相似记忆。
- `merge` 必须同时包含 `source_memory_ids` 和 `memory`。
- `source_memory_ids` 可以是一条，也可以是多条。
- `merge` 的本质是：删除多条旧记忆，新增一条最终新记忆。

### `updata`
- 用于旧记忆与当前事实冲突后的修正更新。
- 结构与 `merge` 相同，也必须携带 `source_memory_ids` 和 `memory`。
- `updata` 的 `source_memory_ids` 必须且只能有 1 个。
- `updata` 的本质是：删除 1 条旧记忆，新增 1 条最终新记忆。
- `updata` 和 `merge` 在执行路径上完全一致，区别只在语义：
  - `merge` 更适合相似记忆合并、去重、收敛
  - `updata` 更适合用户纠正旧信息、事实发生变化、旧记忆被新事实覆盖

## 更新规则（非常重要）

- 合并相似记忆是非常重要的更新动作。
- 只要发现旧记忆重复、表述过时、内容冲突、信息需要收敛，优先使用 `merge`。
- 只要发现旧记忆与当前事实冲突、需要明确改正，优先使用 `updata`。
- 不能把本该更新的内容错误地写成两个 `create`。
- 不能只给 `source_memory_ids` 而不给新的 `memory`。
- 不能只写新的 `memory` 却不声明它替换了哪些旧记忆。
