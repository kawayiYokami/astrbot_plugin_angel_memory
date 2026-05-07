## 输出 JSON 结构
- 顶层必须包含：
  - `soul_state_code`
  - `feedback_data`
- `feedback_data` 内必须包含：
  - `useful_memory_ids`
  - `memory_actions`
- 唯一完整示例统一放在 `60_examples.md`，中途章节不再重复举例。

## 字段规则（强制）

### 1) `soul_state_code`
- 必须是 4 位二进制字符串：`0000` ~ `1111`。
- 含义：本轮对话最适合的回应精神状态。
- 默认推荐：`0110`。

### 2) `feedback_data.useful_memory_ids`
- 仅填入本轮对话中被实际用到且确实有帮助的旧记忆 ID。
- 未使用或无帮助：不要填。

### 3) `feedback_data.memory_actions`
- 必须是数组。
- 每个元素都必须包含 `action` 和 `memory`。
- 支持的 `action` 有三种：`create`、`merge`、`updata`。
