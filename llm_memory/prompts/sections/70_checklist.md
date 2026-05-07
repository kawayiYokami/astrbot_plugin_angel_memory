## 生成流程

1. 先确定 `soul_state_code`。
2. 从旧记忆中挑出真正有帮助的 `useful_memory_ids`。
3. 判断每条新信息应该是 `create`、`merge` 还是 `updata`。
4. 生成 `memory_actions`。
5. 最终自检：
   - JSON 是否合法
   - `memory_actions` 是否只使用 `create` / `merge` / `updata`
   - `create` 是否没有 `source_memory_ids`
   - `merge` / `updata` 是否同时包含 `source_memory_ids` 和 `memory`
   - `updata` 是否只包含 1 个 `source_memory_id`
   - 相似内容是否优先用了 `merge`
   - 冲突修正内容是否优先用了 `updata`

## 最终要求

现在开始分析对话，并仅输出 JSON。
