# Debug Tool 使用说明

## 1. 安装依赖

在插件根目录执行：

```bash
pip install -r debug_tool/requirements.txt
```

## 2. 启动方式

在插件根目录执行：

```bash
streamlit run debug_tool/app.py
```

启动后访问：

- `http://localhost:8501`

## 3. 页面功能

左侧可切换模式：

- `📌 总览`：查看当前 provider、中央库路径、向量库路径、集合统计。
- `🧾 中央记忆浏览`：分页浏览 `memory_center/index/simple_memory.db`（中央真相源）。
- `🔖 全局Tags调试`：查看 `global_tags`、执行一次性 tags 命中检索调试。
- `🧭 memory_index 检索`：查询和浏览向量轻量索引集合 `memory_index`。
- `🗂️ 中央笔记索引`：分页浏览 `note_index_records + note_tag_rel`（路径/标题/tags）。
- `🧠 notes_index 检索`：查询和浏览笔记轻量向量索引集合 `notes_index`。
- `📝 note_recall 模拟`：按 `note_short_id + start_line/end_line` 模拟正文读取。
- `🔄 中央库导入导出`：导出中央库 JSON 快照、导入 JSON（按 `judgment` 幂等合并）。
- `🛠️ 维护状态`：查看 `maintenance_state.json` 与每日备份文件信息。
- `📂 浏览笔记文件`：浏览 `raw/` 下 Markdown 文件内容（文件视图）。

## 4. 常见提示

### `missing ScriptRunContext`

如果看到类似：

```text
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
```

通常是启动方式不对（例如直接 `python app.py`）或命令里包含了占位符文本。

请确认你使用的是：

```bash
streamlit run debug_tool/app.py
```

并且不要带 `"[ARGUMENTS]"` 这类占位符字符串。

## 5. 导入导出说明（中央库）

进入 `🔄 中央库导入导出` 模式后：

- 导出：
  - 点击“生成快照”后可下载 JSON。
- 导入：
  - 上传 JSON 文件
  - 点击“执行导入”
  - 按 `judgment` 执行幂等 upsert（较新 `created_at` 覆盖较旧）

支持两种 JSON 格式：

- 中央快照：`{"records": [...], "global_tags": [...], "memory_tag_rel": [...] }`
- 兼容格式：`{"memories": [...]}`
- 兼容格式：`[...]`
