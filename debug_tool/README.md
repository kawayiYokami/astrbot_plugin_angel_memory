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

- `🔍 混合检索`：查询向量记忆/笔记
- `📖 浏览记忆`：浏览向量记忆集合
- `🧾 浏览Simple记忆`：浏览 `simple_memory.db`
- `🔄 导入导出`：导出/导入 Simple 记忆 JSON
- `📂 浏览笔记`：查看 `raw/` 下的 Markdown 笔记

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

## 5. 导入导出说明

进入 `🔄 导入导出` 模式后：

- 导出：
  - 可选填写 `scope`（留空导出全部）
  - 点击“生成导出文件”后可下载 JSON
- 导入：
  - 上传 JSON 文件
  - 点击“执行导入”
  - 按 `judgment` 执行去重 upsert（较新 `created_at` 覆盖较旧）

支持两种 JSON 格式：

- 对象格式：`{"memories": [...]}`
- 数组格式：`[...]`
