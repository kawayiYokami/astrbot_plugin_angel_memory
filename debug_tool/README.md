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

## 3. 常见提示

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

## 4. 功能说明（当前版本）

- 纯向量检索（已移除 FlashRank 重排）
- 支持按 `memory_scope` 过滤记忆数据
  - 不筛选
  - 仅 public
  - scope + public
  - 仅 scope
