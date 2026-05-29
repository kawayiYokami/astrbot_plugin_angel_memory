# WebUI Plugin Pages 计划

## 目标

将现有 debug_tool（Streamlit）的功能迁移到 AstrBot Plugin Pages 机制，使用 Vue 3 + Vite + Vuetify 3 构建美观的管理界面，用户无需额外启动服务即可在 AstrBot WebUI 中直接使用。

## 技术栈

- **前端**：Vue 3 + Vite + Vuetify 3 + TypeScript
- **后端**：AstrBot `register_web_api`（Quart 路由）
- **通信**：`window.AstrBotPluginPage` Bridge API（apiGet / apiPost）
- **路由**：Hash 路由（`createWebHashHistory`）
- **打包**：Vite `base: './'`，产物输出到 `pages/memory-dashboard/`

## 目录结构

```
astrbot_plugin_angel_memory/
├─ pages/
│  └─ memory-dashboard/          ← Vite build 产物（git tracked）
│     ├─ index.html
│     └─ assets/
│        ├─ index-[hash].js
│        └─ index-[hash].css
├─ webui/                         ← Vue 源码（开发用）
│  ├─ src/
│  │  ├─ App.vue
│  │  ├─ main.ts
│  │  ├─ router/
│  │  │  └─ index.ts
│  │  ├─ views/
│  │  │  ├─ OverviewView.vue       ← 总览
│  │  │  ├─ MemoryBrowseView.vue   ← 中央记忆浏览
│  │  │  ├─ TagsDebugView.vue      ← 全局Tags调试
│  │  │  ├─ VectorSearchView.vue   ← memory_index / notes_index 检索
│  │  │  ├─ NoteIndexView.vue      ← 中央笔记索引
│  │  │  ├─ NoteRecallView.vue     ← note_recall 模拟
│  │  │  ├─ ImportExportView.vue   ← 导入导出
│  │  │  └─ MaintenanceView.vue    ← 维护状态
│  │  ├─ composables/
│  │  │  └─ useBridge.ts           ← Bridge API 封装
│  │  └─ types/
│  │     └─ index.ts
│  ├─ package.json
│  ├─ vite.config.ts
│  └─ tsconfig.json
├─ web_api/                        ← 后端 API 路由模块
│  ├─ __init__.py
│  ├─ routes.py                    ← 路由注册入口
│  ├─ memory_api.py                ← 记忆相关 API
│  ├─ notes_api.py                 ← 笔记相关 API
│  ├─ tags_api.py                  ← 标签相关 API
│  └─ maintenance_api.py           ← 维护/导入导出 API
```

## 功能模块（对应 debug_tool）

### 1. 总览（Overview）
- 显示 provider 状态、路径信息、scope 列表
- 记忆/标签/笔记计数统计
- 向量索引状态

### 2. 中央记忆浏览（Memory Browse）
- 分页浏览所有记忆
- 按 scope 过滤
- 按关键词搜索（judgment/reasoning/tags）
- 查看记忆详情（元数据展开）
- 删除记忆（带确认）

### 3. 全局 Tags 调试
- 输入查询文本，展示命中的 tags 和对应记忆
- 浏览全局 tags 列表（含引用计数）
- 按关键词筛选 tags

### 4. 向量索引检索（memory_index / notes_index）
- 输入查询文本，执行向量检索
- 展示 Top-K 结果及相似度分数
- 原始浏览（分页）

### 5. 中央笔记索引
- 分页浏览笔记索引记录
- 按关键词搜索（路径/标题/tags）
- 展示笔记详情

### 6. note_recall 模拟
- 输入 note_short_id + 行范围
- 展示读取到的笔记内容

### 7. 导入导出
- 导出中央记忆快照（JSON 下载）
- 导入 JSON 文件（上传）

### 8. 维护状态
- 展示 maintenance_state.json
- 列出备份文件

## 后端 API 设计

所有路由前缀：`/astrbot_plugin_angel_memory/`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/overview` | 总览统计 |
| GET | `/memories` | 分页浏览记忆（query: scope, keyword, page, page_size） |
| DELETE | `/memories/<id>` | 删除记忆 |
| GET | `/tags` | 全局标签列表（query: keyword, limit, offset） |
| POST | `/tags/hit-search` | 标签命中搜索（body: query, scope, limit） |
| GET | `/vector/search` | 向量检索（query: collection, text, top_k） |
| GET | `/vector/browse` | 向量浏览（query: collection, page, page_size） |
| GET | `/notes` | 笔记索引浏览（query: keyword, page, page_size） |
| POST | `/notes/recall` | note_recall 模拟（body: note_short_id, start_line, end_line） |
| GET | `/export` | 导出快照 |
| POST | `/import` | 导入快照 |
| GET | `/maintenance` | 维护状态 |

## 实现步骤

### Phase 1：后端 API 骨架
1. 创建 `web_api/` 模块
2. 在 `main.py` 的 `__init__` 中注册所有 web API 路由
3. 实现各 API（复用现有 `memory_sql_manager` 和 `plugin_context` 组件）

### Phase 2：前端骨架
1. 初始化 Vue 3 + Vite + Vuetify 3 项目
2. 配置 hash 路由、Bridge API 封装
3. 实现侧边栏导航 + 各页面基础布局

### Phase 3：功能实现
1. 逐页面实现前端交互
2. 联调后端 API
3. 处理加载状态、错误提示、空状态

### Phase 4：打包与集成
1. Vite build 输出到 `pages/memory-dashboard/`
2. 测试 Plugin Pages 加载
3. 清理 debug_tool 中的重复功能说明

## 注意事项

- Bridge API 的 `apiGet`/`apiPost` endpoint 不能以 `/` 开头
- 前端资源必须用相对路径，AstrBot 自动处理 URL 重写
- 后端路由注册时路径必须包含插件名前缀
- 向量检索依赖 embedding provider 可用性，API 需优雅降级
- 导入导出使用 Bridge 的 `upload`/`download` 方法
