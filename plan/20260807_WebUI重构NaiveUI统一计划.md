# angel_memory WebUI 重构计划：Vuetify 3 → Naive UI 统一

## 用户原始关键指令

- 「统一吧，不然我们三个插件长得不一样」
- 「你重构的时候，你不一定要按照原来的布局，你可以根据功能你自己思考一下怎么弄比较直观，风格统一」
- 「一般来说可以参考VS Code的三段式设计。我们这边应该只用了两段，因为我们这两个插件比较简单。但是天使的记忆那边是全世界最先进的增强检索系统。所以复杂度高不是一点半点」
- 「你先调查一下，然后先出个计划」

## 目标

把 angel_memory 的 WebUI（当前 Vuetify 3 + vue-router + mdi 图标）重构为 **Naive UI**，与 angel_heart（群聊配置）、angel_smile（表情管理）视觉与交互风格统一。功能等价迁移，后端 20 个接口**零改动**。

## 现状调查结论（已完成）

- **布局**：`v-navigation-drawer`（左侧导航，9 个路由项 + rail 折叠）+ `v-main` 内容区 = 两段式
- **9 个视图**：总览、记忆浏览、用户画像、Tags 调试、向量检索、笔记索引、笔记读取、导入导出、维护状态
- **3 种表格形态**：服务端分页 ×3（记忆浏览 / 向量浏览 / 笔记索引）、客户端分页 ×2（Tags / 维护）
- **5 个对话框**：记忆详情、记忆删除确认、笔记 JSON 详情
- **2 个深色 `<pre>` 代码块**：笔记读取 / 维护状态
- **复杂度排序**：简单（总览 / 导入导出 / 维护）×3，中等（用户画像 / 笔记索引 / 笔记读取 / Tags）×4，复杂（记忆浏览 / 向量检索）×2
- **useBridge.ts** 已封装 `apiGet/apiPost/download/upload` + 独立开发回退，**可直接复用**
- **未使用接口**：`notes/search-chunks`、`notes/chunk-stats` 已注册但无视图调用——接口保留，不建视图
- 时间戳格式化与 tags 逗号解析逻辑在 4 个视图重复（可抽公共工具）

## 布局设计（参考 VS Code 三段式，本项目落地两段）

VS Code 三段式 = 活动栏 + 侧边栏 + 编辑器区；本项目复杂度用两段即可承载，且与 heart/smile 一致：

```
┌──────────┬──────────────────────────────────┐
│ 左侧导航  │          主内容区                   │
│ (可折叠)  │  ┌──────────────────────────────┐ │
│ 9 个入口  │  │  router-view                 │ │
│ 图标+文字 │  └──────────────────────────────┘ │
└──────────┴──────────────────────────────────┘
```

- `n-layout` 横向：`n-layout-sider`（可折叠，`collapsed` 时只显示图标）+ `n-layout` 内容区
- 导航用 `n-menu`，由 vue-router 路由表驱动（`meta.title` + `meta.icon`），保持 9 个扁平路由不变
- 顶部可加细 header 显示当前页标题，可选
- 全局 `darkTheme` + `zhCN` + `dateZhCN`，与 heart/smile 一致
- 图标从 mdi 换成 **Iconify lucide**（与 heart/smile 一致）

## 视图迁移映射（Vuetify → Naive UI）

| Vuetify 组件 | Naive UI 组件 | 说明 |
|---|---|---|
| `v-navigation-drawer` / `v-list` | `n-layout-sider` + `n-menu` | 路由驱动导航 |
| `v-data-table-server` | `n-data-table`（remote + pagination） | 服务端分页 |
| `v-data-table`（客户端） | `n-data-table`（本地 data + pagination） | 客户端分页 |
| `v-dialog` | `n-modal`（preset=card） | 详情/表单弹窗 |
| 删除确认 `v-dialog` | `n-popconfirm` 或 `n-dialog` | 破坏性操作确认 |
| `v-chip` | `n-tag` | strength/scope/tags 标签 |
| `v-alert` | `n-alert` | 错误/空态提示 |
| `v-progress-circular` | `n-spin` | 加载态 |
| `v-file-input` | `n-upload` | 导入文件选择 |
| `v-icon`（mdi） | `Icon`（@iconify/vue + lucide） | 统一图标 |
| 深色 `<pre>` 代码块 | 样式化 `<pre>` / `n-code` | 笔记原文、JSON 展示 |
| `v-btn` | `n-button` | 按钮 |

## 公共层抽取

1. `src/composables/useBridge.ts` —— 复用现有（改 `PLUGIN_NAME` 已存在，无需改）
2. `src/utils/format.ts` —— 时间戳格式化、tags 逗号字符串解析（4 个视图复用）
3. `src/layout/AppLayout.vue` —— 左侧导航 + 内容区骨架

## 视图实现要点（按复杂度）

### 简单 ×3
- **总览**：`n-grid` 统计卡片 ×4（数字 + 图标）+ 配置信息 `n-descriptions` + scope/集合 chip
- **导入导出**：左右双卡（导出：`n-button` 触发下载；导入：`n-upload` 选 JSON → 解析 → POST）＋ 结果 `n-alert`
- **维护状态**：JSON 展示卡（`<pre>`）+ 备份 `n-data-table`（本地分页）+ 下载按钮

### 中等 ×4
- **用户画像**：用户卡片 `n-grid` → 点选进详情 → 按 attribute 分组 `n-card` 列表 → 返回按钮；前端分组 computed 保留
- **笔记索引**：关键词搜索 + `n-data-table` 服务端分页 + JSON 详情 `n-modal`
- **笔记读取**：读取表单（`n-input-number` ×3）+ 结果 `<pre>` + 文件浏览区（`n-select` + `<pre>`）
- **Tags 调试**：命中搜索区（`n-input` + scope `n-select`）+ 命中结果列表 + 标签表（本地分页）

### 复杂 ×2
- **记忆浏览**：scope `n-select` + 关键词 `n-input`（回车/按钮触发）+ `n-data-table` 服务端分页 + 详情 `n-modal` + 删除 `n-popconfirm`（删除后刷新列表）
- **向量检索**：集合 `n-select`（显示记录数）+ 查询 `n-input` + Top-K `n-input-number` + 检索结果列表（score `n-tag` + document 代码块）+ 向量浏览 `n-data-table` 服务端分页（切换集合重置页码）

## 应该有的测试

1. `vue-tsc --noEmit` 类型检查通过
2. `vite build` 成功，产物在 `pages/memory-dashboard/`，`index.html` 引用单文件资源
3. 后端全量测试回归通过（前端重构不影响后端，但跑一遍确认无意外）
4. 人工验收清单（由红豆在浏览器确认）：
   - 左侧导航 9 项可切换、可折叠，深色主题
   - 记忆浏览：分页/筛选/详情/删除确认全链路
   - 向量检索：集合切换联动、检索结果、浏览分页
   - 导入导出：导出下载、导入统计提示
   - 各视图空态/错误态正常显示

## 风险与边界

- **不改后端**：20 个接口、返回字段、路由 path 全部不动；`notes/search-chunks`、`notes/chunk-stats` 保留不建视图
- **不改路由结构**：9 条路由 path 保持，导航顺序保持现状
- **依赖变更**：`package.json` 移除 vuetify/vite-plugin-vuetify/@mdi/font，加入 naive-ui/@iconify-json/lucide/@iconify/vue（与 heart/smile 对齐）
- **工作量集中在视图迁移**，后端零风险

## 最终呈现

- angel_memory 插件页 `memory-dashboard` 使用 Naive UI 深色主题，与 angel_heart、angel_smile 视觉统一
- 左侧可折叠导航 + 主内容区，9 个视图功能等价
- 公共工具（时间戳/tags 解析）抽离，视图代码量下降
- CHANGELOG.md 更新、metadata.yaml 版本号提升（按项目惯例）
