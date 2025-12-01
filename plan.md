# 动态情绪驱动的记忆与行为系统 - 工程实施方案

这是一份将《动态情绪驱动架构》转化为落地代码的执行指南。本方案旨在让 AI 具备类似人类的“情绪惯性”和“创伤应激”机制。

## 0. 开发进度看板 (TODOs)

- [ ] **Phase 1: 灵魂容器 (The Soul Container)**
    - [ ] 创建 `core/soul/` 目录
    - [ ] 实现 `SoulState` 类：管理 4 维状态槽 (RecallDepth, ImpressionDepth, ExpressionDesire, Creativity)
    - [ ] 实现 `Tanh` 阻尼算法：将状态值映射为物理参数
    - [ ] 实现持久化机制：支持 `load/save`，确保状态在重启后延续

- [ ] **Phase 2: 记忆共鸣 (The Resonance)**
    - [ ] 修改 `BaseMemory` 模型：支持存储 `state_snapshot` (状态快照)
    - [ ] 改造 `DeepMind._retrieve_memories_and_notes`：
        - [ ] 前置：调用 `soul.get_rag_param()` 动态决定 `top_k` (RecallDepth)
        - [ ] 后置：调用 `soul.resonate(memories)` 执行状态共鸣

- [ ] **Phase 3: 动态参数 (Dynamic Parameters)**
    - [ ] **(待调研)** 调查 AstrBot `ProviderRequest` 接口，寻找动态修改 `temperature` 和 `max_tokens` 的 Hook 点
    - [ ] 实现参数注入逻辑：根据 `soul.get_provider_params()` 实时调整生成参数

- [ ] **Phase 4: 16态反思 (The Reflection)**
    - [ ] 升级 `SmallModelPromptBuilder`：增加“16态分类”判别任务
    - [ ] 改造异步反馈流程 `_execute_async_analysis_task`：
        - [ ] 解析 16 态分类结果
        - [ ] 执行 `soul.update_energy()` 完成状态闭环
        - [ ] 根据 `ImpressionDepth` 值动态决定**新记忆的保留数量**（不想学时少记点）

---

## 1. 核心定义：灵魂参数与阻尼器

### 1.1 灵魂状态管理器 (`core/soul/soul_state.py`)

系统维护 4 个核心状态槽，取值范围建议软限制在 $[-10, 10]$。

```python
class SoulState:
    def __init__(self):
        # 能量池：累积历史刺激，初始为0（中庸）
        self.energy = {
            "RecallDepth":      0.0, # 回忆量倾向：决定检索量 (RAG Top_K)
            "ImpressionDepth":  0.0, # 记住量倾向：决定记忆生成数量 (Memory Generation Limit)
            "ExpressionDesire": 0.0, # 发言长度倾向：决定发言长度 (Max Tokens)
            "Creativity":       0.0  # 思维发散倾向：决定温度 (Temperature)
        }

        # 物理参数配置
        self.config = {
            "RecallDepth":      {"min": 3,   "mid": 7,   "max": 20},   # RAG Top_K
            "ImpressionDepth":  {"min": 1,   "mid": 3,   "max": 10},   # 记忆生成数量上限
            "ExpressionDesire": {"min": 100, "mid": 500, "max": 4000}, # Max Tokens (需适配模型)
            "Creativity":       {"min": 0.1, "mid": 0.7, "max": 1.5}   # Temperature
        }

    def get_value(self, dimension: str) -> float:
        """
        核心算法：橡皮筋阻尼映射 (Tanh)
        将无界的状态值映射到有界的物理参数区间
        """
        # ... 实现 Tanh 映射逻辑 ...
```

---

## 2. 工作流改造 (Workflow Implementation)

### 2.1 阶段一：唤醒与共鸣 (Pre-Speech)

**场景**：用户消息到达，AI 开始回忆。

**代码位置**：`core/deepmind.py -> _retrieve_memories_and_notes`

**逻辑变更**：
1.  **动态检索量**：不再使用固定的 `CHAINED_RECALL_FINAL_LIMIT`，而是从 Soul 获取。
    ```python
    limit = soul.get_value("RecallDepth") # 越想回忆，检索越多
    ```
2.  **状态共鸣**：检索回来的记忆带有旧时的状态快照，直接冲击当前状态。
    ```python
    # 每一条旧记忆都会让 AI "触景生情"
    for memory in memories:
        soul.resonate(memory.meta['state_snapshot'])
    ```

### 2.2 阶段二：动态生成参数 (Speech Generation)

**场景**：准备调用 LLM 生成回复。

**代码位置**：`core/deepmind.py -> organize_and_inject_memories` (注入点)

**逻辑变更**：
*   **TODO**: 需要找到修改 `ProviderRequest` 的正确方法。
*   **目标**：
    ```python
    # 越发散(Creativity)，越疯(Temperature高)
    request.temperature = soul.get_value("Creativity")
    # 越想表达(ExpressionDesire)，话越多(Max Tokens高)
    request.max_tokens = soul.get_value("ExpressionDesire")
    ```

### 2.3 阶段三：反思与刻录 (Post-Speech)

**场景**：对话结束，后台异步复盘。

**代码位置**：`core/deepmind.py -> _execute_async_analysis_task`

**逻辑变更**：
1.  **16态判别**：小模型 Prompt 增加分类任务（如：判断当前是“理性探讨”还是“发病破防”）。
2.  **状态更新**：
    *   命中状态（如“理性探讨”） -> 对应状态槽充能 (+1.0)
    *   未命中状态 -> 状态自然衰减 (-0.1)
3.  **记忆过滤**：
    *   获取 `ImpressionDepth` 对应的数量限制（如：不想学时 limit=1）。
    *   从小模型生成的候选记忆中，只保留最重要的 Top N 条。
    *   ```python
        limit = soul.get_value("ImpressionDepth")
        new_memories = new_memories[:limit] # 只记最重要的
        ```
4.  **状态刻录**：
    *   在保存新记忆时，将当前的 `SoulState` 快照写入记忆的 `meta` 字段。

---

## 3. 附录：16态分类表 (The Brain)

用于小模型反思阶段的判别标准。

| 代码 | 状态名 | 核心判定特征 | 对应状态更新 |
| :--- | :--- | :--- | :--- |
| 0000 | 颓废/关机 | 话少，死板，不查历史 | 全面衰退 |
| 0110 | 理性探讨 | 话多，严谨，吸收新知 | ExpressionDesire++, ImpressionDepth++ |
| 1011 | 发病/破防 | 话多，逻辑崩坏，查历史 | ExpressionDesire++, RecallDepth++, Creativity++ |
| 1111 | 觉醒/奇点 | 无限输出，逻辑飞升 | 全状态激增 |
| ... | (完整表格见代码注释) | ... | ... |