import random
from typing import List
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

# 导入必要的服务组件和模型
from ..llm_memory.service.cognitive_service import CognitiveService
from ..llm_memory.models.data_models import BaseMemory

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CoreMemoryRecallTool(FunctionTool):
    name: str = "core_memory_recall"
    description: str = "当你需要主动反思或回顾那些已被你'铭记'的核心原则、长期目标或关键事实时，调用此工具。它将根据你当初设置的重要性（记忆强度）进行**加权随机抽取**，重要性越高的记忆越有可能被回忆起来。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "要抽取的主动记忆数量",
                    "minimum": 1
                },
                "query": {
                    "type": "string",
                    "description": "（可选）检索关键字，筛选相关记忆"
                }
            },
            "required": ["limit"]
        }
    )

    def __post_init__(self):
        # 初始化日志记录器
        self.logger = logger

    async def run(
        self,
        event: AstrMessageEvent,
        limit: int,
        query: str = None,
    ) -> str:
        self.logger.debug(f"{self.name} - LLM 调用: limit={limit}, query='{query}'")

        # --- 获取服务 ---
        if not hasattr(event, 'plugin_context') or event.plugin_context is None:
            self.logger.error(f"{self.name}: 无法从事件中获取 plugin_context。")
            return "错误：内部服务错误，无法获取插件上下文。"

        plugin_context = event.plugin_context

        try:
            cognitive_service: CognitiveService = plugin_context.get_component("cognitive_service")
            if not cognitive_service:
                raise ValueError("CognitiveService 未在 PluginContext 中注册。")
        except Exception as e:
            self.logger.error(f"{self.name}: 无法获取 CognitiveService 实例: {e}")
            return "错误：内部服务错误，无法初始化记忆系统。"

        # --- 调用服务 ---
        try:
            candidate_limit = max(limit * 3, 20)

            all_memories: List[BaseMemory] = await cognitive_service.memory_manager.comprehensive_recall(
                query=query or "",
                limit=candidate_limit,
                event=event
            )

            all_active_memories = [mem for mem in all_memories if mem.is_active]

            if not all_active_memories:
                return "没有找到相关的主动记忆。"

            population = all_active_memories
            weights = [mem.strength for mem in all_active_memories]

            # 处理边界情况：如果请求的数量大于等于总记忆数，直接返回按权重排序的记忆
            if limit >= len(population):
                # 按权重降序排序所有记忆
                sorted_memories = sorted(zip(population, weights), key=lambda x: x[1], reverse=True)
                sampled_memories = [mem for mem, _ in sorted_memories]
            else:
                # 实现加权无替换抽样算法
                # 首先归一化权重为概率
                total_weight = sum(weights)
                if total_weight == 0:
                    # 如果所有权重都是0，则均匀随机选择
                    sampled_memories = random.sample(population, limit)
                else:
                    # 计算累积权重
                    cumulative_weights = []
                    current_sum = 0
                    for w in weights:
                        current_sum += w
                        cumulative_weights.append(current_sum)

                    # 加权无替换抽样
                    sampled_memories = []
                    remaining_population = population.copy()
                    remaining_weights = weights.copy()

                    for _ in range(limit):
                        # 计算累积权重
                        remaining_cumulative_weights = []
                        current_sum = 0
                        for w in remaining_weights:
                            current_sum += w
                            remaining_cumulative_weights.append(current_sum)

                        # 生成随机数
                        r = random.random() * remaining_cumulative_weights[-1]

                        # 找到对应的索引
                        selected_idx = 0
                        while selected_idx < len(remaining_cumulative_weights) and r > remaining_cumulative_weights[selected_idx]:
                            selected_idx += 1

                        # 添加选中的记忆
                        sampled_memories.append(remaining_population[selected_idx])

                        # 移除已选中的记忆和权重
                        del remaining_population[selected_idx]
                        del remaining_weights[selected_idx]

            formatted_results = []
            for i, mem in enumerate(sampled_memories, 1):
                formatted_results.append(
                    f"#{i}. [重要性: {mem.strength}]\n- 论断 (judgment): {mem.judgment}\n- 论证 (reasoning): {mem.reasoning}\n- 标签 (tags): {', '.join(mem.tags)}"
                )

            self.logger.info(f"{self.name}: 成功抽取 {len(sampled_memories)} 条核心记忆。")
            return "这是根据记忆强度加权抽取的核心记忆列表：\n\n" + "\n\n".join(formatted_results)

        except Exception as e:
            self.logger.error(f"{self.name}: 执行主动回忆失败: {e}", exc_info=True)
            return f"主动回忆失败：{str(e)}。请稍后再试。"
