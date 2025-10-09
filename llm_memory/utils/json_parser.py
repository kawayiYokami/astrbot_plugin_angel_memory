"""
Angel Eye 插件 - JSON 解析工具
提供健壮的 JSON 提取功能，用于从模型返回的文本中安全地解析 JSON 数据
"""

import json
from typing import Dict, Optional, Any, List

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)



def _strip_code_fences(text: str) -> str:
    """
    去除常见的 Markdown 代码块围栏，避免干扰解析。
    例如: ```json ... ``` 或 ``` ... ```
    """
    if not text:
        return text
    # 仅移除围栏标记，不移除内部内容
    return (
        text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
    )


def _find_json_candidates(text: str) -> List[str]:
    """
    在文本中扫描并返回所有"平衡的大括号"子串，作为潜在的 JSON 候选。
    - 跳过字符串字面量中的花括号
    - 支持嵌套
    返回顺序为出现顺序（从左到右）
    """
    candidates: List[str] = []
    if not text:
        return candidates

    in_string = False
    escape = False
    depth = 0
    start_idx: Optional[int] = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            # 字符串中不处理花括号
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidates.append(text[start_idx:i + 1])
                    start_idx = None

    return candidates


class JsonParser:
    """
    高鲁棒性 JSON 解析器类。

    负责从 LLM 响应中提取 JSON 部分并转换为结构化数据。
    使用智能候选识别和评分机制，确保在各种情况下都能正确解析。
    """

    def __init__(self):
        """初始化 JSON 解析器。"""
        self.logger = logger

    def parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应文本中解析出 feedback_data 字典。

        使用高鲁棒性解析策略：
        - 智能识别所有可能的JSON候选
        - 根据字段完整性评分选择最佳候选
        - 支持部分字段缺失的情况

        Args:
            response_text: LLM 的原始响应文本

        Returns:
            解析后的 feedback_data 字典，失败时返回 None
        """
        # 尝试提取JSON数据
        json_data = self.extract_json(response_text)

        if json_data is None:
            self.logger.warning("JsonParser: 未能从响应中提取有效的 JSON")
            return None

        # 从 JSON 中提取 feedback_data
        if "feedback_data" in json_data:
            feedback_data = json_data["feedback_data"]

            # 如果 feedback_data 是字符串，尝试再次解析
            if isinstance(feedback_data, str):
                try:
                    feedback_data = json.loads(feedback_data)
                    self.logger.debug("JsonParser: feedback_data 是字符串，已重新解析为字典")
                except json.JSONDecodeError:
                    self.logger.warning("JsonParser: feedback_data 是字符串但无法解析为 JSON")
                    return None

            return feedback_data
        else:
            # 如果没有 feedback_data 包装，直接返回整个 JSON
            self.logger.debug("JsonParser: JSON 中未找到 'feedback_data' 字段，返回整个 JSON")
            return json_data

    def extract_json(
        self,
        text: str,
        separator: str = "---JSON---",
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        从可能包含其他文本的字符串中，智能地提取最符合条件的单个JSON对象。

        公共方法，用于从LLM响应中提取JSON数据。

        提取策略:
        1.  **分割**: 如果存在分隔符，则优先处理分隔符之后的内容。
        2.  **清理**: 自动去除常见的Markdown代码块围栏。
        3.  **扫描**: 通过"平衡大括号"算法，找出所有结构上闭合的JSON候选片段。
        4.  **筛选**: (如果提供了`required_fields`) 只保留那些包含所有必须字段的JSON对象。
        5.  **评分**: (如果提供了`optional_fields`) 根据包含的可选字段数量为每个合格的JSON对象打分。
        6.  **决策**: 返回分数最高的对象。如果分数相同，则选择在原文中位置最靠后的那一个。
        7.  **回退**: 如果上述策略找不到，则尝试一次"从第一个'{'到最后一个'}'"的大包围策略。

        :param text: 包含JSON的模型原始输出字符串。
        :param separator: 用于分割内容的分隔符。
        :param required_fields: 一个列表，JSON对象必须包含其中所有的字段才算合格。
        :param optional_fields: 一个列表，用于对合格的JSON对象进行评分，包含的可选字段越多，分数越高。
        :return: 最符合条件的JSON对象（字典），如果找不到则返回None。
        """
        if not isinstance(text, str):
            self.logger.warning(f"JsonParser: 输入不是字符串，而是 {type(text)} 类型，无法解析")
            return None

        if not text.strip():
            self.logger.debug("JsonParser: 输入为空字符串")
            return None

        # 1) 分隔符处理
        json_part = text
        if separator in text:
            self.logger.debug(f"JsonParser: 找到分隔符 '{separator}' 进行分割")
            parts = text.split(separator, 1)
            if len(parts) > 1:
                json_part = parts[1].strip()
            else:
                self.logger.warning("JsonParser: 分隔符后无内容")
                return None
        else:
            self.logger.debug(f"JsonParser: 未找到分隔符 '{separator}'，将处理整个文本")

        # 2) 去掉代码围栏
        json_part = _strip_code_fences(json_part)

        # 3) 扫描所有平衡的大括号候选
        candidates = _find_json_candidates(json_part)
        self.logger.debug(f"JsonParser: 扫描到可能的 JSON 候选数量: {len(candidates)}")

        # 4) 筛选与评分
        qualified_jsons = []
        for candidate_str in candidates:
            try:
                parsed_json = json.loads(candidate_str)
                if not isinstance(parsed_json, dict):
                    continue  # 只处理对象类型的JSON

                # 硬性条件：检查必须字段
                if required_fields:
                    if not all(field in parsed_json for field in required_fields):
                        self.logger.debug(f"候选JSON缺少必须字段，跳过: {candidate_str[:100]}...")
                        continue

                # 计算分数
                score = 0
                if optional_fields:
                    score = sum(1 for field in optional_fields if field in parsed_json)

                qualified_jsons.append({"json": parsed_json, "score": score})
                self.logger.debug(f"一个候选JSON合格，得分: {score}")

            except json.JSONDecodeError:
                continue # 解析失败，不是有效的JSON，跳过

        if not qualified_jsons:
            self.logger.warning("JsonParser: 所有候选均不满足要求（或解析失败）。")
            return None

        # 5) 决策：选择分数最高的，同分则取最后的
        # 先按分数排序（稳定排序），然后取最后一个，这样就能保证在分数相同时，选择原文中位置更靠后的
        qualified_jsons.sort(key=lambda x: x['score'])
        best_json_item = qualified_jsons[-1]

        self.logger.info(f"JsonParser: 提取成功，选择了得分最高的JSON（得分: {best_json_item['score']}）。")
        return best_json_item['json']