"""
笔记上下文构建器

负责构建用于LLM提示词的笔记上下文，包括候选笔记清单和完整的上下文扩展。
"""

from typing import List, Dict
from ...llm_memory.service.note_service import NoteService
from .memory_id_resolver import MemoryIDResolver
from ...llm_memory.utils.token_utils import count_tokens

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# 添加调试日志标记
DEBUG_TAG = "🔍 [NoteContextBuilder]"


class NoteContextBuilder:
    """笔记上下文构建器"""

    @staticmethod
    def build_candidate_list_for_prompt(notes: List[Dict]) -> str:
        """
        为小模型构建候选笔记清单，用于注入到提示词中

        Args:
            notes: 候选笔记片段列表

        Returns:
            格式化的笔记清单字符串
        """
        if not notes:
            return "暂无相关笔记"

        lines = ["\n你检索到以下笔记片段作为参考："]

        for i, note in enumerate(notes, 1):
            # 生成短ID用于显示
            short_id = MemoryIDResolver.generate_short_id(note["id"])
            content = note.get("content", "").strip()
            tags = note.get("tags", [])
            tags_str = f" [标签: {', '.join(tags)}]" if tags else ""

            # 笔记块已被嵌入模型限制长度，不需要截断
            lines.append(f"({i}) [ID: {short_id}]{tags_str} {content}")

        return "\n".join(lines)

    @staticmethod
    def expand_context_from_note_ids(
        note_ids: List[str],
        note_service: NoteService,
        total_token_budget: int,
        note_id_mapping: Dict[str, str] = None,
    ) -> str:
        """
        从选中的笔记ID列表构建完整的上下文

        通过遍历文档块链表（related_block_ids）来扩展上下文，确保每个笔记的上下文完整且连贯。

        Args:
            note_ids: 选中的笔记片段ID列表（短ID）
            note_service: 笔记服务实例
            total_token_budget: 总令牌预算
            note_id_mapping: 短ID到完整ID的映射字典

        Returns:
            拼接好的完整上下文字符串
        """
        if not note_ids:
            return ""

        try:
            # 将短ID转换为完整ID
            full_note_ids = []
            for short_id in note_ids:
                if note_id_mapping and short_id in note_id_mapping:
                    full_id = note_id_mapping[short_id]
                    full_note_ids.append(full_id)
                    logger.info(f"{DEBUG_TAG} 短ID转换成功: {short_id} -> {full_id}")
                else:
                    logger.error(f"{DEBUG_TAG} 无法找到短ID '{short_id}' 对应的完整ID")
                    logger.error(
                        f"{DEBUG_TAG} 可用的映射键: {list(note_id_mapping.keys()) if note_id_mapping else 'NONE'}"
                    )
                    logger.error(f"{DEBUG_TAG} 映射内容: {note_id_mapping}")
                    continue

            # 计算每个文档的令牌配额
            num_notes = len(full_note_ids)
            logger.info(f"{DEBUG_TAG} 成功转换的完整ID数量: {num_notes}")

            if num_notes == 0:
                logger.error(f"{DEBUG_TAG} 没有有效的完整ID，无法构建上下文")
                return ""

            token_per_note = total_token_budget // num_notes

            expanded_contexts = []

            for note_id in full_note_ids:
                try:
                    # 获取中心片段
                    center_block = note_service.get_note(note_id)
                    if not center_block:
                        continue

                    # 提取标签并构建标签头
                    tags = center_block.get("tags", [])
                    tags_header = f"[标签: {', '.join(tags)}]\n" if tags else ""

                    # 使用双向扩展获取完整上下文
                    context_blocks = NoteContextBuilder._expand_bidirectional(
                        note_id, note_service, token_per_note
                    )

                    # 合并该笔记的完整上下文，在开头添加标签
                    note_context = tags_header + "\n\n".join(context_blocks)
                    expanded_contexts.append(note_context)

                except Exception as e:
                    logger.warning(f"扩展笔记 {note_id} 上下文失败: {e}")
                    continue

            # 返回所有扩展上下文的拼接结果
            return "\n\n---\n\n".join(expanded_contexts)

        except Exception as e:
            logger.error(f"构建笔记上下文失败: {e}")
            return ""

    @staticmethod
    def _expand_bidirectional(
        center_block_id: str, note_service: NoteService, max_tokens: int
    ) -> List[str]:
        """
        双向扩展上下文：同时向上和向下扩展，直到达到令牌极限或文档边界

        Args:
            center_block_id: 中心块ID
            note_service: 笔记服务实例
            max_tokens: 最大令牌数限制

        Returns:
            扩展后的上下文块列表
        """
        try:
            # 获取中心块
            center_block = note_service.get_note(center_block_id)
            if not center_block:
                return []

            center_content = center_block["content"]
            source_file_path = center_block.get("metadata", {}).get("source_file_path")

            # 初始化上下文块列表，中心块在中间
            context_blocks = [center_content]
            current_tokens = count_tokens(center_content)

            # 获取关联块信息
            metadata = center_block.get("metadata", {})
            related_block_ids_str = metadata.get("related_block_ids", "")
            if not related_block_ids_str:
                return context_blocks

            related_block_ids = related_block_ids_str.split(",")
            prev_id = (
                related_block_ids[0].strip()
                if len(related_block_ids) > 0 and related_block_ids[0] != "none"
                else None
            )
            next_id = (
                related_block_ids[1].strip()
                if len(related_block_ids) > 1 and related_block_ids[1] != "none"
                else None
            )

            # 双向扩展：交替尝试向上和向下扩展
            while (prev_id or next_id) and current_tokens < max_tokens:
                added = False

                # 先尝试向上扩展
                if prev_id:
                    success, prev_id = NoteContextBuilder._try_expand_direction(
                        prev_id,
                        note_service,
                        context_blocks,
                        current_tokens,
                        max_tokens,
                        source_file_path,
                        direction="backward",
                    )
                    if success:
                        added = True
                        current_tokens = count_tokens("\n\n".join(context_blocks))

                # 再尝试向下扩展
                if next_id and current_tokens < max_tokens:
                    success, next_id = NoteContextBuilder._try_expand_direction(
                        next_id,
                        note_service,
                        context_blocks,
                        current_tokens,
                        max_tokens,
                        source_file_path,
                        direction="forward",
                    )
                    if success:
                        added = True
                        current_tokens = count_tokens("\n\n".join(context_blocks))

                # 如果这一轮都没有成功添加内容，停止扩展
                if not added:
                    break

            return context_blocks

        except Exception as e:
            logger.error(f"双向扩展上下文失败: {e}")
            # 返回只包含中心块的最小上下文
            try:
                center_block = note_service.get_note(center_block_id)
                return [center_block["content"]] if center_block else []
            except Exception:
                return []

    @staticmethod
    def _try_expand_direction(
        current_id: str,
        note_service: NoteService,
        context_blocks: List[str],
        current_tokens: int,
        max_tokens: int,
        source_file_path: str,
        direction: str,
    ) -> tuple[bool, str]:
        """
        尝试沿指定方向扩展一个块

        Args:
            current_id: 当前块ID
            note_service: 笔记服务实例
            context_blocks: 当前上下文块列表
            current_tokens: 当前令牌数
            max_tokens: 最大令牌数限制
            source_file_path: 源文件路径（用于验证同文件）
            direction: 'forward' 或 'backward'

        Returns:
            (是否成功扩展, 下一个块ID)
        """
        try:
            block = note_service.get_note(current_id)
            if not block:
                return False, None

            # 验证是否来自同一个文件
            block_file_path = block.get("metadata", {}).get("source_file_path")
            if block_file_path != source_file_path:
                logger.debug(f"跳过来自不同文件的块: {current_id}")
                return False, None

            block_content = block["content"]
            block_tokens = count_tokens(block_content)

            # 检查是否超出令牌限制
            if current_tokens + block_tokens > max_tokens:
                return False, None

            # 根据方向插入内容
            if direction == "backward":
                context_blocks.insert(0, block_content)  # 插入到开头
            else:
                context_blocks.append(block_content)  # 追加到结尾

            # 获取下一个关联块ID
            metadata = block.get("metadata", {})
            related_block_ids_str = metadata.get("related_block_ids", "")
            if related_block_ids_str:
                related_block_ids = related_block_ids_str.split(",")
                if direction == "backward" and len(related_block_ids) > 0:
                    next_id = (
                        related_block_ids[0].strip()
                        if related_block_ids[0] != "none"
                        else None
                    )
                elif direction == "forward" and len(related_block_ids) > 1:
                    next_id = (
                        related_block_ids[1].strip()
                        if related_block_ids[1] != "none"
                        else None
                    )
                else:
                    next_id = None
            else:
                next_id = None

            return True, next_id

        except Exception as e:
            logger.warning(f"扩展方向 {direction} 时出错: {e}")
            return False, None
