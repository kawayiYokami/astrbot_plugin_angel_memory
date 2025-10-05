"""
笔记服务 - LLM的笔记管理服务

提供笔记的增删改查功能，与记忆系统共享同一个向量数据库实例。
支持目录解析、文档向量化存储和智能查询。
"""

import uuid
import re
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Optional
from pathlib import Path

from ..models.document_models import DocumentBlock
from ..parser.parser_manager import parser_manager

# 导入日志记录器
from astrbot.api import logger
from ..config.system_config import system_config


class NoteServiceError(Exception):
    """笔记服务异常基类"""

    pass


class NoteNotFoundError(NoteServiceError):
    """笔记未找到异常"""

    pass


class NoteOperationError(NoteServiceError):
    """笔记操作失败异常"""

    pass


class NoteService:
    """
    笔记服务类

    提供LLM笔记的完整管理功能，包括：
    - 笔记的添加、删除、修改、查询
    - 目录解析和文档向量化
    - 智能标签提取
    - 语义搜索
    - 与记忆系统的数据隔离
    """

    def __init__(self, vector_store):
        """
        初始化笔记服务。

        Args:
            vector_store: 一个已经初始化好的 VectorStore 实例。
        """
        self.logger = logger
        if not vector_store:
            raise ValueError("必须提供一个 VectorStore 实例。")
        self.vector_store = vector_store

        # 获取解析器管理器（不再直接初始化特定解析器）
        self.parser_manager = parser_manager

        # 通过VectorStore获取集合，确保使用正确的模型和维度检查
        self.main_collection = (
            self.vector_store.get_or_create_collection_with_dimension_check(
                system_config.notes_main_collection_name
            )
        )
        self.sub_collection = (
            self.vector_store.get_or_create_collection_with_dimension_check(
                system_config.notes_sub_collection_name
            )
        )

        self.logger.info("笔记服务初始化完成，已建立专用的主副集合。")

    def add_note(
        self, content: str, tags: List[str] = None, metadata: dict = None, note_id: Optional[str] = None
    ) -> str:
        """
        添加笔记

        Args:
            content: 笔记内容
            tags: 标签列表（可选，会自动提取）
            metadata: 额外元数据
            note_id: 笔记ID（可选，用于测试或数据迁移）

        Returns:
            笔记ID
        """
        try:
            # 如果未提供ID，则生成唯一ID
            if note_id is None:
                note_id = str(uuid.uuid4())

            # 自动提取标签（如果没有提供）
            if tags is None:
                tags = self._extract_tags(content)

            # 输入验证
            if not tags:
                raise ValueError("标签列表不允许为空。")

            # 准备元数据
            note_metadata = {
                "created_at": time.time(),  # 使用正确的时间戳
                "source": "note_service",
                "tags": ",".join(tags),  # 将标签列表转换为逗号分隔的字符串
                **(metadata or {}),
            }

            # 文件哈希值检查（去重）
            file_hash = note_metadata.get("source_file_hash")
            if file_hash:
                existing = self.main_collection.get(
                    where={"source_file_hash": file_hash}, limit=1
                )
                if existing and existing["ids"]:
                    self.logger.info(f"文件哈希值 {file_hash} 已存在，跳过写入。")
                    return existing["ids"][0]  # 返回已存在的ID

            # 准备数据
            tags_text = " ".join(tags)
            fused_content_for_embedding = content + " \n\nTags: " + tags_text
            full_note_data = (
                note_metadata  # 假设 metadata 包含完整的 DocumentBlock 结构
            )

            # 事务性写入，确保数据一致性
            try:
                # 使用高级抽象方法写入主集合
                # 设计决策：documents 和 embedding_texts 使用相同的内容。
                # 这是为了将标签的语义融合到向量和存储的文档中，
                # 以解决文档内容本身可能不包含关键索引词（如人名、地名）的问题，
                # 从而提升在这些场景下的召回率。
                self.vector_store.upsert_documents(
                    collection=self.main_collection,
                    ids=[note_id],
                    embedding_texts=[fused_content_for_embedding],
                    documents=[fused_content_for_embedding],
                    metadatas=full_note_data,
                )
                # 使用高级抽象方法写入副集合
                self.vector_store.upsert_documents(
                    collection=self.sub_collection,
                    ids=[note_id],
                    embedding_texts=[tags_text],
                    documents=[tags_text],
                )
            except Exception as e:
                self.logger.error(f"写入双集合失败，执行回滚: {e}")
                # 尝试回滚（删除可能已写入的部分数据）
                try:
                    self.main_collection.delete(ids=[note_id])
                    self.sub_collection.delete(ids=[note_id])
                except Exception as rollback_error:
                    self.logger.error(f"回滚失败: {rollback_error}")
                raise  # 重新抛出异常，让上层感知到失败

            self.logger.info(f"成功添加笔记: {note_id}, 标签: {tags}")
            return note_id

        except Exception as e:
            self.logger.error(f"添加笔记失败: {e}")
            raise

    def search_notes(
        self, query: str, max_results: int = 10, tag_filter: List[str] = None
    ) -> List[Dict]:
        """
        搜索笔记

        Args:
            query: 查询内容
            max_results: 最大结果数
            tag_filter: 标签过滤

        Returns:
            搜索结果列表
        """
        try:
            # 使用两阶段混合检索策略
            results = self._hybrid_search(query, max_results)
            return results

        except Exception as e:
            self.logger.error(f"搜索笔记失败: {e}")
            return []

    def search_notes_by_token_limit(
        self, query: str, max_tokens: int = 10000, recall_count: int = 100, tag_filter: List[str] = None
    ) -> List[Dict]:
        """
        基于token数量限制搜索笔记

        Args:
            query: 查询内容
            max_tokens: 最大token数限制
            recall_count: 候选结果数量
            tag_filter: 标签过滤

        Returns:
            搜索结果列表（保持相关性排序，按token限制动态截取）
        """
        try:
            # 使用现有的混合检索算法获取高质量排序结果
            candidates = self._hybrid_search(query, recall_count=recall_count, max_results=recall_count)

            # 如果候选结果为空，直接返回
            if not candidates:
                return []

            # 按token限制动态选择结果
            from ..utils.token_utils import count_tokens

            selected_notes = []
            current_tokens = 0
            candidate_count = 0

            for note in candidates:
                note_content = note.get('content', '')
                note_tokens = count_tokens(note_content)

                # 检查是否超出token限制
                if current_tokens + note_tokens <= max_tokens:
                    selected_notes.append(note)
                    current_tokens += note_tokens
                    candidate_count += 1
                else:
                    # 加上这个笔记会超出token限制，停止添加
                    self.logger.debug(
                        f"Token限制({max_tokens})达到，选择{candidate_count}个笔记，共{current_tokens} tokens"
                    )
                    break

            self.logger.debug(
                f"基于token限制搜索完成: 返回{len(selected_notes)}个笔记，共{current_tokens} tokens，从{len(candidates)}个候选中选出"
            )
            return selected_notes

        except Exception as e:
            self.logger.error(f"基于token限制搜索笔记失败: {e}")
            return []

    def _hybrid_search(
        self,
        query: str,
        max_results: int = 10,
        recall_count: int = 100,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        两阶段混合检索: 先过滤，后重排。

        Args:
            query: 查询内容
            max_results: 最大结果数
            recall_count: 第一阶段召回数量
            threshold: 内容相似度过滤阈值

        Returns:
            搜索结果列表
        """

        # 1. 第一阶段：过滤 (Filtering)
        # 在主集合 (内容+标签) 中进行向量搜索，并立即应用阈值
        recall_results = self.main_collection.query(
            query_texts=[query],
            n_results=recall_count,
            include=["metadatas", "documents", "distances"],
        )

        # 处理无结果情况
        if not recall_results["ids"] or not recall_results["ids"][0]:
            return []

        # 构建一个包含所有召回信息的字典列表
        all_recalled_notes = []
        for i, note_id in enumerate(recall_results["ids"][0]):
            distance = recall_results["distances"][0][i]
            similarity = 1 - distance
            if similarity >= threshold:
                all_recalled_notes.append(
                    {
                        "id": note_id,
                        "content": recall_results["documents"][0][i],
                        "metadata": recall_results["metadatas"][0][i],
                        "tags": (
                            recall_results["metadatas"][0][i].get("tags", "").split(",")
                            if recall_results["metadatas"][0][i].get("tags")
                            else []
                        ),
                        "content_similarity": similarity,
                    }
                )

        # 如果没有笔记通过阈值过滤，直接返回
        if not all_recalled_notes:
            return []

        # 文档级去重：在同一文档中只保留最相关的片段
        # 这确保了候选集的文档多样性，防止单一文档占据过多候选位置
        seen_source_files = set()
        deduplicated_notes = []

        for note in all_recalled_notes:
            source_file_path = note["metadata"].get("source_file_path")
            if not source_file_path:
                self.logger.warning(f"Note {note['id']} is missing 'source_file_path' in metadata, skipping deduplication for this item.")
                deduplicated_notes.append(note)  # 对于没有路径的笔记，直接保留
                continue

            if source_file_path not in seen_source_files:
                seen_source_files.add(source_file_path)
                deduplicated_notes.append(note)

        # 更新候选笔记列表为去重后的结果
        all_recalled_notes = deduplicated_notes

        # 2. 第二阶段：重排 (Reranking)
        # 拿着原始查询，在整个副集合 (纯标签) 中进行搜索，以获取所有笔记的标签相关性分数
        all_sub_collection_ids = [note["id"] for note in all_recalled_notes]

        rerank_results = self.sub_collection.query(
            query_texts=[query],
            where={
                "id": {"$in": all_sub_collection_ids}
            },  # 优化：只查询通过第一阶段的ID
            n_results=len(all_sub_collection_ids),
            include=["distances"],
        )

        # 创建一个 "ID -> 标签分数" 的映射
        tag_scores = {}
        if rerank_results["ids"] and rerank_results["ids"][0]:
            for i, note_id in enumerate(rerank_results["ids"][0]):
                tag_scores[note_id] = 1 - rerank_results["distances"][0][i]

        # 3. 最终排序
        # 将标签分数附加到通过第一阶段的笔记上
        for note in all_recalled_notes:
            note["tag_score"] = tag_scores.get(
                note["id"], 0.0
            )  # 如果在副集合中没找到，则标签分为0

        # 根据标签分数进行降序排序
        all_recalled_notes.sort(key=lambda x: x["tag_score"], reverse=True)

        # 4. 组装最终结果
        # 截取所需数量的结果，并格式化输出
        final_results = []
        for note in all_recalled_notes[:max_results]:
            final_results.append(
                {
                    "id": note["id"],
                    "content": note["content"],
                    "metadata": note["metadata"],
                    "tags": note["tags"],
                    "similarity": note[
                        "content_similarity"
                    ],  # 返回内容相似度，因为这更直观
                }
            )

        return final_results

    def get_note(self, note_id: str) -> Dict:
        """
        获取指定笔记

        Args:
            note_id: 笔记ID

        Returns:
            笔记内容字典

        Raises:
            NoteNotFoundError: 当笔记不存在时
            NoteOperationError: 当获取过程中发生其他错误时
        """
        try:
            # 主要信息从主集合获取
            results = self.main_collection.get(ids=[note_id])

            if results and results["ids"] and results["ids"][0]:
                # 组装返回数据
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                document = results["documents"][0] if results["documents"] else ""
                tags = (
                    metadata.get("tags", "").split(",") if metadata.get("tags") else []
                )

                formatted_note = {
                    "id": note_id,
                    "content": document,
                    "tags": tags,
                    "metadata": metadata,
                }

                return formatted_note
            else:
                self.logger.warning(f"笔记不存在: {note_id}")
                raise NoteNotFoundError(f"笔记不存在: {note_id}")

        except NoteServiceError:
            # 重新抛出我们自己的异常
            raise
        except Exception as e:
            self.logger.error(f"获取笔记失败: {e}")
            raise NoteOperationError(f"获取笔记失败: {e}") from e

    def delete_note(self, note_id: str) -> bool:
        """
        删除笔记

        Args:
            note_id: 笔记ID

        Returns:
            是否删除成功
        """
        try:
            # 首先检查笔记是否存在，以提供更好的错误信息
            try:
                self.get_note(note_id)
            except NoteNotFoundError:
                self.logger.warning(f"要删除的笔记不存在: {note_id}")
                return False

            # 事务性删除，确保数据一致性
            try:
                self.main_collection.delete(ids=[note_id])
                self.sub_collection.delete(ids=[note_id])
                self.logger.info(f"成功删除笔记: {note_id}")
                return True
            except Exception as e:
                self.logger.error(f"删除双集合中的 {note_id} 失败: {e}")
                # 注意：这里的回滚比较困难，但至少记录了错误
                return False

        except NoteServiceError:
            # 笔记不存在已经被上面处理，这里处理其他错误
            return False
        except Exception as e:
            self.logger.error(f"删除笔记失败: {e}")
            return False

    def _extract_tags(self, content: str) -> List[str]:
        """
        从内容中自动提取标签

        Args:
            content: 文本内容

        Returns:
            提取的标签列表
        """
        tags = []

        # 1. 提取标题（# ## ###）
        title_pattern = r"^#{1,6}\s+(.+)$"
        titles = re.findall(title_pattern, content, re.MULTILINE)
        tags.extend(titles)

        # 2. 提取加粗文本（**text**）
        bold_pattern = r"\*\*([^*]+)\*\*"
        bold_texts = re.findall(bold_pattern, content)
        tags.extend(bold_texts)

        # 3. 提取引号文本
        quote_patterns = [
            r'"([^"]+)"',  # 提取双引号内的文本
            r"'([^']+)'",  # 提取单引号内的文本
        ]

        for pattern in quote_patterns:
            quotes = re.findall(pattern, content)
            tags.extend(quotes)

        # 4. 提取关键词（简单实现）
        # 可以在这里添加更复杂的关键词提取逻辑

        # 去重和清理
        tags = list(set(tags))  # 去重
        tags = [tag.strip() for tag in tags if tag.strip()]  # 去空格
        tags = [tag for tag in tags if len(tag) > 0]  # 过滤空字符串，但保留单字符标签

        return tags

    def update_note(
        self, note_id: str, content: str = None, tags: List[str] = None
    ) -> None:
        """
        更新笔记

        Args:
            note_id: 笔记ID
            content: 新内容（可选）
            tags: 新标签（可选）

        Raises:
            NoteNotFoundError: 当笔记不存在时
            NoteOperationError: 当更新过程中发生其他错误时
        """
        try:
            # 获取现有笔记以获取当前内容
            old_note = self.get_note(note_id)

            # 使用新内容或保留旧内容
            final_content = content if content is not None else old_note["content"]

            # 使用新标签或重新提取标签
            final_tags = tags if tags is not None else self._extract_tags(final_content)

            # 获取旧的元数据并更新
            metadata = old_note.get("metadata", {}).copy()
            metadata["updated_at"] = time.time()
            metadata["updated"] = True
            metadata["tags"] = ",".join(
                final_tags
            )  # 关键修复：将更新后的标签列表转换为字符串

            # 准备数据
            tags_text = " ".join(final_tags)
            fused_content_for_embedding = final_content + " \n\nTags: " + tags_text

            # 事务性更新，确保数据一致性
            try:
                # 使用高级抽象方法更新主集合
                # 设计决策：documents 和 embedding_texts 使用相同的内容。
                # 理由同 add_note 方法，确保更新时也能将标签语义融合进去。
                self.vector_store.upsert_documents(
                    collection=self.main_collection,
                    ids=[note_id],
                    embedding_texts=[fused_content_for_embedding],
                    documents=[fused_content_for_embedding],
                    metadatas=metadata,
                )
                # 使用高级抽象方法更新副集合
                self.vector_store.upsert_documents(
                    collection=self.sub_collection,
                    ids=[note_id],
                    embedding_texts=[tags_text],
                    documents=[tags_text],
                )
            except Exception as e:
                self.logger.error(f"更新双集合失败: {e}")
                # 注意：这里的回滚比较困难，但至少记录了错误
                raise

            self.logger.info(f"成功更新笔记: {note_id}")

        except NoteServiceError:
            # 重新抛出我们自己的异常
            raise
        except Exception as e:
            self.logger.error(f"更新笔记失败: {e}")
            raise NoteOperationError(f"更新笔记失败: {e}") from e

    # ===== 新增功能：目录解析和文档向量化 =====

    def parse_and_store_directory(self, directory_path: str) -> int:
        """
        解析目录中的所有支持的文件，并存储到向量数据库

        Args:
            directory_path: 目录路径

        Returns:
            处理的文件数量
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"目录不存在: {directory_path}")

            if not directory.is_dir():
                raise ValueError(f"路径不是目录: {directory_path}")

            processed_count = 0

            # 遍历目录中的所有文件
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        # 由服务层决定是否支持该文件类型
                        if self._is_supported_file(str(file_path)):
                            self.parse_and_store_file(str(file_path))
                            processed_count += 1
                            self.logger.info(f"成功处理文件: {file_path}")
                    except Exception as e:
                        self.logger.error(f"处理文件失败: {file_path}, 错误: {e}")
                        continue

            self.logger.info(f"目录解析完成，共处理 {processed_count} 个文件")
            return processed_count

        except Exception as e:
            self.logger.error(f"解析目录失败: {e}")
            raise

    def parse_and_store_file(self, file_path: str) -> int:
        """
        解析单个文件并存储到向量数据库

        Args:
            file_path: 文件路径

        Returns:
            处理的文档块数量
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 检查文件类型是否支持
            if not self._is_supported_file(file_path):
                self.logger.debug(f"文件类型不支持，跳过处理: {file_path}")
                return 0

            # 获取对应的解析器
            parser = self.parser_manager.get_parser_for_file(file_path)
            if not parser:
                self.logger.warning(f"未找到适合的解析器，跳过处理: {file_path}")
                return 0

            # 对于支持异步解析的解析器，使用异步方式处理
            if hasattr(parser, "async_parse"):
                # 在线程池中异步执行解析任务
                loop = asyncio.new_event_loop()
                document_blocks = loop.run_until_complete(parser.async_parse(file_path))
            else:
                # 读取文件内容并解析
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 对于二进制文件，传递空内容，由解析器处理
                    content = ""

                # 解析文档
                document_blocks = parser.parse(content, file_path)

            # BUG修复：检查文件是否已存在，如果存在则先删除旧数据
            # 避免文件修改后产生重复数据的问题，同时避免对新文件进行不必要的清理
            existing_results = self.main_collection.get(where={"source_file_path": file_path})
            if existing_results and existing_results["ids"]:
                self.logger.debug(f"文件已存在，清理旧数据: {file_path}")
                removed_success = self.remove_file_data(file_path)
                if not removed_success:
                    self.logger.warning(f"清理文件旧数据失败，继续处理新数据: {file_path}")

            # 批量存储所有文档块，而不是逐个存储
            if document_blocks:
                self._store_document_blocks_batch(document_blocks)

            return len(document_blocks)

        except Exception as e:
            self.logger.error(f"解析文件失败: {file_path}, 错误: {e}")
            raise

    async def async_parse_and_store_file(self, file_path: str) -> int:
        """
        异步解析单个文件并存储到向量数据库（不阻塞主流程）

        Args:
            file_path: 文件路径

        Returns:
            处理的文档块数量
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 检查文件类型是否支持
            if not self._is_supported_file(file_path):
                self.logger.debug(f"文件类型不支持，跳过处理: {file_path}")
                return 0

            # 获取对应的解析器
            parser = self.parser_manager.get_parser_for_file(file_path)
            if not parser:
                self.logger.warning(f"未找到适合的解析器，跳过处理: {file_path}")
                return 0

            # 对于支持异步解析的解析器，使用异步方式处理
            if hasattr(parser, "async_parse"):
                document_blocks = await parser.async_parse(file_path)
            else:
                # 在线程池中执行CPU密集型任务
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    document_blocks = await loop.run_in_executor(
                        executor, self._parse_file_sync, file_path, parser
                    )

            # BUG修复：检查文件是否已存在，如果存在则先删除旧数据
            # 避免文件修改后产生重复数据的问题，同时避免对新文件进行不必要的清理
            existing_results = self.main_collection.get(where={"source_file_path": file_path})
            if existing_results and existing_results["ids"]:
                self.logger.debug(f"文件已存在，清理旧数据(异步): {file_path}")
                removed_success = self.remove_file_data(file_path)
                if not removed_success:
                    self.logger.warning(f"清理文件旧数据失败，继续处理新数据: {file_path}")
            else:
                self.logger.debug(f"新文件，无需清理旧数据(异步): {file_path}")

            # 批量存储所有文档块，而不是逐个存储
            if document_blocks:
                self._store_document_blocks_batch(document_blocks)

            self.logger.info(
                f"文件异步解析完成: {file_path}, 生成 {len(document_blocks)} 个文档块"
            )
            return len(document_blocks)

        except Exception as e:
            self.logger.error(f"异步解析文件失败: {file_path}, 错误: {e}")
            raise

    def _parse_file_sync(self, file_path: str, parser) -> List[DocumentBlock]:
        """
        同步解析文件的辅助方法

        Args:
            file_path: 文件路径
            parser: 解析器实例

        Returns:
            文档块列表
        """
        try:
            # 读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 对于二进制文件，传递空内容，由解析器处理
                content = ""

            # 解析文档
            return parser.parse(content, file_path)
        except Exception as e:
            self.logger.error(f"同步解析文件失败: {file_path}, 错误: {e}")
            raise

    def _is_supported_file(self, file_path: str) -> bool:
        """
        检查文件是否支持解析

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        return self.parser_manager.is_supported_extension(extension)

    def _store_document_blocks_batch(self, blocks: List[DocumentBlock]):
        """
        批量存储文档块到向量数据库（双集合架构）

        Args:
            blocks: 文档块列表
        """
        try:
            # 准备主集合数据
            main_ids = []
            main_contents = []
            main_metadatas = []

            # 准备副集合数据
            sub_ids = []
            sub_tags_texts = []

            for block in blocks:
                # 主集合数据
                main_ids.append(block.id)
                main_contents.append(block.content)

                # 准备主集合元数据
                main_metadata = {
                    "created_at": block.created_at,
                    "source_file_hash": block.source_file_hash,
                    "related_block_ids": ",".join(block.related_block_ids),
                    "source": "document_parser",
                    "source_file_path": block.source_file_path,
                }
                main_metadatas.append(main_metadata)

                # 副集合数据
                sub_ids.append(block.id)
                sub_tags_texts.append(" ".join(block.tags))  # 标签文本用于副集合

            # 使用高级抽象方法批量添加到主集合
            self.vector_store.upsert_documents(
                collection=self.main_collection,
                ids=main_ids,
                embedding_texts=main_contents,
                documents=main_contents,
                metadatas=main_metadatas,
            )

            # 使用高级抽象方法批量添加到副集合
            self.vector_store.upsert_documents(
                collection=self.sub_collection,
                ids=sub_ids,
                embedding_texts=sub_tags_texts,
                documents=sub_tags_texts,
            )

        except Exception as e:
            self.logger.error(f"批量存储文档块失败: {e}")
            raise

    def remove_file_data(self, file_path: str) -> bool:
        """
        删除与指定文件相关的所有数据

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        try:
            # 构造查询条件，查找所有与该文件相关的记录
            # 假设我们在元数据中存储了 source_file_path 字段
            where_clause = {"source_file_path": file_path}

            # 从主集合中获取所有匹配的记录
            main_results = self.main_collection.get(where=where_clause)

            if main_results and main_results["ids"]:
                # 获取要删除的ID列表
                ids_to_delete = main_results["ids"]

                # 从主集合删除
                self.main_collection.delete(ids=ids_to_delete)

                # 从副集合删除
                self.sub_collection.delete(ids=ids_to_delete)

                self.logger.info(
                    f"成功删除文件相关数据: {file_path}, 共删除 {len(ids_to_delete)} 条记录"
                )
                return True
            else:
                self.logger.info(f"未找到与文件相关的数据: {file_path}")
                return True

        except Exception as e:
            self.logger.error(f"删除文件相关数据失败: {file_path}, 错误: {e}")
            return False
