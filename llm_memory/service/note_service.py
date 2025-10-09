"""
笔记服务 - LLM的笔记管理服务

提供笔记的增删改查功能，与记忆系统共享同一个向量数据库实例。
支持目录解析、文档向量化存储和智能查询。
"""

import re
import asyncio
import concurrent.futures
from typing import List, Dict, Optional
from pathlib import Path

from ..models.note_models import NoteData
from ..parser.parser_manager import parser_manager
from .id_service import IDService

# 导入日志记录器
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
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
    - 批量向量化优化
    """

    def __init__(self, plugin_context=None, vector_store=None):
        """
        初始化笔记服务。

        Args:
            plugin_context: PluginContext插件上下文（可选）
            vector_store: VectorStore实例（向后兼容，如果未提供plugin_context则必需）
        """
        self.logger = logger
        self.plugin_context = plugin_context

        # 优先使用PluginContext，否则使用传入的vector_store
        if plugin_context:
            # 从PluginContext创建IDService
            self.id_service = IDService.from_plugin_context(plugin_context)
            # vector_store需要在ComponentFactory中设置
            self.vector_store = None
            self.collections_initialized = False
            self.logger.info("笔记服务初始化完成（PluginContext模式），等待vector_store设置。")
        elif vector_store:
            # 向后兼容模式
            if not vector_store:
                raise ValueError("必须提供一个 VectorStore 实例。")
            self.vector_store = vector_store

            # 初始化ID服务（使用默认配置）
            self.id_service = IDService()

            # 通过VectorStore获取集合
            self._initialize_collections()
            self.logger.info("笔记服务初始化完成（传统模式），已建立专用的主副集合。")
        else:
            raise ValueError("必须提供 plugin_context 或 vector_store 参数")

        # 获取解析器管理器
        self.parser_manager = parser_manager

        # 创建轻量线程池（仅用于同步解析器的兼容，顺序处理模式下几乎不并发）
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,  # 顺序处理，只需1个线程
            thread_name_prefix="NoteService"
        )

        # 批量向量化优化配置
        self._batch_size = 64  # 每批次最多64个文档块
        self._batch_timeout = 5.0  # 最大等待5秒
        self._embedding_queue = []  # 待向量化的队列
        self._batch_lock = asyncio.Lock()  # 异步锁保护队列
        self.logger.info("NoteService初始化完成（顺序处理模式）")

    def _initialize_collections(self):
        """初始化ChromaDB集合"""
        if not self.vector_store:
            raise ValueError("VectorStore未设置，无法初始化集合")

        # 通过VectorStore获取集合，使用缓存机制避免重复初始化
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
        self.collections_initialized = True

    def set_vector_store(self, vector_store):
        """
        设置VectorStore实例（用于PluginContext模式）

        Args:
            vector_store: VectorStore实例
        """
        if self.vector_store:
            self.logger.warning("VectorStore已存在，将被覆盖")

        self.vector_store = vector_store
        self._initialize_collections()
        self.logger.info("VectorStore已设置，集合初始化完成")

    def ensure_ready(self):
        """确保服务已准备就绪（用于PluginContext模式）"""
        if not self.collections_initialized:
            if not self.vector_store:
                raise RuntimeError("NoteService未设置VectorStore，请先调用set_vector_store()")
            self._initialize_collections()
        return True

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
            # 确保服务已准备就绪
            self.ensure_ready()
            # 自动提取标签（如果没有提供）
            if tags is None:
                tags = self._extract_tags(content)

            # 输入验证
            if not tags:
                raise ValueError("标签列表不允许为空。")

            # 使用ID服务将tag字符串转换为tag_ids
            tag_ids = self.id_service.tags_to_ids(tags)

            # 创建NoteData对象
            note = NoteData.create_user_note(
                content=content,
                tag_ids=tag_ids
            )

            # 如果提供了自定义ID，覆盖生成的ID
            if note_id is not None:
                note.id = note_id

            # 使用新的笔记存储方法
            try:
                # 使用VectorStore的笔记专用方法存储主集合
                self.vector_store.store_note(
                    collection=self.main_collection,
                    note=note
                )

                # 副集合只存储标签文本，用于标签重排
                tag_names = self.id_service.ids_to_tags(tag_ids)
                self.vector_store.upsert_documents(
                    collection=self.sub_collection,
                    ids=[note.id],
                    embedding_texts=[note.get_tags_text(tag_names)],
                    documents=[note.get_tags_text(tag_names)]
                )

            except Exception as e:
                self.logger.error(f"写入双集合失败，执行回滚: {e}")
                # 尝试回滚（删除可能已写入的部分数据）
                try:
                    self.main_collection.delete(ids=[note.id])
                    self.sub_collection.delete(ids=[note.id])
                except Exception as rollback_error:
                    self.logger.error(f"回滚失败: {rollback_error}")
                raise  # 重新抛出异常，让上层感知到失败

            self.logger.info(f"成功添加笔记: {note.id}, 标签: {tags}")
            return note.id

        except Exception as e:
            self.logger.error(f"添加笔记失败: {e}")
            raise

    def search_notes(
        self, query: str, max_results: int = 10, tag_filter: List[str] = None, threshold: float = 0.5
    ) -> List[Dict]:
        """
        搜索笔记

        Args:
            query: 查询内容
            max_results: 最大结果数
            tag_filter: 标签过滤
            threshold: 相似度阈值（0.0-1.0），低于此阈值的结果将被过滤

        Returns:
            搜索结果列表
        """
        try:
            # 确保服务已准备就绪
            self.ensure_ready()
            # 使用两阶段混合检索策略，传递阈值参数
            results = self._hybrid_search(query, max_results, threshold=threshold)
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
        # 使用 VectorStore 的笔记专用检索方法进行向量搜索
        recall_results = self.vector_store.search_notes(
            collection=self.main_collection,
            query=query,
            limit=recall_count
        )

        # 处理无结果情况
        if not recall_results:
            return []

        # 构建一个包含所有召回信息的字典列表，并应用阈值过滤
        all_recalled_notes = []
        for note in recall_results:
            # 获取真实的相似度分数（由VectorStore设置）
            content_similarity = getattr(note, 'similarity', 0.0)

            # 应用阈值过滤：只保留相似度 >= threshold 的结果
            if content_similarity < threshold:
                continue

            # 从 NoteData 对象中提取数据
            tag_names = self.id_service.ids_to_tags(note.tag_ids)
            all_recalled_notes.append(
                {
                    "id": note.id,
                    "content": note.content,
                    "metadata": note.to_dict(),
                    "tags": tag_names,  # 转换为tag_names
                    "content_similarity": content_similarity,  # 使用真实相似度
                }
            )

        # 如果没有笔记通过阈值过滤，直接返回
        if not all_recalled_notes:
            self.logger.debug(f"所有召回结果都低于阈值 {threshold}，返回空列表")
            return []

        # 2. 第一阶段：重排 (Reranking)
        # 拿着原始查询，在整个副集合 (纯标签) 中进行搜索，以获取所有笔记的标签相关性分数
        all_sub_collection_ids = [note["id"] for note in all_recalled_notes]

        # 使用 VectorStore 的笔记专用检索方法进行标签重排
        rerank_results = self.vector_store.search_notes(
            collection=self.sub_collection,
            query=query,
            limit=len(all_sub_collection_ids)
        )

        # 创建一个 "ID -> 标签分数" 的映射，使用真实的相似度分数
        tag_scores = {}
        for note in (rerank_results or []):
            if note.id in all_sub_collection_ids:
                # 使用真实的标签相似度分数（由VectorStore设置）
                tag_similarity = getattr(note, 'similarity', 0.0)
                tag_scores[note.id] = tag_similarity

        # 3. 最终排序
        # 将标签分数附加到通过第一阶段的笔记上
        for note in all_recalled_notes:
            note["tag_score"] = tag_scores.get(
                note["id"], 0.0
            )  # 如果在副集合中没找到，则标签分为0

        # 根据标签分数进行降序排序
        all_recalled_notes.sort(key=lambda x: x["tag_score"], reverse=True)

        # 文档去重：在同一文档中只保留排名最高的片段
        # 由于已按标签分数排序，第一次出现的就是该文档中标签最相关的块
        seen_file_ids = set()
        deduplicated_notes = []

        for note in all_recalled_notes:
            file_id = note["metadata"].get("file_id")
            if file_id is None:
                self.logger.warning(f"Note {note['id']} is missing 'file_id' in metadata, skipping deduplication for this item.")
                deduplicated_notes.append(note)
                continue

            if file_id not in seen_file_ids:
                seen_file_ids.add(file_id)
                deduplicated_notes.append(note)

        # 4. 组装最终结果
        # 截取所需数量的结果，并格式化输出
        final_results = []
        for note in deduplicated_notes[:max_results]:
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
                document = metadata.get("content", "")

                # 从tag_ids转换为tag_names
                tag_ids = metadata.get("tag_ids", [])
                if isinstance(tag_ids, str):
                    import json
                    try:
                        tag_ids = json.loads(tag_ids)
                    except json.JSONDecodeError:
                        tag_ids = []

                # 创建NoteData对象以获取tag_names
                note_data = NoteData.from_dict(metadata)
                tag_names = self.id_service.ids_to_tags(note_data.tag_ids)

                formatted_note = {
                    "id": note_id,
                    "content": document,
                    "tags": tag_names,
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
            old_note_dict = self.get_note(note_id)

            # 将字典转换为NoteData对象
            old_note = NoteData.from_dict(old_note_dict["metadata"])

            # 使用新内容或保留旧内容
            final_content = content if content is not None else old_note.content

            # 使用新标签或重新提取标签
            final_tags = tags if tags is not None else self._extract_tags(final_content)

            # 使用ID服务将tag字符串转换为tag_ids
            final_tag_ids = self.id_service.tags_to_ids(final_tags)

            # 更新笔记对象（直接修改dataclass字段）
            old_note.content = final_content
            old_note.tag_ids = final_tag_ids

            # 使用新的笔记存储方法更新主集合
            try:
                self.vector_store.store_note(
                    collection=self.main_collection,
                    note=old_note
                )

                # 更新副集合
                tag_names = self.id_service.ids_to_tags(old_note.tag_ids)
                self.vector_store.upsert_documents(
                    collection=self.sub_collection,
                    ids=[old_note.id],
                    embedding_texts=[old_note.get_tags_text(tag_names)],
                    documents=[old_note.get_tags_text(tag_names)]
                )

            except Exception as e:
                self.logger.error(f"更新双集合失败: {e}")
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

            # 获取对应的解析器（传递ID服务中的TagManager）
            parser = self.parser_manager.get_parser_for_file(file_path, self.id_service.tag_manager)
            if not parser:
                self.logger.warning(f"未找到适合的解析器，跳过处理: {file_path}")
                return 0

            # 获取文件信息（但不创建文件索引）
            file_path_obj = Path(file_path)
            file_timestamp = int(file_path_obj.stat().st_mtime)
            relative_path = file_path_obj.name  # 使用文件名作为相对路径

            # 注意：不在这里创建文件索引，由调用方在处理成功后创建
            # 使用临时文件ID用于解析过程中的标识
            temp_file_id = hash(relative_path + str(file_timestamp)) % (2**31)  # 生成临时ID

            # 使用带TagManager的解析器
            if hasattr(parser, "async_parse"):
                # 异步解析需要传递TagManager和file_id
                loop = asyncio.new_event_loop()
                document_blocks = loop.run_until_complete(parser.async_parse(file_path, self.id_service.tag_manager, temp_file_id))
            else:
                # 读取文件内容并解析
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 对于二进制文件，传递空内容，由解析器处理
                    content = ""

                # 解析文档，传递临时file_id
                document_blocks = parser.parse(content, temp_file_id, file_path)

            # 批量存储所有笔记，而不是逐个存储（同步调用）
            if document_blocks:
                self._store_notes_batch(document_blocks)

            return len(document_blocks)

        except Exception as e:
            self.logger.error(f"解析文件失败: {file_path}, 错误: {e}")
            raise

    def parse_and_store_file_sync(self, file_path: str, relative_path: str = None) -> tuple:
        """
        同步版本：解析并存储文件（顺序处理优化）
        
        Args:
            file_path: 文件路径
            relative_path: 相对路径
            
        Returns:
            (文档数量, 计时字典)
        """
        import time
        timings = {}
        
        try:
            # 检查文件是否存在
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.warning(f"文件不存在: {file_path}")
                return 0, timings
            
            # 获取解析器
            t_parser = time.time()
            parser = self.parser_manager.get_parser_for_file(file_path, self.id_service.tag_manager)
            timings['parser_select'] = (time.time() - t_parser) * 1000
            
            if parser is None:
                self.logger.warning(f"未找到适合的解析器，跳过处理: {file_path}")
                return 0, timings
            
            # 获取文件信息
            file_path_obj = Path(file_path)
            file_timestamp = int(file_path_obj.stat().st_mtime)
            
            # 优先使用传入的relative_path
            if relative_path is None:
                relative_path = file_path_obj.name
            
            # 获取或创建文件ID
            t_start = time.time()
            file_id = self.id_service.file_to_id(relative_path, file_timestamp)
            timings['id_lookup'] = (time.time() - t_start) * 1000
            
            # 同步解析文件
            t_start = time.time()
            document_blocks = self._parse_file_sync(file_path, parser, file_id)
            timings['parse'] = (time.time() - t_start) * 1000
            
            # 存储notes（完全同步版本，直接存储不走队列）
            if document_blocks:
                t_store_submit = time.time()
                
                # 直接同步存储（跳过批量队列，不更新BM25）
                store_timings = self._store_notes_batch(document_blocks, update_bm25=False)
                
                timings['store_total'] = (time.time() - t_store_submit) * 1000
                
                if store_timings:
                    timings.update(store_timings)
            else:
                timings['store_total'] = 0
            
            return len(document_blocks), timings
            
        except Exception as e:
            self.logger.error(f"同步解析文件失败: {file_path}, 错误: {e}")
            raise
    
    async def async_parse_and_store_file(self, file_path: str, relative_path: str = None) -> tuple:
        """
        异步解析单个文件并存储到向量数据库（不阻塞主流程）

        Args:
            file_path: 文件路径
            relative_path: 相对路径（可选，如果不提供则使用文件名）

        Returns:
            (文档块数量, 计时字典)
        """
        import time
        timings = {}
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 检查文件类型是否支持
            if not self._is_supported_file(file_path):
                self.logger.debug(f"文件类型不支持，跳过处理: {file_path}")
                return 0, timings

            # 获取对应的解析器（传递ID服务中的TagManager）
            parser = self.parser_manager.get_parser_for_file(file_path, self.id_service.tag_manager)
            if not parser:
                self.logger.warning(f"未找到适合的解析器，跳过处理: {file_path}")
                return 0, timings

            # 获取文件信息
            file_path_obj = Path(file_path)
            file_timestamp = int(file_path_obj.stat().st_mtime)

            # 优先使用传入的relative_path，否则使用文件名
            if relative_path is None:
                relative_path = file_path_obj.name

            # 获取或创建文件ID
            t_start = time.time()
            file_id = self.id_service.file_to_id(relative_path, file_timestamp)
            timings['id_lookup'] = (time.time() - t_start) * 1000

            # 使用带TagManager的解析器
            t_start = time.time()
            if hasattr(parser, "async_parse"):
                # 异步解析需要传递TagManager和file_id
                document_blocks = await parser.async_parse(file_path, self.id_service.tag_manager, file_id)
            else:
                # 在共享线程池中执行CPU密集型任务（复用线程，避免无限创建）
                loop = asyncio.get_event_loop()
                document_blocks = await loop.run_in_executor(
                    self._thread_pool, self._parse_file_sync, file_path, parser, file_id
                )
            timings['parse'] = (time.time() - t_start) * 1000

            # 注意：不要在这里close()，因为后续存储notes还需要创建tags
            # close()只应该在整个服务关闭时调用
            timings['id_close'] = 0

            # 使用优化的批量存储（异步执行）
            if document_blocks:
                t_store_submit = time.time()
                
                # 使用优化的批量存储方法
                store_timings = await self._store_notes_batch_optimized(document_blocks)
                
                timings['store_total'] = (time.time() - t_store_submit) * 1000
                
                # 合并store内部的详细计时
                if store_timings:
                    timings.update(store_timings)
            else:
                timings['store_total'] = 0

            return len(document_blocks), timings

        except Exception as e:
            self.logger.error(f"异步解析文件失败: {file_path}, 错误: {e}")
            raise

    async def parse_and_store_with_file_id(
        self,
        file_path: str,
        file_id: int,
        relative_path: str
    ) -> tuple:
        """
        使用领导分配的file_id处理文件（优化版，不查数据库）

        Args:
            file_path: 完整文件路径
            file_id: 领导预先分配的文件ID
            relative_path: 相对路径

        Returns:
            (文档块数量, 计时字典)

        Raises:
            Exception: 处理失败时抛出异常
        """
        # 直接使用领导分配的file_id，不查数据库
        doc_count, timings = await self.async_parse_and_store_file(file_path, relative_path)
        return doc_count, timings

    async def parse_and_store_with_rollback(
        self,
        file_path: str,
        file_index_manager,
        relative_path: str,
        timestamp: int
    ) -> tuple:
        """
        原子性地处理文件：先索引，再向量，失败则回滚索引（旧版本，兼容性保留）

        Args:
            file_path: 完整文件路径
            file_index_manager: 文件索引管理器实例
            relative_path: 相对路径
            timestamp: 文件时间戳

        Returns:
            (文档块数量, 计时字典)

        Raises:
            Exception: 向量库操作失败时抛出异常，索引已被回滚
        """
        # 1. 先创建文件索引（简单快速）
        file_id = file_index_manager.get_or_create_file_id(relative_path, timestamp)

        try:
            # 2. 处理向量库，传递正确的relative_path，返回文档数和计时
            doc_count, timings = await self.async_parse_and_store_file(file_path, relative_path)
            return doc_count, timings
        except Exception as e:
            # 3. 向量库失败，回滚索引
            self.logger.error(f"向量库处理失败，回滚文件索引: {relative_path}, 错误: {e}")
            try:
                file_index_manager.delete_file(file_id)
                self.logger.debug(f"已回滚文件索引: {relative_path} (ID: {file_id})")
            except Exception as rollback_error:
                self.logger.error(f"回滚文件索引失败: {relative_path}, 错误: {rollback_error}")
            raise  # 重新抛出异常，让调用方知道失败了

    def _parse_file_sync(self, file_path: str, parser, file_id: int) -> List[NoteData]:
        """
        同步解析文件的辅助方法

        Args:
            file_path: 文件路径
            parser: 解析器实例
            file_id: 文件索引ID

        Returns:
            笔记数据列表
        """
        try:
            # 读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 对于二进制文件，传递空内容，由解析器处理
                content = ""

            # 解析文档，传递file_id（解析器已经有TagManager）
            return parser.parse(content, file_id, file_path)
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

    def _store_notes_batch(self, notes: List[NoteData], update_bm25: bool = False) -> dict:
        """
        批量存储笔记到向量数据库（同步方法）

        Args:
            notes: 笔记数据列表
            update_bm25: 是否立即更新BM25索引（默认False，延迟更新以提升性能）
            
        Returns:
            计时字典
        """
        import time
        timings = {}
        t_method_start = time.time()  # 方法总体计时
        
        try:
            if not notes:
                self.logger.debug("没有笔记需要存储")
                return timings

            # 准备副集合数据
            t_prep_sub = time.time()
            notes_to_store = notes
            sub_collection_data = []

            for note in notes:
                tag_names = self.id_service.ids_to_tags(note.tag_ids)
                sub_collection_data.append({
                    "id": note.id,
                    "tags_text": note.get_tags_text(tag_names)
                })
            timings['prep_sub'] = (time.time() - t_prep_sub) * 1000

            # === 批量处理主集合（同步调用） ===
            if notes_to_store:
                # 准备主集合数据 - 详细计时
                t_prep_main = time.time()
                
                ids = [note.id for note in notes_to_store]
                timings['prep_main_ids'] = (time.time() - t_prep_main) * 1000
                
                # 准备embedding_texts（包含ids_to_tags调用）
                t_embed_texts = time.time()
                embedding_texts = []
                for note in notes_to_store:
                    tag_names = self.id_service.ids_to_tags(note.tag_ids)
                    embedding_texts.append(note.get_embedding_text(tag_names))
                timings['prep_main_embed_texts'] = (time.time() - t_embed_texts) * 1000
                
                # 准备documents
                t_docs = time.time()
                documents = [note.content for note in notes_to_store]
                timings['prep_main_docs'] = (time.time() - t_docs) * 1000
                
                # 准备metadatas
                t_meta = time.time()
                metadatas = [note.to_dict() for note in notes_to_store]
                timings['prep_main_meta'] = (time.time() - t_meta) * 1000

                # 一次性批量存储所有文档块（同步调用，数据库内部处理并发）
                t_main = time.time()
                upsert_timings = self.vector_store.upsert_documents(
                    collection=self.main_collection,
                    ids=ids,
                    embedding_texts=embedding_texts,
                    documents=documents,
                    metadatas=metadatas,
                    _return_timings=True
                )
                timings['store_main'] = (time.time() - t_main) * 1000
                if upsert_timings:
                    timings['main_embed'] = upsert_timings.get('embed', 0)
                    timings['main_db'] = upsert_timings.get('db_upsert', 0)

                # 可选：批量更新BM25索引
                if update_bm25 and self.vector_store._is_hybrid_search_enabled():
                    collection_name = self.main_collection.name
                    doc_ids = [note.id for note in notes_to_store]
                    contents = [note.content for note in notes_to_store]

                    success = self.vector_store.bm25_retriever.add_documents(
                        collection_name, doc_ids, contents
                    )
                    if success:
                        self.logger.debug(f"📝 BM25索引批量更新完成: {len(notes_to_store)} 个文档")
                    else:
                        self.logger.warning("BM25索引批量更新失败")

            # === 批量处理副集合（同步调用） ===
            if sub_collection_data:
                t_sub = time.time()
                ids = [data["id"] for data in sub_collection_data]
                tags_texts = [data["tags_text"] for data in sub_collection_data]
                sub_upsert_timings = self.vector_store.upsert_documents(
                    collection=self.sub_collection,
                    ids=ids,
                    embedding_texts=tags_texts,
                    documents=tags_texts,
                    _return_timings=True
                )
                timings['store_sub'] = (time.time() - t_sub) * 1000
                if sub_upsert_timings:
                    timings['sub_embed'] = sub_upsert_timings.get('embed', 0)
                    timings['sub_db'] = sub_upsert_timings.get('db_upsert', 0)
            
            # 记录方法总体执行时间
            timings['_batch_method_total'] = (time.time() - t_method_start) * 1000
            
            return timings

        except Exception as e:
            self.logger.error(f"批量存储文档块失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    def remove_file_data(self, file_path: str) -> bool:
        """
        删除与指定文件相关的所有数据（同步版本）

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        try:
            # 尝试通过file_id删除（使用ID服务）
            # 获取相对路径
            relative_path = Path(file_path).name
            file_id = self.id_service.id_to_file(relative_path)
            if file_id:
                return self.remove_file_data_by_file_id(file_id)

            # 临时：仍使用旧的source_file_path查询（向后兼容）
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

    def remove_file_data_by_file_id(self, file_id: int) -> bool:
        """
        根据file_id删除文件相关的所有数据

        Args:
            file_id: 文件ID

        Returns:
            是否删除成功
        """
        try:
            # 查询所有与该file_id相关的笔记
            where_clause = {"file_id": file_id}
            main_results = self.main_collection.get(where=where_clause)

            if main_results and main_results["ids"]:
                # 获取要删除的ID列表
                ids_to_delete = main_results["ids"]

                # 从主集合删除
                self.main_collection.delete(ids=ids_to_delete)

                # 从副集合删除
                self.sub_collection.delete(ids=ids_to_delete)

                self.logger.info(f"成功删除文件ID {file_id} 的 {len(ids_to_delete)} 条笔记记录")
                return True
            else:
                self.logger.debug(f"文件ID {file_id} 没有关联的笔记数据")
                return True

        except Exception as e:
            self.logger.error(f"根据file_id删除文件数据失败: {file_id}, 错误: {e}")
            return False

    async def async_remove_file_data(self, file_path: str) -> bool:
        """
        删除与指定文件相关的所有数据（异步版本）
        使用线程池执行同步的ChromaDB操作，避免阻塞事件循环

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        try:
            # 在线程池中执行同步的remove_file_data
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # 使用默认线程池
                self.remove_file_data,
                file_path
            )
            return result

        except Exception as e:
            self.logger.error(f"异步删除文件相关数据失败: {file_path}, 错误: {e}")
            return False

    def close(self):
        """关闭服务，释放资源"""
        try:
            # 关闭线程池
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True)
                self.logger.debug("线程池已关闭")

            # 关闭ID服务
            if hasattr(self, 'id_service'):
                self.id_service.close()

            self.logger.debug("笔记服务已关闭")
        except Exception as e:
            self.logger.error(f"关闭笔记服务失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def from_plugin_context(cls, plugin_context):
        """
        从PluginContext创建NoteService实例

        Args:
            plugin_context: PluginContext插件上下文

        Returns:
            NoteService实例
        """
        return cls(plugin_context=plugin_context)

    def get_status(self):
        """
        获取服务状态

        Returns:
            包含状态信息的字典
        """
        return {
            "ready": self.collections_initialized,
            "has_vector_store": self.vector_store is not None,
            "has_plugin_context": self.plugin_context is not None,
            "provider_id": self.id_service.provider_id if hasattr(self, 'id_service') else None,
            "batch_queue_size": len(self._embedding_queue) if hasattr(self, '_embedding_queue') else 0
        }

    async def _process_batch_embedding(self):
        """
        处理批量向量化队列
        
        Returns:
            (embedding_texts, documents, metadatas) 已处理的批量数据
        """
        if not self._embedding_queue:
            return [], [], []
            
        # 取出队列中的所有数据
        batch_data = self._embedding_queue.copy()
        self._embedding_queue.clear()
        
        if not batch_data:
            return [], [], []
            
        # 提取各个字段
        embedding_texts = [item['embedding_text'] for item in batch_data]
        documents = [item['document'] for item in batch_data]
        metadatas = [item['metadata'] for item in batch_data]
        
        # self.logger.debug(f"批量向量化: {len(embedding_texts)} 个文档块")  # 注释掉
        
        return embedding_texts, documents, metadatas

    async def _add_to_embedding_queue(self, embedding_text: str, document: str, metadata: dict) -> None:
        """
        将文档添加到向量化队列
        
        Args:
            embedding_text: 待向量化的文本
            document: 原始文档内容
            metadata: 元数据
        """
        async with self._batch_lock:
            self._embedding_queue.append({
                'embedding_text': embedding_text,
                'document': document,
                'metadata': metadata
            })
            
            # 如果队列达到批量大小，触发处理
            if len(self._embedding_queue) >= self._batch_size:
                await self._process_batch_embedding()

    async def _flush_embedding_queue(self):
        """
        强制处理队列中剩余的文档（确保所有文档都被处理）
        """
        async with self._batch_lock:
            if self._embedding_queue:
                # self.logger.debug(f"强制处理剩余 {len(self._embedding_queue)} 个文档块")  # 注释掉
                await self._process_batch_embedding()

    async def _store_notes_batch_optimized(self, notes: List[NoteData]) -> dict:
        """
        优化版本的批量存储笔记，使用队列机制累积文档
        
        Args:
            notes: 笔记数据列表
            
        Returns:
            计时信息字典
        """
        import time
        timings = {}
        
        if not notes:
            return timings
            
        t_start = time.time()
        
        # 准备主集合数据
        main_collection_data = []
        sub_collection_data = []
        
        for note in notes:
            # 主集合数据
            import json
            main_collection_data.append({
                'id': note.id,
                'embedding_text': note.content,
                'document': note.content,
                'metadata': {
                    'file_id': note.file_id,
                    'tag_ids': json.dumps(note.tag_ids)  # ChromaDB不支持list，需转JSON字符串
                }
            })
            
            # 副集合数据（标签文本）
            if note.tag_ids:
                tag_names = self.id_service.ids_to_tags(note.tag_ids)
                tags_text = ' '.join(tag_names)
                sub_collection_data.append({
                    'id': note.id,
                    'embedding_text': tags_text,
                    'document': tags_text,
                    'metadata': {'file_id': note.file_id}
                })
        
        # 将数据添加到批量队列
        main_texts = []
        main_documents = []
        main_metadatas = []
        
        for data in main_collection_data:
            await self._add_to_embedding_queue(
                data['embedding_text'], 
                data['document'], 
                data['metadata']
            )
            main_texts.append(data['embedding_text'])
            main_documents.append(data['document'])
            main_metadatas.append(data['metadata'])
        
        # 处理队列中的所有数据
        await self._flush_embedding_queue()
        
        # 批量插入到主集合
        if main_texts:
            t_main = time.time()
            ids = [note.id for note in notes]
            upsert_timings = self.vector_store.upsert_documents(
                collection=self.main_collection,
                ids=ids,
                embedding_texts=main_texts,
                documents=main_documents,
                metadatas=main_metadatas,
                _return_timings=True
            )
            timings['store_main'] = (time.time() - t_main) * 1000
            if upsert_timings:
                timings['main_embed'] = upsert_timings.get('embed', 0)
                timings['main_db'] = upsert_timings.get('db_upsert', 0)
        
        # 处理副集合
        if sub_collection_data:
            t_sub = time.time()
            sub_ids = [data['id'] for data in sub_collection_data]
            sub_texts = [data['embedding_text'] for data in sub_collection_data]
            sub_documents = [data['document'] for data in sub_collection_data]
            sub_metadatas = [data['metadata'] for data in sub_collection_data]
            
            sub_upsert_timings = self.vector_store.upsert_documents(
                collection=self.sub_collection,
                ids=sub_ids,
                embedding_texts=sub_texts,
                documents=sub_documents,
                metadatas=sub_metadatas,
                _return_timings=True
            )
            timings['store_sub'] = (time.time() - t_sub) * 1000
            if sub_upsert_timings:
                timings['sub_embed'] = sub_upsert_timings.get('embed', 0)
                timings['sub_db'] = sub_upsert_timings.get('db_upsert', 0)
        
        timings['total'] = (time.time() - t_start) * 1000
        return timings
