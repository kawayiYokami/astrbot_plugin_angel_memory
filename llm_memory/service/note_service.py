"""
笔记服务 - LLM的笔记管理服务

提供笔记的增删改查功能，与记忆系统共享同一个向量数据库实例。
支持目录解析、文档向量化存储和智能查询。
"""

import asyncio
import concurrent.futures
from typing import List, Dict
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

        # 2. 第二阶段：重排 (Reranking)
        # 使用轻量级方法查询副集合（只获取标签相关性分数）
        # 副集合不存储 metadata，所以不能使用 search_notes
        tag_scores = self.vector_store._search_vector_scores(
            collection=self.sub_collection,
            query=query,
            limit=len(all_recalled_notes)
        )

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

    # ===== 文件解析和文档向量化 =====

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

                # 直接同步存储（跳过批量队列）
                store_timings = self._store_notes_batch(document_blocks)

                timings['store_total'] = (time.time() - t_store_submit) * 1000

                if store_timings:
                    timings.update(store_timings)
            else:
                timings['store_total'] = 0

            return len(document_blocks), timings

        except Exception as e:
            self.logger.error(f"同步解析文件失败: {file_path}, 错误: {e}")
            raise

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

    def _store_notes_batch(self, notes: List[NoteData]) -> dict:
        """
        批量存储笔记到向量数据库（同步方法）

        存储策略：
        - 主集合：存储完整笔记信息（vector 基于"内容+标签"，metadata 包含所有数据）
        - 副集合：仅存储标签向量（vector 基于"标签文本"，不存储 metadata）

        Args:
            notes: 笔记数据列表

        Returns:
            计时字典
        """
        import time
        timings = {}
        t_method_start = time.time()

        try:
            if not notes:
                self.logger.debug("没有笔记需要存储")
                return timings

            # === 批量处理主集合 ===
            t_prep_main = time.time()

            # 准备主集合数据
            ids = [note.id for note in notes]

            # 准备 embedding_texts（内容 + 标签文本）
            embedding_texts = []
            for note in notes:
                tag_names = self.id_service.ids_to_tags(note.tag_ids)
                embedding_texts.append(note.get_embedding_text(tag_names))

            # 准备 metadatas（包含所有笔记数据）
            metadatas = [note.to_dict() for note in notes]

            timings['prep_main'] = (time.time() - t_prep_main) * 1000

            # 批量存储到主集合
            t_main = time.time()
            upsert_timings = self.vector_store.upsert_documents(
                collection=self.main_collection,
                ids=ids,
                embedding_texts=embedding_texts,
                metadatas=metadatas,
                _return_timings=True
            )
            timings['store_main'] = (time.time() - t_main) * 1000
            if upsert_timings:
                timings['main_embed'] = upsert_timings.get('embed', 0)
                timings['main_db'] = upsert_timings.get('db_upsert', 0)


            # === 批量处理副集合 ===
            t_prep_sub = time.time()

            # 准备副集合数据（仅标签文本）
            sub_ids = []
            sub_tags_texts = []
            for note in notes:
                tag_names = self.id_service.ids_to_tags(note.tag_ids)
                tags_text = note.get_tags_text(tag_names)
                sub_ids.append(note.id)
                sub_tags_texts.append(tags_text)

            timings['prep_sub'] = (time.time() - t_prep_sub) * 1000

            # 批量存储到副集合（不传 metadatas）
            t_sub = time.time()
            sub_upsert_timings = self.vector_store.upsert_documents(
                collection=self.sub_collection,
                ids=sub_ids,
                embedding_texts=sub_tags_texts,
                metadatas=None,  # 副集合不存储 metadata
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
