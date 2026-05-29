"""
笔记切片搜索引擎

基于 Tantivy BM25 的多级搜索，参考 Story 项目的 CJK bigram 分词 + 降级策略。
不接入向量分，仅接入 rerank。

多级搜索算法：
- Stage 0（严格）：所有 bigram 都必须命中
- Stage 1（宽松）：任一 bigram 命中即可
- 如果 Stage 0 结果够多就不走 Stage 1
- 降级：单字查询放大 hit_limit
"""

from __future__ import annotations

import inspect
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

try:
    import tantivy
except Exception:
    tantivy = None


# ============================================================
# 分词工具（CJK bigram，不依赖词典）
# ============================================================


def normalize_search_text(text: str) -> str:
    """统一全角→半角、去引号、小写"""
    if not isinstance(text, str):
        return ""
    result = text.strip().lower()
    # 全角→半角
    result = re.sub(
        r"[\uff01-\uff5e]",
        lambda ch: chr(ord(ch.group(0)) - 0xFEE0),
        result,
    )
    result = result.replace("\u3000", " ")
    # 统一引号
    result = re.sub(r'["\u201c\u201d\u2018\u2019]', '"', result)
    result = re.sub(r'^"+|"+$', "", result)
    return result.strip()


def iter_cjk_fragments(
    text: str,
    include_unigram: bool = True,
    max_ngram: int = 2,
) -> List[str]:
    """提取 CJK bigram + 英文 token"""
    normalized = normalize_search_text(text)
    if not normalized:
        return []

    tokens: List[str] = []
    seen: set = set()
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized)
    for part in parts:
        if re.search(r"[\u4e00-\u9fff]", part):
            min_ngram = 1 if include_unigram else 2
            upper = max(min_ngram, int(max_ngram or min_ngram))
            for ngram_size in range(min_ngram, upper + 1):
                if len(part) < ngram_size:
                    continue
                for i in range(len(part) - ngram_size + 1):
                    fragment = part[i : i + ngram_size]
                    if fragment not in seen:
                        seen.add(fragment)
                        tokens.append(fragment)
        else:
            if part not in seen:
                seen.add(part)
                tokens.append(part)
    return tokens


def select_required_cjk_ngrams(query: str) -> List[str]:
    """选择严格匹配所需的 ngram（2字为主，短词保留原样）"""
    normalized = normalize_search_text(query)
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized)
    terms: List[str] = []
    seen: set = set()
    for part in parts:
        if not re.search(r"[\u4e00-\u9fff]", part):
            if part not in seen:
                seen.add(part)
                terms.append(part)
            continue
        if len(part) <= 2:
            if part not in seen:
                seen.add(part)
                terms.append(part)
            continue
        for index in range(len(part) - 1):
            bigram = part[index : index + 2]
            if bigram not in seen:
                seen.add(bigram)
                terms.append(bigram)
    return terms


def tokenize_for_index(text: str) -> str:
    """为索引生成分词文本（CJK bigram + 英文 token，空格拼接）"""
    return " ".join(iter_cjk_fragments(text, include_unigram=True, max_ngram=2))


# ============================================================
# Tantivy 查询构建
# ============================================================


def _escape_query_term(term: str) -> str:
    """转义 Tantivy query parser 保留字符"""
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    escaped = re.sub(r"([+\-!(){}\[\]^~*?:/])", r"\\\1", escaped)
    return escaped


def _build_all_terms_query(terms: List[str]) -> str:
    """所有词都必须命中"""
    if not terms:
        return ""
    return " ".join(f"+{_escape_query_term(t)}" for t in terms)


def _build_any_terms_query(terms: List[str]) -> str:
    """任一词命中即可"""
    if not terms:
        return ""
    return " ".join(_escape_query_term(t) for t in terms)


def _resolve_hit_limit(query: str, base_limit: int) -> int:
    """短查询放大 hit_limit（降级策略）"""
    compact = re.sub(r"\s+", "", normalize_search_text(query))
    if len(compact) <= 1 and re.search(r"[\u4e00-\u9fff]", compact):
        return max(base_limit, 800)
    if len(compact) <= 3 and re.search(r"[\u4e00-\u9fff]", compact):
        return max(base_limit, 240)
    return base_limit


# ============================================================
# 切片搜索引擎
# ============================================================

# 默认参数
DEFAULT_RECALL_LIMIT = 50
DEFAULT_CHUNKS_PER_FILE = 5
MAX_RECALL_LIMIT = 200


class NoteChunkSearchEngine:
    """基于切片的 Tantivy BM25 多级搜索引擎"""

    def __init__(
        self,
        index_dir: str,
        rerank_provider: Optional[Any] = None,
    ):
        if tantivy is None:
            raise RuntimeError("Tantivy 依赖不可用，请先安装: pip install tantivy")

        self._index_dir = Path(index_dir) / "tantivy_note_chunks"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._rerank_provider = rerank_provider
        self._lock = threading.RLock()
        self._schema = self._build_schema()
        self._index = tantivy.Index(self._schema, path=str(self._index_dir))
        logger.info(f"笔记切片搜索引擎初始化完成: {self._index_dir}")

    @staticmethod
    def _build_schema():
        builder = tantivy.SchemaBuilder()
        # 存储字段
        builder.add_text_field("file_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("source_file_path", stored=True, tokenizer_name="raw")
        builder.add_unsigned_field("chunk_index", stored=True, indexed=True)
        builder.add_unsigned_field("line_start", stored=True, indexed=True)
        builder.add_unsigned_field("line_end", stored=True, indexed=True)
        builder.add_text_field("content_raw", stored=True, tokenizer_name="raw")
        # 搜索字段（CJK bigram 分词后空格拼接，用 default tokenizer 按空白分词）
        builder.add_text_field("content_cjk2", stored=False, tokenizer_name="default")
        builder.add_text_field("path_cjk2", stored=False, tokenizer_name="default")
        return builder.build()

    def has_rerank(self) -> bool:
        return self._rerank_provider is not None and hasattr(self._rerank_provider, "rerank")

    # ============================================================
    # 索引写入
    # ============================================================

    def index_chunks(self, file_id: str, source_file_path: str, chunks: List[Dict]) -> int:
        """
        为一个文件的切片建立索引（先删旧再写新）。

        Args:
            file_id: 文件 ID
            source_file_path: 相对路径
            chunks: chunk_file() 的输出列表

        Returns:
            索引的切片数量
        """
        with self._lock:
            writer = self._index.writer()
            # 删除该文件的旧索引
            writer.delete_documents("file_id", file_id)

            path_cjk2_text = tokenize_for_index(source_file_path.replace("/", " ").replace("\\", " "))

            count = 0
            for chunk in chunks:
                content = str(chunk.get("content") or "").strip()
                if not content:
                    continue
                content_cjk2_text = tokenize_for_index(content)
                if not content_cjk2_text:
                    continue

                doc = tantivy.Document()
                doc.add_text("file_id", file_id)
                doc.add_text("source_file_path", source_file_path)
                doc.add_unsigned("chunk_index", int(chunk.get("chunk_index", 0)))
                doc.add_unsigned("line_start", int(chunk.get("line_start", 0)))
                doc.add_unsigned("line_end", int(chunk.get("line_end", 0)))
                doc.add_text("content_raw", content)
                doc.add_text("content_cjk2", content_cjk2_text)
                doc.add_text("path_cjk2", path_cjk2_text)
                writer.add_document(doc)
                count += 1

            writer.commit()
            self._index.reload()
            return count

    def delete_by_file_id(self, file_id: str) -> None:
        """删除某文件的所有切片索引"""
        with self._lock:
            writer = self._index.writer()
            writer.delete_documents("file_id", file_id)
            writer.commit()
            self._index.reload()

    def rebuild_all(self, all_chunks: List[Dict]) -> int:
        """
        全量重建索引。

        Args:
            all_chunks: 所有切片，每项需包含 file_id, source_file_path, chunk_index, line_start, line_end, content

        Returns:
            索引的切片数量
        """
        with self._lock:
            writer = self._index.writer()
            writer.delete_all_documents()

            count = 0
            for chunk in all_chunks:
                file_id = str(chunk.get("file_id") or "").strip()
                source_file_path = str(chunk.get("source_file_path") or "").strip()
                content = str(chunk.get("content") or "").strip()
                if not file_id or not content:
                    continue

                content_cjk2_text = tokenize_for_index(content)
                path_cjk2_text = tokenize_for_index(
                    source_file_path.replace("/", " ").replace("\\", " ")
                )
                if not content_cjk2_text:
                    continue

                doc = tantivy.Document()
                doc.add_text("file_id", file_id)
                doc.add_text("source_file_path", source_file_path)
                doc.add_unsigned("chunk_index", int(chunk.get("chunk_index", 0)))
                doc.add_unsigned("line_start", int(chunk.get("line_start", 0)))
                doc.add_unsigned("line_end", int(chunk.get("line_end", 0)))
                doc.add_text("content_raw", content)
                doc.add_text("content_cjk2", content_cjk2_text)
                doc.add_text("path_cjk2", path_cjk2_text)
                writer.add_document(doc)
                count += 1

            writer.commit()
            self._index.reload()
            logger.info(f"笔记切片索引全量重建完成: {count} 条")
            return count

    # ============================================================
    # 多级搜索
    # ============================================================

    # 多词分割正则：空格或竖线
    _QUERY_SPLIT_RE = re.compile(r"\s*\|\s*|\s+")

    def search(
        self,
        query: str,
        limit: int = 20,
        max_chunks_per_file: int = DEFAULT_CHUNKS_PER_FILE,
    ) -> List[Dict]:
        """
        多词搜索：按空格/竖线拆词，每个词独立搜索，合并后按命中词数排序。

        流程：
        1. 拆词（空格/|）
        2. 每个词独立执行多级搜索（2-gram AND → 1+2-gram OR 降级）
        3. 合并结果，统计每个切片命中了多少个词
        4. 后过滤：content 必须包含至少一个查询词的 ngram
        5. 排序：命中词数 > BM25 分数
        6. 可选 rerank

        Args:
            query: 查询文本（支持空格和|分隔多词）
            limit: 最终返回数量
            max_chunks_per_file: 每个文件最多返回的切片数
        """
        merged = self._search_candidates(query, limit, max_chunks_per_file)
        if not merged:
            return []

        normalized_query = normalize_search_text(query)
        if self.has_rerank():
            merged = self._apply_rerank(normalized_query, merged, limit)
        else:
            merged = merged[:limit]

        self._clear_internal_fields(merged)
        return merged

    async def search_async(
        self,
        query: str,
        limit: int = 20,
        max_chunks_per_file: int = DEFAULT_CHUNKS_PER_FILE,
    ) -> List[Dict]:
        """异步搜索入口，可正确等待异步 rerank_provider。"""
        merged = self._search_candidates(query, limit, max_chunks_per_file)
        if not merged:
            return []

        normalized_query = normalize_search_text(query)
        if self.has_rerank():
            merged = await self._apply_rerank_async(normalized_query, merged, limit)
        else:
            merged = merged[:limit]

        self._clear_internal_fields(merged)
        return merged

    def _search_candidates(
        self,
        query: str,
        limit: int,
        max_chunks_per_file: int,
    ) -> List[Dict]:
        """执行 BM25 多级搜索并返回待 rerank 候选。"""
        normalized_query = normalize_search_text(query)
        if not normalized_query:
            return []

        # 拆词
        query_terms = [t for t in self._QUERY_SPLIT_RE.split(normalized_query) if t.strip()]
        if not query_terms:
            return []

        # 单词直接走原有逻辑
        if len(query_terms) == 1:
            return self._search_single_term(query_terms[0], limit, max_chunks_per_file)

        # 多词：每个词独立搜索
        per_term_limit = max(limit * 3, 50)
        term_results: Dict[str, List[Dict]] = {}
        for term in query_terms:
            results = self._search_single_term(term, per_term_limit, max_chunks_per_file)
            term_results[term] = results

        # 合并：统计每个切片命中了多少个词
        chunk_map: Dict[str, Dict] = {}  # key = file_id#chunk_index
        chunk_matched_terms: Dict[str, set] = {}

        for term, results in term_results.items():
            for r in results:
                key = f"{r.get('file_id')}#{r.get('chunk_index')}"
                if key not in chunk_map:
                    chunk_map[key] = r
                    chunk_matched_terms[key] = set()
                else:
                    # 保留更高分数
                    if float(r.get("score", 0)) > float(chunk_map[key].get("score", 0)):
                        chunk_map[key] = r
                chunk_matched_terms[key].add(term)

        # 排序：命中词数优先 > 分数
        merged = list(chunk_map.values())
        for item in merged:
            key = f"{item.get('file_id')}#{item.get('chunk_index')}"
            item["_matched_term_count"] = len(chunk_matched_terms.get(key, set()))

        merged.sort(
            key=lambda x: (int(x.get("_matched_term_count", 0)), float(x.get("score", 0))),
            reverse=True,
        )

        if not merged:
            return []

        merged = self._cap_chunks_per_file(merged, max_chunks_per_file)
        return merged

    @staticmethod
    def _cap_chunks_per_file(items: List[Dict], max_chunks_per_file: int) -> List[Dict]:
        cap = max(1, int(max_chunks_per_file or 1))
        counts: Dict[str, int] = {}
        capped: List[Dict] = []
        for item in items:
            file_id = str(item.get("file_id") or "")
            if not file_id:
                capped.append(item)
                continue
            if counts.get(file_id, 0) >= cap:
                continue
            counts[file_id] = counts.get(file_id, 0) + 1
            capped.append(item)
        return capped

    @staticmethod
    def _clear_internal_fields(items: List[Dict]) -> None:
        for item in items:
            item.pop("_matched_term_count", None)

    def _search_single_term(
        self,
        term: str,
        limit: int,
        max_chunks_per_file: int,
    ) -> List[Dict]:
        """单词多级搜索（2-gram AND → 1+2-gram OR 降级）+ 后过滤"""
        normalized = normalize_search_text(term)
        if not normalized:
            return []

        required_terms = select_required_cjk_ngrams(normalized)
        all_cjk2_terms = iter_cjk_fragments(normalized, include_unigram=True, max_ngram=2)

        if not required_terms and not all_cjk2_terms:
            return []

        base_hit_limit = max(50, limit * 8)
        hit_limit = _resolve_hit_limit(normalized, base_hit_limit)
        hit_limit_ceiling = min(20000, hit_limit * 8)
        desired_file_count = max(limit, DEFAULT_RECALL_LIMIT)

        searcher = self._index.searcher()

        # 第一次：2-gram AND
        strict_query = _build_all_terms_query(required_terms)
        candidates = self._search_stage(
            searcher=searcher,
            query_string=strict_query,
            hit_limit=hit_limit,
            hit_limit_ceiling=hit_limit_ceiling,
            desired_file_count=desired_file_count,
            max_chunks_per_file=max_chunks_per_file,
            stage_rank=0,
        )

        # 不够则第二次：1+2-gram OR 补充
        if self._count_unique_files(candidates) < desired_file_count:
            loose_query = _build_any_terms_query(all_cjk2_terms)
            loose_candidates = self._search_stage(
                searcher=searcher,
                query_string=loose_query,
                hit_limit=hit_limit,
                hit_limit_ceiling=hit_limit_ceiling,
                desired_file_count=desired_file_count,
                max_chunks_per_file=max_chunks_per_file,
                stage_rank=1,
            )
            candidates = self._merge_candidates(
                candidates, loose_candidates, desired_file_count, max_chunks_per_file
            )

        return candidates[:limit]

    def _search_stage(
        self,
        searcher,
        query_string: str,
        hit_limit: int,
        hit_limit_ceiling: int,
        desired_file_count: int,
        max_chunks_per_file: int,
        stage_rank: int,
    ) -> List[Dict]:
        """单阶段搜索，自动扩大 hit_limit 直到满足文件数目标"""
        if not query_string:
            return []

        fields = ["content_cjk2", "path_cjk2"]
        parsed_query = self._index.parse_query(query_string, fields)
        fetch_limit = max(1, hit_limit)
        ceiling = max(fetch_limit, hit_limit_ceiling)

        while True:
            candidates: List[Dict] = []
            search_result = searcher.search(parsed_query, limit=fetch_limit)
            seen_chunk_keys: set = set()
            file_chunk_counts: Dict[str, int] = {}

            for score, doc_address in search_result.hits:
                doc = searcher.doc(doc_address)
                file_id = self._doc_first(doc, "file_id")
                source_file_path = self._doc_first(doc, "source_file_path")
                chunk_index = self._doc_first_int(doc, "chunk_index")
                if not file_id:
                    continue

                chunk_key = f"{file_id}#{chunk_index}"
                if chunk_key in seen_chunk_keys:
                    continue
                if file_chunk_counts.get(file_id, 0) >= max_chunks_per_file:
                    continue

                seen_chunk_keys.add(chunk_key)
                file_chunk_counts[file_id] = file_chunk_counts.get(file_id, 0) + 1

                candidates.append({
                    "file_id": file_id,
                    "source_file_path": source_file_path,
                    "chunk_index": chunk_index,
                    "line_start": self._doc_first_int(doc, "line_start"),
                    "line_end": self._doc_first_int(doc, "line_end"),
                    "content": self._doc_first(doc, "content_raw"),
                    "score": float(score),
                    "stage_rank": stage_rank,
                })

                if len(file_chunk_counts) >= desired_file_count:
                    break

            # 检查是否需要扩大搜索
            if self._count_unique_files(candidates) >= desired_file_count:
                return candidates
            if len(search_result.hits) < fetch_limit or fetch_limit >= ceiling:
                return candidates
            fetch_limit = min(ceiling, fetch_limit * 2)

    def _merge_candidates(
        self,
        strict: List[Dict],
        loose: List[Dict],
        desired_file_count: int,
        max_chunks_per_file: int,
    ) -> List[Dict]:
        """合并严格和宽松结果，去重"""
        if not strict:
            return loose

        merged = list(strict)
        seen_keys = {f"{c['file_id']}#{c['chunk_index']}" for c in strict}
        file_chunk_counts: Dict[str, int] = {}
        for c in strict:
            fid = c["file_id"]
            file_chunk_counts[fid] = file_chunk_counts.get(fid, 0) + 1

        unique_files = set(file_chunk_counts.keys())

        for item in loose:
            key = f"{item['file_id']}#{item['chunk_index']}"
            if key in seen_keys:
                continue
            fid = item["file_id"]
            if file_chunk_counts.get(fid, 0) >= max_chunks_per_file:
                continue
            seen_keys.add(key)
            file_chunk_counts[fid] = file_chunk_counts.get(fid, 0) + 1
            merged.append(item)
            unique_files.add(fid)
            if len(unique_files) >= desired_file_count:
                break

        return merged

    def _apply_rerank(self, query: str, candidates: List[Dict], limit: int) -> List[Dict]:
        """调用 rerank_provider 对候选重排"""
        if not candidates:
            return []

        documents = [str(c.get("content") or "") for c in candidates]
        try:
            rerank_method = self._rerank_provider.rerank
            rerank_resp = rerank_method(query=query, documents=documents)
            # 处理异步
            if inspect.isawaitable(rerank_resp):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    del loop
                    if hasattr(rerank_resp, "close"):
                        rerank_resp.close()
                    return candidates[:limit]
                except RuntimeError:
                    rerank_resp = asyncio.run(rerank_resp)

            scored = self._extract_rerank_scores(rerank_resp, candidates)
            if scored:
                scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                return scored[:limit]
        except Exception as e:
            logger.warning(f"笔记切片重排失败，降级为 BM25 排序: {e}")

        # 降级：按原始 BM25 分数排序
        return candidates[:limit]

    async def _apply_rerank_async(self, query: str, candidates: List[Dict], limit: int) -> List[Dict]:
        """异步调用 rerank_provider 对候选重排。"""
        if not candidates:
            return []

        documents = [str(c.get("content") or "") for c in candidates]
        try:
            rerank_method = self._rerank_provider.rerank
            rerank_resp = rerank_method(query=query, documents=documents)
            if inspect.isawaitable(rerank_resp):
                rerank_resp = await rerank_resp

            scored = self._extract_rerank_scores(rerank_resp, candidates)
            if scored:
                scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                return scored[:limit]
        except Exception as e:
            logger.warning(f"笔记切片重排失败，降级为 BM25 排序: {e}")

        return candidates[:limit]

    @staticmethod
    def _extract_rerank_scores(rerank_resp: Any, candidates: List[Dict]) -> List[Dict]:
        """从 rerank 响应中提取分数"""
        if rerank_resp is None:
            return []

        items = None
        if isinstance(rerank_resp, dict):
            if rerank_resp.get("code") not in (None, 0, 200, "0", "200"):
                return []
            items = rerank_resp.get("results")
            if items is None:
                data = rerank_resp.get("data")
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("results") or data.get("items")
        elif isinstance(rerank_resp, list):
            items = rerank_resp

        if not isinstance(items, list):
            return []

        scored: List[Dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", -1))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(candidates):
                continue
            try:
                score = float(item.get("relevance_score", 0.0) or item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            result = dict(candidates[index])
            result["score"] = score
            scored.append(result)

        return scored

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _count_unique_files(candidates: List[Dict]) -> int:
        return len({c.get("file_id") for c in candidates if c.get("file_id")})

    @staticmethod
    def _post_filter(candidates: List[Dict], query_terms: List[str]) -> List[Dict]:
        """后过滤：content 必须包含至少一个原始查询词（精确子串匹配）"""
        if not query_terms:
            return candidates
        filtered = []
        for c in candidates:
            content = str(c.get("content") or "").lower()
            if any(term in content for term in query_terms):
                filtered.append(c)
        return filtered

    @staticmethod
    def _doc_first(doc, field_name: str) -> str:
        """从 tantivy Document 提取字段第一个值"""
        values = None
        try:
            if hasattr(doc, "to_dict"):
                d = doc.to_dict()
                if isinstance(d, dict):
                    values = d.get(field_name)
        except Exception:
            pass
        if values is None:
            try:
                values = doc[field_name]
            except Exception:
                pass
        if values is None:
            try:
                values = getattr(doc, field_name, None)
            except Exception:
                pass
        if values is None:
            return ""
        if isinstance(values, (list, tuple)):
            return str(values[0]) if values else ""
        return str(values)

    @classmethod
    def _doc_first_int(cls, doc, field_name: str) -> int:
        val = cls._doc_first(doc, field_name)
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def close(self) -> None:
        """关闭引擎（释放资源）"""
        pass  # Tantivy Index 无需显式关闭
