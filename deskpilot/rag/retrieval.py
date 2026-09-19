from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from .catalog import RagCatalog
from .tracing import RetrievalTrace
from ..core.api_clients import cosine_similarity, local_hash_embedding
from ..core.models import Chunk, Document, Evidence


REFERENCE_HEADINGS = {"references", "bibliography", "参考文献", "引用文献"}


def is_reference_chunk(chunk: Chunk) -> bool:
    """识别任意层级标题路径中的参考文献区域。"""
    headings = [str(value).strip().casefold() for value in chunk.metadata.get("heading_path", [])]
    if any(heading in REFERENCE_HEADINGS for heading in headings):
        return True
    fragment = chunk.source_label.rsplit("#", 1)[-1]
    parts = [part.strip().casefold() for part in fragment.split("/")]
    return any(part in REFERENCE_HEADINGS for part in parts)


@dataclass(frozen=True)
class RetrievalFilters:
    doc_ids: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    page_min: int | None = None
    page_max: int | None = None
    heading_contains: str = ""
    include_references: bool = False


@dataclass(frozen=True)
class RetrievalRequest:
    original_query: str
    queries: tuple[str, ...]
    filters: RetrievalFilters = RetrievalFilters()
    hyde_text: str | None = None
    dense_k: int = 30
    sparse_k: int = 30
    top_k: int = 5
    use_dense: bool = True
    use_sparse: bool = True


@dataclass
class RetrievalCandidate:
    chunk: Chunk
    score: float = 0.0
    rerank_score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    query_hits: list[str] = field(default_factory=list)
    retriever_hits: list[str] = field(default_factory=list)


class DenseRetriever:
    """小型知识库默认精确扫描；接口可平滑替换为 Chroma provider。"""

    provider = "exact"

    def search(
        self,
        query: str,
        chunks: dict[str, Chunk],
        allowed_ids: set[str],
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        hash_query = local_hash_embedding(query)
        scored: list[tuple[str, float]] = []
        for chunk_id in allowed_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            if len(chunk.embedding) == len(hash_query) and cosine_similarity(
                chunk.embedding, local_hash_embedding(chunk.text)
            ) > 0.999:
                score = max(0.0, cosine_similarity(hash_query, chunk.embedding))
            elif len(chunk.embedding) == len(query_embedding):
                score = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
            else:
                continue
            scored.append((chunk_id, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class SparseRetriever:
    def __init__(self, catalog: RagCatalog) -> None:
        self.catalog = catalog

    def search(self, query: str, allowed_ids: set[str], limit: int) -> list[tuple[str, float]]:
        return self.catalog.sparse_search(query, allowed_ids, limit)


class HybridRetriever:
    """Dense/BM25 多路召回与 RRF 融合，不在粗排前执行强 MMR。"""

    def __init__(self, catalog: RagCatalog, rrf_k: int | None = None) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever(catalog)
        self.rrf_k = rrf_k or max(1, int(os.getenv("RAG_RRF_K", "60")))

    def retrieve(
        self,
        request: RetrievalRequest,
        documents: dict[str, Document],
        chunks: dict[str, Chunk],
        embed: Callable[[list[str]], list[list[float]]],
        trace: RetrievalTrace,
    ) -> list[Evidence]:
        filter_started = perf_counter()
        allowed = self._filter_chunks(request.filters, documents, chunks)
        trace.add("metadata_filter", filter_started, len(chunks), len(allowed), filters=request.filters.__dict__)
        if not allowed:
            return []

        channels: list[tuple[str, str, list[tuple[str, float]]]] = []
        queries = list(dict.fromkeys((request.original_query, *request.queries)))
        for query_index, query in enumerate(queries):
            if request.use_dense:
                embeddings = embed([query])
                dense = self.dense.search(query, chunks, allowed, embeddings[0], request.dense_k)
                channels.append((f"q{query_index}:dense", query, dense))
            if request.use_sparse:
                sparse = self.sparse.search(query, allowed, request.sparse_k)
                channels.append((f"q{query_index}:bm25", query, sparse))
        if request.hyde_text and request.use_dense:
            hyde_embedding = embed([request.hyde_text])[0]
            channels.append((
                "hyde:dense",
                "__hyde__",
                self.dense.search(request.hyde_text, chunks, allowed, hyde_embedding, request.dense_k),
            ))

        fused: dict[str, RetrievalCandidate] = {}
        for channel, query, ranked in channels:
            retriever = "bm25" if channel.endswith("bm25") else "dense"
            for rank, (chunk_id, raw_score) in enumerate(ranked, start=1):
                candidate = fused.setdefault(chunk_id, RetrievalCandidate(chunks[chunk_id]))
                candidate.score += 1.0 / (self.rrf_k + rank)
                candidate.scores[channel] = raw_score
                candidate.ranks[channel] = rank
                if query not in candidate.query_hits:
                    candidate.query_hits.append(query)
                if retriever not in candidate.retriever_hits:
                    candidate.retriever_hits.append(retriever)

        # 允许真正的空结果：只有 Dense 命中且相似度过低的候选不能靠“总有 Top-K”进入证据。
        dense_min_score = max(0.0, float(os.getenv("RAG_DENSE_MIN_SCORE", "0.08")))
        fused = {
            chunk_id: candidate for chunk_id, candidate in fused.items()
            if "bm25" in candidate.retriever_hits
            or max((score for channel, score in candidate.scores.items() if channel.endswith("dense")), default=0.0)
            >= dense_min_score
        }
        # 只折叠完全相同文本；主题多样性留给 P2 的最终证据选择。
        best_by_text: dict[str, RetrievalCandidate] = {}
        for candidate in fused.values():
            key = " ".join(candidate.chunk.text.casefold().split())
            current = best_by_text.get(key)
            if current is None or candidate.score > current.score:
                best_by_text[key] = candidate
        # RRF 只利用各通道名次。FTS 中标题会复制到每个 chunk，因而同一篇文档内的
        # 数据表、结论页可能仅凭标题词排在真正回答“流程/架构”的正文前面。这里使用
        # 查询扩展后的正文术语覆盖做一次轻量重排，不依赖具体论文或业务关键词。
        rerank_terms = self._rerank_terms(request)
        for candidate in best_by_text.values():
            body = candidate.chunk.text.casefold()
            covered = sum(self._term_in_body(term, body) for term in rerank_terms)
            coverage = covered / max(1, len(rerank_terms))
            candidate.rerank_score = candidate.score + coverage * 0.04
        ranked = sorted(
            best_by_text.values(), key=lambda item: (item.rerank_score, item.score), reverse=True,
        )
        trace.candidates = [
            {
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "rrf_score": round(item.score, 8),
                "rerank_score": round(item.rerank_score, 8),
                "scores": {key: round(value, 6) for key, value in item.scores.items()},
                "ranks": item.ranks,
                "query_hits": item.query_hits,
                "retriever_hits": item.retriever_hits,
            }
            for item in ranked[: max(request.top_k * 4, request.top_k)]
        ]
        selected = self._select_with_document_coverage(ranked, request)
        trace.selected_chunk_ids = [item.chunk.chunk_id for item in selected]
        trace.add(
            "rrf_fusion", trace.started_at, sum(len(items) for _, _, items in channels), len(ranked),
            channels=len(channels), rrf_k=self.rrf_k, dense_min_score=dense_min_score,
            required_document_coverage=len(request.filters.doc_ids),
        )
        return [
            Evidence(
                item.chunk.chunk_id,
                item.chunk.doc_id,
                item.chunk.source_label,
                item.chunk.text,
                item.rerank_score,
            )
            for item in selected
        ]

    @staticmethod
    def _rerank_terms(request: RetrievalRequest) -> tuple[str, ...]:
        """提取用于正文重排的英文实词；中文查询由 QueryOptimizer 的跨语言扩展补足。"""
        stopwords = {
            "what", "which", "when", "where", "who", "why", "how", "does", "do", "did",
            "is", "are", "was", "were", "the", "and", "or", "for", "with", "from", "about",
            "work", "works", "working", "model",
        }
        values: list[str] = []
        for query in (request.original_query, *request.queries):
            for term in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", query.casefold()):
                if term not in stopwords and term not in values:
                    values.append(term)
        return tuple(values[:20])

    @staticmethod
    def _term_in_body(term: str, body: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", body))

    @staticmethod
    def _select_with_document_coverage(
        ranked: list[RetrievalCandidate], request: RetrievalRequest,
    ) -> list[RetrievalCandidate]:
        """明确限定多文档时保留每篇证据配额，再按 RRF 分数填充剩余名额。"""
        required = set(request.filters.doc_ids)
        if len(required) < 2:
            return ranked[:request.top_k]
        quota = max(1, request.top_k // len(required))
        selected: list[RetrievalCandidate] = []
        selected_ids: set[str] = set()
        for doc_id in sorted(required):
            for candidate in (item for item in ranked if item.chunk.doc_id == doc_id):
                if sum(item.chunk.doc_id == doc_id for item in selected) >= quota:
                    break
                selected.append(candidate)
                selected_ids.add(candidate.chunk.chunk_id)
        for candidate in ranked:
            if len(selected) >= request.top_k:
                break
            if candidate.chunk.chunk_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.chunk.chunk_id)
        return selected[:request.top_k]

    @staticmethod
    def _filter_chunks(
        filters: RetrievalFilters,
        documents: dict[str, Document],
        chunks: dict[str, Chunk],
    ) -> set[str]:
        doc_ids = set(filters.doc_ids)
        paths = {str(Path(value).resolve()).casefold() for value in filters.paths}
        source_types = {value.casefold() for value in filters.source_types}
        allowed: set[str] = set()
        for chunk_id, chunk in chunks.items():
            document = documents.get(chunk.doc_id)
            if document is None or doc_ids and chunk.doc_id not in doc_ids:
                continue
            if paths and str(Path(document.path).resolve()).casefold() not in paths:
                continue
            if source_types and document.source_type.casefold() not in source_types:
                continue
            page = chunk.metadata.get("page_number")
            if filters.page_min is not None and (not isinstance(page, int) or page < filters.page_min):
                continue
            if filters.page_max is not None and (not isinstance(page, int) or page > filters.page_max):
                continue
            heading = "/".join(str(value) for value in chunk.metadata.get("heading_path", []))
            if filters.heading_contains and filters.heading_contains.casefold() not in heading.casefold():
                continue
            if not filters.include_references and is_reference_chunk(chunk):
                continue
            allowed.add(chunk_id)
        return allowed
