from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from .catalog import RagCatalog
from .reranker import DisabledReranker, RerankDocument, build_reranker
from .tracing import RetrievalTrace
from ..context.budget import estimate_tokens
from ..core.api_clients import cosine_similarity, local_hash_embedding
from ..core.models import Chunk, Document, Evidence


REFERENCE_HEADINGS = {"references", "bibliography", "参考文献", "引用文献"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(float(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def is_reference_chunk(chunk: Chunk) -> bool:
    """识别任意层级标题路径中的参考文献区域。"""
    headings = [str(value).strip().casefold() for value in chunk.metadata.get("heading_path", [])]
    if any(heading in REFERENCE_HEADINGS for heading in headings):
        return True
    fragment = chunk.source_label.rsplit("#", 1)[-1]
    return any(part.strip().casefold() in REFERENCE_HEADINGS for part in fragment.split("/"))


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
    context_expansion: str = "none"
    rerank_provider: str | None = None
    relevance_threshold: float | None = None
    max_context_tokens: int | None = None
    sentence_window_size: int | None = None
    mmr_lambda: float | None = None


@dataclass
class RetrievalCandidate:
    chunk: Chunk
    score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    expanded_text: str = ""
    expansion: str = "none"
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    query_hits: list[str] = field(default_factory=list)
    retriever_hits: list[str] = field(default_factory=list)


class DenseRetriever:
    """小型知识库默认精确扫描，接口可平滑替换成独立向量数据库。"""

    provider = "exact"

    def search(self, query: str, chunks: dict[str, Chunk], allowed_ids: set[str],
               query_embedding: list[float], limit: int) -> list[tuple[str, float]]:
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
    """P2 检索流水线：混合召回、RRF、精排、上下文扩展和覆盖感知 MMR。"""

    def __init__(self, catalog: RagCatalog, rrf_k: int | None = None) -> None:
        self.catalog = catalog
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever(catalog)
        self.rrf_k = rrf_k or max(1, int(os.getenv("RAG_RRF_K", "60")))

    def retrieve(self, request: RetrievalRequest, documents: dict[str, Document],
                 chunks: dict[str, Chunk], embed: Callable[[list[str]], list[list[float]]],
                 trace: RetrievalTrace) -> list[Evidence]:
        filter_started = perf_counter()
        allowed = self._filter_chunks(request.filters, documents, chunks)
        trace.add("metadata_filter", filter_started, len(chunks), len(allowed), filters=request.filters.__dict__)
        if not allowed:
            return []

        channels = self._recall(request, chunks, allowed, embed)
        fusion_started = perf_counter()
        ranked = self._fuse(channels, chunks)
        dense_min_score = max(0.0, float(os.getenv("RAG_DENSE_MIN_SCORE", "0.08")))
        ranked = [item for item in ranked if "bm25" in item.retriever_hits or max(
            (score for channel, score in item.scores.items() if channel.endswith("dense")), default=0.0,
        ) >= dense_min_score]
        trace.add("rrf_fusion", fusion_started, sum(len(items) for _, _, items in channels), len(ranked),
                  channels=len(channels), rrf_k=self.rrf_k, dense_min_score=dense_min_score)

        if not _env_bool("RAG_P2_ENABLED", True):
            for item in ranked:
                item.rerank_score = item.score
                item.final_score = item.score
            selected = self._legacy_document_coverage(ranked, request)
            return self._finish(selected, ranked, request, trace, "disabled")

        ranked, effective_provider = self._rerank(request, ranked, documents, trace)
        ranked = self._apply_threshold(request, ranked, effective_provider, trace)
        expanded = self._expand(request, ranked, trace)
        selected = self._select_mmr(request, expanded, trace)
        return self._finish(selected, ranked, request, trace, effective_provider)

    def _recall(self, request: RetrievalRequest, chunks: dict[str, Chunk], allowed: set[str],
                embed: Callable[[list[str]], list[list[float]]]) -> list[tuple[str, str, list[tuple[str, float]]]]:
        channels: list[tuple[str, str, list[tuple[str, float]]]] = []
        queries = list(dict.fromkeys((request.original_query, *request.queries)))
        for query_index, query in enumerate(queries):
            if request.use_dense:
                channels.append((f"q{query_index}:dense", query, self.dense.search(
                    query, chunks, allowed, embed([query])[0], request.dense_k,
                )))
            if request.use_sparse:
                channels.append((f"q{query_index}:bm25", query,
                                 self.sparse.search(query, allowed, request.sparse_k)))
        if request.hyde_text and request.use_dense:
            embedding = embed([request.hyde_text])[0]
            channels.append(("hyde:dense", "__hyde__", self.dense.search(
                request.hyde_text, chunks, allowed, embedding, request.dense_k,
            )))
        return channels

    def _fuse(self, channels: list[tuple[str, str, list[tuple[str, float]]]],
              chunks: dict[str, Chunk]) -> list[RetrievalCandidate]:
        fused: dict[str, RetrievalCandidate] = {}
        for channel, query, values in channels:
            retriever = "bm25" if channel.endswith("bm25") else "dense"
            for rank, (chunk_id, raw_score) in enumerate(values, start=1):
                candidate = fused.setdefault(chunk_id, RetrievalCandidate(chunks[chunk_id]))
                candidate.score += 1.0 / (self.rrf_k + rank)
                candidate.scores[channel] = raw_score
                candidate.ranks[channel] = rank
                if query not in candidate.query_hits:
                    candidate.query_hits.append(query)
                if retriever not in candidate.retriever_hits:
                    candidate.retriever_hits.append(retriever)
        unique: dict[str, RetrievalCandidate] = {}
        for candidate in fused.values():
            key = " ".join(candidate.chunk.text.casefold().split())
            current = unique.get(key)
            if current is None or candidate.score > current.score:
                unique[key] = candidate
        return sorted(unique.values(), key=lambda item: item.score, reverse=True)

    def _rerank(self, request: RetrievalRequest, ranked: list[RetrievalCandidate],
                documents: dict[str, Document], trace: RetrievalTrace) -> tuple[list[RetrievalCandidate], str]:
        started = perf_counter()
        pool_size = _env_int("RAG_RERANK_CANDIDATES", 30, 1, 200)
        top_n = _env_int("RAG_RERANK_TOP_N", 12, 1, 100)
        pool = ranked[:pool_size]
        requested = request.rerank_provider or os.getenv("RAG_RERANK_PROVIDER", "lexical")
        error = ""
        try:
            reranker = build_reranker(requested)
            scores = reranker.rerank(request.original_query, [RerankDocument(
                chunk_id=item.chunk.chunk_id,
                text=item.chunk.text,
                title=documents.get(item.chunk.doc_id).title if documents.get(item.chunk.doc_id) else "",
                heading_path=tuple(str(value) for value in item.chunk.metadata.get("heading_path", [])),
                page_number=item.chunk.metadata.get("page_number"),
                base_score=item.score,
            ) for item in pool])
            effective = reranker.provider
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reranker = DisabledReranker()
            scores = reranker.rerank(request.original_query, [RerankDocument(
                item.chunk.chunk_id, item.chunk.text, base_score=item.score,
            ) for item in pool])
            effective = "disabled_fallback"
        by_id = {value.chunk_id: value.score for value in scores}
        for item in pool:
            item.rerank_score = float(by_id.get(item.chunk.chunk_id, item.score))
            item.final_score = item.rerank_score
        pool.sort(key=lambda item: (item.rerank_score, item.score), reverse=True)
        output = pool[:top_n]
        trace.add("rerank", started, len(ranked), len(output), requested_provider=requested,
                  effective_provider=effective, candidate_limit=pool_size, top_n=top_n,
                  fallback=bool(error), error=error)
        return output, effective

    def _apply_threshold(self, request: RetrievalRequest, ranked: list[RetrievalCandidate],
                         provider: str, trace: RetrievalTrace) -> list[RetrievalCandidate]:
        started = perf_counter()
        threshold = request.relevance_threshold
        if threshold is None:
            threshold = _env_float("RAG_RELEVANCE_THRESHOLD", 0.0, 0.0, 1.0)
        apply_threshold = provider not in {"disabled", "disabled_fallback"} and threshold > 0
        output = [item for item in ranked if item.rerank_score >= threshold] if apply_threshold else ranked
        trace.add("relevance_threshold", started, len(ranked), len(output), threshold=threshold,
                  applied=apply_threshold, provider=provider)
        return output

    def _expand(self, request: RetrievalRequest, ranked: list[RetrievalCandidate],
                trace: RetrievalTrace) -> list[RetrievalCandidate]:
        started = perf_counter()
        strategy = request.context_expansion if _env_bool("RAG_CONTEXT_EXPANSION_ENABLED", True) else "none"
        if strategy not in {"sentence_window", "parent", "none"}:
            strategy = "none"
        token_limit = request.max_context_tokens or _env_int("RAG_EVIDENCE_MAX_TOKENS", 900, 80, 4000)
        window_size = request.sentence_window_size
        if window_size is None:
            window_size = _env_int("RAG_SENTENCE_WINDOW_SIZE", 2, 0, 8)
        fallback_count = 0
        for item in ranked:
            text = ""
            if strategy == "sentence_window":
                nodes = self.catalog.get_sentence_window(
                    [str(value) for value in item.chunk.metadata.get("sentence_ids", [])], window_size,
                )
                text = "".join(node.text for node in nodes)
            elif strategy == "parent":
                parent = self.catalog.get_parent_chunk(str(item.chunk.metadata.get("parent_chunk_id", "")))
                text = parent.text if parent else ""
            if not text:
                text = item.chunk.text
                item.expansion = "child_fallback" if strategy != "none" else "none"
                fallback_count += int(strategy != "none")
            else:
                item.expansion = strategy
            item.expanded_text = self._bounded_context(text, item.chunk.text, token_limit)

        merged: dict[str, RetrievalCandidate] = {}
        for item in ranked:
            parent_id = str(item.chunk.metadata.get("parent_chunk_id", ""))
            key = parent_id if parent_id and item.expansion in {"parent", "sentence_window"} else item.chunk.chunk_id
            current = merged.get(key)
            if current is None:
                merged[key] = item
                continue
            current.query_hits = list(dict.fromkeys((*current.query_hits, *item.query_hits)))
            current.retriever_hits = list(dict.fromkeys((*current.retriever_hits, *item.retriever_hits)))
            current.scores.update(item.scores)
            current.ranks.update(item.ranks)
        output = list(merged.values())
        trace.add("context_expansion", started, len(ranked), len(output), strategy=strategy,
                  token_limit=token_limit, sentence_window_size=window_size, fallback_count=fallback_count)
        return output

    @staticmethod
    def _bounded_context(text: str, child_text: str, max_tokens: int) -> str:
        if estimate_tokens(text) <= max_tokens:
            return text
        max_chars = max(200, max_tokens * 3)
        position = text.find(child_text)
        if position < 0:
            return text[:max_chars]
        start = max(0, position - max(0, (max_chars - len(child_text)) // 2))
        end = min(len(text), start + max_chars)
        return text[max(0, end - max_chars):end]

    def _select_mmr(self, request: RetrievalRequest, ranked: list[RetrievalCandidate],
                    trace: RetrievalTrace) -> list[RetrievalCandidate]:
        started = perf_counter()
        limit = min(request.top_k, _env_int("RAG_FINAL_TOP_K", request.top_k, 1, 50))
        mmr_lambda = request.mmr_lambda
        if mmr_lambda is None:
            mmr_lambda = _env_float("RAG_MMR_LAMBDA", 0.72, 0.0, 1.0)
        selected: list[RetrievalCandidate] = []
        remaining = list(ranked)

        def add(candidate: RetrievalCandidate) -> None:
            if candidate in remaining and len(selected) < limit:
                selected.append(candidate)
                remaining.remove(candidate)

        for doc_id in request.filters.doc_ids:
            candidate = next((item for item in remaining if item.chunk.doc_id == doc_id), None)
            if candidate:
                add(candidate)
        for query in request.queries:
            candidate = next((item for item in remaining if query in item.query_hits), None)
            if candidate:
                add(candidate)
        max_relevance = max((item.rerank_score for item in ranked), default=1.0) or 1.0
        while remaining and len(selected) < limit:
            best: RetrievalCandidate | None = None
            best_score = float("-inf")
            for candidate in remaining:
                relevance = candidate.rerank_score / max_relevance
                redundancy = max((cosine_similarity(
                    candidate.chunk.embedding or local_hash_embedding(candidate.chunk.text),
                    chosen.chunk.embedding or local_hash_embedding(chosen.chunk.text),
                ) for chosen in selected), default=0.0)
                score = mmr_lambda * relevance - (1.0 - mmr_lambda) * max(0.0, redundancy)
                if score > best_score:
                    best, best_score = candidate, score
            if best is None:
                break
            best.final_score = best_score
            add(best)
        trace.add("final_mmr", started, len(ranked), len(selected), mmr_lambda=mmr_lambda,
                  required_documents=len(request.filters.doc_ids), subqueries=len(request.queries), limit=limit)
        return selected

    def _finish(self, selected: list[RetrievalCandidate], ranked: list[RetrievalCandidate],
                request: RetrievalRequest, trace: RetrievalTrace, provider: str) -> list[Evidence]:
        trace.candidates = [{
            "chunk_id": item.chunk.chunk_id, "doc_id": item.chunk.doc_id,
            "rrf_score": round(item.score, 8), "rerank_score": round(item.rerank_score, 8),
            "scores": {key: round(value, 6) for key, value in item.scores.items()},
            "ranks": item.ranks, "query_hits": item.query_hits,
            "retriever_hits": item.retriever_hits, "expansion": item.expansion,
        } for item in ranked[:max(request.top_k * 4, request.top_k)]]
        trace.selected_chunk_ids = [item.chunk.chunk_id for item in selected]
        return [Evidence(
            item.chunk.chunk_id, item.chunk.doc_id, item.chunk.source_label,
            item.expanded_text or item.chunk.text, item.rerank_score,
            metadata={
                "rrf_score": item.score, "rerank_score": item.rerank_score, "reranker": provider,
                "query_hits": item.query_hits, "retriever_hits": item.retriever_hits,
                "expansion": item.expansion,
                "parent_chunk_id": item.chunk.metadata.get("parent_chunk_id"),
                "page_number": item.chunk.metadata.get("page_number"),
                "heading_path": item.chunk.metadata.get("heading_path", []),
                "context_tokens": estimate_tokens(item.expanded_text or item.chunk.text),
            },
        ) for item in selected]

    @staticmethod
    def _legacy_document_coverage(ranked: list[RetrievalCandidate], request: RetrievalRequest) -> list[RetrievalCandidate]:
        required = set(request.filters.doc_ids)
        if len(required) < 2:
            return ranked[:request.top_k]
        selected: list[RetrievalCandidate] = []
        for doc_id in sorted(required):
            match = next((item for item in ranked if item.chunk.doc_id == doc_id), None)
            if match:
                selected.append(match)
        selected.extend(item for item in ranked if item not in selected)
        return selected[:request.top_k]

    @staticmethod
    def _filter_chunks(filters: RetrievalFilters, documents: dict[str, Document],
                       chunks: dict[str, Chunk]) -> set[str]:
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
