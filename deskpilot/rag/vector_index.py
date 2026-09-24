from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from pathlib import Path
from time import perf_counter

from .catalog import RagCatalog
from .chunker import CHUNKER_VERSION, ChunkingConfig, chunk_document_with_metadata
from .document_parsers import PARSER_VERSION, parse_document, scan_supported_files
from .tracing import RetrievalTrace
from .retrieval import HybridRetriever, RetrievalFilters, RetrievalRequest, is_reference_chunk
from ..core.api_clients import OpenAICompatibleClient, cosine_similarity, local_hash_embedding
from ..core.config import INDEX_DIR, ensure_dirs, load_config
from ..core.encoding_utils import fix_mojibake, is_probably_garbled
from ..core.models import Chunk, Document, Evidence


INDEX_FILE = INDEX_DIR / "index.json"


class DocumentIndex:
    def __init__(self, index_file: Path = INDEX_FILE):
        ensure_dirs()
        self.index_file = index_file
        self.catalog = RagCatalog(index_file.with_suffix(".catalog.sqlite3"))
        self.hybrid_retriever = HybridRetriever(self.catalog)
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.client = OpenAICompatibleClient(load_config())
        self.chunking_config = ChunkingConfig.from_env()
        self.last_trace: RetrievalTrace | None = None
        self.last_index_trace: dict[str, object] = {}
        self._embedding_space = self._configured_embedding_space()
        self.load()

    def load(self) -> None:
        if not self.index_file.exists():
            return
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        self.documents = {
            doc_id: Document.from_dict(doc_data) for doc_id, doc_data in data.get("documents", {}).items()
        }
        self.chunks = {
            chunk_id: Chunk.from_dict(chunk_data) for chunk_id, chunk_data in data.get("chunks", {}).items()
        }
        self.catalog.sync_search_index(self.documents, self.chunks)

    def save(self) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "chunks": {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
        }
        temporary = self.index_file.with_suffix(self.index_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.index_file)

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.catalog.clear_content()
        self.save()

    def add_file(self, path: Path) -> tuple[Document, int]:
        document = parse_document(path)
        return self.add_document(document)

    def add_document(self, document: Document) -> tuple[Document, int]:
        started = perf_counter()
        canonical_path = str(Path(document.path).resolve()) if document.path else document.path
        source_hash = str(document.metadata.get("sha256", ""))
        configured_space = self._configured_embedding_space()
        unchanged_id = self.catalog.unchanged_document(
            canonical_path, source_hash, PARSER_VERSION, CHUNKER_VERSION, configured_space
        )
        if unchanged_id and unchanged_id in self.documents:
            count = sum(chunk.doc_id == unchanged_id for chunk in self.chunks.values())
            self.last_index_trace = {
                "status": "skipped_unchanged",
                "doc_id": unchanged_id,
                "chunks": count,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
            return self.documents[unchanged_id], count

        self._embedding_space = configured_space
        result = chunk_document_with_metadata(
            document,
            config=self.chunking_config,
            embed_sentences=self._embed_cached,
        )
        parsed_chunks = result.chunks
        chunks = [chunk for chunk in parsed_chunks if not is_probably_garbled(chunk.text)]
        skipped = len(parsed_chunks) - len(chunks)
        if skipped:
            document.metadata["skipped_low_quality_chunks"] = skipped
            document.metadata["parsed_chunks"] = len(parsed_chunks)
        document.metadata.update({
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "chunking_strategy": result.strategy,
            "chunking_stats": result.stats,
            "embedding_space": self._embedding_space,
        })
        embeddings = self._embed_cached([chunk.text for chunk in chunks]) if chunks else []
        document.metadata["embedding_space"] = self._embedding_space
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.metadata["embedding_space"] = self._embedding_space

        # 同一路径的新内容会得到新 doc_id；内存快照也必须停用旧版本。
        old_doc_ids = {
            doc_id for doc_id, existing in self.documents.items()
            if doc_id != document.doc_id and existing.path
            and str(Path(existing.path).resolve()) == canonical_path
        }
        old_chunk_ids = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.doc_id == document.doc_id or chunk.doc_id in old_doc_ids
        ]
        for chunk_id in old_chunk_ids:
            del self.chunks[chunk_id]
        for doc_id in old_doc_ids:
            self.documents.pop(doc_id, None)
        self.documents[document.doc_id] = document
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        self.catalog.commit_document(
            document,
            chunks,
            result.sentence_nodes,
            result.parent_chunks,
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_space=self._embedding_space,
        )
        self.save()
        self.last_index_trace = {
            "status": "indexed",
            "doc_id": document.doc_id,
            "strategy": result.strategy,
            "blocks": result.stats.get("blocks", 0),
            "sentences": result.stats.get("sentences", 0),
            "chunks": len(chunks),
            "skipped_low_quality": skipped,
            "embedding_space": self._embedding_space,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        }
        return document, len(chunks)

    def add_folder(self, folder: Path) -> list[tuple[Document, int]]:
        results: list[tuple[Document, int]] = []
        for path in scan_supported_files(folder):
            results.append(self.add_file(path))
        return results

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        hybrid_enabled = os.getenv("RAG_HYBRID_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        if hybrid_enabled:
            title_doc_ids = self.matching_title_doc_ids(query)
            request = RetrievalRequest(
                query,
                (query,),
                filters=RetrievalFilters(doc_ids=tuple(sorted(title_doc_ids))),
                top_k=top_k,
            )
            return self.search_hybrid(request)
        return self._legacy_search(query, top_k)

    def search_hybrid(self, request: RetrievalRequest) -> list[Evidence]:
        """执行 P1 混合召回；P0 路径可通过 RAG_HYBRID_ENABLED=false 回滚。"""
        trace = RetrievalTrace(query=request.original_query, mode="hybrid")
        self.last_trace = trace
        if not self.chunks:
            return []
        searchable_chunks = self.chunks
        if not request.filters.include_references:
            reference_starts = self._reference_start_positions()
            searchable_chunks = {
                chunk_id: chunk for chunk_id, chunk in self.chunks.items()
                if chunk.position < reference_starts.get(chunk.doc_id, sys.maxsize)
            }
        return self.hybrid_retriever.retrieve(
            request,
            self.documents,
            searchable_chunks,
            self.client.embed,
            trace,
        )

    def _legacy_search(self, query: str, top_k: int = 5) -> list[Evidence]:
        trace = RetrievalTrace(query=query)
        self.last_trace = trace
        if not self.chunks:
            return []
        embed_started = perf_counter()
        query_embedding = self.client.embed([query])[0]
        trace.add("embed_query", embed_started, input_count=1, output_count=1,
                  dimensions=len(query_embedding))
        scored: list[tuple[float, Chunk]] = []
        query_terms = self._query_terms(query)
        title_doc_ids = self.matching_title_doc_ids(query)
        reference_starts = self._reference_start_positions()
        hash_query = local_hash_embedding(query)
        score_started = perf_counter()
        for chunk in self.chunks.values():
            # 查询明确包含某个文档标题时，优先在该文档集合内检索。这样既避免同领域
            # 其他论文的参考文献串入答案，也支持一次问题同时点名多份文档。
            if title_doc_ids and chunk.doc_id not in title_doc_ids:
                continue
            if chunk.position >= reference_starts.get(chunk.doc_id, sys.maxsize):
                continue
            if is_probably_garbled(chunk.text):
                continue
            # 旧索引可能混有 API 与本地 hash 向量；绝不能截断维度后计算余弦。
            vector_score = self._vector_score(chunk, query_embedding, hash_query)
            title = self.documents.get(chunk.doc_id)
            lexical_score = self._lexical_score(query_terms, chunk.text + " " + (title.title if title else ""))
            # 先扩大候选集进行轻量 rerank，再用 MMR 抑制近重复片段。
            score = vector_score * 0.72 + lexical_score * 0.28
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        trace.add("score_candidates", score_started, input_count=len(self.chunks), output_count=len(scored),
                  dense_weight=0.72, lexical_weight=0.28)
        candidates = scored[: max(top_k * 4, top_k)]
        trace.candidates = [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "score": round(score, 6)}
            for score, chunk in candidates
        ]
        select_started = perf_counter()
        selected = self._mmr_select(candidates, top_k, lambda_weight=0.58)
        trace.add("mmr_select", select_started, input_count=len(candidates), output_count=len(selected),
                  lambda_weight=0.58)
        trace.selected_chunk_ids = [chunk.chunk_id for _, chunk in selected]
        evidences: list[Evidence] = []
        for score, chunk in selected:
            evidences.append(
                Evidence(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_label=chunk.source_label,
                    text=fix_mojibake(chunk.text),
                    score=score,
                )
            )
        return evidences

    def matching_title_doc_ids(self, query: str) -> set[str]:
        """返回查询中被明确点名的文档。

        标题匹配只处理文件名这一结构化信号，不判断用户意图。这里同时支持
        ``BLIP-2``/``BLIP2`` 这类连字符变体，并允许 ``BLIP`` 这种四字符标题；
        短标题必须作为完整 token 出现，避免把它误匹配到更长的普通单词中。
        """
        normalized_query = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
        if not normalized_query:
            return set()
        padded_query = f" {normalized_query} "
        query_tokens = set(normalized_query.split())
        matches: set[str] = set()
        for doc_id, document in self.documents.items():
            title = Path(document.title).stem.casefold()
            normalized_title = re.sub(r"[^a-z0-9]+", " ", title).strip()
            compact_title = normalized_title.replace(" ", "")
            exact_phrase = len(normalized_title) >= 4 and f" {normalized_title} " in padded_query
            compact_variant = len(compact_title) >= 4 and compact_title in query_tokens
            aliases = {
                str(item).casefold() for item in document.metadata.get("aliases", [])
                if str(item).strip()
            }
            # 兼容旧索引：无需重新导入即可从文档开头的“全称 (缩写)”提取别名。
            for match in re.finditer(
                r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,8})\s*\(([A-Z][A-Z0-9-]{1,9})\)",
                document.content[:6000],
            ):
                aliases.add(match.group(2).casefold())
            alias_match = any(alias in query_tokens for alias in aliases)
            if exact_phrase or compact_variant or alias_match:
                matches.add(doc_id)
        return matches

    # 保留内部兼容入口，避免已有扩展代码因方法改名失效。
    def _matching_title_doc_ids(self, query: str) -> set[str]:
        return self.matching_title_doc_ids(query)

    def _reference_start_positions(self) -> dict[str, int]:
        """识别论文尾部参考文献起点，避免引用条目挤占正文 Top-K。"""
        grouped: dict[str, list[Chunk]] = {}
        for chunk in self.chunks.values():
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        starts: dict[str, int] = {}
        pattern = re.compile(r"(?i)\b(?:references|bibliography)\b\s*(?:\[\d+\]|$)|参考文献")
        for doc_id, chunks in grouped.items():
            ordered = sorted(chunks, key=lambda item: item.position)
            midpoint = ordered[len(ordered) // 2].position
            metadata_candidates = [chunk.position for chunk in ordered if is_reference_chunk(chunk)]
            candidates = metadata_candidates or [
                chunk.position for chunk in ordered if chunk.position >= midpoint and pattern.search(chunk.text)
            ]
            if candidates:
                starts[doc_id] = min(candidates)
        return starts

    def search_collection(
        self, query: str, doc_ids: list[str] | None = None, max_docs: int = 8, chunks_per_doc: int = 2,
        diversify_positions: bool = True,
    ) -> list[Evidence]:
        """跨文档汇总时限制单篇占比，并允许 Planner 从索引目录选定文档。"""
        trace = RetrievalTrace(query=query, mode="collection")
        self.last_trace = trace
        if not self.chunks:
            return []
        selected_ids = set(doc_ids or []) & self.documents.keys()
        if doc_ids and not selected_ids:
            return []
        embed_started = perf_counter()
        embedding = self.client.embed([query])[0]
        trace.add("embed_query", embed_started, input_count=1, output_count=1, dimensions=len(embedding))
        hash_query = local_hash_embedding(query)
        terms = self._query_terms(query)
        grouped: dict[str, list[tuple[float, Chunk]]] = {}
        score_started = perf_counter()
        for chunk in self.chunks.values():
            if selected_ids and chunk.doc_id not in selected_ids or is_probably_garbled(chunk.text):
                continue
            doc = self.documents.get(chunk.doc_id)
            if doc is None:
                continue
            vector = self._vector_score(chunk, embedding, hash_query)
            lexical = self._lexical_score(terms, chunk.text + " " + doc.title)
            grouped.setdefault(chunk.doc_id, []).append((vector * 0.72 + lexical * 0.28, chunk))
        trace.add("group_candidates", score_started, input_count=len(self.chunks),
                  output_count=sum(len(items) for items in grouped.values()), documents=len(grouped))
        for doc_id, scored in list(grouped.items()):
            reference_starts = [
                chunk.position for _, chunk in scored
                if re.search(r"(?im)^[ \t]*(?:references|bibliography|参考文献)[ \t]*$", chunk.text)
            ]
            if reference_starts:
                start = min(reference_starts)
                grouped[doc_id] = [(score, chunk) for score, chunk in scored if chunk.position < start]
        grouped = {doc_id: scored for doc_id, scored in grouped.items() if scored}
        ranked = sorted(grouped.items(), key=lambda item: max(score for score, _ in item[1]), reverse=True)
        evidences: list[Evidence] = []
        for _, scored in ranked[:max_docs]:
            for score, chunk in self._collection_chunks(
                scored, chunks_per_doc, diversify_positions=diversify_positions,
            ):
                evidences.append(Evidence(chunk.chunk_id, chunk.doc_id, chunk.source_label, fix_mojibake(chunk.text), score))
        trace.selected_chunk_ids = [evidence.chunk_id for evidence in evidences]
        return evidences

    def retrieval_hint(self, query: str, limit: int = 3, excerpt_chars: int = 220) -> list[dict[str, object]]:
        """为意图路由提供有界索引提示，不调用 embedding API。

        这里判断的是“问题和索引是否有可核对的内容重合”，不是通过动作关键词
        猜测用户意图。最终是否检索仍由 LLM Router 结合问题语义决定。
        """
        if not self.chunks or not query.strip():
            return []
        query_terms = self._query_terms(query)
        english_stopwords = {
            "what", "which", "when", "where", "who", "why", "how", "does", "do", "did",
            "is", "are", "was", "were", "the", "and", "or", "for", "with", "from", "about",
            "work", "works", "working", "method", "model",
        }
        required_ascii_terms = {
            token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]+", query)
            if token.casefold() not in english_stopwords
        }
        short_acronyms = {
            token.casefold()
            for token in re.findall(r"(?<![A-Za-z0-9])[A-Z]{2,4}(?![A-Za-z0-9])", query)
        }
        technical_entities = {
            token.casefold()
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+|[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z0-9]{4,}",
                query,
            )
        }
        ranked: list[tuple[float, Chunk, list[str], list[str]]] = []
        title_doc_ids = self.matching_title_doc_ids(query)
        reference_starts = self._reference_start_positions()
        for chunk in self.chunks.values():
            # 用户明确点名了索引中的文档标题时，路由提示只展示该文档，避免同领域
            # 论文里偶然出现的普通动词或引用标题干扰 Router。
            if title_doc_ids and chunk.doc_id not in title_doc_ids:
                continue
            if is_probably_garbled(chunk.text):
                continue
            if (
                is_reference_chunk(chunk)
                or chunk.position >= reference_starts.get(chunk.doc_id, sys.maxsize)
            ):
                continue
            document = self.documents.get(chunk.doc_id)
            haystack = f"{chunk.text} {document.title if document else ''}".casefold()
            matched_terms = sorted(
                (term for term in query_terms if self._term_in_text(term, haystack)), key=len, reverse=True,
            )
            matched_entities = sorted(
                (entity for entity in technical_entities if self._term_in_text(entity, haystack)), key=len, reverse=True,
            )
            lexical = len(matched_terms) / max(1, len(query_terms))
            phrase_bonus = 0.45 if matched_entities else 0.0
            score = lexical + phrase_bonus
            if not matched_entities and lexical < 0.22:
                continue
            ranked.append((score, chunk, matched_terms[:8], matched_entities[:4]))
        ranked.sort(key=lambda item: item[0], reverse=True)
        # 多词英文实体必须由候选集合完整覆盖；允许 BLIP/BLIP2 这类比较问题由不同
        # 文档联合覆盖，但不能把只含普通 RL 的论文当成 Agentic RL 的直接证据。
        covered_ascii_terms = {
            term for _, chunk, _, _ in ranked for term in required_ascii_terms
            if self._term_in_text(
                term,
                f"{chunk.text} {self.documents.get(chunk.doc_id).title if self.documents.get(chunk.doc_id) else ''}".casefold(),
            )
        }
        if required_ascii_terms and not required_ascii_terms.issubset(covered_ascii_terms):
            return []
        # GPT、RAG、RAM 这类短缩写在综述、参考描述和示例中经常只是被顺带提及。
        # 若查询只有一个短缩写作为主题，且它没有命中文档标题，则候选不足以证明
        # 索引真的覆盖了该主题，应让通用问答直接回答。用户显式要求查索引时，
        # Router 的 explicit_local_retrieval 仍可绕过这一提示为空的保守判断。
        if (
            short_acronyms
            and required_ascii_terms
            and required_ascii_terms.issubset(short_acronyms)
            and not title_doc_ids
        ):
            return []
        hints: list[dict[str, object]] = []
        seen_docs: set[str] = set()
        for score, chunk, terms, entities in ranked:
            # 先保证文档覆盖；候选不足时再允许同文档多个位置。
            if chunk.doc_id in seen_docs and len(ranked) > limit:
                continue
            seen_docs.add(chunk.doc_id)
            hints.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source_label,
                "matched_terms": terms,
                "matched_entities": entities,
                "strong_match": bool(entities) or score >= 0.5,
                "excerpt": fix_mojibake(chunk.text)[:max(80, min(excerpt_chars, 400))],
            })
            if len(hints) >= max(1, min(limit, 5)):
                break
        return hints

    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        """英文缩写和标识符按 token 边界匹配，避免 RL 命中 world/url 等子串。"""
        if re.fullmatch(r"[a-z0-9_.+-]+", term, flags=re.IGNORECASE):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE))
        return term in text

    def _configured_embedding_space(self) -> str:
        config = self.client.config
        if not config.embedding_api_key:
            return "local-hash-v1:384"
        return f"api:{config.embedding_model}"

    def _embed_cached(self, texts: list[str]) -> list[list[float]]:
        """按内容哈希复用 embedding，并保持输入顺序与重复项。"""
        if not texts:
            return []
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        cached = self.catalog.get_cached_embeddings(hashes, self._embedding_space)
        missing_hashes: list[str] = []
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for digest, text in zip(hashes, texts):
            if digest not in cached and digest not in seen_missing:
                seen_missing.add(digest)
                missing_hashes.append(digest)
                missing_texts.append(text)
        if missing_texts:
            try:
                batch_size = max(1, min(int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16")), 128))
            except ValueError:
                batch_size = 16
            fresh: dict[str, list[float]] = {}
            requested_space = self._embedding_space
            for start in range(0, len(missing_texts), batch_size):
                text_batch = missing_texts[start:start + batch_size]
                hash_batch = missing_hashes[start:start + batch_size]
                vectors = self.client.embed(text_batch)
                if self.client.last_error and vectors and len(vectors[0]) == 384 and requested_space != "local-hash-v1:384":
                    # API 失败后的 fallback 必须整体切换 namespace，不能混用已缓存的 API 向量。
                    self._embedding_space = "local-hash-v1:384"
                    local_vectors = [local_hash_embedding(text) for text in texts]
                    local_values = {digest: vector for digest, vector in zip(hashes, local_vectors)}
                    self.catalog.put_cached_embeddings(local_values, self._embedding_space)
                    return local_vectors
                fresh.update({digest: vector for digest, vector in zip(hash_batch, vectors)})
            self.catalog.put_cached_embeddings(fresh, self._embedding_space)
            cached.update(fresh)
        return [cached[digest] for digest in hashes]

    def _collection_chunks(
        self,
        scored: list[tuple[float, Chunk]],
        limit: int,
        diversify_positions: bool = True,
    ) -> list[tuple[float, Chunk]]:
        """兼顾主题相关性与文档不同位置，避免长论文只命中同一节。"""
        if limit <= 0:
            return []
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        selected = self._mmr_select(
            ranked[:max(12, limit * 3)],
            limit if not diversify_positions else min(2, limit),
            0.7 if not diversify_positions else 0.58,
        )
        if not diversify_positions:
            return sorted(selected[:limit], key=lambda item: item[1].position)
        seen = {chunk.chunk_id for _, chunk in selected}
        ordered = sorted(scored, key=lambda item: item[1].position)
        buckets = max(1, limit - len(selected))
        for bucket in range(buckets):
            start = bucket * len(ordered) // buckets
            end = (bucket + 1) * len(ordered) // buckets
            options = [item for item in ordered[start:end] if item[1].chunk_id not in seen]
            if options:
                best = max(options, key=lambda item: item[0])
                selected.append(best)
                seen.add(best[1].chunk_id)
        for item in ranked:
            if len(selected) >= limit:
                break
            if item[1].chunk_id not in seen:
                selected.append(item)
                seen.add(item[1].chunk_id)
        return sorted(selected[:limit], key=lambda item: item[1].position)

    @staticmethod
    def _vector_score(chunk: Chunk, query_embedding: list[float], hash_query: list[float]) -> float:
        # 384 维旧片段只有确认为本地 hash 向量时，才使用同一算法生成的查询向量。
        if len(chunk.embedding) == len(hash_query) and cosine_similarity(
            chunk.embedding, local_hash_embedding(chunk.text)
        ) > 0.999:
            return max(0.0, cosine_similarity(hash_query, chunk.embedding))
        if len(chunk.embedding) == len(query_embedding):
            return max(0.0, cosine_similarity(query_embedding, chunk.embedding))
        return 0.0

    def _mmr_select(
        self, candidates: list[tuple[float, Chunk]], top_k: int, lambda_weight: float
    ) -> list[tuple[float, Chunk]]:
        remaining = list(candidates)
        selected: list[tuple[float, Chunk]] = []
        while remaining and len(selected) < top_k:
            best_index = 0
            best_score = float("-inf")
            for index, (relevance, chunk) in enumerate(remaining):
                redundancy = max(
                    (max(0.0, cosine_similarity(chunk.embedding, chosen.embedding))
                     for _, chosen in selected if len(chunk.embedding) == len(chosen.embedding)),
                    default=0.0,
                )
                normalized = " ".join(chunk.text.casefold().split())
                if any(normalized == " ".join(chosen.text.casefold().split()) for _, chosen in selected):
                    redundancy = max(redundancy, 1.5)
                mmr = lambda_weight * relevance - (1.0 - lambda_weight) * redundancy
                if mmr > best_score:
                    best_index, best_score = index, mmr
            relevance, chunk = remaining.pop(best_index)
            selected.append((relevance, chunk))
        return selected

    def _query_terms(self, query: str) -> set[str]:
        terms = {item.casefold() for item in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query)}
        for chinese in re.findall(r"[\u4e00-\u9fff]{3,}", query):
            terms.update(chinese[index : index + 2] for index in range(len(chinese) - 1))
        return terms

    def _lexical_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        lowered = text.casefold()
        return sum(term in lowered for term in query_terms) / len(query_terms)

    def stats(self) -> dict[str, int]:
        catalog_stats = self.catalog.stats()
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "sentence_nodes": catalog_stats["sentence_nodes"],
            "cached_embeddings": catalog_stats["cached_embeddings"],
        }
