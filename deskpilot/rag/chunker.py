from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .blocks import ChunkingResult, DocumentBlock, ParentChunk, SentenceNode
from ..context.budget import estimate_tokens
from ..core.api_clients import cosine_similarity, local_hash_embedding
from ..core.models import Chunk, Document


CHUNKER_VERSION = "adaptive-v1"
EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class ChunkingConfig:
    policy: str = "adaptive"
    min_chunk_tokens: int = 160
    target_chunk_tokens: int = 400
    max_chunk_tokens: int = 620
    parent_target_tokens: int = 1200
    overlap_sentences: int = 1
    semantic_breakpoint_percentile: float = 92.0
    semantic_buffer_sentences: int = 2
    sentence_window_size: int = 2

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            policy=os.getenv("RAG_CHUNKING_POLICY", "adaptive").strip().lower(),
            min_chunk_tokens=_env_int("RAG_MIN_CHUNK_TOKENS", 160, 20, 2000),
            target_chunk_tokens=_env_int("RAG_CHILD_TOKENS", 400, 40, 4000),
            max_chunk_tokens=_env_int("RAG_CHILD_MAX_TOKENS", 620, 60, 6000),
            parent_target_tokens=_env_int("RAG_PARENT_TOKENS", 1200, 100, 12000),
            overlap_sentences=_env_int("RAG_OVERLAP_SENTENCES", 1, 0, 5),
            semantic_breakpoint_percentile=_env_float(
                "RAG_SEMANTIC_BREAKPOINT_PERCENTILE", 92.0, 50.0, 99.9
            ),
            semantic_buffer_sentences=_env_int("RAG_SEMANTIC_BUFFER_SENTENCES", 2, 1, 5),
            sentence_window_size=_env_int("RAG_SENTENCE_WINDOW_SIZE", 2, 0, 8),
        )


def chunk_document(document: Document, chunk_size: int = 900, overlap: int = 160) -> list[Chunk]:
    """兼容旧调用面；新实现使用 token、结构和语义边界。

    `chunk_size/overlap` 仅为旧代码保留。调用方显式传入非默认值时，转换成
    保守的 token 目标；新代码应调用 `chunk_document_with_metadata`。
    """
    config = ChunkingConfig.from_env()
    if chunk_size != 900:
        target = max(40, int(chunk_size * 0.65))
        config = ChunkingConfig(
            policy=config.policy,
            min_chunk_tokens=min(config.min_chunk_tokens, target),
            target_chunk_tokens=target,
            max_chunk_tokens=max(target, int(target * 1.5)),
            parent_target_tokens=config.parent_target_tokens,
            overlap_sentences=1 if overlap else 0,
            semantic_breakpoint_percentile=config.semantic_breakpoint_percentile,
            semantic_buffer_sentences=config.semantic_buffer_sentences,
            sentence_window_size=config.sentence_window_size,
        )
    return chunk_document_with_metadata(document, config=config).chunks


def chunk_document_with_metadata(
    document: Document,
    config: ChunkingConfig | None = None,
    embed_sentences: EmbeddingFunction | None = None,
) -> ChunkingResult:
    config = config or ChunkingConfig.from_env()
    blocks = parse_document_blocks(document)
    strategy = _select_strategy(document, config)
    parents: list[ParentChunk] = []
    chunks: list[Chunk] = []
    nodes: list[SentenceNode] = []
    semantic_distances: list[float] = []

    # boundary_key 保证句子窗口和 overlap 不跨页面、标题、表格等结构边界。
    for parent_order, group in enumerate(_parent_groups(blocks, config), start=1):
        sentences = _sentences_for_group(group)
        if not sentences:
            continue
        parent_id = f"{document.doc_id}_parent_{parent_order:04d}"
        parent_metadata = _block_metadata(group[0])
        parent_text = "\n\n".join(block.text for block in group if block.text.strip())
        parents.append(ParentChunk(parent_id, document.doc_id, parent_text, parent_order, parent_metadata))

        sentence_ids = [f"{document.doc_id}_sentence_{len(nodes) + index + 1:06d}" for index in range(len(sentences))]
        for index, sentence in enumerate(sentences):
            window = config.sentence_window_size
            node = SentenceNode(
                sentence_id=sentence_ids[index],
                doc_id=document.doc_id,
                parent_chunk_id=parent_id,
                text=sentence,
                order=len(nodes) + index + 1,
                previous_sentence_ids=sentence_ids[max(0, index - window):index],
                next_sentence_ids=sentence_ids[index + 1:index + 1 + window],
                metadata={**parent_metadata, "token_count": estimate_tokens(sentence)},
            )
            nodes.append(node)

        if strategy == "semantic":
            embeddings = (embed_sentences or _local_embed)(sentences)
            segments, distances = _semantic_segments(sentences, embeddings, config)
            semantic_distances.extend(distances)
        else:
            segments = _token_segments(sentences, config)

        sentence_offset = len(nodes) - len(sentences)
        search_from = 0
        for segment in segments:
            # overlap 会让同一句出现在相邻 child；从上次位置附近寻找对应 sentence ID。
            indexes = _locate_segment_indexes(sentences, segment, search_from)
            if indexes:
                search_from = max(indexes[-1], search_from)
            ids = [nodes[sentence_offset + index].sentence_id for index in indexes]
            chunk_position = len(chunks) + 1
            text = " ".join(segment).strip()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            metadata = {
                "path": document.path,
                "title": document.title,
                "source_type": document.source_type,
                **document.metadata,
                **parent_metadata,
                "parent_chunk_id": parent_id,
                "sentence_ids": ids,
                "previous_sentence_ids": nodes[sentence_offset + indexes[0]].previous_sentence_ids if indexes else [],
                "next_sentence_ids": nodes[sentence_offset + indexes[-1]].next_sentence_ids if indexes else [],
                "window_size": config.sentence_window_size,
                "token_count": estimate_tokens(text),
                "content_hash": digest,
                "chunking_strategy": strategy,
                "chunker_version": CHUNKER_VERSION,
            }
            source_label = _source_label(document, chunk_position, parent_metadata)
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}_chunk_{chunk_position:04d}",
                doc_id=document.doc_id,
                text=text,
                source_label=source_label,
                position=chunk_position,
                metadata=metadata,
            ))

    return ChunkingResult(
        chunks=chunks,
        sentence_nodes=nodes,
        parent_chunks=parents,
        strategy=strategy,
        stats={
            "blocks": len(blocks),
            "parents": len(parents),
            "sentences": len(nodes),
            "chunks": len(chunks),
            "semantic_distance_count": len(semantic_distances),
            "semantic_distance_mean": round(sum(semantic_distances) / len(semantic_distances), 6)
            if semantic_distances else 0.0,
        },
    )


def parse_document_blocks(document: Document) -> list[DocumentBlock]:
    source_type = document.source_type.casefold()
    if source_type in {"md", "markdown", "docx"}:
        return _parse_markdown_blocks(document)
    return _parse_marker_blocks(document)


def _parse_markdown_blocks(document: Document) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    headings: list[str] = []
    buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush(block_type: str = "paragraph") -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            blocks.append(DocumentBlock(
                f"{document.doc_id}_block_{len(blocks) + 1:05d}", block_type, text, len(blocks) + 1,
                heading_path=list(headings),
            ))

    for raw in document.content.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            if in_code:
                code_lines.append(line)
                blocks.append(DocumentBlock(
                    f"{document.doc_id}_block_{len(blocks) + 1:05d}", "code", "\n".join(code_lines),
                    len(blocks) + 1, heading_path=list(headings), metadata={"code_block": True},
                ))
                code_lines = []
                in_code = False
            else:
                flush()
                in_code = True
                code_lines = [line]
            continue
        if in_code:
            code_lines.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            headings[:] = headings[:level - 1]
            headings.append(heading.group(2).strip())
            continue
        if not line.strip():
            block_type = "paragraph"
            if buffer and all(_looks_like_table(item) for item in buffer if item.strip()):
                block_type = "table"
            elif buffer and all(_is_list_line(item) for item in buffer if item.strip()):
                block_type = "list"
            flush(block_type)
            continue
        if _looks_like_table(line):
            if buffer and not all(_looks_like_table(item) for item in buffer):
                flush()
            buffer.append(line)
            continue
        if buffer and all(_looks_like_table(item) for item in buffer) and not _looks_like_table(line):
            flush("table")
        buffer.append(line)
    if in_code and code_lines:
        blocks.append(DocumentBlock(
            f"{document.doc_id}_block_{len(blocks) + 1:05d}", "code", "\n".join(code_lines),
            len(blocks) + 1, heading_path=list(headings), metadata={"code_block": True},
        ))
    flush("table" if buffer and all(_looks_like_table(item) for item in buffer) else "paragraph")
    return blocks


def _parse_marker_blocks(document: Document) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    parts = re.split(r"\n\s*\n", document.content)
    for part in parts:
        text = part.strip()
        if not text:
            continue
        marker = re.match(r"^\[(Page|Slide|Sheet)\s+([^\]]+)\]\s*\n?", text, re.IGNORECASE)
        if marker:
            kind, value = marker.group(1).casefold(), marker.group(2).strip()
            if kind == "page" and value.isdigit():
                page, slide, sheet = int(value), None, None
            elif kind == "slide" and value.isdigit():
                slide, page, sheet = int(value), None, None
            elif kind == "sheet":
                sheet, page, slide = value, None, None
            text = text[marker.end():].strip()
        if not text:
            continue
        block_type = "table" if document.source_type in {"csv", "xlsx"} else "paragraph"
        blocks.append(DocumentBlock(
            f"{document.doc_id}_block_{len(blocks) + 1:05d}", block_type, text, len(blocks) + 1,
            page_number=page, slide_number=slide, sheet_name=sheet,
            metadata={"table_id": f"table-{len(blocks) + 1}"} if block_type == "table" else {},
        ))
    return blocks


def _select_strategy(document: Document, config: ChunkingConfig) -> str:
    if config.policy in {"semantic", "structural", "fixed"}:
        return config.policy
    if document.source_type.casefold() in {"pdf", "txt"}:
        return "semantic"
    return "structural"


def _group_blocks(blocks: list[DocumentBlock]) -> Iterable[list[DocumentBlock]]:
    current: list[DocumentBlock] = []
    key: tuple[object, ...] | None = None
    for block in blocks:
        if current and block.boundary_key != key:
            yield current
            current = []
        key = block.boundary_key
        current.append(block)
        # 表格和代码块天然独立，不能与后续正文混合。
        if block.block_type in {"table", "code"}:
            yield current
            current = []
            key = None
    if current:
        yield current


def _parent_groups(blocks: list[DocumentBlock], config: ChunkingConfig) -> Iterable[list[DocumentBlock]]:
    """在结构边界内控制 parent 大小，避免一个超长章节成为无限大的 parent。"""
    for structural_group in _group_blocks(blocks):
        current: list[DocumentBlock] = []
        current_tokens = 0
        for block in structural_group:
            block_tokens = estimate_tokens(block.text)
            if block_tokens > config.parent_target_tokens:
                if current:
                    yield current
                    current, current_tokens = [], 0
                sentences = _split_sentences(block.text)
                if len(sentences) == 1 and estimate_tokens(sentences[0]) > config.parent_target_tokens:
                    sentences = _split_long_unit(block.text, config.parent_target_tokens)
                parts: list[str] = []
                part_tokens = 0
                for sentence in sentences:
                    tokens = estimate_tokens(sentence)
                    if parts and part_tokens + tokens > config.parent_target_tokens:
                        yield [_clone_block(block, " ".join(parts))]
                        parts, part_tokens = [], 0
                    parts.append(sentence)
                    part_tokens += tokens
                if parts:
                    yield [_clone_block(block, " ".join(parts))]
                continue
            if current and current_tokens + block_tokens > config.parent_target_tokens:
                yield current
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            yield current


def _clone_block(block: DocumentBlock, text: str) -> DocumentBlock:
    return DocumentBlock(
        block_id=block.block_id,
        block_type=block.block_type,
        text=text,
        order=block.order,
        page_number=block.page_number,
        slide_number=block.slide_number,
        sheet_name=block.sheet_name,
        heading_path=list(block.heading_path),
        metadata=dict(block.metadata),
    )


def _sentences_for_group(group: list[DocumentBlock]) -> list[str]:
    result: list[str] = []
    for block in group:
        if block.block_type in {"table", "code"}:
            result.extend(_split_long_unit(block.text, 420))
        else:
            result.extend(_split_sentences(block.text))
    return [item.strip() for item in result if item.strip()]


def _split_sentences(text: str) -> list[str]:
    # 同时覆盖中文句末符号和英文句点；换行仍是次级边界。
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=[A-Za-z0-9][.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _split_long_unit(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    step = max(100, int(len(text) * max_tokens / max(estimate_tokens(text), 1)))
    return [text[start:start + step] for start in range(0, len(text), step)]


def _token_segments(sentences: list[str], config: ChunkingConfig) -> list[list[str]]:
    result: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > config.target_chunk_tokens:
            result.append(current)
            current = current[-config.overlap_sentences:] if config.overlap_sentences else []
            current_tokens = sum(estimate_tokens(item) for item in current)
        if sentence_tokens > config.max_chunk_tokens:
            if current:
                result.append(current)
                current = []
                current_tokens = 0
            for part in _split_long_unit(sentence, config.max_chunk_tokens):
                result.append([part])
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        result.append(current)
    return result


def _semantic_segments(sentences: list[str], embeddings: list[list[float]],
                       config: ChunkingConfig) -> tuple[list[list[str]], list[float]]:
    if len(sentences) < 2 or len(embeddings) != len(sentences):
        return _token_segments(sentences, config), []
    buffered = [
        _mean_vector(embeddings[max(0, index - config.semantic_buffer_sentences + 1):index + 1])
        for index in range(len(embeddings))
    ]
    distances = [1.0 - cosine_similarity(buffered[index], buffered[index + 1])
                 for index in range(len(buffered) - 1)]
    threshold = _percentile(distances, config.semantic_breakpoint_percentile)
    raw: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for index, sentence in enumerate(sentences):
        tokens = estimate_tokens(sentence)
        should_break = bool(current) and (
            current_tokens + tokens > config.max_chunk_tokens
            or (index > 0 and distances[index - 1] >= threshold
                and current_tokens >= config.min_chunk_tokens)
        )
        if should_break:
            raw.append(current)
            current = current[-config.overlap_sentences:] if config.overlap_sentences else []
            current_tokens = sum(estimate_tokens(item) for item in current)
        current.append(sentence)
        current_tokens += tokens
        if current_tokens >= config.target_chunk_tokens and index < len(sentences) - 1:
            raw.append(current)
            current = current[-config.overlap_sentences:] if config.overlap_sentences else []
            current_tokens = sum(estimate_tokens(item) for item in current)
    if current:
        raw.append(current)
    return _merge_short_segments(raw, config), distances


def _merge_short_segments(segments: list[list[str]], config: ChunkingConfig) -> list[list[str]]:
    merged: list[list[str]] = []
    for segment in segments:
        tokens = sum(estimate_tokens(item) for item in segment)
        if merged and tokens < config.min_chunk_tokens:
            combined = merged[-1] + segment[config.overlap_sentences:]
            if sum(estimate_tokens(item) for item in combined) <= config.max_chunk_tokens:
                merged[-1] = combined
                continue
        merged.append(segment)
    return merged


def _locate_segment_indexes(all_sentences: list[str], segment: list[str], start: int) -> list[int]:
    indexes: list[int] = []
    cursor = max(0, start - 2)
    for sentence in segment:
        found = next((index for index in range(cursor, len(all_sentences))
                      if all_sentences[index] == sentence), None)
        if found is None:
            found = next((index for index, value in enumerate(all_sentences) if value == sentence), None)
        if found is not None:
            indexes.append(found)
            cursor = found + 1
    return indexes


def _block_metadata(block: DocumentBlock) -> dict[str, object]:
    return {
        "page_number": block.page_number,
        "slide_number": block.slide_number,
        "sheet_name": block.sheet_name,
        "heading_path": list(block.heading_path),
        "content_type": block.block_type,
        **block.metadata,
    }


def _source_label(document: Document, position: int, metadata: dict[str, object]) -> str:
    label = f"{document.title}"
    if metadata.get("page_number"):
        label += f"#page-{metadata['page_number']}"
    elif metadata.get("slide_number"):
        label += f"#slide-{metadata['slide_number']}"
    elif metadata.get("sheet_name"):
        label += f"#sheet-{metadata['sheet_name']}"
    elif metadata.get("heading_path"):
        label += "#" + "/".join(str(item) for item in metadata["heading_path"])
    else:
        label += f"#chunk-{position}"
    return f"web:{label}" if document.source_type == "web" else label


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    compatible = [vector for vector in vectors if len(vector) == size]
    if not compatible:
        return []
    return [sum(vector[index] for vector in compatible) / len(compatible) for index in range(size)]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _local_embed(texts: list[str]) -> list[list[float]]:
    return [local_hash_embedding(text) for text in texts]


def _looks_like_table(line: str) -> bool:
    return line.count("|") >= 2


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
