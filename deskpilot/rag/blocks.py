from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentBlock:
    """解析后的结构块；分块器不再只能面对一段无结构全文。"""

    block_id: str
    block_type: str
    text: str
    order: int
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def boundary_key(self) -> tuple[Any, ...]:
        return (
            self.page_number,
            self.slide_number,
            self.sheet_name,
            tuple(self.heading_path),
            self.metadata.get("table_id"),
            self.metadata.get("code_block"),
        )


@dataclass
class SentenceNode:
    """用于句子窗口扩展的最小节点，正文仍只保存一份。"""

    sentence_id: str
    doc_id: str
    parent_chunk_id: str
    text: str
    order: int
    previous_sentence_ids: list[str] = field(default_factory=list)
    next_sentence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    parent_chunk_id: str
    doc_id: str
    text: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    chunks: list[Any]
    sentence_nodes: list[SentenceNode]
    parent_chunks: list[ParentChunk]
    strategy: str
    stats: dict[str, Any] = field(default_factory=dict)
