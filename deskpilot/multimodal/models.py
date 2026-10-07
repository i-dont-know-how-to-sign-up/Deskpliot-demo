from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.models import utc_now


@dataclass(frozen=True)
class MediaAsset:
    asset_id: str
    sha256: str
    media_type: str
    original_path: str
    source_path: str | None
    width: int
    height: int
    size_bytes: int
    thumbnail_path: str
    source_kind: str = "chat"
    parent_doc_id: str | None = None
    page_number: int | None = None
    sensitivity: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultimodalChunk:
    chunk_id: str
    asset_id: str
    modality: str
    text: str
    source_label: str
    doc_id: str | None = None
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecord:
    vector_id: str
    owner_type: str
    owner_id: str
    modality: str
    embedding_space: str
    model_id: str
    model_revision: str
    dimension: int
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualEvidence:
    evidence_id: str
    asset_id: str
    source_label: str
    modality: str
    score: float
    text: str = ""
    doc_id: str | None = None
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
