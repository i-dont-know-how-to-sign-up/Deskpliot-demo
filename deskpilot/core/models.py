from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    doc_id: str
    path: str
    title: str
    source_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_label: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class Evidence:
    chunk_id: str
    doc_id: str
    source_label: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    name: str
    status: str
    detail: str


@dataclass
class AnswerResult:
    answer: str
    evidences: list[Evidence]
    steps: list[AgentStep]
    used_llm: bool
    session_id: str = ""
    memory_context: str = ""
    pending_action: dict[str, Any] | None = None


def make_doc_id(path: Path, digest: str) -> str:
    safe_suffix = path.suffix.lower().replace(".", "") or "file"
    return f"{safe_suffix}_{digest[:16]}"
