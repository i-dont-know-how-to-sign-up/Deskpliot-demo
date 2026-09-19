from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.models import utc_now


@dataclass
class SessionMessage:
    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMessage":
        return cls(**data)


@dataclass
class SessionInfo:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        return cls(**data)


@dataclass
class MemoryItem:
    memory_id: str
    scope: str
    memory_type: str
    content: str
    source_session_id: str | None = None
    source_message_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    expires_at: str | None = None
    embedding: list[float] = field(default_factory=list)
    # 运行时检索分数不写入 SQLite，仅供本轮上下文选择使用。
    retrieval_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return cls(**data)
