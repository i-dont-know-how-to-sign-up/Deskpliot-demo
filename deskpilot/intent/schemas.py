from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class IntentDecision:
    mode: str
    reason: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    needs_workspace_files: bool = False
    needs_index_catalog: bool = False
    requires_file_output: bool = False
    # 仅表示用户是否明确要求查询本地索引/知识库，而不是 Router 是否偏好使用 RAG。
    explicit_local_retrieval: bool = False
    explicit_web_retrieval: bool = False
    requires_fresh_information: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlotValidationResult:
    ok: bool
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
