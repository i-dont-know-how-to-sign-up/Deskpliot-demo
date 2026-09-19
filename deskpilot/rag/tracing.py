from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class RetrievalTraceEvent:
    stage: str
    duration_ms: float
    input_count: int = 0
    output_count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    query: str
    mode: str = "search"
    events: list[RetrievalTraceEvent] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_chunk_ids: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=perf_counter, repr=False)

    def add(self, stage: str, started: float, input_count: int = 0, output_count: int = 0,
            **detail: Any) -> None:
        self.events.append(RetrievalTraceEvent(
            stage=stage,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            input_count=input_count,
            output_count=output_count,
            detail=detail,
        ))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_duration_ms"] = round((perf_counter() - self.started_at) * 1000, 3)
        data.pop("started_at", None)
        return data
