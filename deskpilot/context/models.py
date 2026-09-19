from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..memory.memory_models import MemoryItem


@dataclass
class ContextPacket:
    """上下文最小管理单元，记录内容以及选择、压缩所需的元数据。"""

    packet_id: str
    kind: str
    content: str
    source: str
    timestamp: str = ""
    estimated_tokens: int = 0
    relevance_score: float = 0.0
    recency_score: float = 0.0
    priority: int = 0
    scope: str = "session"
    required: bool = False
    compressible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """仅控制动态记忆上下文，不代表底层模型的完整上下文窗口。"""

    max_tokens: int = 6000
    reserve_ratio: float = 0.20
    min_relevance: float = 0.02
    relevance_weight: float = 0.70
    recency_weight: float = 0.30
    recent_message_limit: int = 12
    per_message_tokens: int = 500
    tool_output_tokens: int = 450

    @property
    def usable_tokens(self) -> int:
        return max(64, int(self.max_tokens * (1.0 - self.reserve_ratio)))


@dataclass
class AssembledContext:
    text: str
    debug_lines: list[str] = field(default_factory=list)
    retrieved_memories: list[MemoryItem] = field(default_factory=list)
    packets: list[ContextPacket] = field(default_factory=list)
    dropped_packets: list[ContextPacket] = field(default_factory=list)
    estimated_tokens: int = 0
    quality: dict[str, Any] = field(default_factory=dict)
    role: str = "shared"
    input_budget: int = 0
    output_budget: int = 0
    complexity: int = 0
