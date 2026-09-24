from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class AgentResult:
    """智能体统一返回结构，便于 Supervisor 记录状态和指标。"""

    agent: str
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_calls: int = 0
    tokens: int = 0
    step_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskPlanStep:
    step_id: str
    agent: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    requires_human: bool = False
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    max_retries: int = 1
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    goal: str
    route: str
    steps: list[TaskPlanStep]
    complexity_score: int = 0
    max_agents: int = 3
    max_tool_calls: int = 6
    max_total_tokens: int = 12000
    max_duration_seconds: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)
