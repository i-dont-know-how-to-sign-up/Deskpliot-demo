from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable

from ..tasks.models import serializable


def restore_output(value: Any) -> Any:
    """恢复领域数据类型，不能反序列化任意 Python 类型或可执行对象。"""
    if isinstance(value, list):
        return [restore_output(item) for item in value]
    if not isinstance(value, dict):
        return value
    data = {key: restore_output(item) for key, item in value.items()}
    evidence_keys = {"chunk_id", "doc_id", "source_label", "text", "score", "metadata"}
    if evidence_keys <= data.keys():
        from ..core.models import Evidence
        return Evidence(**{key: data[key] for key in evidence_keys})
    if {"ok", "tool_name", "output", "error"} <= data.keys():
        from ..tools.tool_registry import ToolResult
        names = {item.name for item in fields(ToolResult)}
        return ToolResult(**{key: item for key, item in data.items() if key in names})
    return data


def persist_output(value: Any, *, tool_name: str = "") -> Any:
    """旧工具结果协议也转成已知字段，不保存任意对象或运行时代码。"""
    if isinstance(value, dict):
        name = str(value.get("tool_name") or tool_name)
        return {key: persist_output(item, tool_name=name if key == "tool_result" else "")
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [persist_output(item) for item in value]
    if hasattr(value, "ok") and hasattr(value, "output") and hasattr(value, "error"):
        return {"ok": bool(value.ok), "tool_name": str(getattr(value, "tool_name", tool_name)),
                "output": persist_output(value.output), "error": str(value.error or ""),
                "status": str(getattr(value, "status", "success")),
                "error_code": str(getattr(value, "error_code", ""))}
    return value


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
    checks: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = ""
    attempt: int = 0
    error_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return serializable({item.name: persist_output(getattr(self, item.name)) for item in fields(self)})

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentResult:
        names = {item.name for item in fields(cls)}
        data = {key: item for key, item in value.items() if key in names}
        data["output"] = restore_output(data.get("output", {}))
        return cls(**data)


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
    handler_key: str = ""
    handler_version: str = "1"
    handler_binding: dict[str, Any] = field(default_factory=dict)
    retry_safe: bool = True
    approval_mode: str = "gate"

    def to_dict(self) -> dict[str, Any]:
        return serializable({item.name: getattr(self, item.name) for item in fields(self) if item.name != "handler"})

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskPlanStep:
        names = {item.name for item in fields(cls)} - {"handler"}
        data = {key: item for key, item in value.items() if key in names}
        data.setdefault("step_id", str(value.get("id", "")))
        return cls(**data)


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
    plan_version: str = "1"
    max_total_attempts: int = 12

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "steps"}
        data["steps"] = [step.to_dict() for step in self.steps]
        return serializable(data)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskPlan:
        names = {item.name for item in fields(cls)} - {"steps"}
        data = {key: item for key, item in value.items() if key in names}
        return cls(steps=[TaskPlanStep.from_dict(item) for item in value.get("steps", [])], **data)
