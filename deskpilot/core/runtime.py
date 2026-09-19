from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import AgentStep


PlanHandler = Callable[[dict[str, Any]], Any]


@dataclass
class PlanStep:
    name: str
    handler: PlanHandler
    description: str = ""
    max_retries: int = -1
    retryable: bool = True


@dataclass
class PlanExecution:
    values: dict[str, Any] = field(default_factory=dict)
    agent_steps: list[AgentStep] = field(default_factory=list)
    failed: bool = False
    failure: str = ""

    def add_step(self, name: str, status: str, detail: str) -> None:
        self.agent_steps.append(AgentStep(name, status, detail))


class PlanAndExecuteRuntime:
    """Small plan-and-execute runtime with observable steps and bounded retries."""

    def __init__(self, default_max_retries: int = 1):
        self.default_max_retries = max(0, int(default_max_retries))

    def run(self, plan: list[PlanStep], initial_values: dict[str, Any] | None = None) -> PlanExecution:
        execution = PlanExecution(values=dict(initial_values or {}))
        # 先把整条计划写进步骤列表，方便 UI 和日志直接看出执行路径。
        execution.add_step("plan_task", "success", self._describe_plan(plan))
        for step in plan:
            attempts = 0
            max_retries = step.max_retries if step.max_retries >= 0 else self.default_max_retries
            while True:
                attempts += 1
                try:
                    output = step.handler(execution.values)
                    if isinstance(output, dict):
                        execution.values.update(output)
                    elif output is not None:
                        execution.values[step.name] = output
                    detail = step.description or "step completed"
                    if attempts > 1:
                        detail = f"{detail}; retry_attempt={attempts}"
                    execution.add_step(step.name, "success", detail)
                    break
                except Exception as exc:
                    message = str(exc) or exc.__class__.__name__
                    if step.retryable and attempts <= max_retries:
                        # 只对可恢复错误做小范围重试，不把单步失败直接扩大成整条计划失败。
                        execution.add_step(step.name, "retry", f"{message}; retrying")
                        continue
                    execution.add_step(step.name, "failed", message)
                    execution.failed = True
                    execution.failure = message
                    return execution
        return execution

    def _describe_plan(self, plan: list[PlanStep]) -> str:
        if not plan:
            return "No steps."
        names = " -> ".join(step.name for step in plan)
        return f"Plan-and-execute route: {names}"
