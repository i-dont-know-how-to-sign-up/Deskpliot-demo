from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import time

from .schemas import AgentResult, TaskPlan


@dataclass
class SupervisorResult:
    status: str
    values: dict[str, Any] = field(default_factory=dict)
    results: list[AgentResult] = field(default_factory=list)
    error: str = ""


class SupervisorAgent:
    """P0 顺序 DAG Supervisor，负责依赖、预算和人工确认状态。"""

    def execute(self, plan: TaskPlan, initial_values: dict[str, Any] | None = None, *, parallel: bool = True) -> SupervisorResult:
        started = time.monotonic()
        values = dict(initial_values or {})
        results: list[AgentResult] = []
        completed: set[str] = set()
        outputs_by_step: dict[str, dict[str, Any]] = {}
        total_tools = 0
        total_tokens = 0
        while len(completed) < len(plan.steps):
            ready = [step for step in plan.steps if step.step_id not in completed and set(step.depends_on) <= completed]
            if not ready:
                return SupervisorResult("failed", values, results, "找不到可执行节点，计划可能存在循环依赖")
            batch = ready if parallel and len(ready) > 1 else [ready[0]]
            def run_step(step: Any) -> tuple[Any, AgentResult]:
                if step.handler is None:
                    return step, AgentResult(step.agent, "pending", {"requires_human": step.requires_human}, step_id=step.step_id)
                last_error = ""
                # 子 Agent 只接收初始任务状态和直接依赖节点的结构化输出，避免共享完整全局窗口。
                step_values = dict(initial_values or {})
                for dependency in step.depends_on:
                    step_values.update(outputs_by_step.get(dependency, {}))
                for attempt in range(step.max_retries + 1):
                    try:
                        output = dict(step.handler(dict(step_values)) or {})
                        status = "pending" if output.pop("_waiting_human", False) else "success"
                        tool_calls = int(output.pop("_tool_calls", 0) or 0)
                        tokens = int(output.pop("_tokens", 0) or 0)
                        return step, AgentResult(
                            step.agent, status, output, tool_calls=tool_calls,
                            tokens=tokens, step_id=step.step_id,
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        if attempt < step.max_retries:
                            continue
                return step, AgentResult(step.agent, "failed", error=last_error, step_id=step.step_id)

            if len(batch) > 1:
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    batch_results = list(pool.map(run_step, batch))
            else:
                batch_results = [run_step(batch[0])]
            for step, result in batch_results:
                results.append(result)
                completed.add(step.step_id)
                outputs_by_step[step.step_id] = dict(result.output)
                values.update(result.output)
            total_tools += sum(item.tool_calls for _, item in batch_results)
            total_tokens += sum(item.tokens for _, item in batch_results)
            if total_tools > plan.max_tool_calls or total_tokens > plan.max_total_tokens:
                return SupervisorResult("failed", values, results, "任务超过预算")
            if time.monotonic() - started > plan.max_duration_seconds:
                return SupervisorResult("failed", values, results, "任务超过最大执行时间")
            if any(item.status == "failed" for _, item in batch_results):
                failure = next(item.error for _, item in batch_results if item.status == "failed")
                return SupervisorResult("failed", values, results, failure)
            if any(item.status == "pending" for _, item in batch_results):
                return SupervisorResult("waiting_human", values, results, "等待人工确认")
        return SupervisorResult("success", values, results)
