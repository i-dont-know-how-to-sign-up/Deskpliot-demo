from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable

from ..core.contracts import COMMIT_TOOLS, ToolNames
from ..tasks.control import (
    TaskBudgetExceeded, TaskCancelled, TaskControlError, TaskDeadlineExceeded,
    TaskPaused, execution_scope, get_execution_control,
)
from ..tasks.retry import ErrorCategory, RetryPolicy, UnknownSideEffect, classify_error, retry_after_seconds
from .schemas import AgentResult, TaskPlan, TaskPlanStep


@dataclass
class SupervisorResult:
    status: str
    values: dict[str, Any] = field(default_factory=dict)
    results: list[AgentResult] = field(default_factory=list)
    error: str = ""


def classified_exception(error: Exception) -> Exception:
    """兼容旧 handler 的字符串错误，但只有明确的瞬时错误允许重试。"""
    if classify_error(error) != ErrorCategory.PERMANENT:
        return error
    message = str(error).casefold()
    code = getattr(error, "status_code", getattr(error, "code", None))
    code = getattr(getattr(error, "response", None), "status_code", code)
    if code in {401, 403} or any(word in message for word in (
        "unauthorized", "forbidden", "authentication", "permission denied", "access denied",
        "invalid api key", "认证", "权限", "受保护", "401", "403",
    )):
        return PermissionError(str(error))
    if code in {400, 404, 422} or any(word in message for word in (
        "invalid argument", "invalid parameter", "not found", "不存在", "参数", "没有提供可用路径",
    )):
        return ValueError(str(error))
    if code in {408, 429, 500, 502, 503, 504} or any(word in message for word in (
        "temporary", "temporarily", "rate limit", "too many requests", "connection reset",
        "timed out", "timeout", "service unavailable", "429", "502", "503", "504", "网络抖动",
    )):
        return ConnectionError(str(error))
    return error


def _control_status(error: TaskControlError) -> str:
    if isinstance(error, TaskCancelled):
        return "cancelled"
    if isinstance(error, TaskPaused):
        return "paused"
    return "failed"


def _failed_check(checks: Any) -> bool:
    if checks is False:
        return True
    if isinstance(checks, dict):
        return any(_failed_check(value) for value in checks.values())
    if isinstance(checks, list):
        return any(_failed_check(value) for value in checks)
    return False


def _result_error(output: Any) -> tuple[str, Exception | None]:
    if not isinstance(output, dict):
        if hasattr(output, "ok"):
            return _result_error({
                "ok": output.ok, "status": getattr(output, "status", ""),
                "error": getattr(output, "error", ""), "output": getattr(output, "output", None),
            })
        return "success", None
    permission = output.get("permission") or {}
    if isinstance(permission, dict):
        if permission.get("blocked"):
            return "failed", PermissionError(str(permission.get("reason") or "工具权限被阻止"))
        if permission.get("requires_confirmation"):
            return "pending", None
    status = str(output.get("status") or "")
    if status in {"unknown", "recovery_required"} or output.get("state") == "unknown":
        return "recovery_required", UnknownSideEffect(str(output.get("error") or "外部操作结果未知"))
    if output.get("timed_out"):
        return "failed", TimeoutError(str(output.get("error") or "工具执行超时"))
    exit_code = output.get("exit_code", output.get("returncode", 0))
    if output.get("ok") is False or exit_code not in (None, 0, "0") or status in {"failed", "blocked", "error"}:
        error = str(output.get("error") or output.get("error_code") or f"工具执行失败（exit_code={exit_code}）")
        failure = RuntimeError(error)
        failure.status_code = output.get("status_code", output.get("http_status"))
        failure = classified_exception(failure)
        if isinstance(failure, RuntimeError):
            category = str(output.get("error_category") or output.get("error_code") or "")
            if category in {"authentication", "permission", "permission_denied", "invalid_api_key"}:
                failure = PermissionError(error)
            elif category in {"invalid_input", "invalid_argument", "invalid_parameter"}:
                failure = ValueError(error)
            elif category in {"transient", "rate_limit", "timeout"}:
                failure = ConnectionError(error)
        if output.get("retry_after") is not None:
            failure.headers = {"Retry-After": str(output["retry_after"])}
        return "failed", failure
    if status in {"pending", "waiting_human"}:
        return "pending", None
    for key in ("tool_result", "output"):
        if key in output:
            nested_status, error = _result_error(output[key])
            if nested_status != "success":
                return nested_status, error
    return "success", None


class SupervisorAgent:
    """顺序 DAG 执行、持久检查点和有界恢复；工具强制中断由控制层负责。"""

    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    def execute(
        self, plan: TaskPlan, initial_values: dict[str, Any] | None = None, *, parallel: bool = False,
        on_result: Callable[[AgentResult], None] | None = None,
        on_attempt: Callable[[TaskPlanStep, int], str | None] | None = None,
        completed_results: Iterable[AgentResult] | None = None,
        attempt_counts: dict[str, int] | None = None,
        initial_usage: dict[str, int] | None = None,
        approved_steps: Iterable[str] = (),
        control: Any = None,
    ) -> SupervisorResult:
        started = time.monotonic()
        control = control if control is not None else get_execution_control()
        policy = getattr(control, "retry_policy", None) or self.retry_policy
        initial = dict(initial_values or {})
        values = dict(initial)
        results: list[AgentResult] = []
        completed: set[str] = set()
        outputs_by_step: dict[str, dict[str, Any]] = {}
        counts = dict(attempt_counts or {})
        approved = set(approved_steps)
        ids = {step.step_id for step in plan.steps}
        if len(ids) != len(plan.steps) or any(not node_id for node_id in ids):
            return SupervisorResult("failed", values, results, "计划节点 ID 为空或重复")
        if any(set(step.depends_on) - ids for step in plan.steps):
            return SupervisorResult("failed", values, results, "计划存在缺失依赖")
        remaining = {step.step_id: set(step.depends_on) for step in plan.steps}
        visited: set[str] = set()
        while remaining:
            ready_ids = {key for key, dependencies in remaining.items() if dependencies <= visited}
            if not ready_ids:
                return SupervisorResult("failed", values, results, "计划存在循环依赖")
            visited.update(ready_ids)
            remaining = {key: dependencies for key, dependencies in remaining.items() if key not in ready_ids}
        saved = {item.step_id: item for item in completed_results or ()}
        # 只有校验通过、依赖也已恢复的输出才能成为可复用检查点。
        for step in plan.steps:
            item = saved.get(step.step_id)
            if item is not None and item.status in {"success", "succeeded"}:
                if _failed_check(item.checks) or _result_error(item.output)[0] != "success":
                    return SupervisorResult("recovery_required", values, results, f"节点检查点校验失败：{step.step_id}")
                completed.add(step.step_id)
                outputs_by_step[step.step_id] = dict(item.output)
                values.update(item.output)
                results.append(item)
        if any(not set(step.depends_on) <= completed for step in plan.steps if step.step_id in completed):
            return SupervisorResult("recovery_required", values, results, "已完成检查点缺少依赖输出")
        total_tools = max(sum(item.tool_calls for item in results), int((initial_usage or {}).get("tool_calls", 0)))
        total_tokens = max(sum(item.tokens for item in results), int((initial_usage or {}).get("total_tokens", 0)))
        total_attempts = sum(counts.values())
        control_tools = getattr(control, "max_tool_calls", None)
        control_tokens = getattr(control, "max_tokens", None)
        tool_limit = min(plan.max_tool_calls, control_tools if control_tools is not None else plan.max_tool_calls)
        token_limit = min(plan.max_total_tokens, control_tokens if control_tokens is not None else plan.max_total_tokens)
        baseline = plan.metadata.get("_execution_usage_baseline", {})
        if not isinstance(baseline, dict):
            baseline = {}
        baseline_tools = max(0, int(baseline.get("tool_calls", 0)))
        baseline_tokens = max(0, int(baseline.get("total_tokens", 0)))

        def checkpoint(*, dispatch: bool = False, tools: int = 0, tokens: int = 0) -> None:
            if control is not None:
                control.checkpoint()
            if time.monotonic() - started >= plan.max_duration_seconds:
                raise TaskDeadlineExceeded("任务超过最大执行时间")
            usage = control.snapshot_usage() if control is not None and hasattr(control, "snapshot_usage") else {}
            used_tools = max(total_tools + tools, int(usage.get("tool_calls", 0)) - baseline_tools)
            used_tokens = max(total_tokens + tokens, int(usage.get("total_tokens", 0)) - baseline_tokens)
            outer_tools = int(usage.get("tool_calls", 0))
            outer_tokens = int(usage.get("total_tokens", 0))
            if (control_tools is not None and (outer_tools > control_tools or dispatch and outer_tools >= control_tools)) or (
                control_tokens is not None and (outer_tokens > control_tokens or dispatch and outer_tokens >= control_tokens)
            ) or used_tools > tool_limit or used_tokens > token_limit or (
                dispatch and (used_tools >= tool_limit or used_tokens >= token_limit)
            ):
                raise TaskBudgetExceeded("任务超过预算")

        def emit(result: AgentResult) -> None:
            if on_result is not None:
                on_result(result)

        def run_step(step: TaskPlanStep) -> AgentResult:
            prior = saved.get(step.step_id)
            if prior is not None and prior.status in {"pending", "waiting_human"} and step.step_id not in approved:
                return prior
            if step.requires_human and step.step_id not in approved and step.approval_mode != "prepare":
                result = AgentResult(step.agent, "pending", {"requires_human": True}, step_id=step.step_id,
                                     checks={"approval_required": True})
                emit(result)
                return result
            if step.handler is None:
                result = AgentResult(step.agent, "failed", error="节点没有绑定执行器", step_id=step.step_id,
                                     error_category="invalid_input")
                emit(result)
                return result
            step_values = dict(initial)
            for dependency in step.depends_on:
                step_values.update(outputs_by_step[dependency])
            spent_tools = spent_tokens = 0
            limit = min(max(0, step.max_retries) + 1, policy.max_attempts)
            first_attempt = counts.get(step.step_id, 0) + 1
            if first_attempt > limit:
                return AgentResult(step.agent, "failed", error="节点尝试预算已用尽", step_id=step.step_id,
                                   error_category="budget")
            for attempt in range(first_attempt, limit + 1):
                checkpoint(dispatch=True, tools=spent_tools, tokens=spent_tokens)
                if total_attempts + attempt - first_attempt >= plan.max_total_attempts:
                    raise TaskBudgetExceeded("任务尝试预算已用尽")
                attempt_id = str(on_attempt(step, attempt) or "") if on_attempt is not None else ""
                output: dict[str, Any] = {}
                checks: dict[str, Any] = {}
                error: Exception | None = None
                status = "success"
                usage_before = control.snapshot_usage() if control is not None else {}
                with execution_scope(control, node_id=step.step_id, attempt_id=attempt_id):
                    try:
                        output = dict(step.handler(dict(step_values)) or {})
                        provided_checks = output.pop("_checks", output.get("checks", {}))
                        checks = dict(provided_checks) if isinstance(provided_checks, dict) else {"validation": provided_checks}
                        status, error = _result_error(output)
                        if output.pop("_waiting_human", False) and status in {"success", "pending"}:
                            status, error = "pending", None
                        if status == "success" and _failed_check(checks):
                            status, error = "failed", ValueError("节点输出校验未通过")
                    except TaskControlError as exc:
                        error, status = exc, _control_status(exc)
                    except Exception as exc:
                        error = classified_exception(exc)
                        status = "recovery_required" if isinstance(error, UnknownSideEffect) else "failed"
                    usage_after = control.snapshot_usage() if control is not None else {}
                    tools = max(0, int(output.pop("_tool_calls", getattr(error, "tool_calls", 0)) or 0))
                    tokens = max(0, int(output.pop("_tokens", getattr(error, "tokens", 0)) or 0))
                    tools = max(tools, usage_after.get("tool_calls", 0) - usage_before.get("tool_calls", 0))
                    tokens = max(tokens, usage_after.get("total_tokens", 0) - usage_before.get("total_tokens", 0))
                    spent_tools += tools
                    spent_tokens += tokens
                    category = classify_error(error).value if error is not None else ""
                    unsafe_tools = COMMIT_TOOLS | {
                        ToolNames.SHELL_EXECUTE_COMMAND, ToolNames.CODE_EXECUTE_PYTHON, ToolNames.ASSETS_DELETE,
                    }
                    safe_retry = step.retry_safe and not step.requires_human and not unsafe_tools.intersection(step.allowed_tools)
                    if error is not None and category == "transient" and not safe_retry:
                        status, category = "recovery_required", "unknown_side_effect"
                    checks["result_success"] = status == "success"
                    result = AgentResult(step.agent, status, output, str(error or ""), tools, tokens,
                                         step.step_id, checks, attempt_id, attempt, category)
                    # 检查点写入在重试等待和下游派发之前；存储异常不能被当作 handler 重试。
                    emit(result)
                    if error is None or attempt >= limit or not safe_retry:
                        result.tool_calls, result.tokens = spent_tools, spent_tokens
                        return result
                    checkpoint(dispatch=True, tools=spent_tools, tokens=spent_tokens)
                    headers = getattr(error, "headers", None)
                    retry_after = retry_after_seconds(headers.get("Retry-After")) if headers is not None else None
                    delay = policy.delay(attempt, retry_after)
                    remaining_seconds = plan.max_duration_seconds - (time.monotonic() - started)
                    if getattr(control, "deadline", None) is not None:
                        remaining_seconds = min(remaining_seconds, control.deadline - time.monotonic())
                    if delay >= remaining_seconds or delay > policy.max_delay:
                        result.tool_calls, result.tokens = spent_tools, spent_tokens
                        return result
                    if not policy.allows(error, attempt, control=control, idempotent=safe_retry):
                        result.tool_calls, result.tokens = spent_tools, spent_tokens
                        return result
                    if not policy.wait(attempt, control=control, retry_after=delay):
                        result.tool_calls, result.tokens = spent_tools, spent_tokens
                        return result
            raise AssertionError("unreachable attempt loop")

        while len(completed) < len(plan.steps):
            ready = [step for step in plan.steps if step.step_id not in completed and set(step.depends_on) <= completed]
            batch = ready if parallel and control is None and len(ready) > 1 else [ready[0]]
            try:
                checkpoint(dispatch=True)
                if len(batch) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(batch), max(1, plan.max_agents))) as pool:
                        batch_results = list(pool.map(run_step, batch))
                else:
                    batch_results = [run_step(batch[0])]
                for step, result in zip(batch, batch_results):
                    results.append(result)
                    values.update(result.output)
                    total_tools += result.tool_calls
                    total_tokens += result.tokens
                    total_attempts += max(0, result.attempt - counts.get(step.step_id, 0))
                    if result.status in {"success", "succeeded"}:
                        completed.add(step.step_id)
                        outputs_by_step[step.step_id] = dict(result.output)
                checkpoint()
            except TaskControlError as exc:
                return SupervisorResult(_control_status(exc), values, results, str(exc))
            failure = next((item for item in batch_results if item.status in {"failed", "blocked", "unknown", "recovery_required"}), None)
            if failure is not None:
                status = "recovery_required" if failure.status in {"unknown", "recovery_required"} else "failed"
                return SupervisorResult(status, values, results, failure.error)
            if any(item.status in {"pending", "waiting_human"} for item in batch_results):
                return SupervisorResult("waiting_human", values, results, "等待人工确认")
        return SupervisorResult("success", values, results)
