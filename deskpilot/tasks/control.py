from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


class TaskControlError(RuntimeError):
    """执行控制信号，不得作为普通业务失败重试或降级。"""


class TaskCancelled(TaskControlError):
    pass


class TaskPaused(TaskControlError):
    pass


class TaskDeadlineExceeded(TaskControlError):
    pass


class TaskBudgetExceeded(TaskControlError):
    pass


@dataclass
class ExecutionControl:
    """任务级共享控制；deadline 是 time.monotonic() 的绝对时间。

    ContextVar 不自动传入新线程，worker 应显式进入 execution_scope。
    同一任务的节点复用本对象才能共享预算，计数包含失败的派发。
    """

    task_id: str = ""
    node_id: str = ""
    attempt_id: str = ""
    deadline: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_attempts: int | None = None
    max_node_attempts: int | None = None
    retry_policy: Any = None
    usage: dict[str, int] = field(default_factory=lambda: {
        "tool_calls": 0, "api_attempts": 0, "failed_attempts": 0,
        "retries": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "reported_calls": 0, "unknown_usage_calls": 0,
        "reserved_tokens": 0,
    })
    events: list[dict[str, Any]] = field(default_factory=list)
    _node_attempts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _node_retries: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _reservations: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be a finite monotonic timestamp")
        for name in ("max_tool_calls", "max_tokens", "max_attempts", "max_node_attempts"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.retry_policy is None:
            from .retry import RetryPolicy
            self.retry_policy = RetryPolicy()

    def cancel(self) -> None:
        self.cancel_event.set()

    def request_cancel(self) -> None:
        self.cancel()

    def pause(self) -> None:
        self.pause_event.set()

    def request_pause(self) -> None:
        self.pause()

    def resume(self) -> None:
        self.pause_event.clear()

    def checkpoint(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelled(f"Task {self.task_id} cancelled")
        if self.pause_event.is_set():
            raise TaskPaused(f"Task {self.task_id} paused")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TaskDeadlineExceeded(f"Task {self.task_id} deadline exceeded")

    def timeout(self, default: float) -> float:
        self.checkpoint()
        if not math.isfinite(default) or default <= 0:
            raise ValueError("timeout must be finite and positive")
        remaining = self.deadline - time.monotonic() if self.deadline is not None else default
        if remaining <= 0:
            raise TaskDeadlineExceeded(f"Task {self.task_id} deadline exceeded")
        return min(default, remaining)

    def wait(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wait must be finite and nonnegative")
        end = time.monotonic() + seconds
        while True:
            self.checkpoint()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            self.cancel_event.wait(self.timeout(min(remaining, 0.05)))

    def _ids(self) -> tuple[str, str]:
        attribution = _EXECUTION_ATTRIBUTION.get()
        if attribution is not None and attribution[0] is self:
            return attribution[1], attribution[2]
        return self.node_id, self.attempt_id

    def attribution(self) -> dict[str, str]:
        node_id, attempt_id = self._ids()
        return {"task_id": self.task_id, "node_id": node_id, "attempt_id": attempt_id}

    def snapshot_usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self.usage)

    def remaining_tokens(self) -> int | None:
        with self._lock:
            if self.max_tokens is None:
                return None
            return max(0, self.max_tokens - self.usage["total_tokens"] - self.usage["reserved_tokens"])

    def _event(self, kind: str, **values: Any) -> None:
        self.events.append({**self.attribution(), "kind": kind, **values})

    def record_tool(self, name: str = "") -> None:
        self.checkpoint()
        with self._lock:
            if self.max_tool_calls is not None and self.usage["tool_calls"] >= self.max_tool_calls:
                raise TaskBudgetExceeded("Tool call budget exhausted")
            self.usage["tool_calls"] += 1
            self._event("tool", name=name)

    def record_attempt(self, kind: str = "api", *, reserved_tokens: int = 0) -> int:
        self.checkpoint()
        with self._lock:
            count = self.usage["api_attempts"]
            node_id, _ = self._ids()
            node_count = self._node_attempts.get(node_id, 0)
            if self.max_attempts is not None and count >= self.max_attempts:
                raise TaskBudgetExceeded("Task attempt budget exhausted")
            if self.max_node_attempts is not None and node_count >= self.max_node_attempts:
                raise TaskBudgetExceeded("Node attempt budget exhausted")
            if self.max_tokens is not None and (
                self.usage["total_tokens"] + self.usage["reserved_tokens"] >= self.max_tokens
                or self.usage["total_tokens"] + self.usage["reserved_tokens"] + max(0, reserved_tokens) > self.max_tokens
            ):
                raise TaskBudgetExceeded("Token budget exhausted")
            self.usage["api_attempts"] += 1
            self._reservations[count + 1] = max(0, reserved_tokens)
            self.usage["reserved_tokens"] += max(0, reserved_tokens)
            self._node_attempts[node_id] = node_count + 1
            self._event("attempt", api_kind=kind, sequence=count + 1)
            return count + 1

    def record_usage(self, usage: dict[str, Any] | None = None, *,
                     usage_reported: bool | None = None, success: bool = True,
                     error: str = "", api_attempt: int | None = None, **metadata: Any) -> None:
        # 结算不执行 checkpoint，取消后的已知计费和失败仍须入账。
        values = usage if isinstance(usage, dict) else {}
        reported = any(key in values for key in (
            "prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"
        )) if usage_reported is None else usage_reported
        try:
            prompt = max(0, int(values.get("prompt_tokens", values.get("input_tokens", 0)) or 0))
            completion = max(0, int(values.get("completion_tokens", values.get("output_tokens", 0)) or 0))
            total = max(0, int(values.get("total_tokens", prompt + completion) or 0))
        except (TypeError, ValueError, OverflowError):
            prompt = completion = total = 0
            reported = False
        with self._lock:
            if api_attempt is not None:
                self.usage["reserved_tokens"] -= self._reservations.pop(api_attempt, 0)
            self.usage["failed_attempts"] += int(not success)
            self.usage["reported_calls" if reported else "unknown_usage_calls"] += 1
            if reported:
                self.usage["prompt_tokens"] += prompt
                self.usage["completion_tokens"] += completion
                self.usage["total_tokens"] += total
            self._event("usage", usage_reported=reported, success=success,
                        prompt_tokens=prompt, completion_tokens=completion,
                        total_tokens=total, error=error[:500], api_attempt=api_attempt, **metadata)

    def reserve_retry(self, *, max_task_retries: int, max_node_retries: int) -> bool:
        self.checkpoint()
        with self._lock:
            node_id, _ = self._ids()
            node_count = self._node_retries.get(node_id, 0)
            if self.usage["retries"] >= max_task_retries or node_count >= max_node_retries:
                return False
            self.usage["retries"] += 1
            self._node_retries[node_id] = node_count + 1
            self._event("retry", sequence=self.usage["retries"])
            return True


_EXECUTION_CONTROL: ContextVar[ExecutionControl | None] = ContextVar("deskpilot_execution_control", default=None)
_EXECUTION_ATTRIBUTION: ContextVar[tuple[ExecutionControl, str, str] | None] = ContextVar(
    "deskpilot_execution_attribution", default=None
)


def get_execution_control() -> ExecutionControl | None:
    return _EXECUTION_CONTROL.get()


@contextmanager
def execution_scope(control: ExecutionControl | None, *, node_id: str | None = None,
                    attempt_id: str | None = None) -> Iterator[ExecutionControl | None]:
    token = _EXECUTION_CONTROL.set(control)
    ids = control._ids() if control is not None else ("", "")
    attribution_token = _EXECUTION_ATTRIBUTION.set(
        (control, ids[0] if node_id is None else node_id, ids[1] if attempt_id is None else attempt_id)
        if control is not None else None
    )
    try:
        yield control
    finally:
        _EXECUTION_ATTRIBUTION.reset(attribution_token)
        _EXECUTION_CONTROL.reset(token)
