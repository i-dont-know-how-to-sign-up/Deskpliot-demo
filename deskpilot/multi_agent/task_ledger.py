from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tasks import TaskStore
from .schemas import AgentResult, TaskPlan


_UNSET = object()


class TaskLedger:
    """兼容旧执行回调，实际输入和输出经独立引用持久保存。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.store = TaskStore(self.path)
        # 新实例只回收到期 lease，不打断仍然存活的 worker。
        self.store.expire_leases()
        self.store.recover_orphaned_tasks()

    def start(
        self, plan: TaskPlan | dict[str, Any], *, initial_values: Any = None,
        session_id: str = "", plan_version: str | None = None, executor_version: str = "1",
        status: str = "running", **kwargs: Any,
    ) -> str:
        descriptor = self._plan_metadata(plan)
        metadata = descriptor.get("metadata", {})
        if initial_values is None:
            initial_values = metadata.get("initial_values", {})
        session_id = session_id or str(metadata.get("session_id", ""))
        plan_version = str(plan_version if plan_version is not None else descriptor.get("plan_version", "1"))
        return self.store.create_task(
            plan, initial_values=initial_values, session_id=session_id, plan_version=plan_version,
            executor_version=executor_version, status=status, **kwargs,
        )

    def update_node(self, task_id: str, result: AgentResult, **kwargs: Any) -> None:
        if self.store.get_task(task_id) is None:
            return
        checks = getattr(result, "checks", {})
        if checks:
            kwargs.setdefault("checks", checks)
            kwargs.setdefault("validation", {"ok": result.status in {"success", "succeeded"} and self._checks_pass(checks), "checks": checks})
        kwargs.setdefault("attempt_id", getattr(result, "attempt_id", "") or None)
        kwargs.setdefault("error_category", getattr(result, "error_category", ""))
        self.store.update_node(
            task_id, result.step_id, result.status, output=result.output, error=result.error,
            tool_calls=result.tool_calls, tokens=result.tokens, **kwargs,
        )

    def finish(self, task_id: str, status: str, error: str = "", *, output: Any = _UNSET, **kwargs: Any) -> None:
        if self.store.get_task(task_id) is None:
            return
        if output is not _UNSET:
            kwargs["output"] = output
        self.store.finish_task(task_id, status, error, **kwargs)

    def list_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.list_tasks(**kwargs)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.get_task(task_id)

    def start_attempt(self, task_id: str, node_id: str, attempt: int | None = None, **guard: Any) -> str:
        return self.store.start_attempt(task_id, node_id, **guard)["attempt_id"]

    def finish_attempt(self, attempt_id: str, status: str, **kwargs: Any) -> dict[str, Any]:
        return self.store.finish_attempt(attempt_id, status, **kwargs)

    def completed_results(self, task_id: str) -> list[AgentResult]:
        record = self.store.get_task(task_id)
        if record is None:
            return []
        valid = set(self.store.validate_recovery(task_id)["reusable_nodes"])
        return [AgentResult(
            agent=node.get("agent", "executor"), status=node["status"],
            output=self.store.load_output(task_id, node_id) or {}, error=node.get("error", ""),
            tool_calls=node["tool_calls"], tokens=node["tokens"], step_id=node_id,
            **({"checks": node.get("checks", {})} if "checks" in AgentResult.__dataclass_fields__ else {}),
        ) for node_id, node in record["nodes"].items() if node_id in valid]

    def resume(self, task_id: str, **guard: Any) -> dict[str, Any]:
        record = self.store.get_task(task_id)
        if record is None:
            raise KeyError(task_id)
        if record["legacy"]:
            raise ValueError("Legacy history cannot be resumed")
        for node_id, node in record["nodes"].items():
            if node["status"] in {"unknown", "recovery_required"} or (node["status"] in {"running", "interrupted"} and node.get("side_effect")):
                raise ValueError("Side effects require reconciliation before resume")
            if node["status"] == "running" and record["owner"] is None:
                self.store.update_node(task_id, node_id, "interrupted", record_attempt=True, **guard)
        self.store.clear_requests(task_id)
        return self.store.update_task(task_id, "running", error="", **guard)

    def recovery_candidates(self) -> list[dict[str, Any]]:
        return self.store.recovery_candidates()

    def save_plan(self, task_id: str, plan: Any, **kwargs: Any) -> str:
        return self.store.save_plan(task_id, plan, **kwargs)

    def load_plan(self, task_id: str) -> dict[str, Any] | None:
        return self.store.load_plan(task_id)

    def save_input(self, task_id: str, values: Any, **kwargs: Any) -> str:
        return self.store.save_input(task_id, values, **kwargs)

    def load_input(self, task_id: str) -> Any:
        return self.store.load_input(task_id)

    load_initial_values = load_input

    @staticmethod
    def _checks_pass(value: Any) -> bool:
        if value is False:
            return False
        if isinstance(value, dict):
            return all(TaskLedger._checks_pass(item) for item in value.values())
        if isinstance(value, list):
            return all(TaskLedger._checks_pass(item) for item in value)
        return True

    @staticmethod
    def _plan_metadata(plan: Any) -> dict[str, Any]:
        if isinstance(plan, dict):
            return plan
        return {"metadata": getattr(plan, "metadata", {}), "plan_version": getattr(plan, "plan_version", "1")}

    def save_output(self, task_id: str, output: Any, *, node_id: str | None = None, **kwargs: Any) -> str:
        return self.store.save_output(task_id, output, node_id=node_id, **kwargs)

    def load_output(self, task_id: str, node_id: str | None = None) -> Any:
        return self.store.load_output(task_id, node_id)

    _artifact_paths = staticmethod(TaskStore.artifact_paths)
