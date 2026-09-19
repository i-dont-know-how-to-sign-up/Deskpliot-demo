from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.runtime import PlanAndExecuteRuntime, PlanStep


def test_plan_runtime_retries_retryable_step() -> None:
    attempts = {"count": 0}

    def flaky(values: dict[str, object]) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return {"ok": True}

    runtime = PlanAndExecuteRuntime(default_max_retries=1)
    execution = runtime.run([PlanStep("flaky_step", flaky)])

    assert execution.failed is False
    assert execution.values["ok"] is True
    assert attempts["count"] == 2
    assert any(step.name == "plan_task" for step in execution.agent_steps)
    assert any(step.name == "flaky_step" and step.status == "retry" for step in execution.agent_steps)
    assert any(step.name == "flaky_step" and step.status == "success" for step in execution.agent_steps)


def test_plan_runtime_stops_after_non_retryable_failure() -> None:
    def fail(values: dict[str, object]) -> dict[str, object]:
        raise ValueError("blocked")

    def should_not_run(values: dict[str, object]) -> dict[str, object]:
        return {"unexpected": True}

    runtime = PlanAndExecuteRuntime(default_max_retries=1)
    execution = runtime.run(
        [
            PlanStep("validate", fail, retryable=False),
            PlanStep("execute", should_not_run),
        ]
    )

    assert execution.failed is True
    assert execution.failure == "blocked"
    assert "unexpected" not in execution.values
    assert any(step.name == "validate" and step.status == "failed" for step in execution.agent_steps)
    assert not any(step.name == "execute" for step in execution.agent_steps)


def main() -> None:
    test_plan_runtime_retries_retryable_step()
    test_plan_runtime_stops_after_non_retryable_failure()
    print("Agent runtime tests passed.")


if __name__ == "__main__":
    main()
