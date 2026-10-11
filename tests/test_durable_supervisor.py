from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from deskpilot.core.models import Evidence
from deskpilot.multi_agent.plan_executor import PlanExecutor
from deskpilot.multi_agent.schemas import AgentResult, TaskPlan, TaskPlanStep
from deskpilot.multi_agent.supervisor import SupervisorAgent
from deskpilot.multi_agent.task_ledger import TaskLedger
from deskpilot.tasks.control import ExecutionControl, execution_scope, get_execution_control
from deskpilot.tasks.retry import RetryPolicy, UnknownSideEffect
from deskpilot.tools.tool_registry import ToolResult


def supervisor() -> SupervisorAgent:
    return SupervisorAgent(RetryPolicy(base_delay=0, jitter=0))


def executor(ledger: TaskLedger | None = None) -> PlanExecutor:
    return PlanExecutor(tool_call=lambda name, **kwargs: ToolResult(True, name, {}),
                        research=lambda query: None, llm_call=lambda prompt: "body",
                        supervisor=supervisor(), task_ledger=ledger)


@pytest.mark.parametrize("error", [
    PermissionError("denied"), ValueError("bad argument"), FileNotFoundError("missing"),
    RuntimeError("401 unauthorized"), RuntimeError("403 forbidden"),
    RuntimeError("invalid parameter"), RuntimeError("unexpected bug"),
])
def test_permanent_failures_never_retry(error: Exception) -> None:
    calls = []
    def fail(values):
        calls.append(values)
        raise error
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=fail, max_retries=10),
    ]))
    assert result.status == "failed"
    assert len(calls) == 1
    assert result.results[0].error_category != "transient"


def test_attempt_start_and_result_checkpoint_precede_next_dispatch() -> None:
    events = []
    def fail_then_succeed(values):
        events.append("handler")
        if events.count("handler") == 1:
            raise ConnectionError("connection reset")
        return {"value": "ok", "_checks": {"validated": True}}
    plan = TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=fail_then_succeed),
        TaskPlanStep("b", "knowledge", "b", ["a"], handler=lambda values: {"seen": values["value"]}),
    ])
    def attempt(step, number):
        events.append(f"start:{step.step_id}:{number}")
        return f"attempt-{number}"
    result = supervisor().execute(plan, on_attempt=attempt,
                                  on_result=lambda item: events.append(f"save:{item.step_id}:{item.status}"))
    assert result.status == "success"
    assert events[:6] == ["start:a:1", "handler", "save:a:failed", "start:a:2", "handler", "save:a:success"]
    assert events[6:] == ["start:b:1", "save:b:success"]
    assert result.results[0].checks["validated"] is True
    assert result.results[0].attempt == 2


def test_checkpoint_failure_does_not_retry_handler() -> None:
    calls = []
    plan = TaskPlan("test", "multi_agent", [TaskPlanStep("a", "knowledge", "a", handler=lambda values: calls.append(1) or {})])
    def cannot_store(item):
        raise OSError("disk unavailable")
    with pytest.raises(OSError, match="disk unavailable"):
        supervisor().execute(plan, on_result=cannot_store)
    assert calls == [1]


@pytest.mark.parametrize("output", [
    {"ok": False}, {"exit_code": 9}, {"returncode": 1},
    {"tool_result": ToolResult(True, "command", {"ok": False})},
    {"_checks": {"citation_valid": False}},
    {"tool_result": ToolResult(True, "write", {"permission": {"blocked": True}}), "_waiting_human": True},
])
def test_unsuccessful_outputs_never_unlock_downstream(output) -> None:
    calls = []
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=lambda values: output),
        TaskPlanStep("b", "knowledge", "b", ["a"], handler=lambda values: calls.append(1) or {}),
    ]))
    assert result.status == "failed"
    assert not calls


@pytest.mark.parametrize("signal,status", [("cancel", "cancelled"), ("pause", "paused")])
def test_control_prevents_dispatch(signal, status) -> None:
    control = ExecutionControl()
    getattr(control, signal)()
    calls = []
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=lambda values: calls.append(1) or {}),
    ]), control=control)
    assert result.status == status
    assert not calls


def test_expired_deadline_and_zero_budget_prevent_dispatch() -> None:
    calls = []
    plan = TaskPlan("test", "multi_agent", [TaskPlanStep("a", "knowledge", "a", handler=lambda values: calls.append(1) or {})])
    assert supervisor().execute(plan, control=ExecutionControl(deadline=time.monotonic() - 1)).status == "failed"
    plan.max_tool_calls = 0
    assert supervisor().execute(plan).status == "failed"
    assert not calls


def test_plan_budget_excludes_preplanning_usage_but_limits_execution() -> None:
    control = ExecutionControl(max_tokens=80000)
    control.record_usage({"total_tokens": 12564})
    calls = []

    def first(_values):
        calls.append("first")
        control.record_usage({"total_tokens": 10})
        return {"value": "ready"}

    result = executor().execute_bound_plan(
        {"max_total_tokens": 10, "steps": [
            {"id": "first", "agent": "knowledge"},
            {"id": "second", "agent": "communication", "depends_on": ["first"]},
        ]},
        {"first": first, "second": lambda values: calls.append("second") or {}},
        control=control,
    )
    assert result.status == "failed"
    assert "预算" in result.error
    assert calls == ["first"]
    assert control.snapshot_usage()["total_tokens"] == 12574


def test_outer_task_budget_still_blocks_dispatch_after_plan_baseline() -> None:
    control = ExecutionControl(max_tokens=12564)
    control.record_usage({"total_tokens": 12564})
    calls = []
    result = executor().execute_bound_plan(
        {"steps": [{"id": "read", "agent": "knowledge"}]},
        {"read": lambda values: calls.append("read") or {}}, control=control,
    )
    assert result.status == "failed"
    assert "预算" in result.error
    assert not calls


def test_email_plan_uses_explicit_attachment_not_planner_source(tmp_path: Path) -> None:
    attachment = tmp_path / "attachment.txt"
    attachment.write_text("expected attachment", encoding="utf-8")
    unrelated = tmp_path / "unrelated.pdf"
    unrelated.write_bytes(b"unrelated")
    calls = []

    def tool_call(name, **kwargs):
        calls.append((name, kwargs))
        if name == "files.read_document":
            return ToolResult(True, name, {"path": kwargs["path"], "content": "source"})
        return ToolResult(True, name, {"permission": {"requires_confirmation": True}})

    runner = PlanExecutor(tool_call=tool_call, research=lambda query: None,
                          llm_call=lambda prompt: "Please see the attachment.", supervisor=supervisor())
    result = runner.execute_email_plan({"steps": [
        {"id": "source", "agent": "knowledge", "allowed_tools": ["files.read_document"],
         "arguments": {"paths": [str(unrelated)]}},
        {"id": "commit", "agent": "communication", "allowed_tools": ["email.save_draft"]},
    ]}, {"to": "example@example.com", "subject": "Attachment", "request": "Please see attachment",
         "attachment_paths": [str(attachment)], "email_tool": "email.save_draft"})
    assert result.supervisor.status == "waiting_human"
    assert calls[0] == ("files.read_document", {"path": str(attachment)})
    assert result.call_arguments["attachment_paths"] == [str(attachment)]


def test_pause_interrupts_backoff_and_preserves_attempt_checkpoint() -> None:
    control = ExecutionControl(retry_policy=RetryPolicy(base_delay=0, jitter=0))
    calls = []
    checkpoints = []
    def fail(values):
        calls.append(1)
        raise ConnectionError("temporary")
    def wait(seconds):
        control.pause()
        control.checkpoint()
    control.wait = wait
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=fail),
    ]), control=control, on_result=checkpoints.append)
    assert result.status == "paused"
    assert calls == [1]
    assert checkpoints[0].attempt == 1
    assert checkpoints[0].error_category == "transient"


def test_real_usage_on_failed_attempt_prevents_retry() -> None:
    control = ExecutionControl(max_tool_calls=1, retry_policy=RetryPolicy(base_delay=0, jitter=0))
    checkpoints = []
    def fail(values):
        current = get_execution_control()
        assert current.attribution()["node_id"] == "a"
        assert current.attribution()["attempt_id"] == "id1"
        current.record_tool("read")
        current.record_usage({"total_tokens": 7}, success=False)
        raise ConnectionError("temporary")
    plan = TaskPlan("test", "multi_agent", [TaskPlanStep("a", "knowledge", "a", handler=fail)], max_tool_calls=1)
    result = supervisor().execute(plan, control=control, on_result=checkpoints.append,
                                  on_attempt=lambda step, attempt: "id1")
    assert result.status == "failed"
    assert len(checkpoints) == 1
    assert checkpoints[0].tool_calls == 1
    assert checkpoints[0].tokens == 7
    assert control.snapshot_usage()["failed_attempts"] == 1


def test_unsafe_transient_and_unknown_effect_do_not_retry() -> None:
    for error in [ConnectionError("temporary"), UnknownSideEffect("dispatched")]:
        calls = []
        def fail(values):
            calls.append(1)
            raise error
        result = supervisor().execute(TaskPlan("test", "multi_agent", [
            TaskPlanStep("a", "executor", "a", handler=fail, retry_safe=False, max_retries=10),
        ]))
        assert result.status == "recovery_required"
        assert calls == [1]


def test_restore_codec_evidence_toolresult_and_plan_bindings() -> None:
    evidence = Evidence("c", "d", "source", "text", 0.8)
    result = AgentResult("knowledge", "success", {"evidences": [evidence],
        "tool_result": ToolResult(True, "read", {"nested": [evidence]})})
    restored = AgentResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.output["evidences"][0].doc_id == "d"
    assert restored.output["tool_result"].output["nested"][0].text == "text"
    plan = TaskPlan("test", "multi_agent", [TaskPlanStep("a", "knowledge", "a", handler=lambda values: {},
        handler_key="domain.read", handler_binding={"path": Path("input.md")})])
    descriptor = json.loads(json.dumps(plan.to_dict()))
    assert "handler" not in descriptor["steps"][0]
    assert TaskPlan.from_dict(descriptor).steps[0].handler_binding["path"] == "input.md"


def test_approval_gate_preserves_completed_dependency_and_resumes_downstream() -> None:
    calls = []
    plan = TaskPlan("test", "multi_agent", [
        TaskPlanStep("read", "knowledge", "read", handler=lambda values: {"source": "saved"}),
        TaskPlanStep("commit", "executor", "commit", ["read"], requires_human=True,
                     handler=lambda values: calls.append("commit") or {"sent": True}),
        TaskPlanStep("after", "knowledge", "after", ["commit"], handler=lambda values: {"done": values["sent"]}),
    ])
    first = supervisor().execute(plan)
    assert first.status == "waiting_human"
    assert not calls
    approved = AgentResult("executor", "success", {"sent": True}, step_id="commit", checks={"approved": True})
    resumed = supervisor().execute(plan, completed_results=[first.results[0], approved])
    assert resumed.status == "success"
    assert resumed.values["done"] is True
    assert not calls


def test_gate_waiting_does_not_consume_the_approved_execution_attempt(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    calls = []
    runner.register_handler("domain.commit", lambda binding: lambda values: calls.append(1) or {"sent": True})
    first = runner.execute_bound_plan({"steps": [
        {"id": "commit", "handler_key": "domain.commit", "requires_human": True, "max_retries": 0},
    ]}, {})
    task_id = first.values["task_id"]
    assert ledger.get_task(task_id)["nodes"]["commit"]["status"] == "waiting_human"
    assert not ledger.store.list_attempts(task_id)
    result = runner.resume(task_id, approved_steps=["commit"])
    assert result.status == "success", result.error
    assert calls == [1]
    assert len(ledger.store.list_attempts(task_id)) == 1


def test_restart_uses_persisted_output_and_initial_values(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    calls = []
    preview = {"steps": [
        {"id": "read", "handler_key": "domain.read", "agent": "knowledge"},
        {"id": "write", "handler_key": "domain.write", "agent": "knowledge", "depends_on": ["read"]},
    ]}
    def interrupted(values):
        raise SystemExit("worker killed")
    with pytest.raises(SystemExit):
        runner.execute_bound_plan(preview, {
            "read": lambda values: {"evidences": [Evidence("c", "d", "s", "saved text", 1)]},
            "write": interrupted,
        }, initial_values={"question": "original"}, session_id="session")
    task_id = ledger.list_tasks()[0]["task_id"]
    restarted = executor(TaskLedger(tmp_path / "tasks.json"))
    restarted.register_handler("domain.read", lambda binding: lambda values: calls.append("read") or {})
    restarted.register_handler("domain.write", lambda binding: lambda values: {
        "answer": values["question"] + ":" + values["evidences"][0].text,
    })
    result = restarted.resume(task_id)
    assert result.status == "success", result.error
    assert result.values["answer"] == "original:saved text"
    assert result.values["task_id"] == task_id
    assert not calls


def test_unregistered_closure_cannot_resume(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    first = runner.execute_bound_plan({"steps": [
        {"id": "commit", "requires_human": True},
    ]}, {"commit": lambda values: {"ok": True}})
    resumed = runner.resume(first.values["task_id"])
    assert resumed.status == "recovery_required"
    assert "未注册" in resumed.error


def test_queued_outer_task_is_adopted_without_finishing_lease(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    task_id = ledger.store.create_task(TaskPlan("queued", "multi_agent", []), session_id="session")
    claimed = ledger.store.claim(task_id, "worker")
    control = ExecutionControl(task_id=task_id)
    control.ledger_guard = {"owner": "worker", "lease_token": claimed["lease_token"]}
    with execution_scope(control):
        result = executor(ledger).execute_bound_plan({"steps": [{"id": "read"}]}, {
            "read": lambda values: {"ok": True},
        }, initial_values={"original": True}, session_id="session")
    assert result.status == "success"
    record = ledger.get_task(task_id)
    assert record["owner"] == "worker"
    assert len(ledger.list_tasks()) == 1
    assert ledger.load_initial_values(task_id) == {"original": True}


def test_persisted_approval_result_skips_commit_and_runs_downstream(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    runner.register_handler("domain.after", lambda binding: lambda values: {"done": values["sent"]})
    first = runner.execute_bound_plan({"steps": [
        {"id": "source"}, {"id": "commit", "depends_on": ["source"], "requires_human": True},
        {"id": "after", "handler_key": "domain.after", "depends_on": ["commit"]},
    ]}, {"source": lambda values: {"source": True}, "commit": lambda values: pytest.fail("unapproved effect"),
         "after": lambda values: {"done": True}})
    task_id = first.values["task_id"]
    ledger.update_node(task_id, AgentResult("executor", "success", {"sent": True}, step_id="commit", checks={"approved": True}))
    result = runner.resume(task_id)
    assert result.status == "success", result.error
    assert result.values["done"] is True


def test_retry_attempts_are_persisted_with_classification_and_checks(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    calls = []
    def flaky(values):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("temporary")
        return {"value": "saved", "_checks": {"hash_valid": True}}
    result = executor(ledger).execute_bound_plan({"steps": [{"id": "a"}]}, {"a": flaky})
    attempts = ledger.store.list_attempts(result.values["task_id"], "a")
    assert [item["status"] for item in attempts] == ["failed", "success"]
    assert attempts[0]["error_category"] == "transient"
    node = ledger.get_task(result.values["task_id"])["nodes"]["a"]
    assert node["checks"]["hash_valid"] is True
    assert node["attempt_count"] == 2


def test_restart_cannot_reset_exhausted_attempt_budget(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    first = runner.execute_bound_plan({"steps": [
        {"id": "a", "handler_key": "domain.a", "max_retries": 0},
    ]}, {"a": lambda values: {"ok": False}})
    calls = []
    runner.register_handler("domain.a", lambda binding: lambda values: calls.append(1) or {"ok": True})
    result = runner.resume(first.values["task_id"])
    assert result.status == "failed"
    assert not calls
    assert "预算" in result.error


def test_failed_attempt_tool_usage_survives_resume(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    first = runner.execute_bound_plan({"max_tool_calls": 1, "steps": [
        {"id": "a", "handler_key": "domain.a"},
    ]}, {"a": lambda values: {"ok": False, "error_category": "transient", "_tool_calls": 1}})
    calls = []
    runner.register_handler("domain.a", lambda binding: lambda values: calls.append(1) or {})
    result = runner.resume(first.values["task_id"])
    assert result.status == "failed"
    assert "预算" in result.error
    assert not calls


def test_resume_rejects_changed_handler_binding(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    first = runner.execute_bound_plan({"steps": [
        {"id": "a", "requires_human": True, "arguments": {"path": "one"}},
    ]}, {"a": lambda values: {}})
    result = runner.execute_bound_plan({"steps": [
        {"id": "a", "requires_human": True, "arguments": {"path": "two"}},
    ]}, {"a": lambda values: {}}, resume_task_id=first.values["task_id"])
    assert result.status == "recovery_required"
    assert len(ledger.list_tasks()) == 1


def test_resume_rejects_changed_registered_version(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    first = runner.execute_bound_plan({"steps": [
        {"id": "a", "handler_key": "domain.a", "requires_human": True},
    ]}, {"a": lambda values: {}})
    runner.register_handler("domain.a", lambda binding: lambda values: {}, version="2")
    result = runner.resume(first.values["task_id"])
    assert result.status == "recovery_required"
    assert "版本" in result.error


def test_resume_preserves_serialized_waiting_email_preview(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    runner.research = lambda query: SimpleNamespace(report="evidence", artifact_path="")
    calls = []
    runner.tool_call = lambda name, **kwargs: calls.append(name) or ToolResult(
        True, name, {"permission": {"requires_confirmation": True}})
    first = runner.execute_email_plan({"steps": [{"id": "source", "allowed_tools": ["web.research"]}]}, {
        "subject": "subject", "request": "request", "to": ["a@example.org"],
    })
    resumed = runner.resume(first.supervisor.values["task_id"])
    assert resumed.status == "waiting_human", resumed.error
    waiting = resumed.results[-1]
    assert waiting.output["call_arguments"]["body"] == "body"
    assert waiting.output["tool_result"].output["permission"]["requires_confirmation"]
    assert calls == ["email.send"]


def test_control_zero_budget_is_checked_before_handler() -> None:
    calls = []
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=lambda values: calls.append(1) or {}),
    ]), control=ExecutionControl(max_tool_calls=0))
    assert result.status == "failed"
    assert not calls


def test_synchronous_executor_installs_attributed_deadline_control(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    seen = []
    def handler(values):
        control = get_execution_control()
        seen.append(control.attribution())
        assert 0 < control.timeout(300) <= 120
        control.record_tool("read")
        control.record_usage({"total_tokens": 3})
        return {"ok": True}
    result = executor(ledger).execute_bound_plan({"steps": [{"id": "a"}]}, {"a": handler})
    assert result.status == "success"
    assert seen[0]["task_id"] == result.values["task_id"]
    assert seen[0]["node_id"] == "a"
    assert seen[0]["attempt_id"]
    assert result.results[0].tool_calls == 1
    assert result.results[0].tokens == 3
    assert get_execution_control() is None


def test_shared_retry_policy_caps_attempts_and_waits() -> None:
    control = ExecutionControl(retry_policy=RetryPolicy(max_attempts=3, base_delay=0, jitter=0))
    calls = []
    def fail(values):
        calls.append(1)
        raise ConnectionError("temporary")
    result = supervisor().execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=fail, max_retries=99),
    ]), control=control)
    assert result.status == "failed"
    assert len(calls) == 3
    assert control.snapshot_usage()["retries"] == 2


def test_changed_artifact_is_not_reused_or_replayed(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    artifact = tmp_path / "report.md"
    artifact.write_text("original", encoding="utf-8")
    first = runner.execute_bound_plan({"steps": [{"id": "a"}]}, {
        "a": lambda values: {"artifact_path": str(artifact)},
    })
    artifact.write_text("user changed", encoding="utf-8")
    result = runner.resume(first.values["task_id"])
    assert result.status == "recovery_required"
    assert artifact.read_text(encoding="utf-8") == "user changed"


def test_retry_after_beyond_deadline_never_waits_or_dispatches_again(monkeypatch) -> None:
    calls = []
    policy = RetryPolicy(base_delay=0, jitter=0)
    waits = []
    monkeypatch.setattr(RetryPolicy, "wait", lambda *args, **kwargs: waits.append(1))
    def handler(values):
        calls.append(1)
        return {"ok": False, "http_status": 429, "retry_after": 5}
    result = SupervisorAgent(policy).execute(TaskPlan("test", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=handler),
    ], max_duration_seconds=1))
    assert result.status == "failed"
    assert calls == [1]
    assert not waits


def test_prepare_binding_can_restore_preview_without_replaying_tool(tmp_path: Path) -> None:
    ledger = TaskLedger(tmp_path / "tasks.json")
    runner = executor(ledger)
    calls = []
    def prepare(values):
        calls.append(1)
        return {"tool_name": "files.write_file", "call_arguments": {"content": values["source"]},
                "tool_result": ToolResult(True, "files.write_file", {"permission": {"requires_confirmation": True}})}
    runner.register_handler("domain.prepare", lambda binding: prepare)
    preview = {"steps": [
        {"id": "source"}, {"id": "prepare", "depends_on": ["source"], "requires_human": True,
                             "handler_key": "domain.prepare"},
    ]}
    first = runner.execute_bound_plan(preview, {"source": lambda values: {"source": "saved"}, "prepare": prepare},
                                      prepare_steps=["prepare"])
    assert first.status == "waiting_human"
    resumed = runner.resume(first.values["task_id"])
    assert resumed.status == "waiting_human"
    assert resumed.results[-1].output["call_arguments"]["content"] == "saved"
    assert calls == [1]
