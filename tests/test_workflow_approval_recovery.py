from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import deskpilot.core.agent as agent_module
from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.approval import PendingActionStore
from deskpilot.core.models import AgentStep, AnswerResult, Evidence
from deskpilot.core.sqlite_utils import sqlite_transaction
from deskpilot.memory.session_store import SessionStore
from deskpilot.mcp.email_mcp import EmailMCPService, MockEmailProvider
from deskpilot.multi_agent.plan_executor import PlanExecutor, PlanExecution
from deskpilot.multi_agent.supervisor import SupervisorResult
from deskpilot.multi_agent.task_ledger import TaskLedger
from deskpilot.tasks.control import TaskBudgetExceeded, TaskDeadlineExceeded, TaskPaused, get_execution_control
from deskpilot.tasks.effects import effects_for_task
from deskpilot.tasks.service import TaskService
from deskpilot.tools.permissions import PermissionDecision
from deskpilot.tools.tool_registry import ToolParameter, ToolRegistry, ToolResult, ToolSpec
from deskpilot.tools import write_tools


@pytest.fixture
def agent_factory(tmp_path: Path, monkeypatch):
    calls = Counter()
    agents = []
    provider = MockEmailProvider()
    email = EmailMCPService(provider)
    context = SimpleNamespace(text="fixture context", complexity="medium", debug_lines=[])
    monkeypatch.setattr(write_tools, "assess_path_write", lambda *args, **kwargs: PermissionDecision(
        False, True, False, "high", ["Fixture target requires approval."],
    ))
    monkeypatch.setattr(agent_module, "MemoryTurnManager", lambda *args: SimpleNamespace(
        process=lambda **kwargs: SimpleNamespace(skipped=True, reason="fixture", extracted=0, compacted=False),
    ))

    def make_agent():
        agent = DocumentQAAgent.__new__(DocumentQAAgent)
        agent.workspace_root = tmp_path
        agent._operation_lock = threading.RLock()
        agent._task_reference = {}
        agent.session_store = SessionStore(tmp_path / "sessions")
        agent.task_ledger = TaskLedger(tmp_path / "tasks.sqlite3")
        agent.pending_actions = PendingActionStore(db_path=agent.task_ledger.store.path)
        agent.task_service = TaskService(agent.task_ledger.store)
        agent.task_service.register("core.workflow", agent._run_task_workflow)
        agent._load_memory_context = lambda *args: context
        agent.context_assembler = SimpleNamespace(for_role=lambda *args, **kwargs: context)
        agent.memory_extractor = agent.memory_store = agent.memory_compactor = object()
        agent.client = SimpleNamespace(event_stream=lambda callback: nullcontext(), last_error="")
        agent.web_research_agent = SimpleNamespace(research=lambda query: pytest.fail("unexpected research"))

        def search_collection(*args, **kwargs):
            calls["retrieve"] += 1
            return [Evidence("chunk", "doc", "page 1", "Fixture evidence.", 1.0)]

        agent.index = SimpleNamespace(
            documents={"doc": SimpleNamespace(title="Fixture document", path=str(tmp_path / "source.md"))},
            search_collection=search_collection,
        )

        def chat(*args, **kwargs):
            calls["llm"] += 1
            return "Fixture report [1]."

        agent._chat_with_stage = chat
        registry = ToolRegistry()

        def search(**kwargs):
            calls["search"] += 1
            return [{"title": "Fixture source", "url": "https://example.com/source", "snippet": "Fixture evidence."}]

        registry.register(ToolSpec("web.search", "web", "fixture search", [
            ToolParameter("query", "string"), ToolParameter("limit", "integer"),
        ], search))

        def prepare_file(**kwargs):
            calls["prepare_file"] += 1
            return write_tools.write_file(**kwargs, confirm=False)

        def approve_file(**kwargs):
            calls["approve_file"] += 1
            return write_tools.write_file(**kwargs, confirm=True)

        registry.register(ToolSpec("files.write_file", "files", "fixture write", [
            ToolParameter("path", "string"), ToolParameter("content", "string"),
            ToolParameter("overwrite", "boolean"),
        ], prepare_file, approve_file))

        def prepare_email(**kwargs):
            calls["prepare_email"] += 1
            return email.send(**kwargs, confirm=False)

        def approve_email(**kwargs):
            calls["approve_email"] += 1
            return email.send(**kwargs, confirm=True)

        registry.register(ToolSpec("email.send", "email", "fixture send", [
            ToolParameter("to", "array"), ToolParameter("subject", "string"),
            ToolParameter("body", "string"), ToolParameter("attachment_paths", "array"),
        ], prepare_email, approve_email))
        agent.tool_registry = registry
        agents.append(agent)
        return agent

    yield make_agent, calls, provider
    for agent in agents:
        agent.task_service.close()


def run_domain(agent, domain: str):
    session = agent.session_store.get_or_create()
    question = "fixture workflow"
    user = agent.session_store.append_message(session.session_id, "user", question)
    common = dict(question=question, steps=[], session_id=session.session_id,
                  user_message_id=user.message_id, memory_context=agent._load_memory_context())
    if domain == "web_file":
        return agent._answer_web_file_request(
            request={"query": "fixture", "path": str(agent.workspace_root / "web.md")}, **common,
        )
    if domain == "index_report":
        return agent._answer_index_report(
            request={"query": "fixture", "doc_ids": ["doc"], "path": str(agent.workspace_root / "report.md")}, **common,
        )
    return agent._answer_research_email_request(
        request={"request": "fixture", "subject": "fixture", "to": ["recipient@example.com"],
                 "email_tool": "email.send"},
        plan_preview={"steps": [{"id": "source", "allowed_tools": ["web.search"]}]}, **common,
    )


def runner(agent):
    return PlanExecutor(tool_call=agent.tool_registry.call, research=agent.web_research_agent.research,
                        llm_call=lambda prompt: "fixture", task_ledger=agent.task_ledger)


def test_prepare_approve_and_manual_resume_runs_only_downstream(agent_factory):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    session = agent.session_store.create_session()
    user = agent.session_store.append_message(session.session_id, "user", "fixture DAG")
    target = agent.workspace_root / "downstream.md"

    def source(values):
        calls["source"] += 1
        return {"content": "durable source"}

    def prepare(values):
        result = agent.tool_registry.call("files.write_file", path=str(target), content=values["content"], overwrite=False)
        return {"file_content": values["content"], "write_output": result.output,
                "_waiting_human": True, "_checks": {"content_ready": True}}

    first = runner(agent).execute_bound_plan({"steps": [
        {"id": "source"},
        {"id": "commit", "depends_on": ["source"], "allowed_tools": ["files.write_file"], "requires_human": True},
        {"id": "downstream", "depends_on": ["commit"], "handler_key": "fixture.downstream"},
    ]}, {"source": source, "commit": prepare, "downstream": lambda values: pytest.fail("premature downstream")},
        prepare_steps={"commit"}, session_id=session.session_id)
    assert first.status == "waiting_human"
    task_id = first.values["task_id"]
    assert agent.task_ledger.store.get_task(task_id)["nodes"]["commit"]["checks"]["result_success"] is False
    agent._task_reference = {"task_id": task_id, "node_id": "commit"}
    waiting = agent._finalize_answer(
        "fixture prepared", [], [], False, session.session_id, user.message_id, "fixture DAG",
        agent._load_memory_context(), pending_action={"tool_name": "files.write_file",
            "kwargs": {"path": str(target), "content": "durable source", "overwrite": False}},
    )

    reopened = make_agent()

    def continue_workflow(payload):
        restored = runner(reopened)

        def downstream(values):
            calls["downstream"] += 1
            assert values["write_output"]["written"] is True
            assert values["write_output"]["permission"]["requires_confirmation"] is False
            assert values["file_content"] == "durable source"
            return {"done": True}

        # 已完成源/提交节点不需要重新注册 closure，只绑定剩余领域 handler。
        restored.register_handler("fixture.downstream", lambda binding: downstream)
        result = restored.resume(get_execution_control().task_id)
        assert result.status == "success", result.error
        return AnswerResult("downstream complete", [], [AgentStep("supervisor", result.status, "")], False)

    reopened.task_service.register("fixture.workflow", continue_workflow)
    reopened.task_service.bind(task_id, "fixture.workflow", {})
    approved = reopened.approve_pending_action(waiting.pending_action, session_id=session.session_id)
    assert approved.pending_action is None
    store = reopened.task_ledger.store
    assert store.get_task(task_id)["status"] == "paused"
    assert calls["downstream"] == 0
    checks = store.get_task(task_id)["nodes"]["commit"]["checks"]
    assert checks == {"content_ready": True, "result_success": True,
                      "approval_consumed": True, "approved_tool_success": True}
    checkpoint = store.load_checkpoint(task_id, "commit")
    assert checkpoint["validation"]["ok"] is True
    assert checkpoint["validation"]["checks"] == checks
    assert store.validate_recovery(task_id)["reusable_nodes"] == ["commit", "source"]
    assert not reopened.pending_actions.list_pending(task_id=task_id)
    reopened.task_service.resume(task_id)
    assert reopened.task_service.run_once()
    assert store.get_task(task_id)["status"] == "succeeded"
    assert calls["source"] == calls["prepare_file"] == calls["approve_file"] == calls["downstream"] == 1
    assert target.read_text(encoding="utf-8") == "durable source"


@pytest.mark.parametrize("domain", ["web_file", "index_report", "email"])
def test_domain_resume_after_approval_checkpoint_does_not_replay_or_reapprove(agent_factory, monkeypatch, domain):
    make_agent, calls, provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, domain)
    pending = first.pending_action
    assert pending
    task_id, node_id = pending["task_id"], pending["node_id"]
    before = calls.copy()
    store = agent.task_ledger.store

    def lose_terminal_commit(*args, **kwargs):
        raise RuntimeError("fixture terminal commit lost")

    with monkeypatch.context() as patch:
        patch.setattr(store, "finish_task", lose_terminal_commit)
        with pytest.raises(RuntimeError, match="terminal commit lost"):
            agent.approve_pending_action(pending, session_id=first.session_id)
    record = store.get_task(task_id)
    assert record["nodes"][node_id]["status"] == "success"
    output = store.load_output(task_id, node_id)
    assert output["tool_result"]["status"] == "success"
    assert output["_waiting_human"] is False
    if domain != "email":
        assert output["write_output"]["written"] is True
        assert output["write_output"] == output["tool_result"]["output"]
    else:
        assert output["tool_result"]["output"]["status"] == "sent"
    # 模拟收尾失败后的安全租约释放，再由真实 TaskService 恢复同一个任务。
    store.release_lease(task_id, record["owner"], record["lease_token"], status="paused")
    reopened = make_agent()
    delivered = []
    reopened.task_service.on_result = lambda task, result: delivered.append((task, result))
    reopened.task_service.resume(task_id)
    assert reopened.task_service.run_once()
    assert reopened.task_ledger.store.get_task(task_id)["status"] == "succeeded"
    assert delivered[0][0] == task_id
    assert delivered[0][1].pending_action is None
    assert not any(step.status == "waiting_human" for step in delivered[0][1].steps)
    for key in ("search", "retrieve", "llm", "prepare_file", "prepare_email"):
        assert calls[key] == before[key]
    assert calls["approve_email" if domain == "email" else "approve_file"] == 1
    assert len(provider.sent) == (1 if domain == "email" else 0)
    assert not reopened.pending_actions.list_pending(task_id=task_id)


@pytest.mark.parametrize("domain", ["web_file", "index_report", "email"])
def test_terminal_approval_completes_original_task(agent_factory, domain):
    make_agent, _calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, domain)
    task_id = first.pending_action["task_id"]
    result = agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    assert result.pending_action is None
    record = agent.task_ledger.store.get_task(task_id)
    assert record["status"] == "succeeded"
    assert record["owner"] is None
    assert all(node["status"] == "success" for node in record["nodes"].values())
    with pytest.raises(ValueError):
        agent.approve_pending_action(first.pending_action, session_id=first.session_id)


@pytest.mark.parametrize("domain", ["web_file", "index_report", "email"])
@pytest.mark.parametrize("status, message", [
    ("paused", "已暂停"), ("cancelled", "已取消"), ("recovery_required", "需要人工核对"),
])
def test_stopped_domain_does_not_report_success_or_create_approval(agent_factory, monkeypatch, domain, status, message):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    stopped = SupervisorResult(status, {"write_output": {"written": True}, "target": "stale.md"}, [], "fixture stop")
    monkeypatch.setattr(PlanExecutor, "execute_bound_plan", lambda *args, **kwargs: stopped)
    monkeypatch.setattr(PlanExecutor, "execute_email_plan", lambda *args, **kwargs: PlanExecution(
        stopped, "email.send", ToolResult(True, "email.send", {"status": "sent"}), {},
    ))
    result = run_domain(agent, domain)
    assert message in result.answer
    assert "fixture stop" in result.answer
    assert result.pending_action is None
    assert not any(step.name in {"write_file", "write_report", "request_email_confirmation"} for step in result.steps)
    assert next(step.status for step in result.steps if step.name == "supervisor") == status
    assert not calls


@pytest.mark.parametrize("method", ["initialize_metrics", "persist_metrics"])
def test_approval_metrics_failure_does_not_block_success_checkpoint(agent_factory, monkeypatch, caplog, method):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")

    def fail_metrics(control):
        raise sqlite3.OperationalError("fixture metrics failure")

    monkeypatch.setattr(agent.task_service, method, fail_metrics)
    result = agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    task_id, node_id = first.pending_action["task_id"], first.pending_action["node_id"]
    store = agent.task_ledger.store
    assert result.pending_action is None
    assert store.get_task(task_id)["status"] == "succeeded"
    assert store.load_checkpoint(task_id, node_id)["validation"]["ok"] is True
    assert calls["approve_file"] == 1
    assert "Approval metrics" in caplog.text


def test_approval_metrics_failure_does_not_mask_tool_error(agent_factory, monkeypatch):
    make_agent, _calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")

    def fail_tool(*args, **kwargs):
        raise ValueError("fixture tool error")

    def fail_metrics(control):
        raise sqlite3.OperationalError("fixture metrics failure")

    monkeypatch.setattr(agent.tool_registry, "call_approved", fail_tool)
    monkeypatch.setattr(agent.task_service, "persist_metrics", fail_metrics)
    with pytest.raises(ValueError, match="fixture tool error"):
        agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    assert agent.task_ledger.store.get_task(first.pending_action["task_id"])["status"] == "recovery_required"


def test_approval_heartbeat_covers_tool_fingerprint_and_terminal_commit(agent_factory, monkeypatch):
    make_agent, _calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")
    store = agent.task_ledger.store
    stage = ["tool"]
    observed = {name: threading.Event() for name in ("tool", "fingerprint", "terminal")}
    threads = set()
    heartbeat = store.heartbeat
    call_approved = agent.tool_registry.call_approved
    prepare_checkpoint = store._prepare_checkpoint
    finish_task = store.finish_task

    def observe_heartbeat(*args, **kwargs):
        result = heartbeat(*args, **kwargs)
        threads.add(threading.current_thread())
        observed[stage[0]].set()
        return result

    def wait_for_heartbeat(name):
        stage[0] = name
        assert observed[name].wait(5), f"approval heartbeat stopped during {name}"

    def slow_tool(*args, **kwargs):
        wait_for_heartbeat("tool")
        return call_approved(*args, **kwargs)

    def slow_fingerprint(*args, **kwargs):
        assert (agent.workspace_root / "web.md").exists()
        wait_for_heartbeat("fingerprint")
        return prepare_checkpoint(*args, **kwargs)

    def slow_terminal(*args, **kwargs):
        wait_for_heartbeat("terminal")
        return finish_task(*args, **kwargs)

    monkeypatch.setattr(store, "heartbeat", observe_heartbeat)
    monkeypatch.setattr(agent.tool_registry, "call_approved", slow_tool)
    monkeypatch.setattr(store, "_prepare_checkpoint", slow_fingerprint)
    monkeypatch.setattr(store, "finish_task", slow_terminal)
    agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    assert all(event.is_set() for event in observed.values())
    assert threads and all(not thread.is_alive() for thread in threads)
    assert store.get_task(first.pending_action["task_id"])["status"] == "succeeded"


def save_job_budget(agent, task_id, *, timeout=600, elapsed=0, tool_calls=0):
    usage = agent.task_service.descriptor(task_id)["usage"]
    usage["tool_calls"] = tool_calls
    with sqlite_transaction(agent.task_ledger.store.path) as connection:
        connection.execute("UPDATE runtime_jobs SET timeout_seconds=?,elapsed=?,usage_json=? WHERE task_id=?",
                           (timeout, elapsed, json.dumps(usage), task_id))


@pytest.mark.parametrize("timeout, elapsed, remaining", [(10, 8, 2), (600, 450, 120)])
def test_approval_uses_persisted_remaining_deadline_and_tool_budget(agent_factory, monkeypatch, timeout, elapsed, remaining):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")
    task_id = first.pending_action["task_id"]
    save_job_budget(agent, task_id, timeout=timeout, elapsed=elapsed, tool_calls=39)
    call_approved = agent.tool_registry.call_approved
    observed = {}
    started = time.monotonic()

    def inspect_control(*args, **kwargs):
        control = get_execution_control()
        observed.update(deadline=control.deadline, inspected=time.monotonic())
        assert control.max_tool_calls == 40
        assert control.snapshot_usage()["tool_calls"] == 39
        return call_approved(*args, **kwargs)

    monkeypatch.setattr(agent.tool_registry, "call_approved", inspect_control)
    agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    assert started + remaining <= observed["deadline"] <= observed["inspected"] + remaining
    assert calls["approve_file"] == 1
    assert agent.task_service.descriptor(task_id)["usage"]["tool_calls"] == 40
    assert agent.task_ledger.store.get_task(task_id)["status"] == "succeeded"


@pytest.mark.parametrize("tool_calls", [40, 41])
def test_approval_exhausted_tool_budget_is_not_dispatched_or_unknown(agent_factory, tool_calls):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")
    task_id, node_id = first.pending_action["task_id"], first.pending_action["node_id"]
    save_job_budget(agent, task_id, tool_calls=tool_calls)
    with pytest.raises(TaskBudgetExceeded, match="审批工具未派发"):
        agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    store = agent.task_ledger.store
    record = store.get_task(task_id)
    assert record["status"] == "failed"
    assert record["owner"] is None
    assert record["nodes"][node_id]["status"] == "waiting_human"
    attempt = store.list_attempts(task_id, node_id)[-1]
    assert attempt["certainty"] == "confirmed"
    assert attempt["tool_calls"] == 0
    assert calls["approve_file"] == 0
    assert effects_for_task(store, task_id) == []
    assert not (agent.workspace_root / "web.md").exists()


@pytest.mark.parametrize("elapsed", [10, 11])
def test_approval_exhausted_deadline_is_not_dispatched_or_unknown(agent_factory, elapsed):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")
    task_id, node_id = first.pending_action["task_id"], first.pending_action["node_id"]
    save_job_budget(agent, task_id, timeout=10, elapsed=elapsed)
    with pytest.raises(TaskDeadlineExceeded, match="审批工具未派发"):
        agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    store = agent.task_ledger.store
    assert store.get_task(task_id)["status"] == "failed"
    assert store.get_task(task_id)["nodes"][node_id]["status"] == "waiting_human"
    assert store.list_attempts(task_id, node_id)[-1]["certainty"] == "confirmed"
    assert calls["approve_file"] == 0
    assert effects_for_task(store, task_id) == []


def test_standalone_approval_without_job_keeps_one_tool_budget(agent_factory, monkeypatch):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    session = agent.session_store.create_session()
    user = agent.session_store.append_message(session.session_id, "user", "fixture standalone")
    first = agent._finalize_answer(
        "fixture prepared", [], [], False, session.session_id, user.message_id, "fixture standalone",
        agent._load_memory_context(), pending_action={"tool_name": "files.write_file", "kwargs": {
            "path": str(agent.workspace_root / "standalone.md"), "content": "fixture", "overwrite": False,
        }},
    )
    assert agent.task_service.descriptor(first.pending_action["task_id"]) is None
    call_approved = agent.tool_registry.call_approved

    def inspect_control(*args, **kwargs):
        control = get_execution_control()
        assert control.max_tool_calls == 1
        assert 119 < control.deadline - time.monotonic() <= 120
        return call_approved(*args, **kwargs)

    monkeypatch.setattr(agent.tool_registry, "call_approved", inspect_control)
    agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    assert calls["approve_file"] == 1


def test_pause_before_approval_dispatch_preserves_preview_for_manual_resume(agent_factory, monkeypatch):
    make_agent, calls, _provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "web_file")
    task_id, node_id = first.pending_action["task_id"], first.pending_action["node_id"]
    initialize_metrics = agent.task_service.initialize_metrics

    def pause_before_dispatch(control):
        initialize_metrics(control)
        assert task_id not in agent.task_service._active
        agent.task_service.request(task_id, "pause")

    with monkeypatch.context() as patch:
        patch.setattr(agent.task_service, "initialize_metrics", pause_before_dispatch)
        with pytest.raises(TaskPaused, match="审批工具未派发"):
            agent.approve_pending_action(first.pending_action, session_id=first.session_id)
    store = agent.task_ledger.store
    assert store.get_task(task_id)["status"] == "paused"
    assert store.get_task(task_id)["nodes"][node_id]["status"] == "waiting_human"
    assert store.list_attempts(task_id, node_id)[-1]["certainty"] == "confirmed"
    assert calls["approve_file"] == 0
    assert effects_for_task(store, task_id) == []
    delivered = []
    agent.task_service.on_result = lambda task, result: delivered.append(result)
    agent.task_service.resume(task_id)
    assert agent.task_service.run_once()
    renewed = delivered[0].pending_action
    assert renewed and renewed["task_id"] == task_id and renewed["node_id"] == node_id
    assert renewed["action_id"] != first.pending_action["action_id"]
    assert calls["search"] == calls["prepare_file"] == 1
    agent.approve_pending_action(renewed, session_id=first.session_id)
    assert store.get_task(task_id)["status"] == "succeeded"
    assert calls["approve_file"] == 1


@pytest.mark.parametrize("committed", [False, True])
def test_approval_heartbeat_observes_service_pause_outside_active_worker(agent_factory, committed):
    make_agent, _calls, provider = agent_factory
    agent = make_agent()
    first = run_domain(agent, "email")
    task_id = first.pending_action["task_id"]
    spec = agent.tool_registry.get_tool_spec("email.send")
    observed = []

    def pause_in_flight(**kwargs):
        control = get_execution_control()
        assert task_id not in agent.task_service._active
        result = spec.approval_handler(**kwargs) if committed else None
        agent.task_service.request(task_id, "pause")
        try:
            control.wait(5)
        except TaskPaused as exc:
            observed.append(control.pause_event.is_set())
            if committed:
                exc.side_effect_state = "committed"
                exc.result = result
            raise
        pytest.fail("approval heartbeat did not observe pause")

    agent.tool_registry.register(replace(spec, approval_handler=pause_in_flight))
    if committed:
        agent.approve_pending_action(first.pending_action, session_id=first.session_id)
        assert agent.task_ledger.store.get_task(task_id)["status"] == "succeeded"
        assert len(provider.sent) == 1
    else:
        with pytest.raises(TaskPaused, match="审批工具已派发，结果尚未确认"):
            agent.approve_pending_action(first.pending_action, session_id=first.session_id)
        assert agent.task_ledger.store.get_task(task_id)["status"] == "recovery_required"
        assert len(provider.sent) == 0
        with pytest.raises(ValueError, match="结果不确定"):
            agent.task_service.resume(task_id)
    assert observed == [True]
