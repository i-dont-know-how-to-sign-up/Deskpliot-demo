from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from deskpilot.tasks.control import ExecutionControl, TaskCancelled, TaskPaused, get_execution_control
from deskpilot.tasks.models import LeaseConflict, VersionConflict
from deskpilot.tasks.service import TaskService
from deskpilot.tasks.store import TaskStore


@pytest.fixture
def service(tmp_path: Path) -> TaskService:
    value = TaskService(TaskStore(tmp_path / "tasks.sqlite3"))
    value.register("fixture", lambda payload: {"answer": "done"})
    yield value
    value.close()


def expire(store: TaskStore, task_id: str) -> None:
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE task_id=?", (task_id,))
    assert store.expire_leases() == [task_id]


def raise_observer(*args, **kwargs) -> None:
    raise RuntimeError("UI disconnected")


@pytest.mark.parametrize("error", [None, RuntimeError, TaskCancelled, TaskPaused])
@pytest.mark.parametrize("reclaimed", [False, True])
def test_stale_worker_cannot_write_after_expiry_or_new_completion(service, error, reclaimed) -> None:
    store = service.store
    task_id = service.submit("fixture", {})
    saved = {}

    def handler(payload):
        get_execution_control().record_tool("old worker")
        expire(store, task_id)
        if reclaimed:
            store.resume_runtime_task(task_id)
            current = store.claim(task_id, "replacement")
            store.finish_worker_task(task_id, "succeeded", output={"answer": "replacement"},
                                     owner="replacement", lease_token=current["lease_token"])
        saved["task"] = store.get_task(task_id)
        saved["events"] = store.events(task_id)
        if error:
            raise error("old worker result")
        return {"answer": "old worker"}

    service.register("fixture", handler)
    delivered = []
    service.on_result = lambda *args: delivered.append(args)
    assert service.run_once()
    assert store.get_task(task_id) == saved["task"]
    assert store.events(task_id) == saved["events"]
    assert store.load_output(task_id) == ({"answer": "replacement"} if reclaimed else None)
    assert service.descriptor(task_id)["usage"] == {}
    assert not delivered
    assert not service._active


def test_result_callback_failure_keeps_success_and_does_not_replay(service) -> None:
    calls = []
    service.register("fixture", lambda payload: calls.append(payload) or {"answer": "done"})
    service.on_result = service.on_error = raise_observer
    first = service.submit("fixture", {"sequence": 1})
    second = service.submit("fixture", {"sequence": 2})
    assert service.run_once()
    assert service.store.get_task(first)["status"] == "succeeded"
    assert service.store.load_output(first) == {"answer": "done"}
    with pytest.raises(ValueError):
        service.resume(first)
    assert service.run_once()
    assert service.store.get_task(second)["status"] == "succeeded"
    assert calls == [{"sequence": 1}, {"sequence": 2}]
    assert not service._active


def test_error_callback_and_metrics_failure_do_not_break_cleanup(service, monkeypatch) -> None:
    def handler(payload):
        raise RuntimeError("business failure")

    service.register("fixture", handler)
    service.on_error = raise_observer
    monkeypatch.setattr(service, "persist_metrics", raise_observer)
    task_id = service.submit("fixture", {})
    assert service.run_once()
    task = service.store.get_task(task_id)
    assert task["status"] == "failed" and task["error"] == "business failure"
    assert not service._active


def test_fenced_result_notifies_ui_even_if_observer_raises(service) -> None:
    task_id = service.submit("fixture", {})
    notified = []

    def handler(payload):
        expire(service.store, task_id)
        return {"answer": "stale"}

    def observer(task, error):
        notified.append((task, error))
        raise_observer()

    service.register("fixture", handler)
    service.on_error = observer
    assert service.run_once()
    assert len(notified) == 1 and notified[0][0] == task_id
    assert "Tasks" in notified[0][1]
    assert service.store.get_task(task_id)["status"] == "interrupted"
    assert service.store.load_output(task_id) is None


def test_fenced_result_and_final_metrics_failure_do_not_restart_run_forever(service, monkeypatch) -> None:
    task_id = service.submit("fixture", {})
    errors = []

    def handler(payload):
        expire(service.store, task_id)
        return {"answer": "stale"}

    service.register("fixture", handler)
    service.on_error = lambda task, error: errors.append((task, error))
    monkeypatch.setattr(service, "persist_metrics", raise_observer)
    original = service.run_once
    calls = []

    def two_iterations():
        calls.append(True)
        if len(calls) == 1:
            return original()
        service._stop.set()
        return False

    monkeypatch.setattr(service, "run_once", two_iterations)
    monkeypatch.setattr(service._wake, "wait", lambda timeout: None)
    service.run_forever()
    assert len(calls) == 2
    assert len(errors) == 1 and errors[0][0] == task_id
    assert "worker failure" not in errors[0][1]
    assert service.store.get_task(task_id)["status"] == "interrupted"
    assert not service._active


@pytest.mark.parametrize("raise_late", [False, True])
def test_expired_side_effect_remains_unknown_and_requires_recovery(service, raise_late) -> None:
    task_id = service.store.create_task({"goal": "send", "steps": [{"id": "send", "requires_human": True}]})
    service.bind(task_id, "fixture", {})

    def handler(payload):
        control = get_execution_control()
        service.store.start_attempt(task_id, "send", **control.ledger_guard)
        expire(service.store, task_id)
        if raise_late:
            raise RuntimeError("late effect failure")
        return {"answer": "late effect result"}

    service.register("fixture", handler)
    assert service.run_once()
    task = service.store.get_task(task_id)
    assert task["status"] == "recovery_required" and task["nodes"]["send"]["status"] == "unknown"
    assert service.store.list_attempts(task_id)[0]["status"] == "unknown"
    assert service.store.load_output(task_id) is None
    with pytest.raises(ValueError):
        service.resume(task_id)


@pytest.mark.parametrize("fails_after_commit", [False, True])
def test_handler_committed_terminal_is_not_downgraded_or_overwritten(service, fails_after_commit) -> None:
    task_id = service.submit("fixture", {})

    def handler(payload):
        control = get_execution_control()
        service.store.finish_task(task_id, "succeeded", output={"answer": "committed"}, **control.ledger_guard)
        if fails_after_commit:
            raise RuntimeError("after commit")
        return {"answer": "late result"}

    service.register("fixture", handler)
    assert service.run_once()
    assert service.store.get_task(task_id)["status"] == "succeeded"
    assert service.store.load_output(task_id) == {"answer": "committed"}


@pytest.mark.parametrize("cancel_after_release", [False, True])
def test_intentional_waiting_release_saves_output_but_cancel_fences_it(service, cancel_after_release) -> None:
    task_id = service.submit("fixture", {})

    def handler(payload):
        control = get_execution_control()
        service.store.finish_task(task_id, "waiting_human", **control.ledger_guard)
        if cancel_after_release:
            TaskService(TaskStore(service.store.path)).request(task_id, "cancel")
        return {"pending_action": "approval"}

    service.register("fixture", handler)
    service.on_result = raise_observer
    assert service.run_once()
    assert service.store.get_task(task_id)["status"] == ("cancelled" if cancel_after_release else "waiting_human")
    assert service.store.load_output(task_id) == (None if cancel_after_release else {"pending_action": "approval"})


@pytest.mark.parametrize("action", ["cancel", "pause"])
def test_cross_instance_request_is_honored_atomically_at_worker_completion(service, action) -> None:
    task_id = service.submit("fixture", {})
    other = TaskService(TaskStore(service.store.path))

    def handler(payload):
        other.request(task_id, action)
        task = service.store.get_task(task_id)
        assert task["status"] == "running" and task["owner"] == service.owner
        return {"answer": "completed despite request"}

    service.register("fixture", handler)
    assert service.run_once()
    assert service.store.get_task(task_id)["status"] == ("cancelled" if action == "cancel" else "paused")
    other.close()


def test_request_uses_live_state_when_other_worker_claims(service, monkeypatch) -> None:
    task_id = service.submit("fixture", {})
    original = service.store.request_control

    def claimed_before_request(task, action, **kwargs):
        service.store.claim(task, "other worker")
        return original(task, action, **kwargs)

    monkeypatch.setattr(service.store, "request_control", claimed_before_request)
    service.request(task_id, "cancel")
    task = service.store.get_task(task_id)
    assert task["status"] == "running" and task["owner"] == "other worker"
    assert task["cancel_requested"]


def test_cancel_between_resume_snapshot_and_transaction_is_not_erased(service, monkeypatch) -> None:
    task_id = service.submit("fixture", {})
    service.request(task_id, "pause")
    other = TaskService(TaskStore(service.store.path))
    original = service.store.resume_runtime_task

    def cancelled_before_resume(task, **kwargs):
        other.request(task, "cancel")
        return original(task, **kwargs)

    monkeypatch.setattr(service.store, "resume_runtime_task", cancelled_before_resume)
    with pytest.raises(VersionConflict):
        service.resume(task_id)
    task = service.store.get_task(task_id)
    assert task["status"] == "cancelled" and task["cancel_requested"]
    assert service.descriptor(task_id)["restarts"] == 0
    other.close()


def test_two_instances_resume_only_once_at_restart_limit(service, monkeypatch) -> None:
    task_id = service.submit("fixture", {})
    service.request(task_id, "pause")
    other = TaskService(TaskStore(service.store.path))
    with sqlite3.connect(service.store.path) as connection:
        connection.execute("UPDATE runtime_jobs SET restarts=2 WHERE task_id=?", (task_id,))
    barrier = threading.Barrier(2)
    for participant in (service, other):
        original = participant.store.get_task

        def same_snapshot(task, original=original):
            value = original(task)
            barrier.wait(timeout=5)
            return value

        monkeypatch.setattr(participant.store, "get_task", same_snapshot)

    def resume(participant):
        try:
            participant.resume(task_id)
            return "resumed"
        except VersionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(resume, (service, other)))
    monkeypatch.undo()
    assert sorted(results) == ["conflict", "resumed"]
    assert service.descriptor(task_id)["restarts"] == 3
    service.request(task_id, "pause")
    with pytest.raises(ValueError, match="3"):
        service.resume(task_id)
    assert service.descriptor(task_id)["restarts"] == 3
    other.close()


@pytest.mark.parametrize("action", ["resume", "cancel"])
def test_control_transaction_rolls_back_all_changes_on_event_failure(service, monkeypatch, action) -> None:
    task_id = service.submit("fixture", {})
    service.request(task_id, "pause")
    before = service.store.get_task(task_id)
    monkeypatch.setattr(service.store, "_event", raise_observer)
    with pytest.raises(RuntimeError, match="UI disconnected"):
        service.resume(task_id) if action == "resume" else service.request(task_id, "cancel")
    assert service.store.get_task(task_id) == before
    assert service.descriptor(task_id)["restarts"] == 0


def test_heartbeat_cancels_on_owner_loss_without_waiting_for_renewal(service) -> None:
    task_id = service.submit("fixture", {})
    task = service.store.claim(task_id, service.owner)
    with sqlite3.connect(service.store.path) as connection:
        connection.execute("UPDATE tasks SET owner='other worker' WHERE task_id=?", (task_id,))
    control = ExecutionControl(task_id=task_id)

    class OneTick:
        def wait(self, seconds):
            return control.cancel_event.is_set()

    service._heartbeat(task_id, task["lease_token"], control, OneTick())
    assert control.cancel_event.is_set()


@pytest.mark.parametrize("state", ["dispatched", "unknown", "committed"])
def test_resume_refuses_uncertain_or_committed_short_task_effect(service, state) -> None:
    task_id = service.submit("fixture", {})
    service.request(task_id, "pause")
    with sqlite3.connect(service.store.path) as connection:
        connection.execute("CREATE TABLE runtime_tool_effects(task_id TEXT, state TEXT)")
        connection.execute("INSERT INTO runtime_tool_effects VALUES(?,?)", (task_id, state))
    before = service.store.get_task(task_id)
    with pytest.raises(ValueError):
        service.resume(task_id)
    assert service.store.get_task(task_id) == before
    assert service.descriptor(task_id)["restarts"] == 0


def test_bind_preserves_timeout_on_rebind(service) -> None:
    task_id = service.submit("fixture", {"value": 1}, timeout_seconds=123)
    service.bind(task_id, "fixture", {"value": 2}, timeout_seconds=999)
    assert service.descriptor(task_id)["timeout_seconds"] == 123
    assert service.descriptor(task_id)["payload"] == {"value": 2}


def node_attempt(service):
    store = service.store
    task_id = store.create_task({"goal": "fixture", "steps": [{"id": "read", "agent": "knowledge"}]})
    store.update_node(task_id, "read", "pending", checks={"result_success": False})
    task = store.claim(task_id, service.owner)
    guard = {"owner": service.owner, "lease_token": task["lease_token"]}
    attempt = store.start_attempt(task_id, "read", **guard)
    return task_id, attempt["attempt_id"], guard


@pytest.mark.parametrize("failed_check", [False, True])
def test_finish_attempt_atomically_updates_checks_and_checkpoint(service, failed_check) -> None:
    task_id, attempt_id, guard = node_attempt(service)
    checks = {"result_success": True, "nested": [not failed_check], "api_key": "must-not-persist"}
    service.store.finish_attempt(attempt_id, "success", output={"answer": "approved"}, checks=checks, **guard)
    reopened = TaskStore(service.store.path)
    saved_checks = reopened.get_task(task_id)["nodes"]["read"]["checks"]
    assert saved_checks["result_success"] is True
    checkpoint = reopened.load_checkpoint(task_id, "read")
    assert checkpoint["validation"]["checks"] == saved_checks
    assert checkpoint["validation"]["ok"] is not failed_check
    assert "must-not-persist" not in str(saved_checks) + str(checkpoint)
    assert len(reopened.list_attempts(task_id)) == 1


def test_finish_attempt_checks_and_checkpoint_roll_back_together(service, monkeypatch) -> None:
    task_id, attempt_id, guard = node_attempt(service)
    store = service.store
    before = store.get_task(task_id)
    original = store._event

    def fail_after_checkpoint(connection, task, event_type, **kwargs):
        if event_type == "attempt_finished":
            raise RuntimeError("event failure")
        return original(connection, task, event_type, **kwargs)

    monkeypatch.setattr(store, "_event", fail_after_checkpoint)
    with pytest.raises(RuntimeError, match="event failure"):
        store.finish_attempt(attempt_id, "success", output={"answer": "done"}, checks={"result_success": True}, **guard)
    assert store.get_task(task_id) == before
    assert store.load_checkpoint(task_id, "read") is None
    assert store.load_output(task_id, "read") is None
    assert store.list_attempts(task_id)[0]["status"] == "running"
