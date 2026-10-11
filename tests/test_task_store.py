from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deskpilot.core.approval import PendingActionStore
from deskpilot.multi_agent.schemas import AgentResult, TaskPlan, TaskPlanStep
from deskpilot.multi_agent.task_ledger import TaskLedger
from deskpilot.tasks import ApprovalConflict, LeaseConflict, SerializationError, TaskStore, TaskStoreError, VersionConflict
from deskpilot.tasks.models import serializable
from deskpilot.tools.tool_registry import ToolResult


def plan(*, effect: bool = False, dependent: bool = False) -> dict:
    steps = [{"id": "read", "agent": "knowledge", "handler_key": "domain.read", "arguments": {"query": "original"}}]
    if dependent:
        steps.append({"id": "write", "agent": "knowledge", "handler_key": "domain.write", "depends_on": ["read"]})
    if effect:
        steps.append({"id": "send", "agent": "communication", "requires_human": True,
                      "allowed_tools": ["email.send"], "depends_on": ["read"]})
    return {"goal": "actual private request", "route": "multi_agent", "steps": steps}


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.sqlite3")


def expired_lease(store: TaskStore, task_id: str) -> None:
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE task_id=?", (task_id,))


def desktop_approval(store: TaskStore) -> tuple[str, PendingActionStore, dict]:
    task_id = store.create_task(plan(effect=True), session_id="session")
    store.update_node(task_id, "read", "success", output={"evidence": "saved"})
    store.update_node(task_id, "send", "pending", output={"requires_human": True})
    store.finish_task(task_id, "waiting_human")
    pending = PendingActionStore(db_path=store.path)
    action = pending.create("session", {"tool_name": "email.send", "task_id": task_id, "node_id": "send",
                                        "plan_version": 1, "kwargs": {"to": ["reader@example.test"], "body": "actual body"}})
    return task_id, pending, action


def test_wal_full_sync_and_schema(store: TaskStore) -> None:
    with store._connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "nodes", "attempts", "events", "checkpoints", "output_refs", "approvals"} <= tables


def test_real_plan_inputs_and_outputs_roundtrip_without_summary_body(store: TaskStore) -> None:
    task_id = store.create_task(plan(), initial_values={"question": "private source", "source": Path("input.md")}, session_id="session")
    assert store.load_plan(task_id)["goal"] == "actual private request"
    assert store.load_input(task_id) == {"question": "private source", "source": "input.md"}
    assert store.load_node_input(task_id, "read") == {"query": "original"}
    output = {"answer": "actual private answer", "tool_result": ToolResult(True, "read", {"content": "full source"})}
    store.update_node(task_id, "read", "success", output=output)
    store.finish_task(task_id, "succeeded", output={"report": "final report"})
    reopened = TaskStore(store.path)
    assert reopened.load_output(task_id, "read")["tool_result"]["output"]["content"] == "full source"
    assert reopened.load_output(task_id) == {"report": "final report"}
    summary = str(reopened.get(task_id)) + str(reopened.list()) + str(reopened.events(task_id))
    assert "actual private request" not in summary
    assert "actual private answer" not in summary
    assert "private source" not in summary
    assert "full source" not in summary


def test_credentials_never_reach_objects_events_or_database(store: TaskStore) -> None:
    task_id = store.create_task(plan(), initial_values={"question": "keep this", "api_key": "raw-key-A", "smtp_password": "raw-pass-B",
                                                       "nested": {"authorization": "Bearer raw-C", "body": "mail正文"}})
    store.update_node(task_id, "read", "failed", output={"secret": "raw-secret-D", "tokens": 12,
                      "content": "password=raw-E Bearer raw-F https://user:raw-G@example.test"}, error="api_key=raw-H")
    assert store.load_input(task_id) == {"question": "keep this", "nested": {"body": "mail正文"}}
    assert store.load_output(task_id, "read")["tokens"] == 12
    with sqlite3.connect(store.path) as connection:
        dump = "\n".join(connection.iterdump())
    for value in ("raw-key-A", "raw-pass-B", "raw-C", "raw-secret-D", "raw-E", "raw-F", "raw-G", "raw-H"):
        assert value not in dump


@pytest.mark.parametrize("value", [object(), lambda: None, {"closure": lambda: None}, float("nan"), {1: "invalid"}])
def test_nonserializable_task_values_are_rejected_atomically(store: TaskStore, value) -> None:
    with pytest.raises(SerializationError):
        store.create_task(plan(), initial_values=value)
    assert store.list_tasks() == []
    assert store.list_events() == []


def test_dataclass_codec_skips_plan_handler_and_rejects_cycles(store: TaskStore) -> None:
    bound = TaskPlan("actual goal", "multi_agent", [TaskPlanStep("read", "knowledge", "read", handler=lambda values: {},
                                                                            arguments={"path": Path("input.md")})])
    task_id = store.create_task(bound)
    assert "handler" not in store.load_plan(task_id)["steps"][0]
    loop = []
    loop.append(loop)
    with pytest.raises(SerializationError, match="Circular"):
        serializable(loop)
    @dataclass
    class Data:
        output: object
    assert serializable(Data(ToolResult(True, "read", {"value": 1})))["output"]["ok"] is True


def test_explicit_plan_version_is_authoritative_and_matches_descriptor(store: TaskStore) -> None:
    bound = TaskPlan("goal", "multi_agent", [TaskPlanStep("read", "knowledge", "read")])
    ledger = TaskLedger(store.path)
    task_id = ledger.start(bound, plan_version="2")
    assert ledger.get_task(task_id)["plan_version"] == ledger.load_plan(task_id)["plan_version"] == "2"
    ledger.save_plan(task_id, bound, plan_version="3")
    assert ledger.get_task(task_id)["plan_version"] == ledger.load_plan(task_id)["plan_version"] == "3"
    bound.plan_version = "4"
    task_id = store.create_task(bound)
    assert store.get_task(task_id)["plan_version"] == store.load_plan(task_id)["plan_version"] == "4"


def test_state_event_output_attempt_and_checkpoint_rollback_together(store: TaskStore, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id = store.create_task(plan())
    before_events = store.list_events()
    original = store._event
    def failed_event(*args, **kwargs):
        if len(args) > 2 and args[2] == "node_updated":
            raise OSError("disk full")
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "_event", failed_event)
    with pytest.raises(OSError, match="disk full"):
        store.update_node(task_id, "read", "success", output={"answer": "saved"})
    assert store.get_task(task_id)["nodes"]["read"]["status"] == "pending"
    assert store.load_output(task_id, "read") is None
    assert store.load_checkpoint(task_id, "read") is None
    assert store.list_attempts(task_id) == []
    assert store.list_events() == before_events


def test_multiinstance_updates_do_not_lose_nodes(store: TaskStore) -> None:
    descriptor = {"steps": [{"id": f"n{index}"} for index in range(24)]}
    task_id = store.create_task(descriptor)
    def update(index: int) -> None:
        TaskStore(store.path).update_node(task_id, f"n{index}", "success", output={"value": index})
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(update, range(24)))
    nodes = store.get_task(task_id)["nodes"]
    assert len(nodes) == 24 and all(node["status"] == "success" for node in nodes.values())
    assert len(store.list_attempts(task_id)) == 24
    sequences = [event["event_seq"] for event in store.list_events(task_id)]
    assert sequences == sorted(set(sequences))


def test_cas_rejects_stale_task_and_node_versions(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    initial = store.get_task(task_id)
    store.update_node(task_id, "read", "failed", error="network", expected_version=initial["version"], expected_node_version=1)
    with pytest.raises(VersionConflict):
        store.finish_task(task_id, "failed", expected_version=initial["version"])
    with pytest.raises(VersionConflict):
        store.update_node(task_id, "read", "success", output={}, expected_node_version=1)


def test_attempt_usage_and_compatibility_callback_do_not_double_count(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    attempt = store.start_attempt(task_id, "read")
    store.finish_attempt(attempt["attempt_id"], "success", output={"answer": "done"}, tool_calls=2, tokens=13,
                         usage={"total_tokens": 13}, certainty="confirmed")
    store.update_node(task_id, "read", "success", output={"answer": "done"}, tool_calls=2, tokens=13)
    store.update_node(task_id, "read", "success", output={"answer": "done"}, tool_calls=2, tokens=13,
                      attempt_id=attempt["attempt_id"], record_attempt=False)
    assert len(store.list_attempts(task_id)) == 1
    assert store.get_task(task_id)["nodes"]["read"]["tool_calls"] == 2
    assert store.get_task(task_id)["nodes"]["read"]["tokens"] == 13


def test_failed_attempts_usage_and_numbers_are_retained(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    first = store.start_attempt(task_id, "read")
    store.finish_attempt(first["attempt_id"], "failed", error_category="transient", tool_calls=1, tokens=7, usage={"reported": False})
    second = store.start_attempt(task_id, "read")
    store.finish_attempt(second["attempt_id"], "success", output={"answer": "done"}, tool_calls=2, tokens=10, certainty="confirmed")
    assert second["attempt_number"] == 2
    record = store.get_task(task_id)["nodes"]["read"]
    assert record["attempt_count"] == 2
    assert record["tool_calls"] == 3 and record["tokens"] == 17
    with pytest.raises(TaskStoreError, match="already finished"):
        store.finish_attempt(second["attempt_id"], "failed")


def test_checkpoint_validates_files_versions_inputs_and_downstream(store: TaskStore, tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("original report", encoding="utf-8")
    task_id = store.create_task(plan(dependent=True), initial_values={"question": "original"})
    store.update_node(task_id, "read", "success", output={"artifact_path": artifact, "evidence": "source"})
    store.update_node(task_id, "write", "success", output={"answer": "report"})
    store.finish_task(task_id, "failed")
    assert store.validate_recovery(task_id)["reusable_nodes"] == ["read", "write"]
    assert store.validate_recovery(task_id, executor_version="2")["reusable_nodes"] == []
    assert not store.validate_recovery(task_id, handler_versions={"domain.read": "2", "domain.write": "1"})["can_resume"]
    artifact.write_text("user modified", encoding="utf-8")
    validation = store.validate_recovery(task_id)
    assert validation["reusable_nodes"] == []
    assert "dependency_checkpoint_invalid" in validation["invalid_nodes"]["write"]
    assert any(reason.startswith("artifact_missing_or_changed:") for reason in validation["invalid_nodes"]["read"])
    assert validation["can_resume"]  # Read-only nodes may be recomputed, never reused blindly.


def test_node_and_task_input_changes_invalidate_checkpoints(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    store.update_node(task_id, "read", "success", output={"value": 1})
    store.save_node_input(task_id, "read", {"query": "changed"})
    assert "node_input_changed" in store.validate_recovery(task_id)["invalid_nodes"]["read"]
    store.save_input(task_id, {"question": "new"})
    assert not store.validate_recovery(task_id, initial_values={"question": "old"})["can_resume"]


def test_same_path_changed_input_file_invalidates_checkpoint(store: TaskStore, tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"first document")
    descriptor = plan()
    descriptor["steps"][0]["handler_binding"] = {"source_path": source}
    task_id = store.create_task(descriptor, initial_values={"request": {"attachment_paths": [source]}})
    store.update_node(task_id, "read", "success", output={"evidence": "read first document"})
    assert store.validate_recovery(task_id)["reusable_nodes"] == ["read"]
    source.write_bytes(b"different document")
    assert store.validate_recovery(task_id)["reusable_nodes"] == []
    assert any(reason.startswith("artifact_missing_or_changed:") for reason in store.validate_recovery(task_id)["invalid_nodes"]["read"])


def test_changed_binding_under_same_plan_version_cannot_reuse_old_output(store: TaskStore) -> None:
    descriptor = plan()
    descriptor["plan_version"] = "1"
    descriptor["steps"][0]["handler_binding"] = {"model": "old"}
    task_id = store.create_task(descriptor)
    store.update_node(task_id, "read", "success", output={"evidence": "old"})
    descriptor["steps"][0]["handler_binding"] = {"model": "new"}
    store.save_plan(task_id, descriptor, plan_version="1")
    assert store.validate_recovery(task_id)["reusable_nodes"] == []
    assert "node_input_changed" in store.validate_recovery(task_id)["invalid_nodes"]["read"]


def test_file_hashing_does_not_hold_writer_lock_and_detects_input_race(store: TaskStore, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id = store.create_task(plan())
    claimed = store.claim_next("worker")
    guard = {"owner": "worker", "lease_token": claimed["lease_token"]}
    original = store._fingerprints
    def hash_with_heartbeat(paths):
        store.heartbeat(task_id, **guard)
        return original(paths)
    monkeypatch.setattr(store, "_fingerprints", hash_with_heartbeat)
    store.update_node(task_id, "read", "success", output={"evidence": "saved"}, **guard)
    assert store.load_checkpoint(task_id, "read") is not None
    def hash_with_input_change(paths):
        store.save_node_input(task_id, "read", {"query": "concurrent change"}, **guard)
        return original(paths)
    monkeypatch.setattr(store, "_fingerprints", hash_with_input_change)
    with pytest.raises(VersionConflict, match="during file validation"):
        store.save_checkpoint(task_id, "read", {"batch": 2}, **guard)
    assert store.load_checkpoint(task_id, "read")["checkpoint"] is None


def test_corrupted_object_reference_is_not_reused(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    store.update_node(task_id, "read", "success", output={"value": 1})
    output_ref = store.get_task(task_id)["nodes"]["read"]["output_ref"]
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE output_refs SET content_json='{}' WHERE ref_id=?", (output_ref,))
    with pytest.raises(TaskStoreError, match="digest mismatch"):
        store.load_output(task_id, "read")
    assert "object_reference_invalid" in store.validate_recovery(task_id)["invalid_nodes"]["read"]


def test_partial_batch_checkpoint_roundtrip(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    store.start_attempt(task_id, "read")
    checkpoint_id = store.save_checkpoint(task_id, "read", {"document_sha": "a" * 64, "completed_pages": [1, 2],
                                                               "ocr_model": "ocr-v1", "vector_space": "text-v2"}, output={"batch": 2})
    checkpoint = store.load_checkpoint(task_id, "read")
    assert checkpoint["checkpoint_id"] == checkpoint_id
    assert checkpoint["checkpoint"]["completed_pages"] == [1, 2]
    assert store.validate_recovery(task_id)["resumable_nodes"] == ["read"]


def test_claim_race_fencing_and_active_constructor_does_not_interrupt(store: TaskStore) -> None:
    task_id = store.create_task(plan())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda owner: TaskStore(store.path).claim_next(owner), ["worker-a", "worker-b"]))
    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    task = claimed[0]
    assert TaskLedger(store.path).get_task(task_id)["status"] == "running"
    with pytest.raises(LeaseConflict):
        store.update_node(task_id, "read", "success", output={})
    with pytest.raises(LeaseConflict):
        store.heartbeat(task_id, "stale-owner", task["lease_token"])
    guard = {"owner": task["owner"], "lease_token": task["lease_token"]}
    attempt = store.start_attempt(task_id, "read", **guard)
    expired_lease(store, task_id)
    assert store.expire_leases() == [task_id]
    with pytest.raises(LeaseConflict):
        store.finish_attempt(attempt["attempt_id"], "success", output={}, **guard)
    assert store.get_task(task_id)["status"] == "interrupted"


def test_expired_side_effect_is_unknown_and_never_auto_claimed(store: TaskStore) -> None:
    task_id = store.create_task(plan(effect=True))
    claimed = store.claim_next("worker")
    guard = {"owner": "worker", "lease_token": claimed["lease_token"]}
    attempt = store.start_attempt(task_id, "send", **guard)
    expired_lease(store, task_id)
    TaskLedger(store.path)
    task = store.get_task(task_id)
    assert task["status"] == "recovery_required"
    assert task["nodes"]["send"]["status"] == "unknown"
    assert store.list_attempts(task_id, "send")[0]["attempt_id"] == attempt["attempt_id"]
    assert store.list_attempts(task_id, "send")[0]["certainty"] == "unknown"
    assert store.claim_next("another-worker") is None
    assert not store.validate_recovery(task_id)["safe_to_resume"]
    store.update_task(task_id, "queued")
    assert store.claim_next("another-worker") is None
    with pytest.raises(TaskStoreError, match="reconciled"):
        store.start_attempt(task_id, "send")


def test_approval_gate_without_dispatch_does_not_create_fake_attempt(store: TaskStore) -> None:
    task_id = store.create_task(plan(effect=True))
    store.update_node(task_id, "send", "pending", output={"requires_human": True})
    assert store.list_attempts(task_id, "send") == []


def test_expired_queued_deadline_does_not_block_session_forever(store: TaskStore) -> None:
    first = store.create_task(plan(), session_id="session", deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    second = store.create_task(plan(), session_id="session")
    assert store.claim_next("worker")["task_id"] == second
    assert store.get_task(first)["status"] == "failed"


def test_queue_is_ordered_per_session_and_respects_backoff_stop_deadline(store: TaskStore) -> None:
    first = store.create_task(plan(), session_id="a")
    second = store.create_task(plan(), session_id="a")
    other = store.create_task(plan(), session_id="b")
    assert {item["task_id"] for item in store.query_queue()} == {first, other}
    store.request_pause(first)
    assert [item["task_id"] for item in store.query_queue()] == [other]
    store.request_cancel(other)
    assert store.query_queue() == []
    store.clear_requests(first)
    store.finish_task(first, "succeeded")
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    store.update_task(second, "retry_wait", retry_at=future)
    assert store.query_queue() == []
    store.update_task(second, "queued", retry_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert [item["task_id"] for item in store.query_queue()] == [second]
    store.update_task(second, deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert store.query_queue() == []


def test_legacy_json_is_read_only_idempotent_history(tmp_path: Path) -> None:
    source = tmp_path / "tasks.json"
    original = json.dumps([{"task_id": "old", "goal": "read -> send", "status": "running", "safe_to_resume": True,
                             "nodes": {"read": {"status": "success", "artifact_paths": ["report.md"]}}}]).encode()
    source.write_bytes(original)
    ledger = TaskLedger(source)
    task = ledger.list_tasks()[0]
    assert task["legacy"] and task["status"] == "interrupted" and not task["safe_to_resume"]
    assert ledger.load_plan("old") is None and ledger.load_output("old", "read") is None
    assert not ledger.recovery_candidates()[0]["safe_to_resume"]
    with pytest.raises(TaskStoreError, match="read-only"):
        ledger.finish("old", "success")
    assert len(TaskLedger(source).list_tasks()) == 1
    assert source.read_bytes() == original


def test_corrupt_legacy_json_is_not_silently_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "tasks.json"
    source.write_text("{broken", encoding="utf-8")
    with pytest.raises(TaskStoreError, match="source was not modified"):
        TaskLedger(source)
    assert source.read_text(encoding="utf-8") == "{broken"


def test_active_task_history_is_not_truncated_to_200(store: TaskStore) -> None:
    for index in range(205):
        store.create_task({"goal": str(index), "steps": []})
    assert len(store.list_tasks()) == 205


def test_native_approval_binding_once_only_and_atomic_task_transition(store: TaskStore) -> None:
    task_id = store.create_task(plan(effect=True), session_id="session")
    store.finish_task(task_id, "waiting_human")
    approval = store.save_approval(task_id, "send", arguments={"body": "private mail", "secret": "never stored"}, preview={"body": "private mail"})
    assert store.get_approval(approval["approval_id"])["arguments"] == {"body": "private mail"}
    assert "private mail" not in str(store.list_approvals())
    with pytest.raises(ApprovalConflict, match="session"):
        store.decide_approval(approval["approval_id"], "approved", session_id="other")
    store.decide_approval(approval["approval_id"], "approved", session_id="session", content_digest=approval["content_digest"])
    store.consume_approval(approval["approval_id"], task_status="queued", node_status="pending")
    assert store.get_task(task_id)["status"] == "queued"
    with pytest.raises(ApprovalConflict):
        store.consume_approval(approval["approval_id"])


def test_native_approval_expiry_attachment_and_plan_change(store: TaskStore, tmp_path: Path) -> None:
    attachment = tmp_path / "attachment.txt"
    attachment.write_text("original", encoding="utf-8")
    task_id = store.create_task(plan(effect=True))
    approval = store.save_approval(task_id, "send", arguments={"attachment_paths": [attachment]})
    attachment.write_text("changed", encoding="utf-8")
    with pytest.raises(ApprovalConflict, match="changed"):
        store.decide_approval(approval["approval_id"], "approved")
    expired = store.save_approval(task_id, "send", arguments={}, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(ApprovalConflict, match="expired"):
        store.decide_approval(expired["approval_id"], "approved")
    fresh = store.save_approval(task_id, "send", arguments={})
    store.save_plan(task_id, plan(effect=True), plan_version="2")
    assert store.get_approval(fresh["approval_id"])["status"] == "invalidated"


def test_desktop_approval_claim_is_atomic_leased_and_once_only(store: TaskStore) -> None:
    task_id, pending, action = desktop_approval(store)
    attempt = store.claim_desktop_approval(action["action_id"], "session", task_id, "send", 1)
    guard = {"owner": attempt["owner"], "lease_token": attempt["lease_token"]}
    assert attempt["owner"].startswith("approval_") and attempt["lease_token"] > 0
    task = store.get_task(task_id)
    assert task["status"] == task["nodes"]["send"]["status"] == "running"
    assert task["lease_expires_at"] == attempt["lease_expires_at"]
    assert pending.inspect(action["action_id"], "session") is None
    assert store.heartbeat(task_id, **guard)["status"] == "running"
    with pytest.raises(ApprovalConflict):
        store.claim_desktop_approval(action["action_id"], "session", task_id, "send", 1)
    store.finish_attempt(attempt["attempt_id"], "success", output={"sent": True}, certainty="confirmed", **guard)
    store.finish_task(task_id, "succeeded", **guard)
    assert store.get_task(task_id)["owner"] is None


def test_desktop_dispatch_rollback_keeps_pending_and_no_attempt(store: TaskStore, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id, pending, action = desktop_approval(store)
    def cannot_store(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(store, "_event", cannot_store)
    with pytest.raises(OSError):
        store.claim_desktop_approval(action["action_id"], "session", task_id, "send", 1)
    assert pending.inspect(action["action_id"], "session") is not None
    assert store.get_task(task_id)["status"] == "waiting_human"
    assert store.get_task(task_id)["owner"] is None
    assert all(attempt["node_id"] != "send" or attempt["status"] != "running" for attempt in store.list_attempts(task_id))


def test_desktop_dispatch_crash_expires_to_unknown_and_fences_callback(store: TaskStore) -> None:
    task_id, _, action = desktop_approval(store)
    attempt = store.claim_desktop_approval(action["action_id"], "session", task_id, "send", 1)
    expired_lease(store, task_id)
    TaskLedger(store.path)
    assert store.get_task(task_id)["status"] == "recovery_required"
    assert store.get_task(task_id)["nodes"]["send"]["status"] == "unknown"
    with pytest.raises(LeaseConflict):
        store.finish_attempt(attempt["attempt_id"], "success", output={"sent": True}, owner=attempt["owner"], lease_token=attempt["lease_token"])


def test_real_process_exit_preserves_checkpoint_and_attempt(tmp_path: Path) -> None:
    path = tmp_path / "crash.sqlite3"
    code = """
import os, sys
from deskpilot.tasks import TaskStore
store = TaskStore(sys.argv[1])
task_id = store.create_task({'steps': [{'id': 'read'}, {'id': 'next', 'depends_on': ['read']}]}, task_id='crash')
task = store.claim_next('crashed-worker', lease_seconds=15)
guard = {'owner': task['owner'], 'lease_token': task['lease_token']}
attempt = store.start_attempt(task_id, 'read', **guard)
store.finish_attempt(attempt['attempt_id'], 'success', output={'evidence': 'saved'}, certainty='confirmed', **guard)
store.start_attempt(task_id, 'next', **guard)
store.heartbeat(task_id, guard['owner'], guard['lease_token'], lease_seconds=0.1)
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert result.returncode == 17, result.stderr.decode(errors="replace")
    time.sleep(0.12)
    reopened = TaskLedger(path)
    assert reopened.get_task("crash")["status"] == "interrupted"
    assert reopened.load_output("crash", "read") == {"evidence": "saved"}
    assert reopened.store.validate_recovery("crash")["reusable_nodes"] == ["read"]
    assert [attempt["status"] for attempt in reopened.store.list_attempts("crash")] == ["success", "interrupted"]


def test_unleased_sync_crash_is_recovered_without_interrupting_live_process(tmp_path: Path) -> None:
    path = tmp_path / "sync.sqlite3"
    live = TaskLedger(path)
    live_id = live.start(plan())
    assert TaskLedger(path).get_task(live_id)["status"] == "running"
    code = """
import os, sys
from deskpilot.multi_agent.task_ledger import TaskLedger
ledger = TaskLedger(sys.argv[1])
task = ledger.start({'steps': [{'id': 'send', 'requires_human': True}]}, task_id='sync-crash')
ledger.store.start_attempt(task, 'send')
os._exit(18)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert result.returncode == 18, result.stderr.decode(errors="replace")
    reopened = TaskLedger(path)
    assert reopened.get_task(live_id)["status"] == "running"
    assert reopened.get_task("sync-crash")["status"] == "recovery_required"
    assert reopened.get_task("sync-crash")["nodes"]["send"]["status"] == "unknown"


def test_recycled_pid_is_not_mistaken_for_original_worker(store: TaskStore, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id = store.create_task(plan(), status="running")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE task_origins SET process_birth='old-process' WHERE task_id=?", (task_id,))
    monkeypatch.setattr(store, "_process_state", lambda pid: (True, "new-process"))
    assert store.recover_orphaned_tasks() == [task_id]
    assert store.get_task(task_id)["status"] == "interrupted"
