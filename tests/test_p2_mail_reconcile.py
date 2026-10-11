from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskpilot.mcp.email_mcp import ImapSmtpEmailProvider
from deskpilot.core.approval import PendingActionStore
from deskpilot.tasks import TaskStore
from deskpilot.tasks.control import ExecutionControl, execution_scope
from deskpilot.tasks.effects import begin_effect, effects_for_task, finish_effect
from deskpilot.tasks.mail_reconcile import reconcile_email_effect
from deskpilot.tools.tool_registry import ToolResult, ToolRegistry, ToolSpec, ToolParameter


def logged_send(store: TaskStore, state: str, sent_copy_saved: bool = False) -> str:
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    with execution_scope(control):
        operation_id = begin_effect(control, "email.send", {"to": ["a@example.com"], "body": "hello"})
        finish_effect(control, operation_id, ToolResult(
            state == "committed", "email.send",
            {"message_id": f"<deskpilot-{operation_id}@example.com>",
             "status": "sent" if state == "committed" else "unknown",
             "sent_copy_saved": sent_copy_saved}, status="success" if state == "committed" else "unknown",
        ))
    return operation_id


def test_stable_message_id_and_unknown_result_never_resends(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    with execution_scope(control):
        operation_id = begin_effect(control, "email.send", {"to": ["a@example.com"]})
        provider = ImapSmtpEmailProvider.__new__(ImapSmtpEmailProvider)
        provider.username = "me@example.com"
        message = provider._build_message(["a@example.com"], "subject", "body")
        assert message["Message-ID"] == f"<deskpilot-{operation_id}@example.com>"
        finish_effect(control, operation_id, ToolResult(False, "email.send",
            {"message_id": message["Message-ID"]}, status="unknown"))
    class Provider:
        def find_sent_copy(self, message_id):
            assert message_id == message["Message-ID"]
            return False
        def send(self, *args, **kwargs):
            raise AssertionError("unknown must not resend")
    result = reconcile_email_effect(store, Provider(), operation_id)
    assert result["status"] == "unknown"
    assert effects_for_task(store, "task-mail")[0]["state"] == "unknown"
    with execution_scope(control), pytest.raises(RuntimeError, match="不能自动重复提交"):
        begin_effect(control, "email.send", {"to": ["a@example.com"]})


def test_sent_copy_failure_retries_only_imap_append(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    operation_id = logged_send(store, "committed", sent_copy_saved=False)
    calls = []
    class Provider:
        def find_sent_copy(self, message_id):
            calls.append("search")
            return False
        def repair_sent_copy(self, op, message_id, store_path):
            calls.append("append")
            assert op == operation_id and store_path == store.path
            return True
        def send(self, *args, **kwargs):
            raise AssertionError("SMTP must not be repeated")
    result = reconcile_email_effect(store, Provider(), operation_id)
    assert result["action"] == "sent_copy_repaired"
    assert calls == ["search", "append"]
    detail = json.loads(effects_for_task(store, "task-mail")[0]["result_json"])
    assert detail["sent_copy_saved"] is True


def test_unknown_send_found_in_sent_mailbox_becomes_verified(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    operation_id = logged_send(store, "unknown")
    class Provider:
        def find_sent_copy(self, message_id):
            return True
    result = reconcile_email_effect(store, Provider(), operation_id)
    assert result["action"] == "found_sent_copy"
    assert effects_for_task(store, "task-mail")[0]["state"] == "committed"


def test_mime_snapshot_repairs_sent_copy_without_smtp(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    provider = ImapSmtpEmailProvider.__new__(ImapSmtpEmailProvider)
    provider.username = "me@example.com"
    appended = []
    provider.find_sent_copy = lambda message_id: False
    provider._append_to_mailbox = lambda message, mailbox, flags: appended.append(message) or True
    provider.sent_mailbox = "Sent"
    with execution_scope(control):
        operation_id = begin_effect(control, "email.send", {"to": ["a@example.com"]})
        message = provider._build_message(["a@example.com"], "subject", "body")
        provider._persist_mime(message)
        finish_effect(control, operation_id, ToolResult(True, "email.send", {
            "message_id": message["Message-ID"], "sent_copy_saved": False,
        }))
    assert provider.repair_sent_copy(operation_id, message["Message-ID"], store.path)
    assert len(appended) == 1
    assert appended[0]["Message-ID"] == message["Message-ID"]


def test_interrupted_smtp_dispatch_cannot_replay_and_can_reconcile(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    provider = ImapSmtpEmailProvider.__new__(ImapSmtpEmailProvider)
    provider.username = "me@example.com"
    with execution_scope(control):
        operation_id = begin_effect(control, "email.send", {"to": ["a@example.com"]})
        message = provider._build_message(["a@example.com"], "subject", "body")
        provider._persist_mime(message)
    assert effects_for_task(store, "task-mail")[0]["state"] == "dispatched"
    with execution_scope(control), pytest.raises(RuntimeError, match="interrupted after dispatch"):
        begin_effect(control, "email.send", {"to": ["a@example.com"]})
    class Provider:
        def find_sent_copy(self, message_id):
            assert message_id == message["Message-ID"]
            return True
    assert reconcile_email_effect(store, Provider(), operation_id)["action"] == "found_sent_copy"
    assert effects_for_task(store, "task-mail")[0]["state"] == "committed"


def test_interrupted_smtp_without_sent_copy_stays_unknown(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    provider = ImapSmtpEmailProvider.__new__(ImapSmtpEmailProvider)
    provider.username = "me@example.com"
    with execution_scope(control):
        operation_id = begin_effect(control, "email.send", {"to": ["a@example.com"]})
        provider._persist_mime(provider._build_message(["a@example.com"], "subject", "body"))
    class Provider:
        def find_sent_copy(self, message_id):
            return False
        def send(self, *args, **kwargs):
            raise AssertionError("SMTP must not be retried")
    assert reconcile_email_effect(store, Provider(), operation_id)["status"] == "unknown"
    assert effects_for_task(store, "task-mail")[0]["state"] == "dispatched"


def test_approval_preview_and_changed_attachment_invalidates_renewal(tmp_path: Path) -> None:
    attachment = tmp_path / "attachment.txt"
    attachment.write_text("original", encoding="utf-8")
    approvals = PendingActionStore(db_path=tmp_path / "tasks.sqlite")
    request = approvals.create("session", {
        "tool_name": "email.save_draft", "description": "Save draft",
        "kwargs": {"to": ["a@example.com"], "subject": "review", "body": "Please review",
                   "attachment_paths": [str(attachment)]},
    })
    assert "a@example.com" in request["preview"]
    assert "attachment.txt" in request["preview"]
    assert "SHA256" in request["preview"]
    attachment.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        approvals.renew(request["action_id"], "session")
    assert approvals.inspect(request["action_id"], "session") is None


def test_interrupted_dispatch_is_unknown_to_tool_registry(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    control = ExecutionControl(task_id="task-mail", node_id="send")
    control.task_store = store
    calls = []
    registry = ToolRegistry()
    registry.register(ToolSpec("email.send", "email", "fixture", [ToolParameter("to", "array")],
                               lambda to: calls.append(to)))
    with execution_scope(control):
        begin_effect(control, "email.send", {"to": ["a@example.com"]})
        result = registry.call("email.send", to=["a@example.com"])
    assert result.status == "unknown" and result.error_code == "SIDE_EFFECT_UNKNOWN"
    assert not calls
