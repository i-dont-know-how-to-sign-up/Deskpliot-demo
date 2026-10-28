from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from deskpilot.api.app import create_app
from deskpilot.api.security import TokenCatalog
from deskpilot.api.service import PrivateTaskService
from deskpilot.api.worker import register_api_handlers
from deskpilot.core.approval import PendingActionStore
from deskpilot.core.sqlite_utils import sqlite_transaction


@pytest.fixture
def api(tmp_path: Path):
    tokens = {
        "alice": "alice-" + "a" * 40,
        "bob": "bob-" + "b" * 40,
        "reader": "reader-" + "r" * 40,
        "expired": "expired-" + "e" * 40,
    }
    records = [
        {
            "principal": name,
            "sha256": hashlib.sha256(token.encode()).hexdigest(),
            "scopes": ["read"] if name == "reader" else ["submit", "read", "control", "approve"],
            **({"expires_at": "2000-01-01T00:00:00Z"} if name == "expired" else {}),
        }
        for name, token in tokens.items()
    ]
    catalog_path = tmp_path / "tokens.json"
    catalog_path.write_text(json.dumps({"tokens": records}), encoding="utf-8")
    service = PrivateTaskService(tmp_path / "index" / "index.json")
    client = TestClient(create_app(service, TokenCatalog(catalog_path)))
    return client, service, tokens, catalog_path, records


def auth(token: str, key: str | None = None) -> dict[str, str]:
    result = {"Authorization": f"Bearer {token}"}
    if key is not None:
        result["Idempotency-Key"] = key
    return result


def submit(client: TestClient, token: str, key: str, question: str = "What is RAG?",
           session_id: str | None = None):
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/v1/tasks", json=body, headers=auth(token, key))


class FakeEmailAgent:
    def __init__(self, pending_actions: PendingActionStore, task_service, email_handler) -> None:
        self.pending_actions = pending_actions
        self.task_service = task_service
        self.email_handler = email_handler

    def approve_pending_action(self, action: dict, *, session_id: str):
        item = self.pending_actions.consume(action["action_id"], session_id)
        self.email_handler(item.tool_name, item.kwargs)
        return SimpleNamespace(steps=[SimpleNamespace(status="succeeded")], answer="fixture accepted")

    def cancel_pending_action(self, action: dict, *, session_id: str) -> bool:
        return self.pending_actions.cancel(action["action_id"], session_id)


def fake_worker(service: PrivateTaskService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_API_WORKER", "1")
    pending = PendingActionStore()
    calls: list[tuple[str, dict]] = []

    def fake_email_handler(tool_name: str, kwargs: dict) -> None:
        calls.append((tool_name, dict(kwargs)))

    agent = FakeEmailAgent(pending, service.queue, fake_email_handler)
    register_api_handlers(agent)
    return agent, calls


def pending_email(agent: FakeEmailAgent, *, task_id: str, session_id: str,
                  attachment: Path | None = None) -> dict:
    kwargs = {"to": ["nobody@example.invalid"], "subject": "Fixture", "body": "No live send"}
    if attachment is not None:
        kwargs["attachment_paths"] = [str(attachment)]
    action = agent.pending_actions.create(session_id, {
        "task_id": task_id, "node_id": "email_send", "tool_name": "email.send", "kwargs": kwargs,
    })
    item = agent.pending_actions.inspect(action["action_id"], session_id)
    assert item is not None
    return {
        "action_id": action["action_id"], "session_id": session_id,
        "parent_task_id": task_id, "preview_digest": PrivateTaskService._digest(item.fingerprint),
    }


def test_foreign_principal_cannot_reuse_session_or_control_task(api) -> None:
    client, service, tokens, _, _ = api
    first = submit(client, tokens["alice"], "alice-session-0001")
    assert first.status_code == 202
    task_id, session_id = first.json()["task_id"], first.json()["session_id"]

    for route in (f"/v1/tasks/{task_id}", f"/v1/tasks/{task_id}/events"):
        assert client.get(route, headers=auth(tokens["bob"])).status_code == 404
    for action in ("cancel", "resume"):
        assert client.post(f"/v1/tasks/{task_id}/{action}", headers=auth(tokens["bob"])).status_code == 404
    foreign = submit(client, tokens["bob"], "bob-session-0001", session_id=session_id)
    assert foreign.status_code == 404
    assert service.store.get_task(task_id)["status"] == "queued"

    own_followup = submit(client, tokens["alice"], "alice-session-0002", "Follow up", session_id)
    assert own_followup.status_code == 202
    assert own_followup.json()["session_id"] == session_id
    assert own_followup.json()["task_id"] != task_id


def test_idempotency_is_principal_scoped_and_conflict_precedes_quota(api) -> None:
    client, service, tokens, _, _ = api
    service.max_active = 1
    key = "shared-key-0001"
    alice = submit(client, tokens["alice"], key, "Original")
    assert alice.status_code == 202

    replay = submit(client, tokens["alice"], key, "Original")
    assert replay.status_code == 202
    assert replay.json()["task_id"] == alice.json()["task_id"]
    assert replay.json()["replayed"] is True
    conflict = submit(client, tokens["alice"], key, "Changed")
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "idempotency_conflict"

    bob = submit(client, tokens["bob"], key, "Original")
    assert bob.status_code == 202
    assert bob.json()["task_id"] != alice.json()["task_id"]
    assert len(service.store.list_tasks()) == 2


def test_in_progress_idempotency_retry_never_creates_second_task(api, monkeypatch) -> None:
    client, service, tokens, catalog_path, _ = api
    other_service = PrivateTaskService(service.index_path)
    other_client = TestClient(create_app(other_service, TokenCatalog(catalog_path)))
    entered, release = threading.Event(), threading.Event()
    original_enqueue = service._enqueue

    def paused_enqueue(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return original_enqueue(*args, **kwargs)

    monkeypatch.setattr(service, "_enqueue", paused_enqueue)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(submit, client, tokens["alice"], "race-key-0001", "Race")
        assert entered.wait(timeout=10)
        in_progress = submit(other_client, tokens["alice"], "race-key-0001", "Race")
        assert in_progress.status_code == 425
        assert int(in_progress.headers["Retry-After"]) > 0
        release.set()
        first = pending.result(timeout=10)

    assert first.status_code == 202
    replay = submit(client, tokens["alice"], "race-key-0001", "Race")
    assert replay.status_code == 202
    assert replay.json()["task_id"] == first.json()["task_id"]
    assert replay.json()["replayed"] is True
    assert len(service.store.list_tasks()) == 1


def test_concurrent_distinct_keys_cannot_exceed_active_quota(api) -> None:
    client, service, tokens, catalog_path, _ = api
    service.max_active = 1
    other_service = PrivateTaskService(service.index_path, max_active=1)
    other_client = TestClient(create_app(other_service, TokenCatalog(catalog_path)))
    barrier = threading.Barrier(2)

    def raced_submit(target: TestClient, key: str):
        barrier.wait(timeout=10)
        return submit(target, tokens["alice"], key, key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(raced_submit, client, "quota-race-0001")
        second = pool.submit(raced_submit, other_client, "quota-race-0002")
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert sorted(result.status_code for result in results) == [202, 429]
    assert len(service.store.list_tasks()) == 1
    denied = next(result for result in results if result.status_code == 429)
    assert int(denied.headers["Retry-After"]) > 0


@pytest.mark.parametrize("quota,limit", [("max_per_minute", 1), ("max_daily_reserved_tokens", 2048)])
def test_rate_and_cost_quota_reject_new_work_but_allow_replay(api, quota: str, limit: int) -> None:
    client, service, tokens, _, _ = api
    setattr(service, quota, limit)
    first = submit(client, tokens["alice"], "quota-first-0001")
    assert first.status_code == 202
    denied = submit(client, tokens["alice"], "quota-second-0001")
    assert denied.status_code == 429
    assert denied.json()["error"] == "quota_exceeded"
    assert submit(client, tokens["alice"], "quota-first-0001").json()["task_id"] == first.json()["task_id"]
    assert submit(client, tokens["bob"], "quota-first-0001").status_code == 202
    assert len(service.store.list_tasks()) == 2


def test_foreign_and_tampered_approval_cannot_enqueue_decision(api) -> None:
    client, service, tokens, _, _ = api
    created = submit(client, tokens["alice"], "approval-parent-0001")
    task_id, session_id = created.json()["task_id"], created.json()["session_id"]
    service.store.update_task(task_id, "waiting_human")
    action = service.approvals.create(session_id, {
        "task_id": task_id,
        "node_id": "mail_send",
        "tool_name": "email.save_draft",
        "kwargs": {"to": ["fixture@example.invalid"], "subject": "Fixture", "body": "No delivery"},
    })
    pending = client.get(f"/v1/tasks/{task_id}", headers=auth(tokens["alice"])).json()["pending_actions"]
    assert len(pending) == 1
    digest = pending[0]["preview_digest"]
    assert action["action_id"] == pending[0]["action_id"]

    body = {"action_id": action["action_id"], "preview_digest": digest}
    assert client.get(f"/v1/tasks/{task_id}", headers=auth(tokens["bob"])).status_code == 404
    for decision in ("approve", "reject"):
        denied = client.post(f"/v1/tasks/{task_id}/{decision}", json=body, headers=auth(tokens["bob"]))
        assert denied.status_code == 404
    tampered = client.post(f"/v1/tasks/{task_id}/approve",
                           json={**body, "preview_digest": "0" * 64}, headers=auth(tokens["alice"]))
    assert tampered.status_code == 409
    injected = client.post(f"/v1/tasks/{task_id}/approve",
                           json={**body, "confirm": True}, headers=auth(tokens["alice"]))
    assert injected.status_code == 422
    assert len(service.store.list_tasks()) == 1
    assert service.approvals.inspect(action["action_id"], session_id) is not None

    other = submit(client, tokens["alice"], "approval-other-0001", "Other task", session_id)
    assert other.status_code == 202
    service.store.update_task(other.json()["task_id"], "waiting_human")
    wrong_task = client.post(f"/v1/tasks/{other.json()['task_id']}/approve",
                             json=body, headers=auth(tokens["alice"]))
    assert wrong_task.status_code == 404

    accepted = client.post(f"/v1/tasks/{task_id}/approve", json=body, headers=auth(tokens["alice"]))
    assert accepted.status_code == 202
    assert accepted.json()["task_id"] != task_id
    assert len(service.store.list_tasks()) == 3
    assert service.approvals.inspect(action["action_id"], session_id) is not None


def test_approval_retry_replays_but_opposite_decision_conflicts(api) -> None:
    client, service, tokens, _, _ = api
    created = submit(client, tokens["alice"], "approval-replay-0001")
    task_id, session_id = created.json()["task_id"], created.json()["session_id"]
    service.store.update_task(task_id, "waiting_human")
    action = service.approvals.create(session_id, {
        "task_id": task_id,
        "node_id": "mail_send",
        "tool_name": "email.save_draft",
        "kwargs": {"to": ["fixture@example.invalid"], "subject": "Replay", "body": "No delivery"},
    })
    digest = client.get(f"/v1/tasks/{task_id}", headers=auth(tokens["alice"])).json()["pending_actions"][0]["preview_digest"]
    body = {"action_id": action["action_id"], "preview_digest": digest}
    approved = client.post(f"/v1/tasks/{task_id}/approve", json=body, headers=auth(tokens["alice"]))
    assert approved.status_code == 202
    again = client.post(f"/v1/tasks/{task_id}/approve", json=body, headers=auth(tokens["alice"]))
    assert again.status_code == 202
    assert again.json()["task_id"] == approved.json()["task_id"]
    assert again.json()["replayed"] is True
    opposite = client.post(f"/v1/tasks/{task_id}/reject", json=body, headers=auth(tokens["alice"]))
    assert opposite.status_code == 409
    assert len(service.store.list_tasks()) == 2


@pytest.mark.parametrize("bad_header,expected", [(None, 401), ("Basic anything", 401),
                                                    ("Bearer invalid", 401), ("reader", 403),
                                                    ("expired", 401)])
def test_bad_auth_never_persists_request(api, bad_header: str | None, expected: int) -> None:
    client, service, tokens, _, _ = api
    token = tokens.get(bad_header, bad_header) if bad_header is not None else None
    headers = {"Idempotency-Key": "invalid-auth-0001"}
    if token is not None:
        headers["Authorization"] = token if token.startswith(("Basic ", "Bearer ")) else f"Bearer {token}"
    result = client.post("/v1/tasks", json={"question": "No dispatch"}, headers=headers)
    assert result.status_code == expected
    assert service.store.list_tasks() == []


def test_token_revocation_is_applied_without_restart(api) -> None:
    client, service, tokens, catalog_path, records = api
    first = submit(client, tokens["alice"], "revocation-0001")
    assert first.status_code == 202
    catalog_path.write_text(json.dumps({"tokens": [row for row in records if row["principal"] != "alice"]}), encoding="utf-8")
    assert client.get(f"/v1/tasks/{first.json()['task_id']}", headers=auth(tokens["alice"])).status_code == 401
    assert submit(client, tokens["alice"], "revocation-0002").status_code == 401
    assert len(service.store.list_tasks()) == 1


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": 123},
    {"question": "ok", "principal": "bob"},
    {"question": "ok", "session_id": {"id": "spoof"}},
])
def test_bad_body_cannot_create_task_or_choose_principal(api, body: dict) -> None:
    client, service, tokens, _, _ = api
    result = client.post("/v1/tasks", json=body, headers=auth(tokens["alice"], "invalid-body-0001"))
    assert result.status_code == 422
    assert service.store.list_tasks() == []


def test_oversized_body_and_bad_idempotency_key_fail_closed(api) -> None:
    client, service, tokens, _, _ = api
    too_large = client.post("/v1/tasks", json={"question": "x" * 17_000},
                            headers=auth(tokens["alice"], "large-body-0001"))
    assert too_large.status_code == 413
    for key in (None, "short", "key with space"):
        result = submit(client, tokens["alice"], key) if key is not None else client.post(
            "/v1/tasks", json={"question": "ok"}, headers=auth(tokens["alice"]))
        assert result.status_code == 400
    assert service.store.list_tasks() == []


def test_sse_cursor_replays_only_owned_task_events(api) -> None:
    client, service, tokens, _, _ = api
    alice = submit(client, tokens["alice"], "sse-alice-0001")
    alice_id = alice.json()["task_id"]
    first_seq = service.store.list_events(alice_id)[0]["event_seq"]
    bob = submit(client, tokens["bob"], "sse-bob-0001")
    bob_id = bob.json()["task_id"]
    bob_sequences = {row["event_seq"] for row in service.store.list_events(bob_id)}
    service.store.update_task(alice_id, "succeeded")

    response = client.get(f"/v1/tasks/{alice_id}/events?after=999999",
                          headers={**auth(tokens["alice"]), "Last-Event-ID": str(first_seq)})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    ids = [int(line.removeprefix("id: ")) for line in response.text.splitlines() if line.startswith("id: ")]
    expected = [row["event_seq"] for row in service.store.list_events(alice_id, after_seq=first_seq)]
    assert ids == expected
    assert all(sequence > first_seq and sequence not in bob_sequences for sequence in ids)
    assert client.get(f"/v1/tasks/{alice_id}/events",
                      headers={**auth(tokens["alice"]), "Last-Event-ID": str(ids[-1])}).text.find("id: ") == -1

    assert client.get(f"/v1/tasks/{alice_id}/events",
                      headers={**auth(tokens["alice"]), "Last-Event-ID": "-1"}).status_code == 400
    assert client.get(f"/v1/tasks/{alice_id}/events",
                      headers={**auth(tokens["bob"]), "Last-Event-ID": str(first_seq)}).status_code == 404


def test_worker_decision_approves_once_and_rejects_without_email(api, monkeypatch) -> None:
    _, service, _, _, _ = api
    agent, email_calls = fake_worker(service, monkeypatch)
    decision = service.queue.handlers["core.api_decision"]

    approve_payload = pending_email(agent, task_id="task_fixture_approve", session_id="session_fixture_approve")
    accepted = decision({**approve_payload, "decision": "approve"})
    assert accepted == {"approved": True, "answer": "fixture accepted"}
    assert email_calls == [("email.send", {
        "to": ["nobody@example.invalid"], "subject": "Fixture", "body": "No live send",
    })]
    with pytest.raises(ValueError, match="Approval no longer belongs"):
        decision({**approve_payload, "decision": "approve"})
    assert len(email_calls) == 1

    reject_payload = pending_email(agent, task_id="task_fixture_reject", session_id="session_fixture_reject")
    assert decision({**reject_payload, "decision": "reject"}) == {"rejected": True}
    assert agent.pending_actions.inspect(reject_payload["action_id"], reject_payload["session_id"]) is None
    with pytest.raises(ValueError, match="Approval no longer belongs"):
        decision({**reject_payload, "decision": "reject"})
    assert len(email_calls) == 1


@pytest.mark.parametrize("mutation", ["digest", "attachment", "arguments"])
def test_worker_decision_refuses_changed_preview_or_attachment(api, monkeypatch, tmp_path: Path,
                                                               mutation: str) -> None:
    _, service, _, _, _ = api
    agent, email_calls = fake_worker(service, monkeypatch)
    attachment = tmp_path / "approval-attachment.txt"
    attachment.write_text("approved bytes", encoding="utf-8")
    payload = pending_email(agent, task_id="task_fixture_changed", session_id="session_fixture_changed",
                            attachment=attachment)
    if mutation == "digest":
        payload["preview_digest"] = "0" * 64
    elif mutation == "attachment":
        attachment.write_text("replaced bytes", encoding="utf-8")
    else:
        item = agent.pending_actions.inspect(payload["action_id"], payload["session_id"])
        assert item is not None
        item.kwargs["subject"] = "Unapproved subject"

    with pytest.raises(ValueError, match="Approval preview changed|Approval inputs changed|审批内容或附件"):
        service.queue.handlers["core.api_decision"]({**payload, "decision": "approve"})
    assert email_calls == []
    assert agent.pending_actions.inspect(payload["action_id"], payload["session_id"]) is not None


def test_queued_worker_decision_runs_while_parent_waits_for_approval(api, monkeypatch) -> None:
    client, service, tokens, _, _ = api
    agent, email_calls = fake_worker(service, monkeypatch)
    service.approvals = agent.pending_actions
    parent = submit(client, tokens["alice"], "worker-parent-0001")
    assert parent.status_code == 202
    parent_id, session_id = parent.json()["task_id"], parent.json()["session_id"]
    service.store.update_task(parent_id, "waiting_human")
    payload = pending_email(agent, task_id=parent_id, session_id=session_id)
    queued = client.post(f"/v1/tasks/{parent_id}/approve", json={
        "action_id": payload["action_id"], "preview_digest": payload["preview_digest"],
    }, headers=auth(tokens["alice"]))
    assert queued.status_code == 202
    decision_id = queued.json()["task_id"]
    assert service.store.get_task(decision_id)["status"] == "queued"

    assert service.queue.run_once(), "queued decision must not be blocked by its waiting_human parent"
    assert service.store.get_task(decision_id)["status"] == "succeeded"
    assert len(email_calls) == 1
    assert agent.pending_actions.inspect(payload["action_id"], session_id) is None


@pytest.mark.parametrize("partial_task", [False, True])
def test_stale_creating_reservation_recovers_same_task_after_lease(api, partial_task: bool) -> None:
    client, service, tokens, _, _ = api
    key = "stale-create-0001"
    question = "Recover this submission"
    task_id, session_id, replayed = service._reserve(
        "alice", key, {"question": question}, None, "core.api_direct",
    )
    assert replayed is False
    if partial_task:
        service.store.create_task({"goal": "core.api_direct", "steps": []},
                                  session_id=session_id, status="paused", task_id=task_id)
    immediate = submit(client, tokens["alice"], key, question)
    assert immediate.status_code == 425

    with sqlite_transaction(service.store.path) as connection:
        connection.execute("UPDATE api_submissions SET claimed_at=? WHERE task_id=?", (time.time() - 31, task_id))
    recovered = submit(client, tokens["alice"], key, question)
    assert recovered.status_code == 202
    assert recovered.json()["task_id"] == task_id
    assert recovered.json()["session_id"] == session_id
    assert service.store.get_task(task_id)["status"] == "queued"
    assert service.queue.descriptor(task_id)["handler_key"] == "core.api_direct"
    assert len(service.store.list_tasks()) == 1
    assert submit(client, tokens["alice"], key, question).json()["replayed"] is True
