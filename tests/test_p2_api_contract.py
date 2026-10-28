from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from deskpilot.api.app import create_app
from deskpilot.api.credentials import create_catalog
from deskpilot.api.security import TokenCatalog
from deskpilot.api.service import PrivateTaskService
from deskpilot.api.worker import register_api_handlers
from deskpilot.tasks.service import TaskService
from deskpilot.tools.permissions import assess_command, assess_python_code


def api_client(base: Path, *, max_active: int = 4, max_per_minute: int = 20):
    tokens = {"alice": "alice-" + "a" * 40, "bob": "bob-" + "b" * 40}
    records = [{
        "principal": principal, "sha256": hashlib.sha256(token.encode()).hexdigest(),
        "scopes": ["submit", "read", "control", "approve"],
    } for principal, token in tokens.items()]
    catalog = base / "tokens.json"
    catalog.write_text(json.dumps({"tokens": records}), encoding="utf-8")
    service = PrivateTaskService(base / "index" / "index.json", max_active=max_active,
                                 max_per_minute=max_per_minute)
    return TestClient(create_app(service, TokenCatalog(catalog))), service, tokens


def headers(token: str, key: str = "request-key-0001") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Idempotency-Key": key}


def test_p2_submit_replay_owner_and_restart(tmp_path: Path) -> None:
    client, service, tokens = api_client(tmp_path)
    first = client.post("/v1/tasks", json={"question": "What is RAG?"}, headers=headers(tokens["alice"]))
    assert first.status_code == 202
    task_id = first.json()["task_id"]
    assert first.json()["replayed"] is False
    assert service.store.get_task(task_id)["status"] == "queued"
    assert service.task_view("alice", task_id)["status"] == "queued"

    repeated = client.post("/v1/tasks", json={"question": "What is RAG?"}, headers=headers(tokens["alice"]))
    assert repeated.status_code == 202
    assert repeated.json()["task_id"] == task_id
    assert repeated.json()["replayed"] is True
    assert client.get(f"/v1/tasks/{task_id}", headers=headers(tokens["bob"])).status_code == 404
    assert client.get(f"/v1/tasks/{task_id}", headers=headers(tokens["alice"])).status_code == 200

    reopened = PrivateTaskService(tmp_path / "index" / "index.json")
    assert reopened.owned_task("alice", task_id)["task_id"] == task_id


def test_p2_auth_input_limit_and_idempotency_conflict(tmp_path: Path) -> None:
    client, service, tokens = api_client(tmp_path)
    assert client.post("/v1/tasks", json={"question": "hello"}).status_code == 401
    assert client.post("/v1/tasks", json={"question": "hello"}, headers=headers("wrong" * 10)).status_code == 401
    assert client.post("/v1/tasks", json={"question": "x" * 5000}, headers=headers(tokens["alice"])).status_code == 422
    assert client.post("/v1/tasks", json={"question": "hello", "user_id": "bob"}, headers=headers(tokens["alice"])).status_code == 422
    assert service.store.list_tasks() == []

    first = client.post("/v1/tasks", json={"question": "hello"}, headers=headers(tokens["alice"]))
    conflict = client.post("/v1/tasks", json={"question": "different"}, headers=headers(tokens["alice"]))
    assert first.status_code == 202
    assert conflict.status_code == 409
    assert len(service.store.list_tasks()) == 1


def test_p2_quota_and_session_ownership(tmp_path: Path) -> None:
    client, service, tokens = api_client(tmp_path, max_active=1, max_per_minute=2)
    first = client.post("/v1/tasks", json={"question": "hello"}, headers=headers(tokens["alice"]))
    assert first.status_code == 202
    session_id = first.json()["session_id"]
    second = client.post("/v1/tasks", json={"question": "next"}, headers=headers(tokens["alice"], "request-key-0002"))
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    foreign = client.post("/v1/tasks", json={"question": "foreign", "session_id": session_id},
                          headers=headers(tokens["bob"], "request-key-0003"))
    assert foreign.status_code == 404
    assert len(service.store.list_tasks()) == 1


def test_p2_sse_and_control_do_not_expose_foreign_task(tmp_path: Path) -> None:
    client, service, tokens = api_client(tmp_path)
    created = client.post("/v1/tasks", json={"question": "hello"}, headers=headers(tokens["alice"]))
    task_id = created.json()["task_id"]
    assert client.post(f"/v1/tasks/{task_id}/cancel", headers=headers(tokens["bob"])).status_code == 404
    assert client.get(f"/v1/tasks/{task_id}/events", headers=headers(tokens["bob"])).status_code == 404
    assert client.get(f"/v1/tasks/{task_id}/events", headers={**headers(tokens["alice"]), "Last-Event-ID": "oops"}).status_code == 400
    assert service.store.get_task(task_id)["status"] == "queued"
    cancelled = client.post(f"/v1/tasks/{task_id}/cancel", headers=headers(tokens["alice"]))
    assert cancelled.status_code == 202


def test_p2_separate_worker_result_and_answer_artifact(tmp_path: Path) -> None:
    client, service, tokens = api_client(tmp_path)
    created = client.post("/v1/tasks", json={"question": "hello"}, headers=headers(tokens["alice"]))
    task_id = created.json()["task_id"]
    assert service.store.get_task(task_id)["status"] == "queued"
    worker = TaskService(service.store)
    worker.register("core.api_direct", lambda payload: {"answer": "fixture answer", "session_id": payload["session_id"]})
    assert worker.run_once() is True
    worker.close()

    own = client.get(f"/v1/tasks/{task_id}", headers=headers(tokens["alice"]))
    assert own.json()["status"] == "succeeded"
    artifact_id = own.json()["artifacts"][0]["id"]
    assert client.get(f"/v1/artifacts/{artifact_id}", headers=headers(tokens["alice"])).text == "fixture answer"
    assert client.get(f"/v1/artifacts/{artifact_id}", headers=headers(tokens["bob"])).status_code == 404
    stream = client.get(f"/v1/tasks/{task_id}/events", headers=headers(tokens["alice"]))
    assert stream.status_code == 200
    assert "event: task" in stream.text
    assert "fixture answer" not in stream.text


def test_p2_token_catalog_and_worker_fail_closed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "credentials" / "tokens.json"
    token = create_catalog(path, "owner")
    assert token not in path.read_text(encoding="utf-8")
    assert TokenCatalog(path).authenticate(f"Bearer {token}", "submit").principal_id == "owner"
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_API_WORKER", "1")
    assert assess_command("python -c 'print(1)'").blocked
    assert assess_python_code("print(1)").blocked

    class FakeQueue:
        handlers = {"core.answer": lambda _: "unsafe"}

        def register(self, key, handler):
            self.handlers[key] = handler

    class FakeClient:
        def chat(self, messages, **kwargs):
            assert len(messages) == 2
            return "direct answer"

    class FakeAgent:
        task_service = FakeQueue()
        client = FakeClient()

    agent = FakeAgent()
    register_api_handlers(agent)
    assert set(agent.task_service.handlers) == {"core.api_direct", "core.api_decision"}
    assert agent.task_service.handlers["core.api_direct"]({"question": "What is RAG?", "session_id": "test"})["answer"] == "direct answer"


def test_p2_health_distinguishes_live_from_ready(tmp_path: Path) -> None:
    client, service, _ = api_client(tmp_path)
    assert client.get("/health/live").json()["status"] == "live"
    assert client.get("/health/ready").status_code == 503
    assert service.worker_ready() is False
