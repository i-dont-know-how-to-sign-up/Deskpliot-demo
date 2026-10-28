from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from deskpilot.api.app import create_app
from deskpilot.api.credentials import create_catalog
from deskpilot.api.security import TokenCatalog
from deskpilot.api.service import PrivateTaskService
from deskpilot.api.worker import register_api_handlers


def _app(tmp_path: Path, *, sandbox_enabled: bool, execute_scope: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    token = "test-token-" + "x" * 40
    catalog = tmp_path / "tokens.json"
    scopes = ["submit", "read", "control", "approve"]
    if execute_scope:
        scopes.append("execute")
    catalog.write_text(json.dumps({"tokens": [{
        "principal": "owner", "sha256": hashlib.sha256(token.encode()).hexdigest(),
        "scopes": scopes,
    }]}), encoding="utf-8")
    service = PrivateTaskService(tmp_path / "index" / "index.json", sandbox_enabled=sandbox_enabled)
    return TestClient(create_app(service, TokenCatalog(catalog))), service, token


def _submit(client: TestClient, token: str, key: str = "sandbox-key-001", **body):
    return client.post("/v1/sandbox/tasks", json=body or {"code": "print(1)", "inputs": {}},
                       headers={"Authorization": f"Bearer {token}", "Idempotency-Key": key})


def test_sandbox_endpoint_is_disabled_and_scope_is_separate(tmp_path: Path) -> None:
    client, service, token = _app(tmp_path / "disabled", sandbox_enabled=False)
    assert _submit(client, token).status_code == 503
    assert service.store.list_tasks() == []

    client, service, token = _app(tmp_path / "no-scope", sandbox_enabled=True, execute_scope=False)
    assert _submit(client, token).status_code == 403
    assert service.store.list_tasks() == []
    assert _submit(client, "wrong-token").status_code == 401
    assert service.store.list_tasks() == []


def test_sandbox_rejects_host_paths_and_bad_payloads_before_enqueue(tmp_path: Path) -> None:
    client, service, token = _app(tmp_path, sandbox_enabled=True)
    for bad in (
        {"code": "print(1)", "inputs": {"../secret.txt": "x"}},
        {"code": "print(1)", "inputs": {"C:\\secret.txt": "x"}},
        {"code": "print(1)", "inputs": {"a.txt": "x"}, "host_path": "C:\\secret.txt"},
        {"code": " "},
        {"code": "x" * 9000},
    ):
        assert _submit(client, token, key="invalid-" + str(len(service.store.list_tasks())) + "-0000", **bad).status_code == 422
    assert service.store.list_tasks() == []


def test_sandbox_runs_only_in_worker_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    client, service, token = _app(tmp_path, sandbox_enabled=True)
    body = {"code": "print(open('/inputs/a.txt').read())", "inputs": {"a.txt": "fixture"}}
    queued = _submit(client, token, **body)
    assert queued.status_code == 202
    task_id = queued.json()["task_id"]
    assert service.store.get_task(task_id)["status"] == "queued"
    replay = _submit(client, token, **body)
    assert replay.status_code == 202
    assert replay.json()["task_id"] == task_id
    assert replay.json()["replayed"] is True
    assert _submit(client, token, code="print(2)", inputs={}).status_code == 409

    calls = []

    class FakeRunner:
        def __init__(self, image: str):
            assert image == "sha256:" + "a" * 64

        def run_python(self, code: str, inputs: dict[str, bytes]):
            calls.append((code, inputs))
            return {"ok": True, "stdout": "fixture\n", "stderr": "", "returncode": 0,
                    "timed_out": False, "output_limited": False, "sandbox": "docker"}

    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_API_WORKER", "1")
    monkeypatch.setenv("DESKPILOT_ENABLE_DOCKER_SANDBOX", "1")
    monkeypatch.setenv("DESKPILOT_DOCKER_IMAGE", "sha256:" + "a" * 64)
    monkeypatch.setattr("deskpilot.api.worker.DockerSandboxRunner", FakeRunner)
    register_api_handlers(SimpleNamespace(task_service=service.queue))
    assert service.queue.run_once()
    view = client.get(f"/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {token}"})
    assert view.status_code == 200
    assert view.json()["status"] == "succeeded"
    assert view.json()["answer"] == "fixture\n"
    assert calls == [(body["code"], {"a.txt": b"fixture"})]


def test_catalog_execute_scope_is_opt_in(tmp_path: Path) -> None:
    default_path = tmp_path / "default.json"
    token = create_catalog(default_path, "owner")
    assert "execute" not in TokenCatalog(default_path).authenticate(f"Bearer {token}", "read").scopes
    enabled_path = tmp_path / "enabled.json"
    token = create_catalog(enabled_path, "owner", allow_sandbox_execute=True)
    assert "execute" in TokenCatalog(enabled_path).authenticate(f"Bearer {token}", "execute").scopes
