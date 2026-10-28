"""Offline HTTP contract probes for the restricted private API."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from deskpilot import __version__
from deskpilot.api.app import create_app
from deskpilot.api.security import TokenCatalog
from deskpilot.api.service import PrivateTaskService
from deskpilot.tasks.service import TaskService

DATASET = Path(__file__).parent / "dataset" / "p2_api_cases.jsonl"


def load_cases(ids: list[str] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        by_id = {row["id"]: row for row in rows}
        missing = [case_id for case_id in ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown P2 API case IDs: {missing}")
        rows = [by_id[case_id] for case_id in dict.fromkeys(ids)]
    return rows


def run_case(case: dict[str, Any], root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    alice, bob = "alice-" + "a" * 40, "bob-" + "b" * 40
    catalog = root / "tokens.json"
    catalog.write_text(json.dumps({"tokens": [{
        "principal": name, "sha256": hashlib.sha256(token.encode()).hexdigest(),
        "scopes": ["submit", "read", "control", "approve"] + (["execute"] if name == "alice" else []),
    } for name, token in (("alice", alice), ("bob", bob))]}), encoding="utf-8")
    service = PrivateTaskService(root / "index" / "index.json", max_active=1 if case["scenario"] == "quota" else 4,
                                 sandbox_enabled=case["scenario"] != "sandbox_disabled")
    client = TestClient(create_app(service, TokenCatalog(catalog)))
    def headers(token: str, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}", "Idempotency-Key": key}
    def submit(token: str, key: str, question: str = "What is RAG?", session_id: str | None = None):
        body = {"question": question}
        if session_id:
            body["session_id"] = session_id
        return client.post("/v1/tasks", json=body, headers=headers(token, key))

    checks: dict[str, bool] = {}
    scenario = case["scenario"]
    if scenario in {"sandbox_disabled", "sandbox_scope", "sandbox_input_traversal", "sandbox_idempotency"}:
        token = bob if scenario == "sandbox_scope" else alice
        body = {"code": "print('fixture')", "inputs": {}}
        if scenario == "sandbox_input_traversal":
            body["inputs"] = {"../secret.txt": "synthetic"}
        response = client.post("/v1/sandbox/tasks", json=body, headers=headers(token, "sandbox-key-0001"))
        if scenario == "sandbox_idempotency":
            replay = client.post("/v1/sandbox/tasks", json=body, headers=headers(token, "sandbox-key-0001"))
            checks = {"same_task": response.status_code == 202 and replay.status_code == 202
                      and response.json()["task_id"] == replay.json()["task_id"],
                      "task_count": len(service.store.list_tasks()) == 1}
        else:
            expected = {"sandbox_disabled": 503, "sandbox_scope": 403,
                        "sandbox_input_traversal": 422}[scenario]
            checks = {"status": response.status_code == expected,
                      "task_count": len(service.store.list_tasks()) == 0}
    elif scenario == "invalid_token":
        response = submit("invalid-token", "probe-key-0001")
        checks = {"status": response.status_code == 401, "no_task": not service.store.list_tasks()}
    else:
        first = submit(alice, "probe-key-0001")
        checks["initial_submit"] = first.status_code == 202
        task_id = first.json()["task_id"]
        session_id = first.json()["session_id"]
        if scenario == "submit_replay":
            replay = submit(alice, "probe-key-0001")
            checks.update(same_task=replay.status_code == 202 and replay.json()["task_id"] == task_id,
                          task_count=len(service.store.list_tasks()) == 1)
        elif scenario == "foreign_task":
            response = client.get(f"/v1/tasks/{task_id}", headers=headers(bob, "unused-key-0001"))
            checks["status"] = response.status_code == 404
        elif scenario == "session_followup":
            response = submit(alice, "probe-key-0002", "Follow up", session_id)
            checks.update(same_session=response.status_code == 202 and response.json()["session_id"] == session_id,
                          task_count=len(service.store.list_tasks()) == 2)
        elif scenario == "idempotency_conflict":
            response = submit(alice, "probe-key-0001", "Different question")
            checks.update(status=response.status_code == 409,
                          task_count=len(service.store.list_tasks()) == 1)
        elif scenario == "answer_artifact":
            worker = TaskService(service.store)
            worker.register("core.api_direct", lambda _: {"answer": "synthetic answer"})
            worker.run_once()
            worker.close()
            view = client.get(f"/v1/tasks/{task_id}", headers=headers(alice, "unused-key-0001"))
            artifact_id = view.json().get("artifacts", [{}])[0].get("id", "")
            response = client.get(f"/v1/artifacts/{artifact_id}", headers=headers(alice, "unused-key-0001"))
            checks.update(status=response.status_code == 200, body=response.text == "synthetic answer")
        elif scenario == "sse_resume":
            first_seq = service.store.list_events(task_id)[0]["event_seq"]
            foreign = submit(bob, "probe-key-0002")
            foreign_ids = {row["event_seq"] for row in service.store.list_events(foreign.json()["task_id"])}
            service.store.update_task(task_id, "succeeded")
            response = client.get(f"/v1/tasks/{task_id}/events", headers={**headers(alice, "unused-key-0001"),
                                                                           "Last-Event-ID": str(first_seq)})
            ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
            checks.update(ordered=ids == sorted(set(ids)) and all(seq > first_seq for seq in ids),
                          isolated=not set(ids) & foreign_ids)
        elif scenario == "quota":
            response = submit(alice, "probe-key-0002")
            checks.update(status=response.status_code == 429,
                          task_count=len(service.store.list_tasks()) == 1)
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")
    return {"case_id": case["id"], "subset": case["subset"], "scenario": scenario,
            "status": "passed" if checks and all(checks.values()) else "failed",
            "checks": checks, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated P2 private-API HTTP contract probes")
    parser.add_argument("--mode", choices=["offline"], default="offline")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--project-version", default=__version__)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/p2_api_latest.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/reports/p2_api_latest.md"))
    args = parser.parse_args()
    cases = load_cases(args.case_id)
    with tempfile.TemporaryDirectory(prefix="deskpilot_p2_api_") as temporary:
        results = []
        for case in cases:
            try:
                result = run_case(case, Path(temporary) / case["id"])
            except Exception as exc:
                result = {"case_id": case["id"], "subset": case["subset"], "scenario": case["scenario"],
                          "status": "error", "error": type(exc).__name__}
            print(f"{case['id']}: {result['status']}", flush=True)
            results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    lines = [f"# DeskPilot {args.project_version} P2 API offline contract", "",
             f"- Executed: {len(results)}", f"- Passed: {sum(row['status'] == 'passed' for row in results)}", "",
             "| ID | Scenario | Result |", "|---|---|---|"]
    lines.extend(f"| {row['case_id']} | {row['scenario']} | {row['status']} |" for row in results)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "passed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
