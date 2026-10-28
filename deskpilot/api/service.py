from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

from ..core.approval import PendingActionStore
from ..core.sqlite_utils import sqlite_transaction
from ..tasks.service import TaskService
from ..tasks.store import TaskStore
from ..tasks.models import sanitize_text
from ..sandbox.broker import SandboxBroker


class APIServiceError(Exception):
    def __init__(self, status: int, code: str, *, retry_after: int | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.retry_after = retry_after


class PrivateTaskService:
    """Persistent admission and ownership boundary; never starts a worker here."""

    def __init__(self, index_path: Path, *, max_active: int = 4, max_per_minute: int = 20,
                 max_daily_reserved_tokens: int = 200_000, sandbox_enabled: bool = False) -> None:
        self.index_path = index_path.resolve()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = TaskStore(self.index_path.parent / "task_ledger.sqlite3")
        self.queue = TaskService(self.store)
        self.queue.register("core.api_direct", self._must_run_in_worker)
        self.queue.register("core.api_decision", self._must_run_in_worker)
        self.queue.register("core.api_sandbox_python", self._must_run_in_worker)
        self.approvals = PendingActionStore(db_path=self.store.path)
        self.max_active = max(1, max_active)
        self.max_per_minute = max(1, max_per_minute)
        self.max_daily_reserved_tokens = max(1, max_daily_reserved_tokens)
        self.sandbox_enabled = sandbox_enabled
        with sqlite_transaction(self.store.path) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS api_sessions (
                    session_id TEXT PRIMARY KEY, principal TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS api_submissions (
                    principal TEXT NOT NULL, request_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL, task_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL, state TEXT NOT NULL,
                    created_at REAL NOT NULL, claimed_at REAL NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    PRIMARY KEY(principal, request_key));
                CREATE INDEX IF NOT EXISTS api_submissions_task ON api_submissions(task_id);
            """)

    @staticmethod
    def _must_run_in_worker(_: dict[str, Any]) -> None:
        raise RuntimeError("API process cannot execute task handlers")

    @staticmethod
    def _digest(value: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _reserve(self, principal: str, key: str, body: dict[str, Any], session_id: str | None,
                 handler: str) -> tuple[str, str, bool]:
        if not 8 <= len(key) <= 128 or not key.isascii() or any(ord(char) < 33 for char in key):
            raise APIServiceError(400, "invalid_idempotency_key")
        request_hash = self._digest({"body": body, "session_id": session_id or "", "handler": handler})
        now = time.time()
        reserve_tokens = max(2048, min(20_000, len(json.dumps(body, ensure_ascii=False)) * 3))
        with sqlite_transaction(self.store.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM api_submissions WHERE principal=? AND request_key=?", (principal, key)
            ).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise APIServiceError(409, "idempotency_conflict")
                if row["state"] == "ready":
                    return row["task_id"], row["session_id"], True
                if now - row["claimed_at"] < 30:
                    raise APIServiceError(425, "submission_in_progress", retry_after=2)
                connection.execute(
                    "UPDATE api_submissions SET claimed_at=? WHERE principal=? AND request_key=?",
                    (now, principal, key),
                )
                return row["task_id"], row["session_id"], False
            active = connection.execute("""
                SELECT COUNT(*) FROM api_submissions s LEFT JOIN tasks t ON t.task_id=s.task_id
                WHERE s.principal=? AND (s.state='creating' OR
                    (s.state='ready' AND t.status IN ('queued','running','retry_wait','waiting_human')))
            """, (principal,)).fetchone()[0]
            global_active = connection.execute("""
                SELECT COUNT(*) FROM api_submissions s LEFT JOIN tasks t ON t.task_id=s.task_id
                WHERE s.state='creating' OR
                    (s.state='ready' AND t.status IN ('queued','running','retry_wait','waiting_human'))
            """).fetchone()[0]
            recent = connection.execute(
                "SELECT COUNT(*) FROM api_submissions WHERE principal=? AND created_at>?",
                (principal, now - 60),
            ).fetchone()[0]
            daily = connection.execute(
                "SELECT COALESCE(SUM(reserved_tokens),0) FROM api_submissions WHERE principal=? AND created_at>?",
                (principal, now - 86400),
            ).fetchone()[0]
            if (active >= self.max_active or global_active >= self.queue.queue_limit
                    or recent >= self.max_per_minute or daily + reserve_tokens > self.max_daily_reserved_tokens):
                raise APIServiceError(429, "quota_exceeded", retry_after=60)
            if session_id:
                owner = connection.execute("SELECT principal FROM api_sessions WHERE session_id=?", (session_id,)).fetchone()
                if owner is None or owner[0] != principal:
                    raise APIServiceError(404, "session_not_found")
                selected_session = session_id
            else:
                selected_session = "sess_api_" + secrets.token_urlsafe(18)
                connection.execute("INSERT INTO api_sessions VALUES (?,?)", (selected_session, principal))
            task_id = "task_api_" + secrets.token_urlsafe(18)
            connection.execute(
                "INSERT INTO api_submissions VALUES (?,?,?,?,?,?,?,?,?)",
                (principal, key, request_hash, task_id, selected_session, "creating", now, now, reserve_tokens),
            )
            return task_id, selected_session, False

    def submit(self, principal: str, key: str, question: str, session_id: str | None = None) -> dict[str, Any]:
        body = {"question": question}
        task_id, selected_session, replayed = self._reserve(principal, key, body, session_id, "core.api_direct")
        if replayed:
            task = self.store.get_task(task_id)
            return {"task_id": task_id, "session_id": selected_session, "status": task["status"] if task else "unknown", "replayed": True}
        payload = {"question": question, "session_id": selected_session}
        self._enqueue(task_id, selected_session, "core.api_direct", payload)
        return {"task_id": task_id, "session_id": selected_session, "status": "queued", "replayed": False}

    def submit_sandbox(self, principal: str, key: str, code: str,
                       inputs: dict[str, str] | None = None) -> dict[str, Any]:
        if not self.sandbox_enabled:
            raise APIServiceError(503, "sandbox_disabled")
        try:
            code_bytes = code.encode("utf-8") if isinstance(code, str) else b""
        except UnicodeError as exc:
            raise APIServiceError(422, "invalid_sandbox_code") from exc
        if not isinstance(code, str) or not code.strip() or b"\x00" in code_bytes or len(code_bytes) > 8192:
            raise APIServiceError(422, "invalid_sandbox_code")
        if not isinstance(inputs, dict) and inputs is not None:
            raise APIServiceError(422, "invalid_sandbox_inputs")
        try:
            encoded = {name: value.encode("utf-8") for name, value in (inputs or {}).items()}
            SandboxBroker().validate(encoded)
        except (AttributeError, TypeError, ValueError) as exc:
            raise APIServiceError(422, "invalid_sandbox_inputs") from exc
        body = {"code": code, "inputs": inputs or {}}
        task_id, session_id, replayed = self._reserve(principal, key, body, None, "core.api_sandbox_python")
        if replayed:
            task = self.store.get_task(task_id)
            return {"task_id": task_id, "session_id": session_id,
                    "status": task["status"] if task else "unknown", "replayed": True}
        self._enqueue(task_id, session_id, "core.api_sandbox_python", body)
        return {"task_id": task_id, "session_id": session_id, "status": "queued", "replayed": False}

    def _enqueue(self, task_id: str, session_id: str, handler: str, payload: dict[str, Any],
                 *, queue_session_id: str | None = None) -> None:
        task = self.store.get_task(task_id)
        if task is None:
            # Approval decisions must not wait behind their waiting_human parent.
            self.store.create_task({"goal": handler, "steps": []},
                                   session_id=session_id if queue_session_id is None else queue_session_id,
                                   status="paused", task_id=task_id)
            task = self.store.get_task(task_id)
        if task["status"] == "paused":
            self.queue.bind(task_id, handler, payload, timeout_seconds=600)
            self.store.update_task(task_id, "queued")
        with sqlite_transaction(self.store.path) as connection:
            connection.execute("UPDATE api_submissions SET state='ready' WHERE task_id=?", (task_id,))

    def owned_task(self, principal: str, task_id: str) -> dict[str, Any]:
        with sqlite_transaction(self.store.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM api_submissions WHERE principal=? AND task_id=? AND state='ready'",
                (principal, task_id),
            ).fetchone()
        if row is None:
            raise APIServiceError(404, "task_not_found")
        task = self.store.get_task(task_id)
        if task is None:
            raise APIServiceError(404, "task_not_found")
        return task

    def task_view(self, principal: str, task_id: str) -> dict[str, Any]:
        task = self.owned_task(principal, task_id)
        output = self.store.load_output(task_id) if task["output_ref"] else None
        answer = output.get("answer") if isinstance(output, dict) else None
        sandbox = output.get("sandbox") if isinstance(output, dict) else None
        pending = []
        for item in self.approvals.list_pending(session_id=task["session_id"], task_id=task_id):
            public = self.approvals.public(item)
            public["preview_digest"] = self._digest(item.fingerprint)
            pending.append(public)
        return {
            "task_id": task_id, "session_id": task["session_id"], "status": task["status"],
            "created_at": task["created_at"], "updated_at": task["updated_at"],
            "answer": answer, "sandbox": sandbox, "error": sanitize_text(task["error"]), "pending_actions": pending,
            "artifacts": ([{"id": "answer_" + task_id, "name": "answer.txt"}] if answer else []),
        }

    def answer_artifact(self, principal: str, artifact_id: str) -> str:
        if not artifact_id.startswith("answer_task_api_"):
            raise APIServiceError(404, "artifact_not_found")
        task_id = artifact_id.removeprefix("answer_")
        view = self.task_view(principal, task_id)
        if not isinstance(view["answer"], str) or not view["answer"]:
            raise APIServiceError(404, "artifact_not_found")
        return view["answer"]

    def control(self, principal: str, task_id: str, action: str) -> dict[str, Any]:
        self.owned_task(principal, task_id)
        if action == "resume":
            self.queue.resume(task_id)
        else:
            self.queue.request(task_id, action)
        return self.task_view(principal, task_id)

    def decide(self, principal: str, task_id: str, action_id: str, decision: str,
               preview_digest: str) -> dict[str, Any]:
        task = self.owned_task(principal, task_id)
        key = "decision:" + action_id
        payload = {"action_id": action_id, "session_id": task["session_id"], "decision": decision,
                   "parent_task_id": task_id, "preview_digest": preview_digest}
        with sqlite_transaction(self.store.path) as connection:
            prior = connection.execute(
                "SELECT 1 FROM api_submissions WHERE principal=? AND request_key=?", (principal, key)
            ).fetchone()
        if prior:
            decision_task_id, _, replayed = self._reserve(principal, key, payload,
                                                            task["session_id"], "core.api_decision")
            return {"task_id": decision_task_id, "status": "queued", "replayed": replayed}
        if task["status"] != "waiting_human":
            raise APIServiceError(409, "task_not_waiting_for_approval")
        pending = self.approvals.inspect(action_id, task["session_id"])
        if pending is None or pending.task_id != task_id:
            raise APIServiceError(404, "approval_not_found")
        if self._digest(pending.fingerprint) != preview_digest:
            raise APIServiceError(409, "approval_preview_changed")
        if pending.tool_name not in {"email.send", "email.save_draft"}:
            raise APIServiceError(403, "approval_tool_not_available")
        decision_task_id, session_id, replayed = self._reserve(principal, key, payload,
                                                                task["session_id"], "core.api_decision")
        if not replayed:
            self._enqueue(decision_task_id, session_id, "core.api_decision", payload, queue_session_id="")
        return {"task_id": decision_task_id, "status": "queued", "replayed": replayed}

    def events(self, principal: str, task_id: str, after_seq: int, limit: int = 100) -> list[dict[str, Any]]:
        self.owned_task(principal, task_id)
        return [{"id": row["event_seq"], "type": row["event_type"], "time": row["created_at"]}
                for row in self.store.list_events(task_id, after_seq=after_seq, limit=min(100, limit))]

    def worker_ready(self) -> bool:
        with sqlite_transaction(self.store.path) as connection:
            row = connection.execute("SELECT owner,expires FROM runtime_worker_lock WHERE id=1").fetchone()
        return bool(row and str(row[0]).startswith("api_worker_") and float(row[1]) > time.time())
