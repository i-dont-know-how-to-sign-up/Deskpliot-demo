from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..core.sqlite_utils import connect_sqlite
from .models import (
    NODE_STATES, RECOVERY_STATES, SUCCESS_STATES, TASK_STATES, TERMINAL_STATES,
    ApprovalConflict, LeaseConflict, SerializationError, TaskStoreError, VersionConflict,
    digest, json_data, plan_data, sanitize_text, serializable, timestamp, utc_now,
)


_UNSET = object()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL DEFAULT '',
    goal TEXT NOT NULL DEFAULT '', goal_digest TEXT NOT NULL DEFAULT '',
    plan_version TEXT NOT NULL, executor_version TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1, plan_ref TEXT, input_ref TEXT, output_ref TEXT,
    error TEXT NOT NULL DEFAULT '', deadline TEXT, budget_json TEXT NOT NULL DEFAULT '{}',
    owner TEXT, lease_token INTEGER NOT NULL DEFAULT 0, lease_expires_at TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0, pause_requested INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT, legacy INTEGER NOT NULL DEFAULT 0, safe_to_resume INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nodes (
    task_id TEXT NOT NULL REFERENCES tasks(task_id), node_id TEXT NOT NULL,
    status TEXT NOT NULL, data_json TEXT NOT NULL, input_ref TEXT, input_digest TEXT NOT NULL,
    output_ref TEXT, error TEXT NOT NULL DEFAULT '', tool_calls INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0, artifact_paths_json TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id, node_id)
);
CREATE TABLE IF NOT EXISTS output_refs (
    ref_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id), node_id TEXT,
    kind TEXT NOT NULL, content_json TEXT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, node_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
    finished_at TEXT, error TEXT NOT NULL DEFAULT '', error_category TEXT NOT NULL DEFAULT '',
    retry_at TEXT, tool_calls INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
    usage_json TEXT NOT NULL DEFAULT '{}', certainty TEXT NOT NULL DEFAULT 'unknown',
    owner TEXT, lease_token INTEGER NOT NULL DEFAULT 0, output_ref TEXT,
    observed INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(task_id, node_id) REFERENCES nodes(task_id, node_id),
    UNIQUE(task_id, node_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, node_id TEXT NOT NULL,
    plan_version TEXT NOT NULL, executor_version TEXT NOT NULL, handler_version TEXT NOT NULL,
    input_digest TEXT NOT NULL, output_ref TEXT, output_digest TEXT,
    artifacts_json TEXT NOT NULL DEFAULT '[]', validation_json TEXT NOT NULL DEFAULT '{}',
    payload_ref TEXT, created_at TEXT NOT NULL,
    FOREIGN KEY(task_id, node_id) REFERENCES nodes(task_id, node_id),
    UNIQUE(task_id, node_id)
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, node_id TEXT NOT NULL,
    session_id TEXT NOT NULL, plan_version TEXT NOT NULL, content_digest TEXT NOT NULL,
    arguments_ref TEXT NOT NULL, preview_ref TEXT, status TEXT NOT NULL DEFAULT 'pending',
    risk TEXT NOT NULL DEFAULT '', expires_at TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(task_id, node_id) REFERENCES nodes(task_id, node_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES tasks(task_id),
    node_id TEXT, attempt_id TEXT, event_type TEXT NOT NULL, data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_imports (
    source_path TEXT PRIMARY KEY, source_digest TEXT NOT NULL, imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_origins (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), process_id INTEGER NOT NULL, process_birth TEXT
);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks(status, retry_at, created_at);
CREATE INDEX IF NOT EXISTS tasks_session ON tasks(session_id, status, created_at);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, event_seq);
CREATE INDEX IF NOT EXISTS attempts_node ON attempts(task_id, node_id, attempt_number);
CREATE INDEX IF NOT EXISTS approvals_task ON approvals(task_id, status);
"""


class TaskStore:
    """短事务 WAL 存储；外部工具执行不应放在数据库事务内。"""

    def __init__(self, path: str | Path, *, legacy_path: str | Path | None = None) -> None:
        requested = Path(path)
        self.path = requested.with_suffix(".sqlite3") if requested.suffix.lower() == ".json" else requested
        self.path = self.path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise TaskStoreError(f"Unsupported task schema version: {version}")
            connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + "\nPRAGMA user_version=1;\nCOMMIT;")
        source = Path(legacy_path) if legacy_path is not None else requested if requested.suffix.lower() == ".json" else None
        if source is not None:
            self.migrate_legacy_json(source)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = connect_sqlite(self.path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _require_task(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    @staticmethod
    def _require_node(connection: sqlite3.Connection, task_id: str, node_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM nodes WHERE task_id=? AND node_id=?", (task_id, node_id)).fetchone()
        if row is None:
            raise KeyError((task_id, node_id))
        return row

    @staticmethod
    def _guard(
        task: sqlite3.Row, *, expected_version: int | None = None,
        owner: str | None = None, lease_token: int | None = None,
    ) -> None:
        if task["legacy"]:
            raise TaskStoreError("Imported legacy tasks are read-only history")
        if expected_version is not None and task["version"] != expected_version:
            raise VersionConflict("Task version changed")
        if task["owner"] is not None or owner is not None or lease_token is not None:
            if (
                owner != task["owner"] or lease_token != task["lease_token"]
                or task["lease_expires_at"] is None or task["lease_expires_at"] <= utc_now()
            ):
                raise LeaseConflict("Task lease is missing, expired, or fenced")

    @staticmethod
    def _bump(connection: sqlite3.Connection, task_id: str) -> None:
        connection.execute("UPDATE tasks SET version=version+1, updated_at=? WHERE task_id=?", (utc_now(), task_id))

    @staticmethod
    def _process_state(process_id: int) -> tuple[bool, str | None]:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, process_id)
            if not handle:
                # 无权限时不能把仍活着的其他进程当成已崩溃。
                return ctypes.get_last_error() != 87, None
            try:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True, None
                if code.value != 259:
                    return False, None
                times = [wintypes.FILETIME() for _ in range(4)]
                if kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                    return True, str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
                return True, None
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False, None
        except PermissionError:
            return True, None
        try:
            birth = Path(f"/proc/{process_id}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            birth = None
        return True, birth

    def _record_origin(self, connection: sqlite3.Connection, task_id: str) -> None:
        _, birth = self._process_state(os.getpid())
        connection.execute("INSERT INTO task_origins VALUES(?,?,?) ON CONFLICT(task_id) DO UPDATE SET process_id=excluded.process_id,process_birth=excluded.process_birth", (task_id, os.getpid(), birth))

    @staticmethod
    def _event(
        connection: sqlite3.Connection, task_id: str, event_type: str, *, node_id: str | None = None,
        attempt_id: str | None = None, data: Any = None,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO events(task_id,node_id,attempt_id,event_type,data_json,created_at) VALUES(?,?,?,?,?,?)",
            (task_id, node_id, attempt_id, event_type, json_data(data or {}), utc_now()),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _put_ref(connection: sqlite3.Connection, task_id: str, kind: str, value: Any, node_id: str | None = None) -> str:
        content = json_data(value)
        ref_id = "ref_" + secrets.token_urlsafe(16)
        connection.execute(
            "INSERT INTO output_refs VALUES(?,?,?,?,?,?,?)",
            (ref_id, task_id, node_id, kind, content, hashlib.sha256(content.encode("utf-8")).hexdigest(), utc_now()),
        )
        return ref_id

    @staticmethod
    def _load_ref(connection: sqlite3.Connection, ref_id: str | None) -> Any:
        if ref_id is None:
            return None
        row = connection.execute("SELECT content_json,sha256 FROM output_refs WHERE ref_id=?", (ref_id,)).fetchone()
        if row is None:
            raise TaskStoreError("Missing persisted object reference")
        if hashlib.sha256(row["content_json"].encode("utf-8")).hexdigest() != row["sha256"]:
            raise TaskStoreError("Persisted object digest mismatch")
        return json.loads(row["content_json"])

    @staticmethod
    def _node_record(row: sqlite3.Row) -> dict[str, Any]:
        data = json.loads(row["data_json"])
        data.update({key: row[key] for key in (
            "node_id", "status", "input_ref", "input_digest", "output_ref", "error",
            "tool_calls", "tokens", "version", "updated_at",
        )})
        data["artifact_paths"] = json.loads(row["artifact_paths_json"])
        if "attempt_count" in row.keys():
            data["attempt_count"] = row["attempt_count"]
            data["tool_calls"] = row["total_tool_calls"]
            data["tokens"] = row["total_tokens"]
        return data

    def _task_record(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["budget"] = json.loads(record.pop("budget_json"))
        for key in ("legacy", "safe_to_resume", "cancel_requested", "pause_requested"):
            record[key] = bool(record[key])
        record["nodes"] = {
            node["node_id"]: self._node_record(node)
            for node in connection.execute("""SELECT n.*,
                (SELECT COALESCE(MAX(a.attempt_number),0) FROM attempts a WHERE a.task_id=n.task_id AND a.node_id=n.node_id) AS attempt_count,
                (SELECT COALESCE(SUM(a.tool_calls),n.tool_calls) FROM attempts a WHERE a.task_id=n.task_id AND a.node_id=n.node_id) AS total_tool_calls,
                (SELECT COALESCE(SUM(a.tokens),n.tokens) FROM attempts a WHERE a.task_id=n.task_id AND a.node_id=n.node_id) AS total_tokens
                FROM nodes n WHERE n.task_id=? ORDER BY n.rowid""", (row["task_id"],))
        }
        return record

    def create_task(
        self, plan: Any, *, initial_values: Any = None, session_id: str = "", status: str = "queued",
        plan_version: str | None = None, executor_version: str = "1", task_id: str | None = None,
        deadline: str | datetime | None = None, budget: dict[str, Any] | None = None,
    ) -> str:
        if status not in TASK_STATES:
            raise ValueError(f"Unknown task state: {status}")
        descriptor = plan_data(plan, executor_version=executor_version)
        plan_version = str(plan_version if plan_version is not None else descriptor.get("plan_version", "1"))
        descriptor["plan_version"] = plan_version
        inputs = serializable(initial_values if initial_values is not None else {})
        task_id = task_id or "task_" + secrets.token_urlsafe(12)
        now = utc_now()
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO tasks(task_id,session_id,goal,goal_digest,plan_version,executor_version,status,
                created_at,updated_at,deadline,budget_json,safe_to_resume) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task_id, session_id, " -> ".join(step["step_id"] for step in descriptor["steps"]),
                 hashlib.sha256(str(descriptor.get("goal", "")).encode("utf-8")).hexdigest()[:16],
                 str(plan_version), str(executor_version), status, now, now,
                 timestamp(deadline) if deadline is not None else None, json_data(budget or {}),
                 int(not any(step["side_effect"] for step in descriptor["steps"]))),
            )
            plan_ref = self._put_ref(connection, task_id, "plan", descriptor)
            input_ref = self._put_ref(connection, task_id, "input", inputs)
            connection.execute("UPDATE tasks SET plan_ref=?,input_ref=? WHERE task_id=?", (plan_ref, input_ref, task_id))
            self._insert_nodes(connection, task_id, descriptor, inputs)
            if status == "running":
                self._record_origin(connection, task_id)
            self._event(connection, task_id, "task_created", data={"status": status, "plan_version": str(plan_version)})
        return task_id

    def _insert_nodes(self, connection: sqlite3.Connection, task_id: str, plan: dict[str, Any], inputs: Any) -> None:
        for step in plan["steps"]:
            node_id = step["step_id"]
            arguments = step["arguments"]
            input_ref = self._put_ref(connection, task_id, "node_input", arguments, node_id)
            summary = {key: step[key] for key in (
                "agent", "handler_key", "handler_version", "depends_on", "allowed_tools", "requires_human", "side_effect",
            )}
            summary["binding_digest"] = digest(step.get("handler_binding", {}))
            connection.execute(
                """INSERT INTO nodes(task_id,node_id,status,data_json,input_ref,input_digest,updated_at)
                VALUES(?,?,?,?,?,?,?)""",
                (task_id, node_id, "pending", json_data(summary), input_ref,
                 self._input_digest(inputs, arguments, summary), utc_now()),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return self._task_record(connection, row) if row is not None else None

    def list_tasks(
        self, *, session_id: str | None = None, statuses: list[str] | set[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses, arguments = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            arguments.append(session_id)
        if statuses is not None:
            if not statuses:
                return []
            clauses.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            arguments.extend(statuses)
        sql = "SELECT * FROM tasks" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at,task_id"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be nonnegative")
            sql += " LIMIT ?"
            arguments.append(limit)
        with self._connection() as connection:
            connection.execute("BEGIN")
            return [self._task_record(connection, row) for row in connection.execute(sql, arguments).fetchall()]

    def list_events(self, task_id: str | None = None, *, after_seq: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        sql = "SELECT * FROM events WHERE event_seq>?"
        arguments: list[Any] = [after_seq]
        if task_id is not None:
            sql += " AND task_id=?"
            arguments.append(task_id)
        sql += " ORDER BY event_seq LIMIT ?"
        arguments.append(limit)
        with self._connection() as connection:
            records = []
            for row in connection.execute(sql, arguments):
                record = dict(row)
                record["data"] = json.loads(record.pop("data_json"))
                records.append(record)
            return records

    get = get_task
    list = list_tasks
    events = list_events

    def update_task(
        self, task_id: str, status: str | None = None, *, error: str | None = None,
        expected_version: int | None = None, owner: str | None = None, lease_token: int | None = None,
        retry_at: str | datetime | None = None, deadline: str | datetime | None = None,
        budget: dict[str, Any] | None = None, output: Any = _UNSET,
    ) -> dict[str, Any]:
        if status is not None and status not in TASK_STATES:
            raise ValueError(f"Unknown task state: {status}")
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, expected_version=expected_version, owner=owner, lease_token=lease_token)
            values: dict[str, Any] = {}
            if status is not None:
                values["status"] = status
            if error is not None:
                values["error"] = sanitize_text(str(error))[:1000]
            if retry_at is not None:
                values["retry_at"] = timestamp(retry_at)
            if deadline is not None:
                values["deadline"] = timestamp(deadline)
            if budget is not None:
                values["budget_json"] = json_data(budget)
            if output is not _UNSET:
                values["output_ref"] = self._put_ref(connection, task_id, "output", output)
            if status in TERMINAL_STATES or status in {"queued", "retry_wait", "waiting_human", "paused", "recovery_required", "interrupted"}:
                values.update(owner=None, lease_expires_at=None)
            if values:
                connection.execute("UPDATE tasks SET " + ",".join(key + "=?" for key in values) + " WHERE task_id=?", [*values.values(), task_id])
            if status == "running" and task["owner"] is None:
                self._record_origin(connection, task_id)
            self._bump(connection, task_id)
            event = {"status": status or task["status"], "previous_status": task["status"]}
            if task["owner"] is not None and "owner" in values and values["owner"] is None:
                event.update(released_owner=task["owner"], lease_generation=task["lease_token"])
            self._event(connection, task_id, "task_updated", data=event)
            return self._task_record(connection, self._require_task(connection, task_id))

    def finish_task(self, task_id: str, status: str, error: str = "", **guard: Any) -> dict[str, Any]:
        return self.update_task(task_id, status, error=error, **guard)

    def finish_worker_task(
        self, task_id: str, status: str, error: str = "", *, owner: str, lease_token: int,
        output: Any = _UNSET,
    ) -> dict[str, Any]:
        """worker 的终态提交始终检查原 lease，并与控制请求原子排序。"""
        if status not in TASK_STATES or status == "running":
            raise ValueError("Worker completion requires a non-running state")
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, owner=owner, lease_token=lease_token)
            if task["status"] != "running":
                raise LeaseConflict("Worker task is no longer running")
            if status != "recovery_required":
                if task["cancel_requested"]:
                    status = "cancelled"
                elif task["pause_requested"]:
                    status = "paused"
            ref = task["output_ref"] if output is _UNSET else self._put_ref(connection, task_id, "output", output)
            connection.execute(
                "UPDATE tasks SET status=?,error=?,output_ref=?,owner=NULL,lease_expires_at=NULL WHERE task_id=?",
                (status, sanitize_text(str(error))[:1000], ref, task_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "task_updated", data={
                "status": status, "previous_status": task["status"], "released_owner": owner, "lease_generation": lease_token,
            })
            return self._task_record(connection, self._require_task(connection, task_id))

    def _save_value(self, task_id: str, kind: str, value: Any, *, node_id: str | None = None, **guard: Any) -> str:
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            node = self._require_node(connection, task_id, node_id) if node_id is not None else None
            ref = self._put_ref(connection, task_id, kind, value, node_id)
            column = "input_ref" if kind in {"input", "node_input"} else "output_ref"
            if node is None:
                connection.execute(f"UPDATE tasks SET {column}=? WHERE task_id=?", (ref, task_id))
            else:
                connection.execute(f"UPDATE nodes SET {column}=?,version=version+1,updated_at=? WHERE task_id=? AND node_id=?", (ref, utc_now(), task_id, node_id))
            if kind == "input":
                for row in connection.execute("SELECT * FROM nodes WHERE task_id=?", (task_id,)).fetchall():
                    self._refresh_input_digest(connection, task_id, row["node_id"])
            elif kind == "node_input":
                self._refresh_input_digest(connection, task_id, node_id)
            self._bump(connection, task_id)
            self._event(connection, task_id, kind + "_saved", node_id=node_id, data={"ref": ref})
            return ref

    def _refresh_input_digest(self, connection: sqlite3.Connection, task_id: str, node_id: str) -> None:
        task, node = self._require_task(connection, task_id), self._require_node(connection, task_id, node_id)
        value = self._input_digest(self._load_ref(connection, task["input_ref"]), self._load_ref(connection, node["input_ref"]), json.loads(node["data_json"]))
        connection.execute("UPDATE nodes SET input_digest=? WHERE task_id=? AND node_id=?", (value, task_id, node_id))

    @staticmethod
    def _input_digest(inputs: Any, arguments: Any, metadata: dict[str, Any]) -> str:
        value = {"initial_values": inputs, "arguments": arguments}
        if "binding_digest" in metadata:
            value["binding_digest"] = metadata["binding_digest"]
        return digest(value)

    def _load_value(self, task_id: str, column: str, node_id: str | None = None) -> Any:
        with self._connection() as connection:
            row = self._require_task(connection, task_id) if node_id is None else self._require_node(connection, task_id, node_id)
            return self._load_ref(connection, row[column])

    def save_input(self, task_id: str, values: Any, **guard: Any) -> str:
        return self._save_value(task_id, "input", values, **guard)

    def load_input(self, task_id: str) -> Any:
        return self._load_value(task_id, "input_ref")

    def save_node_input(self, task_id: str, node_id: str, values: Any, **guard: Any) -> str:
        return self._save_value(task_id, "node_input", values, node_id=node_id, **guard)

    def load_node_input(self, task_id: str, node_id: str) -> Any:
        return self._load_value(task_id, "input_ref", node_id)

    def save_output(self, task_id: str, output: Any, *, node_id: str | None = None, **guard: Any) -> str:
        return self._save_value(task_id, "output", output, node_id=node_id, **guard)

    def save_released_output(
        self, task_id: str, output: Any, *, owner: str, lease_token: int, expected_version: int,
    ) -> str:
        """仅补存同一 worker 主动释放后的输出，不修改已持久化状态。"""
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            if task["legacy"] or task["lease_token"] != lease_token or task["owner"] is not None:
                raise LeaseConflict("Released task belongs to a fenced worker")
            if task["version"] != expected_version:
                raise VersionConflict("Released task version changed")
            if task["status"] not in TERMINAL_STATES | {"waiting_human", "paused", "recovery_required"}:
                raise LeaseConflict("Task was not intentionally stopped by this worker")
            row = connection.execute(
                "SELECT data_json FROM events WHERE task_id=? AND event_type IN ('task_updated','lease_released') "
                "ORDER BY event_seq DESC LIMIT 1", (task_id,),
            ).fetchone()
            release = json.loads(row[0]) if row else {}
            if (release.get("released_owner") != owner or release.get("lease_generation") != lease_token
                    or release.get("status") != task["status"]):
                raise LeaseConflict("No matching intentional lease release")
            if task["output_ref"] is not None and task["status"] in TERMINAL_STATES:
                return task["output_ref"]
            ref = self._put_ref(connection, task_id, "output", output)
            connection.execute("UPDATE tasks SET output_ref=? WHERE task_id=?", (ref, task_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "output_saved", data={"ref": ref, "lease_generation": lease_token})
            return ref

    def load_output(self, task_id: str, node_id: str | None = None) -> Any:
        return self._load_value(task_id, "output_ref", node_id)

    def load_plan(self, task_id: str) -> dict[str, Any] | None:
        return self._load_value(task_id, "plan_ref")

    def save_plan(
        self, task_id: str, plan: Any, *, plan_version: str | None = None,
        executor_version: str | None = None, **guard: Any,
    ) -> str:
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            executor_version = str(executor_version or task["executor_version"])
            descriptor = plan_data(plan, executor_version=executor_version)
            version = str(plan_version) if plan_version is not None else str(descriptor.get("plan_version") or "plan_" + digest(descriptor)[:16])
            descriptor["plan_version"] = version
            previous = self._load_ref(connection, task["plan_ref"])
            ref = self._put_ref(connection, task_id, "plan", descriptor)
            # 重规划保留旧输出和执行日志，不自动删除或重放已有副作用。
            existing = {row["node_id"] for row in connection.execute("SELECT node_id FROM nodes WHERE task_id=?", (task_id,))}
            inputs = self._load_ref(connection, task["input_ref"])
            for step in descriptor["steps"]:
                if step["step_id"] not in existing:
                    self._insert_nodes(connection, task_id, {"steps": [step]}, inputs)
                    continue
                summary = {key: step[key] for key in (
                    "agent", "handler_key", "handler_version", "depends_on", "allowed_tools", "requires_human", "side_effect",
                )}
                summary["binding_digest"] = digest(step.get("handler_binding", {}))
                input_ref = self._put_ref(connection, task_id, "node_input", step["arguments"], step["step_id"])
                connection.execute(
                    "UPDATE nodes SET data_json=?,input_ref=?,version=version+1,updated_at=? WHERE task_id=? AND node_id=?",
                    (json_data(summary), input_ref, utc_now(), task_id, step["step_id"]),
                )
                self._refresh_input_digest(connection, task_id, step["step_id"])
            connection.execute(
                "UPDATE tasks SET plan_ref=?,plan_version=?,executor_version=?,safe_to_resume=0 WHERE task_id=?",
                (ref, version, executor_version, task_id),
            )
            if previous is not None and digest(previous) != digest(descriptor):
                connection.execute("UPDATE checkpoints SET validation_json=? WHERE task_id=?", (json_data({"ok": False, "reason": "plan_changed"}), task_id))
            connection.execute("UPDATE approvals SET status='invalidated',version=version+1,updated_at=? WHERE task_id=? AND status IN ('pending','approved')", (utc_now(), task_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "plan_saved", data={"ref": ref, "plan_version": version})
            return ref

    @staticmethod
    def artifact_paths(output: Any) -> list[str]:
        if not isinstance(output, dict):
            return []
        values: list[str] = []
        for key in ("artifact_path", "path", "resolved_paths", "attachment_paths"):
            raw = output.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw if item)
            elif isinstance(raw, str) and raw.strip():
                values.append(raw)
        return list(dict.fromkeys(values))[:20]

    @staticmethod
    def _fingerprints(paths: list[str]) -> list[dict[str, Any]]:
        records = []
        for value in paths:
            path = Path(value).resolve()
            try:
                with path.open("rb") as source:
                    sha = hashlib.file_digest(source, "sha256").hexdigest()
                records.append({"path": str(path), "sha256": sha, "size": path.stat().st_size})
            except OSError:
                records.append({"path": str(path), "sha256": None, "missing": True})
        return records

    @staticmethod
    def _input_paths(value: Any) -> list[str]:
        values: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"path", "paths", "source_path", "input_path", "document_path", "attachment_paths", "resolved_paths"}:
                    candidates = item if isinstance(item, list) else [item]
                    values.extend(str(path) for path in candidates if isinstance(path, (str, Path)) and path and "://" not in str(path))
                values.extend(TaskStore._input_paths(item))
        elif isinstance(value, list):
            for item in value:
                values.extend(TaskStore._input_paths(item))
        return list(dict.fromkeys(values))

    def _prepare_checkpoint(
        self, task_id: str, node_id: str, *, output: Any = _UNSET, artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            task, node = self._require_task(connection, task_id), self._require_node(connection, task_id, node_id)
            inputs = self._load_ref(connection, task["input_ref"])
            arguments = self._load_ref(connection, node["input_ref"])
            descriptor = self._load_ref(connection, task["plan_ref"])
            binding = next((step.get("handler_binding", {}) for step in (descriptor or {}).get("steps", []) if step["step_id"] == node_id), {})
            value = self._load_ref(connection, node["output_ref"]) if output is _UNSET else serializable(output)
            paths = artifacts if artifacts is not None else self.artifact_paths(value)
            paths = list(dict.fromkeys([*paths, *self._input_paths(inputs), *self._input_paths(arguments), *self._input_paths(binding)]))
            snapshot = {"node_version": node["version"], "node_input_ref": node["input_ref"],
                        "task_input_ref": task["input_ref"], "plan_ref": task["plan_ref"]}
        snapshot["artifacts"] = self._fingerprints(paths)
        return snapshot

    @staticmethod
    def _check_checkpoint_snapshot(task: sqlite3.Row, node: sqlite3.Row, snapshot: dict[str, Any]) -> None:
        if (node["version"] != snapshot["node_version"] or node["input_ref"] != snapshot["node_input_ref"]
            or task["input_ref"] != snapshot["task_input_ref"] or task["plan_ref"] != snapshot["plan_ref"]):
            raise VersionConflict("Checkpoint inputs or node changed during file validation")

    def _update_node_tx(
        self, connection: sqlite3.Connection, task_id: str, node_id: str, status: str,
        *, output: Any = _UNSET, error: str = "", tool_calls: int = 0, tokens: int = 0,
        checkpoint: bool = True, validation: dict[str, Any] | None = None,
        prepared_checkpoint: dict[str, Any] | None = None,
    ) -> str | None:
        node = self._require_node(connection, task_id, node_id)
        ref = node["output_ref"]
        paths = json.loads(node["artifact_paths_json"])
        if output is not _UNSET:
            normalized = serializable(output)
            ref = self._put_ref(connection, task_id, "output", normalized, node_id)
            paths = self.artifact_paths(normalized)
        connection.execute(
            """UPDATE nodes SET status=?,output_ref=?,error=?,tool_calls=?,tokens=?,artifact_paths_json=?,
            version=version+1,updated_at=? WHERE task_id=? AND node_id=?""",
            (status, ref, sanitize_text(str(error))[:1000], tool_calls, tokens, json_data(paths), utc_now(), task_id, node_id),
        )
        metadata = json.loads(node["data_json"])
        if metadata.get("side_effect") or metadata.get("requires_human") or status in {"pending", "waiting_human", "unknown"}:
            connection.execute("UPDATE tasks SET safe_to_resume=0 WHERE task_id=?", (task_id,))
        if status in SUCCESS_STATES and checkpoint and ref is not None:
            self._checkpoint_tx(connection, task_id, node_id, validation=validation, prepared_checkpoint=prepared_checkpoint)
        return ref

    def update_node(
        self, task_id: str, node_id: str, status: str, *, output: Any = _UNSET,
        error: str = "", tool_calls: int = 0, tokens: int = 0,
        expected_version: int | None = None, expected_node_version: int | None = None,
        owner: str | None = None, lease_token: int | None = None,
        attempt_id: str | None = None, checkpoint: bool = True,
        validation: dict[str, Any] | None = None, record_attempt: bool = True,
        checks: dict[str, Any] | None = None, error_category: str = "", certainty: str | None = None,
    ) -> dict[str, Any]:
        if status not in NODE_STATES:
            raise ValueError(f"Unknown node state: {status}")
        if tool_calls < 0 or tokens < 0:
            raise ValueError("Usage must be nonnegative")
        prepared = self._prepare_checkpoint(task_id, node_id, output=output) if status in SUCCESS_STATES and checkpoint else None
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, expected_version=expected_version, owner=owner, lease_token=lease_token)
            node = self._require_node(connection, task_id, node_id)
            if expected_node_version is not None and node["version"] != expected_node_version:
                raise VersionConflict("Node version changed")
            if prepared is not None:
                self._check_checkpoint_snapshot(task, node, prepared)
            if attempt_id is not None:
                attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
                if attempt is None or attempt["task_id"] != task_id or attempt["node_id"] != node_id:
                    raise TaskStoreError("Attempt does not belong to the active node execution")
                if attempt["status"] != "running":
                    if attempt["status"] == status and node["status"] == status and (output is _UNSET or digest(output) == digest(self._load_ref(connection, node["output_ref"]))):
                        return self._node_record(node)
                    raise TaskStoreError("Attempt already finished with a different result")
            else:
                attempt = connection.execute(
                    "SELECT * FROM attempts WHERE task_id=? AND node_id=? AND status='running' ORDER BY attempt_number DESC LIMIT 1",
                    (task_id, node_id),
                ).fetchone()
            completed = connection.execute(
                "SELECT * FROM attempts WHERE task_id=? AND node_id=? ORDER BY attempt_number DESC LIMIT 1",
                (task_id, node_id),
            ).fetchone()
            duplicate_result = (
                attempt is None and completed is not None and completed["status"] == status
                and node["status"] == status and completed["tool_calls"] == tool_calls
                and completed["tokens"] == tokens and completed["error"] == sanitize_text(str(error))[:1000]
                and (output is _UNSET or (node["output_ref"] is not None
                     and digest(output) == digest(self._load_ref(connection, node["output_ref"]))))
            )
            if duplicate_result:
                return self._node_record(node)
            ref = self._update_node_tx(
                connection, task_id, node_id, status, output=output, error=error,
                tool_calls=tool_calls, tokens=tokens, checkpoint=checkpoint, validation=validation,
                prepared_checkpoint=prepared,
            )
            if checks is not None:
                metadata = json.loads(node["data_json"])
                metadata["checks"] = serializable(checks)
                connection.execute("UPDATE nodes SET data_json=? WHERE task_id=? AND node_id=?", (json_data(metadata), task_id, node_id))
            if status != "running" and record_attempt and (attempt is not None or status not in {"pending", "waiting_human", "blocked", "skipped"}):
                if attempt is not None:
                    attempt_id = attempt["attempt_id"]
                    connection.execute(
                        "UPDATE attempts SET status=?,finished_at=?,error=?,tool_calls=?,tokens=?,output_ref=?,certainty=? WHERE attempt_id=?",
                        (status, utc_now(), sanitize_text(str(error))[:1000], tool_calls, tokens, ref,
                         "confirmed" if status in SUCCESS_STATES else "unknown", attempt_id),
                    )
                else:
                    # 兼容回调只观察到最终结果，不能声称记录了 handler 内部所有重试。
                    attempt_id = self._new_attempt_tx(connection, task_id, node_id, task["owner"], task["lease_token"], observed=True)
                    connection.execute(
                        "UPDATE attempts SET status=?,finished_at=?,error=?,tool_calls=?,tokens=?,output_ref=?,certainty=? WHERE attempt_id=?",
                        (status, utc_now(), sanitize_text(str(error))[:1000], tool_calls, tokens, ref,
                         "confirmed" if status in SUCCESS_STATES else "unknown", attempt_id),
                    )
                connection.execute("UPDATE attempts SET error_category=?,certainty=COALESCE(?,certainty) WHERE attempt_id=?", (sanitize_text(error_category), certainty, attempt_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "node_updated", node_id=node_id, attempt_id=attempt_id,
                        data={"status": status, "previous_status": node["status"], "output_ref": ref})
            return self._node_record(self._require_node(connection, task_id, node_id))

    def _new_attempt_tx(
        self, connection: sqlite3.Connection, task_id: str, node_id: str,
        owner: str | None, lease_token: int | None, *, observed: bool = False,
    ) -> str:
        number = connection.execute(
            "SELECT COALESCE(MAX(attempt_number),0)+1 FROM attempts WHERE task_id=? AND node_id=?", (task_id, node_id),
        ).fetchone()[0]
        attempt_id = "attempt_" + secrets.token_urlsafe(12)
        connection.execute(
            """INSERT INTO attempts(attempt_id,task_id,node_id,attempt_number,status,started_at,owner,lease_token,observed)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (attempt_id, task_id, node_id, number, "running", utc_now(), owner, lease_token or 0, int(observed)),
        )
        return attempt_id

    def start_attempt(self, task_id: str, node_id: str, **guard: Any) -> dict[str, Any]:
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            node = self._require_node(connection, task_id, node_id)
            if task["status"] in TERMINAL_STATES or task["cancel_requested"] or task["pause_requested"]:
                raise TaskStoreError("Task is finished or has a pending stop request")
            if node["status"] in SUCCESS_STATES:
                raise TaskStoreError("Completed nodes must be explicitly invalidated before another attempt")
            if json.loads(node["data_json"]).get("side_effect") and node["status"] in {"unknown", "recovery_required", "interrupted"}:
                raise TaskStoreError("Uncertain side effects must be reconciled before another attempt")
            if connection.execute("SELECT 1 FROM attempts WHERE task_id=? AND node_id=? AND status='running'", (task_id, node_id)).fetchone():
                raise TaskStoreError("Node already has a running attempt")
            attempt_id = self._new_attempt_tx(connection, task_id, node_id, task["owner"], task["lease_token"])
            connection.execute("UPDATE tasks SET status='running' WHERE task_id=?", (task_id,))
            if task["owner"] is None:
                self._record_origin(connection, task_id)
            connection.execute("UPDATE nodes SET status='running',version=version+1,updated_at=? WHERE task_id=? AND node_id=?", (utc_now(), task_id, node_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "attempt_started", node_id=node_id, attempt_id=attempt_id)
            return self._attempt_record(connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    @staticmethod
    def _attempt_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["usage"] = json.loads(record.pop("usage_json"))
        record["observed"] = bool(record["observed"])
        return record

    def finish_attempt(
        self, attempt_id: str, status: str, *, output: Any = _UNSET, error: str = "",
        error_category: str = "", retry_at: str | datetime | None = None,
        tool_calls: int = 0, tokens: int = 0, usage: dict[str, Any] | None = None,
        certainty: str = "unknown", validation: dict[str, Any] | None = None,
        checks: dict[str, Any] | None = None, **guard: Any,
    ) -> dict[str, Any]:
        if status not in NODE_STATES or status == "running":
            raise ValueError("An attempt must finish with a non-running node state")
        if tool_calls < 0 or tokens < 0:
            raise ValueError("Usage must be nonnegative")
        if checks is not None:
            checks = serializable(checks)

            def failed_check(value: Any) -> bool:
                if isinstance(value, dict):
                    return any(failed_check(item) for item in value.values())
                if isinstance(value, list):
                    return any(failed_check(item) for item in value)
                return value is False

            validation = {**(validation if validation is not None else {"ok": True, "scope": "persistence"}),
                          "checks": checks}
            if failed_check(checks):
                validation["ok"] = False
        prepared = None
        if status in SUCCESS_STATES:
            with self._connection() as connection:
                existing = connection.execute("SELECT task_id,node_id FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
                if existing is None:
                    raise KeyError(attempt_id)
            prepared = self._prepare_checkpoint(existing["task_id"], existing["node_id"], output=output)
        with self._transaction() as connection:
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            task_id, node_id = attempt["task_id"], attempt["node_id"]
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            if prepared is not None:
                self._check_checkpoint_snapshot(task, self._require_node(connection, task_id, node_id), prepared)
            if attempt["status"] != "running":
                raise TaskStoreError("Attempt already finished")
            if attempt["owner"] != task["owner"] or attempt["lease_token"] != task["lease_token"]:
                raise LeaseConflict("Attempt belongs to a fenced worker")
            if checks is not None:
                node = self._require_node(connection, task_id, node_id)
                metadata = json.loads(node["data_json"])
                metadata["checks"] = checks
                connection.execute("UPDATE nodes SET data_json=? WHERE task_id=? AND node_id=?",
                                   (json_data(metadata), task_id, node_id))
            ref = self._update_node_tx(connection, task_id, node_id, status, output=output, error=error,
                                       tool_calls=tool_calls, tokens=tokens, validation=validation, prepared_checkpoint=prepared)
            connection.execute(
                """UPDATE attempts SET status=?,finished_at=?,error=?,error_category=?,retry_at=?,tool_calls=?,tokens=?,
                usage_json=?,certainty=?,output_ref=? WHERE attempt_id=?""",
                (status, utc_now(), sanitize_text(str(error))[:1000], sanitize_text(error_category),
                 timestamp(retry_at) if retry_at is not None else None, tool_calls, tokens,
                 json_data(usage or {}), certainty, ref, attempt_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "attempt_finished", node_id=node_id, attempt_id=attempt_id,
                        data={"status": status, "error_category": error_category, "certainty": certainty, "output_ref": ref})
            return self._attempt_record(connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def list_attempts(self, task_id: str, node_id: str | None = None) -> list[dict[str, Any]]:
        sql, arguments = "SELECT * FROM attempts WHERE task_id=?", [task_id]
        if node_id is not None:
            sql += " AND node_id=?"
            arguments.append(node_id)
        with self._connection() as connection:
            return [self._attempt_record(row) for row in connection.execute(sql + " ORDER BY started_at,attempt_number", arguments)]

    def _checkpoint_tx(
        self, connection: sqlite3.Connection, task_id: str, node_id: str, *, checkpoint: Any = None,
        validation: dict[str, Any] | None = None, artifacts: list[str] | None = None,
        input_digest: str | None = None, handler_version: str | None = None,
        prepared_checkpoint: dict[str, Any] | None = None,
    ) -> str:
        task, node = self._require_task(connection, task_id), self._require_node(connection, task_id, node_id)
        metadata = json.loads(node["data_json"])
        output = self._load_ref(connection, node["output_ref"])
        if prepared_checkpoint is None:
            raise TaskStoreError("Checkpoint file validation must precede the write transaction")
        fingerprints = prepared_checkpoint["artifacts"]
        checkpoint_id = "checkpoint_" + secrets.token_urlsafe(12)
        payload_ref = self._put_ref(connection, task_id, "checkpoint", checkpoint, node_id) if checkpoint is not None else None
        connection.execute(
            """INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(task_id,node_id) DO UPDATE SET checkpoint_id=excluded.checkpoint_id,
            plan_version=excluded.plan_version,executor_version=excluded.executor_version,
            handler_version=excluded.handler_version,input_digest=excluded.input_digest,output_ref=excluded.output_ref,
            output_digest=excluded.output_digest,artifacts_json=excluded.artifacts_json,
            validation_json=excluded.validation_json,payload_ref=excluded.payload_ref,created_at=excluded.created_at""",
            (checkpoint_id, task_id, node_id, task["plan_version"], task["executor_version"],
             str(handler_version or metadata["handler_version"]), input_digest or node["input_digest"], node["output_ref"],
             digest(output) if node["output_ref"] is not None else None, json_data(fingerprints),
             json_data(validation if validation is not None else {"ok": True, "scope": "persistence"}), payload_ref, utc_now()),
        )
        self._event(connection, task_id, "checkpoint_saved", node_id=node_id, data={"checkpoint_id": checkpoint_id, "output_ref": node["output_ref"]})
        return checkpoint_id

    def save_checkpoint(
        self, task_id: str, node_id: str, checkpoint: Any = None, *, output: Any = _UNSET,
        validation: dict[str, Any] | None = None, artifacts: list[str] | None = None,
        input_digest: str | None = None, handler_version: str | None = None, **guard: Any,
    ) -> str:
        prepared = self._prepare_checkpoint(task_id, node_id, output=output, artifacts=artifacts)
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            node = self._require_node(connection, task_id, node_id)
            self._check_checkpoint_snapshot(task, node, prepared)
            if output is not _UNSET:
                self._update_node_tx(connection, task_id, node_id, node["status"], output=output,
                                     tool_calls=node["tool_calls"], tokens=node["tokens"], checkpoint=False)
            ref = self._checkpoint_tx(connection, task_id, node_id, checkpoint=checkpoint, validation=validation,
                                      artifacts=artifacts, input_digest=input_digest, handler_version=handler_version, prepared_checkpoint=prepared)
            self._bump(connection, task_id)
            return ref

    def load_checkpoint(self, task_id: str, node_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM checkpoints WHERE task_id=? AND node_id=?", (task_id, node_id)).fetchone()
            if row is None:
                return None
            record = dict(row)
            record["artifacts"] = json.loads(record.pop("artifacts_json"))
            record["validation"] = json.loads(record.pop("validation_json"))
            record["checkpoint"] = self._load_ref(connection, record["payload_ref"])
            return record

    @staticmethod
    def _lease_deadline(seconds: float) -> str:
        if seconds <= 0:
            raise ValueError("Lease duration must be positive")
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def _queue_rows(self, connection: sqlite3.Connection, session_id: str | None, limit: int) -> list[sqlite3.Row]:
        now = utc_now()
        sql = """SELECT t.* FROM tasks t WHERE t.legacy=0
            AND t.status IN ('queued','retry_wait') AND t.owner IS NULL
            AND t.cancel_requested=0 AND t.pause_requested=0
            AND (t.retry_at IS NULL OR t.retry_at<=?) AND (t.deadline IS NULL OR t.deadline>?)
            AND NOT EXISTS (SELECT 1 FROM nodes uncertain WHERE uncertain.task_id=t.task_id AND uncertain.status IN ('unknown','recovery_required'))
            AND NOT EXISTS (
                SELECT 1 FROM tasks earlier WHERE t.session_id<>'' AND earlier.session_id=t.session_id
                AND earlier.legacy=0 AND earlier.task_id<>t.task_id
                AND (earlier.created_at<t.created_at OR (earlier.created_at=t.created_at AND earlier.task_id<t.task_id))
                AND earlier.status NOT IN ('success','succeeded','failed','cancelled','partial')
            )"""
        arguments: list[Any] = [now, now]
        if session_id is not None:
            sql += " AND t.session_id=?"
            arguments.append(session_id)
        sql += " ORDER BY t.created_at,t.task_id LIMIT ?"
        arguments.append(limit)
        return connection.execute(sql, arguments).fetchall()

    def query_queue(self, *, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        with self._connection() as connection:
            connection.execute("BEGIN")
            return [self._task_record(connection, row) for row in self._queue_rows(connection, session_id, limit)]

    def claim_next(self, owner: str, *, lease_seconds: float = 15, session_id: str | None = None) -> dict[str, Any] | None:
        if not owner:
            raise ValueError("A worker owner is required")
        expires = self._lease_deadline(lease_seconds)
        with self._transaction() as connection:
            self._expire_leases_tx(connection)
            self._expire_deadlines_tx(connection)
            rows = self._queue_rows(connection, session_id, 1)
            if not rows:
                return None
            task = rows[0]
            connection.execute(
                """UPDATE tasks SET status='running',owner=?,lease_token=lease_token+1,lease_expires_at=?,
                version=version+1,updated_at=? WHERE task_id=?""", (owner, expires, utc_now(), task["task_id"]),
            )
            self._event(connection, task["task_id"], "lease_claimed", data={"lease_token": task["lease_token"] + 1})
            return self._task_record(connection, self._require_task(connection, task["task_id"]))

    def claim(self, task_id: str, owner: str, *, lease_seconds: float = 15) -> dict[str, Any]:
        if not owner:
            raise ValueError("A worker owner is required")
        expires = self._lease_deadline(lease_seconds)
        with self._transaction() as connection:
            self._expire_leases_tx(connection)
            self._expire_deadlines_tx(connection)
            task = self._require_task(connection, task_id)
            if task["legacy"] or task["owner"] is not None or task["status"] not in {"queued", "retry_wait"}:
                raise LeaseConflict("Task is not claimable")
            if not any(row["task_id"] == task_id for row in self._queue_rows(connection, task["session_id"], 1)):
                raise LeaseConflict("Task is not the ready head of its session")
            connection.execute(
                "UPDATE tasks SET status='running',owner=?,lease_token=lease_token+1,lease_expires_at=?,version=version+1,updated_at=? WHERE task_id=?",
                (owner, expires, utc_now(), task_id),
            )
            self._event(connection, task_id, "lease_claimed", data={"lease_token": task["lease_token"] + 1})
            return self._task_record(connection, self._require_task(connection, task_id))

    def heartbeat(self, task_id: str, owner: str, lease_token: int, *, lease_seconds: float = 15) -> dict[str, Any]:
        expires = self._lease_deadline(lease_seconds)
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, owner=owner, lease_token=lease_token)
            connection.execute("UPDATE tasks SET lease_expires_at=? WHERE task_id=?", (expires, task_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "lease_heartbeat", data={"lease_token": lease_token})
            return self._task_record(connection, self._require_task(connection, task_id))

    def release_lease(self, task_id: str, owner: str, lease_token: int, *, status: str = "queued") -> dict[str, Any]:
        if status not in TASK_STATES or status == "running":
            raise ValueError("Release requires a non-running task state")
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, owner=owner, lease_token=lease_token)
            running = connection.execute("SELECT 1 FROM attempts WHERE task_id=? AND status='running'", (task_id,)).fetchone()
            if running is not None:
                raise LeaseConflict("Finish or reconcile running attempts before releasing the lease")
            connection.execute("UPDATE tasks SET owner=NULL,lease_expires_at=NULL,status=? WHERE task_id=?", (status, task_id))
            self._bump(connection, task_id)
            self._event(connection, task_id, "lease_released",
                        data={"status": status, "lease_generation": lease_token, "released_owner": owner})
            return self._task_record(connection, self._require_task(connection, task_id))

    def _expire_leases_tx(self, connection: sqlite3.Connection) -> list[str]:
        tasks = connection.execute("SELECT * FROM tasks WHERE owner IS NOT NULL AND lease_expires_at<=? AND legacy=0", (utc_now(),)).fetchall()
        changed = []
        for task in tasks:
            self._interrupt_task_tx(connection, task, "lease_expired")
            changed.append(task["task_id"])
        return changed

    def _interrupt_task_tx(self, connection: sqlite3.Connection, task: sqlite3.Row, event_type: str) -> None:
        task_id, unsafe = task["task_id"], False
        for node in connection.execute("SELECT * FROM nodes WHERE task_id=?", (task_id,)).fetchall():
            metadata = json.loads(node["data_json"])
            if node["status"] == "running":
                state = "unknown" if metadata.get("side_effect") else "interrupted"
                unsafe |= state == "unknown"
                connection.execute("UPDATE nodes SET status=?,version=version+1,updated_at=? WHERE task_id=? AND node_id=?", (state, utc_now(), task_id, node["node_id"]))
                connection.execute("UPDATE attempts SET status=?,finished_at=?,certainty='unknown' WHERE task_id=? AND node_id=? AND status='running'", (state, utc_now(), task_id, node["node_id"]))
            if metadata.get("side_effect") and node["status"] in {"unknown", "success", "succeeded", "pending", "waiting_human"}:
                unsafe = True
        state = "recovery_required" if unsafe else "interrupted"
        connection.execute(
            "UPDATE tasks SET status=?,owner=NULL,lease_expires_at=NULL,lease_token=lease_token+1,safe_to_resume=? WHERE task_id=?",
            (state, int(not unsafe), task_id),
        )
        self._bump(connection, task_id)
        self._event(connection, task_id, event_type, data={"status": state, "automatic_replay": False})

    def recover_orphaned_tasks(self) -> list[str]:
        with self._connection() as connection:
            origins = connection.execute("SELECT o.* FROM task_origins o JOIN tasks t ON t.task_id=o.task_id WHERE t.owner IS NULL AND t.status='running' AND t.legacy=0").fetchall()
        dead = []
        for origin in origins:
            alive, birth = self._process_state(origin["process_id"])
            if not alive or (birth is not None and origin["process_birth"] is not None and birth != origin["process_birth"]):
                dead.append(origin)
        if not dead:
            return []
        changed = []
        with self._transaction() as connection:
            for original in dead:
                task = self._require_task(connection, original["task_id"])
                current = connection.execute("SELECT * FROM task_origins WHERE task_id=?", (task["task_id"],)).fetchone()
                if task["owner"] is None and task["status"] == "running" and dict(current) == dict(original):
                    self._interrupt_task_tx(connection, task, "process_exited")
                    changed.append(task["task_id"])
        return changed

    def expire_leases(self) -> list[str]:
        with self._transaction() as connection:
            return self._expire_leases_tx(connection)

    def _expire_deadlines_tx(self, connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute("SELECT task_id FROM tasks WHERE legacy=0 AND owner IS NULL AND status IN ('queued','retry_wait') AND deadline<=?", (utc_now(),)).fetchall()
        task_ids = [row["task_id"] for row in rows]
        for task_id in task_ids:
            connection.execute("UPDATE tasks SET status='failed',error='Task deadline exceeded before dispatch' WHERE task_id=?", (task_id,))
            self._bump(connection, task_id)
            self._event(connection, task_id, "task_deadline_expired", data={"status": "failed"})
        return task_ids

    def expire_deadlines(self) -> list[str]:
        with self._transaction() as connection:
            return self._expire_deadlines_tx(connection)

    def _request(self, task_id: str, *, cancel: bool | None = None, pause: bool | None = None, expected_version: int | None = None) -> dict[str, Any]:
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            if task["legacy"]:
                raise TaskStoreError("Imported legacy tasks are read-only history")
            if expected_version is not None and expected_version != task["version"]:
                raise VersionConflict("Task version changed")
            connection.execute(
                "UPDATE tasks SET cancel_requested=?,pause_requested=? WHERE task_id=?",
                (int(cancel) if cancel is not None else task["cancel_requested"],
                 int(pause) if pause is not None else task["pause_requested"], task_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "task_request", data={"cancel_requested": cancel, "pause_requested": pause})
            return self._task_record(connection, self._require_task(connection, task_id))

    def request_cancel(self, task_id: str, *, expected_version: int | None = None) -> dict[str, Any]:
        return self._request(task_id, cancel=True, expected_version=expected_version)

    def request_pause(self, task_id: str, *, expected_version: int | None = None) -> dict[str, Any]:
        return self._request(task_id, pause=True, expected_version=expected_version)

    def clear_requests(self, task_id: str, *, expected_version: int | None = None) -> dict[str, Any]:
        return self._request(task_id, cancel=False, pause=False, expected_version=expected_version)

    def request_control(
        self, task_id: str, action: str, *, expected_version: int | None = None,
    ) -> dict[str, Any]:
        if action not in {"pause", "cancel"}:
            raise ValueError("Unknown task control action")
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            if task["legacy"] or task["status"] in TERMINAL_STATES:
                raise ValueError("任务不存在或已结束")
            if expected_version is not None and task["version"] != expected_version:
                raise VersionConflict("Task version changed")
            cancel = bool(task["cancel_requested"]) or action == "cancel"
            pause = bool(task["pause_requested"]) or action == "pause"
            state = task["status"]
            # 无 lease 的 running 也可能是仍在执行的同步审批，只发送控制信号。
            settled = task["owner"] is None and state not in {"running", "compensating"}
            if settled:
                state = "cancelled" if cancel else "paused"
            connection.execute(
                "UPDATE tasks SET cancel_requested=?,pause_requested=?,status=?,lease_token=lease_token+? WHERE task_id=?",
                (int(cancel), int(pause), state, int(settled), task_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "task_request",
                        data={"cancel_requested": cancel, "pause_requested": pause, "status": state})
            return self._task_record(connection, self._require_task(connection, task_id))

    def resume_runtime_task(
        self, task_id: str, *, expected_version: int | None = None, max_restarts: int = 3,
    ) -> dict[str, Any]:
        """恢复描述、取消请求和重启预算与入队在同一写事务内校验。"""
        if max_restarts < 1:
            raise ValueError("max_restarts must be positive")
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            if task["legacy"]:
                raise ValueError("该历史任务没有可恢复描述")
            if expected_version is not None and task["version"] != expected_version:
                raise VersionConflict("Task version changed")
            has_jobs = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_jobs'",
            ).fetchone()
            job = connection.execute("SELECT * FROM runtime_jobs WHERE task_id=?", (task_id,)).fetchone() if has_jobs else None
            if job is None:
                raise ValueError("该历史任务没有可恢复描述")
            if task["owner"] is not None or task["cancel_requested"]:
                raise ValueError("任务正在执行或已请求取消，不能恢复")
            if task["status"] not in {"paused", "failed", "interrupted", "partial", "recovery_required"}:
                raise ValueError("当前状态不能恢复；等待审批的任务请先重新预览")
            nodes = [self._node_record(row) for row in connection.execute("SELECT * FROM nodes WHERE task_id=?", (task_id,))]
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_tool_effects'",
            ).fetchone()
            effects = connection.execute("SELECT state FROM runtime_tool_effects WHERE task_id=?", (task_id,)).fetchall() if exists else []
            if any(effect["state"] in {"dispatched", "unknown"} for effect in effects):
                raise ValueError("存在已派发但结果不确定的副作用，禁止自动重放；请人工核对")
            if not nodes and any(effect["state"] == "committed" for effect in effects):
                raise ValueError("该短任务已有提交记录，不能重新执行；请查看任务结果")
            if any(node["status"] in {"unknown", "running"} for node in nodes):
                raise ValueError("存在结果不确定的外部操作，必须先人工核对，禁止自动重放")
            if task["status"] == "recovery_required" and any(
                node.get("side_effect") and node["status"] not in SUCCESS_STATES | {"pending", "waiting_human"}
                for node in nodes
            ):
                raise ValueError("副作用结果需人工核对，不能直接重试")
            remaining = job["timeout_seconds"] - job["elapsed"]
            if remaining <= 0:
                raise ValueError("任务总执行时间预算已耗尽，请新建任务")
            if job["restarts"] >= max_restarts:
                raise ValueError(f"已达到 {max_restarts} 次恢复上限，请检查失败原因后新建任务")
            connection.execute("UPDATE runtime_jobs SET restarts=restarts+1 WHERE task_id=?", (task_id,))
            connection.execute(
                "UPDATE tasks SET status='queued',error='',pause_requested=0,lease_token=lease_token+1,deadline=? WHERE task_id=?",
                (self._lease_deadline(remaining), task_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "task_resumed", data={"restarts": job["restarts"] + 1})
            return self._task_record(connection, self._require_task(connection, task_id))

    def validate_recovery(
        self, task_id: str, *, plan_version: str | None = None, executor_version: str | None = None,
        handler_versions: dict[str, str] | None = None, initial_values: Any = _UNSET,
    ) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            task = self._require_task(connection, task_id)
            reasons: list[str] = []
            if task["legacy"]:
                reasons.append("legacy_history_has_no_reconstructable_plan")
            if plan_version is not None and str(plan_version) != task["plan_version"]:
                reasons.append("plan_version_changed")
            if executor_version is not None and str(executor_version) != task["executor_version"]:
                reasons.append("executor_version_changed")
            try:
                plan = self._load_ref(connection, task["plan_ref"])
                inputs = self._load_ref(connection, task["input_ref"])
            except (TaskStoreError, ValueError):
                plan, inputs = None, None
                reasons.append("plan_or_input_reference_invalid")
            if plan is None:
                reasons.append("missing_plan")
            if initial_values is not _UNSET and digest(initial_values) != digest(inputs):
                reasons.append("task_input_changed")
            nodes = {row["node_id"]: row for row in connection.execute("SELECT * FROM nodes WHERE task_id=?", (task_id,))}
            checkpoints = {row["node_id"]: row for row in connection.execute("SELECT * FROM checkpoints WHERE task_id=?", (task_id,))}
            active_ids = {step["step_id"] for step in plan["steps"]} if plan is not None else set(nodes)
            invalid: dict[str, list[str]] = {}
            reusable, resumable, bindings = [], [], []
            for node_id in active_ids:
                node = nodes.get(node_id)
                if node is None:
                    invalid[node_id] = ["missing_node"]
                    continue
                metadata = json.loads(node["data_json"])
                if metadata.get("side_effect") and node["status"] in {"running", "unknown", "success", "succeeded", "waiting_human"}:
                    reasons.append("side_effect_requires_reconciliation:" + node_id)
                if metadata.get("requires_human") and node["status"] in {"pending", "waiting_human"}:
                    reasons.append("approval_required:" + node_id)
                if handler_versions is not None:
                    if handler_versions.get(metadata["handler_key"]) != metadata["handler_version"]:
                        reasons.append("handler_unavailable_or_changed:" + node_id)
                else:
                    bindings.append(metadata["handler_key"])
                checkpoint = checkpoints.get(node_id)
                if checkpoint is None:
                    if node["status"] in SUCCESS_STATES:
                        invalid[node_id] = ["missing_checkpoint"]
                    continue
                failures = []
                if checkpoint["plan_version"] != task["plan_version"] or "plan_version_changed" in reasons:
                    failures.append("plan_version_changed")
                if checkpoint["executor_version"] != task["executor_version"] or "executor_version_changed" in reasons:
                    failures.append("executor_version_changed")
                if checkpoint["handler_version"] != metadata["handler_version"]:
                    failures.append("handler_version_changed")
                if handler_versions is not None and handler_versions.get(metadata["handler_key"]) != checkpoint["handler_version"]:
                    failures.append("handler_version_changed")
                if checkpoint["input_digest"] != node["input_digest"] or "task_input_changed" in reasons:
                    failures.append("node_input_changed")
                if not json.loads(checkpoint["validation_json"]).get("ok", False):
                    failures.append("validation_failed")
                try:
                    arguments = self._load_ref(connection, node["input_ref"])
                    if self._input_digest(inputs, arguments, metadata) != node["input_digest"]:
                        failures.append("input_reference_changed")
                    output = self._load_ref(connection, checkpoint["output_ref"])
                    self._load_ref(connection, checkpoint["payload_ref"])
                    if checkpoint["output_ref"] != node["output_ref"]:
                        failures.append("output_reference_changed")
                    if checkpoint["output_ref"] is not None and digest(output) != checkpoint["output_digest"]:
                        failures.append("output_digest_changed")
                    if checkpoint["output_ref"] is None and node["status"] in SUCCESS_STATES:
                        failures.append("missing_output")
                except (TaskStoreError, ValueError):
                    failures.append("object_reference_invalid")
                artifacts = json.loads(checkpoint["artifacts_json"])
                current = self._fingerprints([item["path"] for item in artifacts])
                for original, actual in zip(artifacts, current):
                    if original.get("sha256") is None or original.get("sha256") != actual.get("sha256"):
                        failures.append("artifact_missing_or_changed:" + original["path"])
                if failures:
                    invalid[node_id] = list(dict.fromkeys(failures))
                elif node["status"] in SUCCESS_STATES:
                    reusable.append(node_id)
                elif checkpoint["payload_ref"] is not None:
                    resumable.append(node_id)
            # 依赖输出失效会连带使下游失效，不能仅检查节点自身参数摘要。
            changed = True
            while changed:
                changed = False
                for node_id in [*reusable, *resumable]:
                    dependencies = json.loads(nodes[node_id]["data_json"])["depends_on"]
                    if any(dependency not in reusable for dependency in dependencies):
                        invalid[node_id] = ["dependency_checkpoint_invalid"]
                        if node_id in reusable:
                            reusable.remove(node_id)
                        if node_id in resumable:
                            resumable.remove(node_id)
                        changed = True
            if task["owner"] is not None and task["lease_expires_at"] is not None and task["lease_expires_at"] > utc_now():
                reasons.append("active_worker_lease")
            reasons = list(dict.fromkeys(reasons))
            return {
                "task_id": task_id, "can_resume": not reasons, "safe_to_resume": not reasons,
                "reusable_nodes": sorted(reusable), "resumable_nodes": sorted(resumable),
                "invalid_nodes": invalid, "reasons": reasons,
                "requires_handler_binding": sorted(set(bindings)),
            }

    def recovery_candidates(self) -> list[dict[str, Any]]:
        self.expire_leases()
        self.recover_orphaned_tasks()
        records = self.list_tasks(statuses=RECOVERY_STATES)
        for record in records:
            validation = self.validate_recovery(record["task_id"])
            record["safe_to_resume"] = validation["safe_to_resume"]
            record["recovery"] = validation
        return records

    @staticmethod
    def _approval_digest(arguments: Any) -> str:
        paths = arguments.get("attachment_paths", []) if isinstance(arguments, dict) else []
        if isinstance(paths, str):
            paths = [paths]
        return digest({"arguments": arguments, "attachments": TaskStore._fingerprints(paths)})

    def save_approval(
        self, task_id: str, node_id: str, *, arguments: Any, approval_id: str | None = None,
        preview: Any = None, expires_at: str | datetime | None = None, risk: str = "",
        plan_version: str | None = None, **guard: Any,
    ) -> dict[str, Any]:
        arguments = serializable(arguments)
        content_digest = self._approval_digest(arguments)
        approval_id = approval_id or "approval_" + secrets.token_urlsafe(18)
        now = utc_now()
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            self._guard(task, **guard)
            self._require_node(connection, task_id, node_id)
            if plan_version is not None and str(plan_version) != task["plan_version"]:
                raise ApprovalConflict("Approval plan version is stale")
            arguments_ref = self._put_ref(connection, task_id, "approval_arguments", arguments, node_id)
            preview_ref = self._put_ref(connection, task_id, "approval_preview", preview, node_id) if preview is not None else None
            connection.execute(
                """INSERT INTO approvals(approval_id,task_id,node_id,session_id,plan_version,content_digest,
                arguments_ref,preview_ref,risk,expires_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (approval_id, task_id, node_id, task["session_id"], task["plan_version"], content_digest,
                 arguments_ref, preview_ref, sanitize_text(risk), timestamp(expires_at) if expires_at is not None else None, now, now),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "approval_created", node_id=node_id,
                        data={"approval_id": approval_id, "content_digest": content_digest})
            return self._approval_record(connection, connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone(), include_content=True)

    def _approval_record(self, connection: sqlite3.Connection, row: sqlite3.Row, *, include_content: bool) -> dict[str, Any]:
        record = dict(row)
        if include_content:
            record["arguments"] = self._load_ref(connection, row["arguments_ref"])
            record["preview"] = self._load_ref(connection, row["preview_ref"])
        return record

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            return self._approval_record(connection, row, include_content=True) if row is not None else None

    def list_approvals(
        self, task_id: str | None = None, *, session_id: str | None = None, status: str | None = None,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        clauses, arguments = [], []
        for key, value in (("task_id", task_id), ("session_id", session_id), ("status", status)):
            if value is not None:
                clauses.append(key + "=?")
                arguments.append(value)
        sql = "SELECT * FROM approvals" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at,approval_id"
        with self._connection() as connection:
            connection.execute("BEGIN")
            return [self._approval_record(connection, row, include_content=include_content) for row in connection.execute(sql, arguments).fetchall()]

    def _check_approval(
        self, connection: sqlite3.Connection, row: sqlite3.Row, *, session_id: str | None = None,
        plan_version: str | None = None, content_digest: str | None = None, expected_version: int | None = None,
    ) -> sqlite3.Row:
        task = self._require_task(connection, row["task_id"])
        if task["legacy"] or row["plan_version"] != task["plan_version"]:
            raise ApprovalConflict("Approval belongs to an old plan")
        if task["cancel_requested"] or task["status"] in TERMINAL_STATES:
            raise ApprovalConflict("Task was cancelled or finished")
        if row["expires_at"] is not None and row["expires_at"] <= utc_now():
            raise ApprovalConflict("Approval expired")
        if session_id is not None and row["session_id"] != session_id:
            raise ApprovalConflict("Approval session mismatch")
        if plan_version is not None and row["plan_version"] != str(plan_version):
            raise ApprovalConflict("Approval plan version mismatch")
        if expected_version is not None and row["version"] != expected_version:
            raise VersionConflict("Approval version changed")
        arguments = self._load_ref(connection, row["arguments_ref"])
        if self._approval_digest(arguments) != row["content_digest"] or (content_digest is not None and content_digest != row["content_digest"]):
            raise ApprovalConflict("Approval arguments or attachments changed")
        return task

    def _transition_approval(
        self, approval_id: str, status: str, required_status: str, *, task_status: str | None = None,
        node_status: str | None = None, **checks: Any,
    ) -> dict[str, Any]:
        if task_status is not None and task_status not in TASK_STATES:
            raise ValueError("Unknown task state")
        if node_status is not None and node_status not in NODE_STATES:
            raise ValueError("Unknown node state")
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            if row is None:
                raise KeyError(approval_id)
            if row["status"] != required_status:
                raise ApprovalConflict("Approval was already decided or consumed")
            task = self._check_approval(connection, row, **checks)
            if (task_status is not None or node_status is not None) and task["owner"] is not None:
                raise LeaseConflict("Cannot dispatch an approval while another worker owns the task")
            connection.execute("UPDATE approvals SET status=?,version=version+1,updated_at=? WHERE approval_id=?", (status, utc_now(), approval_id))
            if task_status is not None:
                connection.execute("UPDATE tasks SET status=? WHERE task_id=?", (task_status, row["task_id"]))
            if node_status is not None:
                connection.execute("UPDATE nodes SET status=?,version=version+1,updated_at=? WHERE task_id=? AND node_id=?", (node_status, utc_now(), row["task_id"], row["node_id"]))
            self._bump(connection, row["task_id"])
            self._event(connection, row["task_id"], "approval_" + status, node_id=row["node_id"],
                        data={"approval_id": approval_id, "status": status, "task_status": task_status, "node_status": node_status})
            return self._approval_record(connection, connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone(), include_content=True)

    def decide_approval(self, approval_id: str, decision: str, **checks: Any) -> dict[str, Any]:
        status = {"approve": "approved", "reject": "rejected", "cancel": "cancelled"}.get(decision, decision)
        if status not in {"approved", "rejected", "cancelled"}:
            raise ValueError("Decision must be approved, rejected, or cancelled")
        return self._transition_approval(approval_id, status, "pending", **checks)

    def consume_approval(self, approval_id: str, **checks: Any) -> dict[str, Any]:
        return self._transition_approval(approval_id, "consumed", "approved", **checks)

    def claim_desktop_approval(
        self, action_id: str, session_id: str, task_id: str, node_id: str, plan_version: str | int,
    ) -> dict[str, Any]:
        """桌面审批消费与执行意图原子提交；文件核验由审批服务在调用前完成。"""
        with self._transaction() as connection:
            task = self._require_task(connection, task_id)
            if task["status"] != "waiting_human" or task["plan_version"] != str(plan_version):
                raise ApprovalConflict("Task is not waiting for this approval plan")
            self._guard(task)
            if task["session_id"] != session_id or task["cancel_requested"] or task["pause_requested"]:
                raise ApprovalConflict("Approval session mismatch or task stop requested")
            node = self._require_node(connection, task_id, node_id)
            if node["status"] not in {"pending", "waiting_human"}:
                raise ApprovalConflict("Node is not waiting for approval")
            try:
                action = connection.execute("SELECT * FROM desktop_approvals WHERE action_id=?", (action_id,)).fetchone()
            except sqlite3.OperationalError as exc:
                raise ApprovalConflict("Desktop approval table is not initialized") from exc
            if action is None or action["state"] != "pending" or action["session_id"] != session_id or action["task_id"] != task_id:
                raise ApprovalConflict("Desktop approval is missing, consumed, or belongs to another task")
            payload = json.loads(action["payload"])
            if payload.get("node_id") != node_id or str(payload.get("plan_version")) != str(plan_version):
                raise ApprovalConflict("Desktop approval node or plan mismatch")
            if float(payload.get("expires_at", 0)) <= datetime.now(timezone.utc).timestamp():
                raise ApprovalConflict("Desktop approval expired")
            if connection.execute("SELECT 1 FROM attempts WHERE task_id=? AND node_id=? AND status='running'", (task_id, node_id)).fetchone():
                raise ApprovalConflict("Node already has a running attempt")
            changed = connection.execute(
                "UPDATE desktop_approvals SET state='consumed' WHERE action_id=? AND session_id=? AND task_id=? AND state='pending'",
                (action_id, session_id, task_id),
            ).rowcount
            if changed != 1:
                raise ApprovalConflict("Desktop approval was concurrently consumed")
            owner = f"approval_{os.getpid()}_{secrets.token_hex(8)}"
            lease_token = task["lease_token"] + 1
            expires = self._lease_deadline(15)
            attempt_id = self._new_attempt_tx(connection, task_id, node_id, owner, lease_token)
            metadata = json.loads(node["data_json"])
            metadata["side_effect"] = True
            connection.execute(
                "UPDATE nodes SET status='running',data_json=?,version=version+1,updated_at=? WHERE task_id=? AND node_id=?",
                (json_data(metadata), utc_now(), task_id, node_id),
            )
            connection.execute(
                "UPDATE tasks SET status='running',safe_to_resume=0,owner=?,lease_token=?,lease_expires_at=? WHERE task_id=?",
                (owner, lease_token, expires, task_id),
            )
            self._bump(connection, task_id)
            self._event(connection, task_id, "desktop_approval_dispatched", node_id=node_id, attempt_id=attempt_id,
                        data={"action_id": action_id, "plan_version": str(plan_version), "lease_token": lease_token})
            self._event(connection, task_id, "attempt_started", node_id=node_id, attempt_id=attempt_id)
            result = self._attempt_record(connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())
            result["lease_expires_at"] = expires
            return result

    def migrate_legacy_json(self, path: str | Path) -> int:
        source = Path(path).resolve()
        if not source.is_file():
            return 0
        with self._connection() as connection:
            if connection.execute("SELECT 1 FROM legacy_imports WHERE source_path=?", (str(source),)).fetchone():
                return 0
        try:
            content = source.read_bytes()
            records = json.loads(content.decode("utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise TaskStoreError("Legacy JSON is unreadable; source was not modified") from exc
        if not isinstance(records, list):
            raise TaskStoreError("Legacy JSON must contain a list; source was not modified")
        imported = 0
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM legacy_imports WHERE source_path=?", (str(source),)).fetchone():
                return 0
            for value in records:
                if not isinstance(value, dict) or not isinstance(value.get("task_id"), str) or not value["task_id"]:
                    continue
                record = serializable(value)
                task_id = record["task_id"]
                if connection.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone():
                    continue
                now = utc_now()
                state = record.get("status", "interrupted")
                state = "interrupted" if state == "running" or state not in TASK_STATES else state
                connection.execute(
                    """INSERT INTO tasks(task_id,session_id,goal,goal_digest,plan_version,executor_version,status,
                    created_at,updated_at,error,legacy,safe_to_resume) VALUES(?,?,?,?,?,?,?,?,?,?,1,0)""",
                    (task_id, str(record.get("session_id", "")), str(record.get("goal", "")), str(record.get("goal_digest", "")),
                     "legacy", "unknown", state, now, now, sanitize_text(str(record.get("error", "")))[:1000]),
                )
                nodes = record.get("nodes", {})
                if isinstance(nodes, dict):
                    for node_id, node in nodes.items():
                        if not isinstance(node, dict):
                            continue
                        summary = {key: node.get(key, default) for key, default in (
                            ("agent", "executor"), ("depends_on", []), ("allowed_tools", []), ("requires_human", False),
                        )}
                        summary.update(handler_key="", handler_version="unknown", side_effect=bool(summary["requires_human"]))
                        node_status = node.get("status", "pending")
                        if node_status == "running" or node_status not in NODE_STATES:
                            node_status = "interrupted"
                        connection.execute(
                            """INSERT INTO nodes(task_id,node_id,status,data_json,input_digest,error,artifact_paths_json,updated_at)
                            VALUES(?,?,?,?,?,?,?,?)""",
                            (task_id, node_id, node_status, json_data(summary), "", sanitize_text(str(node.get("error", "")))[:1000],
                             json_data(node.get("artifact_paths", [])), now),
                        )
                self._event(connection, task_id, "legacy_imported", data={"history_only": True})
                imported += 1
            connection.execute("INSERT INTO legacy_imports VALUES(?,?,?)", (str(source), hashlib.sha256(content).hexdigest(), utc_now()))
        return imported
