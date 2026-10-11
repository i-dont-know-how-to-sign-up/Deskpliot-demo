from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..core.sqlite_utils import sqlite_transaction
from .control import (
    ExecutionControl, TaskCancelled, TaskDeadlineExceeded, TaskPaused, execution_scope,
)
from .models import LeaseConflict, VersionConflict, json_data, sanitize_text, utc_now
from .retry import UnknownSideEffect
from .store import TaskStore


_LOGGER = logging.getLogger(__name__)


class TaskService:
    """单 worker 的持久队列；UI 是订阅者，不拥有任务的唯一状态。"""

    def __init__(self, store: TaskStore, *, queue_limit: int = 50) -> None:
        self.store = store
        self.queue_limit = max(1, queue_limit)
        self.owner = f"worker_{os.getpid()}_{secrets.token_hex(6)}"
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._active: dict[str, ExecutionControl] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self.on_result: Callable[[str, Any], None] | None = None
        self.on_error: Callable[[str, str], None] | None = None
        with sqlite_transaction(store.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS runtime_jobs "
                               "(task_id TEXT PRIMARY KEY, handler_key TEXT NOT NULL, payload TEXT NOT NULL, "
                               "timeout_seconds REAL NOT NULL, elapsed REAL NOT NULL DEFAULT 0, "
                               "usage_json TEXT NOT NULL DEFAULT '{}', restarts INTEGER NOT NULL DEFAULT 0)")
            connection.execute("CREATE TABLE IF NOT EXISTS runtime_worker_lock "
                               "(id INTEGER PRIMARY KEY CHECK(id=1), owner TEXT, expires REAL NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS runtime_metrics "
                               "(task_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, "
                               "PRIMARY KEY(task_id,sequence))")

    def register(self, key: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.handlers[key] = handler

    def bind(self, task_id: str, key: str, payload: dict[str, Any], *, timeout_seconds: float = 600) -> None:
        if key not in self.handlers:
            raise ValueError(f"Unregistered task handler: {key}")
        encoded = json_data(payload)
        with sqlite_transaction(self.store.path) as connection:
            connection.execute("INSERT INTO runtime_jobs(task_id,handler_key,payload,timeout_seconds) VALUES(?,?,?,?) "
                               "ON CONFLICT(task_id) DO UPDATE SET handler_key=excluded.handler_key,payload=excluded.payload",
                               (task_id, key, encoded, timeout_seconds))

    def descriptor(self, task_id: str) -> dict[str, Any] | None:
        with sqlite_transaction(self.store.path) as connection:
            row = connection.execute("SELECT * FROM runtime_jobs WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        value["usage"] = json.loads(value.pop("usage_json"))
        with sqlite_transaction(self.store.path) as connection:
            events = [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM runtime_metrics WHERE task_id=? AND "
                "json_extract(payload,'$.kind') IN ('pdf_started','pdf_checkpoint') ORDER BY sequence", (task_id,))]
        with self._lock:
            active = self._active.get(task_id)
        if active is not None:
            value["usage"] = active.snapshot_usage()
            value["elapsed"] = active.base_elapsed + time.monotonic() - active.started_at
            with active._lock:
                events.extend(event for event in active.events if event["kind"] in {"pdf_started", "pdf_checkpoint"})
        progress: dict[str, Any] = {}
        completed_pages: set[int] = set()
        reused_pages: set[int] = set()
        for event in events:
            if event["kind"] == "pdf_started":
                progress = {"total": event["total_pages"], "completed": 0, "reused": 0}
                completed_pages.clear()
                reused_pages.clear()
            elif progress:
                completed_pages.add(event["page"])
                if event.get("reused"):
                    reused_pages.add(event["page"])
                progress.update(completed=len(completed_pages), reused=len(reused_pages))
        value["pdf_progress"] = progress
        return value

    def persist_metrics(self, control: ExecutionControl) -> None:
        with control._lock:
            events = list(control.events)
        guard = getattr(control, "ledger_guard", {})
        with sqlite_transaction(self.store.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute("SELECT owner,lease_token,lease_expires_at FROM tasks WHERE task_id=?",
                                      (control.task_id,)).fetchone()
            if task is None or task["lease_token"] != guard.get("lease_token", task["lease_token"]):
                return
            if task["owner"] is not None and (
                task["owner"] != guard.get("owner") or not task["lease_expires_at"] or task["lease_expires_at"] <= utc_now()
            ):
                return
            base = getattr(control, "event_base", 0)
            for index, event in enumerate(events, base + 1):
                connection.execute("INSERT OR IGNORE INTO runtime_metrics VALUES(?,?,?)",
                                   (control.task_id, index, json_data(event)))
            elapsed = control.base_elapsed + time.monotonic() - control.started_at
            connection.execute("UPDATE runtime_jobs SET elapsed=?,usage_json=? WHERE task_id=?",
                               (elapsed, json_data(control.snapshot_usage()), control.task_id))

    def initialize_metrics(self, control: ExecutionControl) -> None:
        job = self.descriptor(control.task_id)
        control.base_elapsed = job["elapsed"] if job else 0
        control.started_at = time.monotonic()
        with sqlite_transaction(self.store.path) as connection:
            control.event_base = int(connection.execute("SELECT COALESCE(MAX(sequence),0) FROM runtime_metrics WHERE task_id=?",
                                                       (control.task_id,)).fetchone()[0])

    def submit(self, key: str, payload: dict[str, Any], *, session_id: str = "",
               timeout_seconds: float = 600) -> str:
        if key not in self.handlers or not 0 < timeout_seconds <= 86400:
            raise ValueError("Unknown handler or invalid task timeout")
        with self._lock:
            if len(self.store.list_tasks(statuses={"queued", "running", "retry_wait"})) >= self.queue_limit:
                raise ValueError("任务队列已满，请先处理已有任务")
            # 先保存不可派发状态，再落恢复描述，防止独立 worker 读到半初始化任务。
            task_id = self.store.create_task({"goal": payload.get("question", key), "steps": []},
                                             session_id=session_id, status="paused")
            self.bind(task_id, key, payload, timeout_seconds=timeout_seconds)
            self.store.update_task(task_id, "queued")
            self._wake.set()
            return task_id

    def request(self, task_id: str, action: str) -> None:
        with self._lock:
            try:
                task = self.store.request_control(task_id, action)
            except KeyError as exc:
                raise ValueError("任务不存在或已结束") from exc
            control = self._active.get(task_id)
            if control:
                if task["cancel_requested"]:
                    control.request_cancel()
                elif task["pause_requested"]:
                    control.request_pause()
        self._wake.set()

    def resume(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError("该历史任务没有可恢复描述")
        self.store.resume_runtime_task(task_id, expected_version=task["version"])
        self._wake.set()

    def _worker_lease(self) -> bool:
        with sqlite_transaction(self.store.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner,expires FROM runtime_worker_lock WHERE id=1").fetchone()
            if row is not None and row[0] != self.owner and row[1] > time.time():
                return False
            connection.execute("INSERT INTO runtime_worker_lock VALUES(1,?,?) "
                               "ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,expires=excluded.expires",
                               (self.owner, time.time() + 15))
            return True

    def _heartbeat(self, task_id: str, token: int, control: ExecutionControl, done: threading.Event) -> None:
        tick = 0
        while not done.wait(0.5):
            try:
                task = self.store.get_task(task_id)
                if task is None or task["lease_token"] != token:
                    control.cancel()
                    return
                if task["owner"] is None and task["status"] != "running":
                    return
                if (task["owner"] != self.owner or not task["lease_expires_at"]
                        or task["lease_expires_at"] <= utc_now()):
                    control.cancel()
                    return
                if task["cancel_requested"]:
                    control.cancel()
                if task["pause_requested"]:
                    control.pause()
                tick += 1
                if tick % 10 == 0:
                    self.persist_metrics(control)
                    if not self._worker_lease():
                        control.cancel()
                        return
                    self.store.heartbeat(task_id, self.owner, token)
            except Exception as exc:
                control.cancel()
                _LOGGER.warning("Task heartbeat stopped for %s: %s", task_id, sanitize_text(str(exc)))
                return

    @staticmethod
    def _notify(callback: Callable | None, task_id: str, value: Any) -> None:
        if callback is not None:
            try:
                callback(task_id, value)
            except Exception as exc:
                _LOGGER.warning("Task observer failed for %s: %s", task_id, sanitize_text(str(exc)))

    def _finish_failure(self, control: ExecutionControl, status: str, error: str) -> None:
        try:
            current = self.store.get_task(control.task_id)
            if current is None or current["lease_token"] != control.ledger_guard["lease_token"]:
                control.cancel()
                return
            if current["status"] != "running":
                return
            self.store.finish_worker_task(control.task_id, status, error, **control.ledger_guard)
        except LeaseConflict:
            # 失联 worker 只能停止，不得去掉 guard 重试或覆盖新 owner 的结果。
            control.cancel()

    def run_once(self) -> bool:
        if not self._worker_lease():
            return False
        self.store.expire_leases()
        task = self.store.claim_next(self.owner)
        if task is None:
            return False
        task_id = task["task_id"]
        job = self.descriptor(task_id)
        if job is None or job["handler_key"] not in self.handlers:
            try:
                self.store.finish_worker_task(task_id, "recovery_required", "恢复 handler 未注册，未执行任何操作",
                                              owner=self.owner, lease_token=task["lease_token"])
            except LeaseConflict:
                pass
            return True
        remaining = max(0, job["timeout_seconds"] - job["elapsed"])
        control = ExecutionControl(task_id=task_id, deadline=time.monotonic() + remaining,
                                   max_tool_calls=40, max_tokens=80000, max_attempts=40)
        control.usage.update(job["usage"])
        control.ledger_guard = {"owner": self.owner, "lease_token": task["lease_token"]}
        control.task_store = self.store
        control.base_elapsed = job["elapsed"]
        control.started_at = time.monotonic()
        with self._lock:
            self._active[task_id] = control
        done = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(task_id, task["lease_token"], control, done), daemon=True)
        try:
            self.initialize_metrics(control)
            self.store.update_task(task_id, deadline=datetime.now(timezone.utc) + timedelta(seconds=remaining), **control.ledger_guard)
            heartbeat.start()
            with execution_scope(control):
                control.checkpoint()
                result = self.handlers[job["handler_key"]](job["payload"])
            current = self.store.get_task(task_id)
            if current is None:
                raise LeaseConflict("Worker task no longer exists")
            if current["status"] == "running":
                status = "waiting_human" if getattr(result, "pending_action", None) else "succeeded"
                steps = getattr(result, "steps", [])
                supervisor_state = next((getattr(step, "status", "") for step in reversed(steps)
                                         if getattr(step, "name", "") == "supervisor"), "")
                if supervisor_state in {"paused", "cancelled", "recovery_required", "waiting_human", "failed"}:
                    status = supervisor_state
                elif any(getattr(step, "status", "") in {"failed", "unknown"} for step in steps):
                    status = "failed"
                self.store.finish_worker_task(task_id, status, output=result, **control.ledger_guard)
            else:
                self.store.save_released_output(task_id, result, expected_version=current["version"], **control.ledger_guard)
        except (LeaseConflict, VersionConflict) as exc:
            control.cancel()
            _LOGGER.warning("Worker result fenced for %s: %s", task_id, sanitize_text(str(exc)))
            self._notify(self.on_error, task_id, "执行权已变化，旧结果未回写；请查看 Tasks")
        except (TaskCancelled, TaskPaused, TaskDeadlineExceeded) as exc:
            status = "cancelled" if isinstance(exc, TaskCancelled) else "paused" if isinstance(exc, TaskPaused) else "failed"
            self._finish_failure(control, status, str(exc))
            self._notify(self.on_error, task_id, sanitize_text(str(exc)))
        except Exception as exc:
            current = self.store.get_task(task_id)
            unsafe = isinstance(exc, UnknownSideEffect) or any(
                node["status"] in {"unknown", "running"} and node.get("side_effect")
                for node in (current["nodes"].values() if current else [])
            )
            self._finish_failure(control, "recovery_required" if unsafe else "failed", str(exc))
            self._notify(self.on_error, task_id, sanitize_text(str(exc)))
        else:
            self._notify(self.on_result, task_id, result)
        finally:
            done.set()
            try:
                if heartbeat.ident is not None:
                    heartbeat.join(timeout=2)
                self.persist_metrics(control)
            except Exception as exc:
                _LOGGER.warning("Task metrics failed for %s: %s", task_id, sanitize_text(str(exc)))
            finally:
                with self._lock:
                    self._active.pop(task_id, None)
        return True

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self.run_forever, name="deskpilot-task-worker", daemon=True)
        self._worker.start()

    def run_forever(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    self._wake.wait(0.5)
                    self._wake.clear()
                failures = 0
            except Exception as exc:
                failures += 1
                self._notify(self.on_error, "", sanitize_text(f"worker failure {failures}/3: {exc}"))
                if failures >= 3:
                    self._stop.set()
                else:
                    self._stop.wait(min(2 ** failures, 5))

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            for task_id, control in self._active.items():
                self.store.request_pause(task_id)
                control.pause()
        if self._worker is not None:
            self._worker.join(timeout=3)
        # 仍在外部调用中的 worker 不能提前释放锁，等待 lease 失效并核对结果。
        if not self._active:
            with sqlite_transaction(self.store.path) as connection:
                connection.execute("DELETE FROM runtime_worker_lock WHERE owner=?", (self.owner,))
