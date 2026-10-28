from __future__ import annotations

import hashlib
from typing import Any

from ..core.sqlite_utils import sqlite_transaction
from .models import digest, json_data
from .retry import UnknownSideEffect


def _database(control: Any):
    store = getattr(control, "task_store", None)
    return store.path if store is not None else None


def begin_effect(control: Any, name: str, arguments: dict[str, Any]) -> str:
    path = _database(control)
    side_effect = name.startswith(("files.write", "files.delete", "email.send", "email.save_draft", "code.execute", "shell.execute", "files.apply_move"))
    if path is None or not side_effect:
        return ""
    ids = control.attribution()
    input_digest = digest(arguments)
    operation_id = hashlib.sha256(json_data({"task_id": ids["task_id"], "node_id": ids["node_id"],
                                             "tool": name, "input_digest": input_digest}).encode()).hexdigest()
    with sqlite_transaction(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS runtime_tool_effects "
                           "(operation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, node_id TEXT NOT NULL, "
                           "tool_name TEXT NOT NULL, input_digest TEXT NOT NULL, state TEXT NOT NULL, result_json TEXT NOT NULL)")
        previous = connection.execute("SELECT state FROM runtime_tool_effects WHERE operation_id=?",
                                      (operation_id,)).fetchone()
        if previous and previous["state"] == "dispatched":
            raise UnknownSideEffect(f"Side effect {operation_id[:12]} was interrupted after dispatch; manual review required")
        if previous and previous["state"] in {"compensating", "compensation_unknown"}:
            raise UnknownSideEffect(f"Side effect {operation_id[:12]} has an unfinished compensation; manual review required")
        if previous and previous["state"] in {"committed", "unknown", "compensated"}:
            raise RuntimeError(f"副作用 {operation_id[:12]} 已{previous['state']}，不能自动重复提交")
        if previous:
            connection.execute("UPDATE runtime_tool_effects SET state='dispatched' WHERE operation_id=?", (operation_id,))
        else:
            connection.execute("INSERT INTO runtime_tool_effects VALUES(?,?,?,?,?,'dispatched','{}')",
                               (operation_id, ids["task_id"], ids["node_id"], name, input_digest))
    control.active_operation_id = operation_id
    return operation_id


def finish_effect(control: Any, operation_id: str, result: Any) -> None:
    path = _database(control)
    if not operation_id or path is None:
        return
    state = "prepared" if result.status == "waiting_human" else "committed" if result.status == "success" else "unknown" if result.status == "unknown" else "failed"
    # 任意代码失败也可能修改过本地状态，不能靠非零退出码推导“没有副作用”。
    if result.tool_name.startswith(("shell.execute", "code.execute")) and state == "failed":
        state = "unknown"
    payload = result.output if isinstance(result.output, dict) else {}
    summary = {key: payload[key] for key in (
        "path", "source", "destination", "status", "returncode", "written", "sent_copy_saved",
        "sent_copy_error", "message_id", "refused_recipients", "before_sha256", "after_sha256",
        "backup_path", "compensatable",
    ) if key in payload}
    with sqlite_transaction(path) as connection:
        connection.execute("UPDATE runtime_tool_effects SET state=?,result_json=? WHERE operation_id=?",
                           (state, json_data(summary), operation_id))
    if getattr(control, "active_operation_id", "") == operation_id:
        control.active_operation_id = ""


def effects_for_task(store: Any, task_id: str) -> list[dict[str, Any]]:
    with sqlite_transaction(store.path) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_tool_effects'").fetchone()
        if not exists:
            return []
        return [dict(row) for row in connection.execute(
            "SELECT * FROM runtime_tool_effects WHERE task_id=? ORDER BY rowid DESC", (task_id,)
        )]
