from __future__ import annotations

import json
import hashlib
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import AgentResult, TaskPlan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskLedger:
    """持久化复杂任务的计划和节点状态，不保存工具敏感参数。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._mark_interrupted_tasks()

    def start(self, plan: TaskPlan) -> str:
        task_id = f"task_{secrets.token_urlsafe(12)}"
        record = {
            "task_id": task_id,
            "goal": " -> ".join(step.step_id for step in plan.steps),
            "goal_digest": hashlib.sha256(plan.goal.encode("utf-8")).hexdigest()[:16],
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "nodes": {
                step.step_id: {
                    "agent": step.agent,
                    "status": "pending",
                    "depends_on": list(step.depends_on),
                    "allowed_tools": list(step.allowed_tools),
                    "requires_human": step.requires_human,
                }
                for step in plan.steps
            },
            "safe_to_resume": True,
        }
        self._upsert(record)
        return task_id

    def update_node(self, task_id: str, result: AgentResult) -> None:
        records = self._read()
        record = next((item for item in records if item.get("task_id") == task_id), None)
        if record is None:
            return
        node = record.setdefault("nodes", {}).setdefault(result.step_id, {})
        node.update({
            "status": result.status,
            "error": result.error[:1000],
            "tool_calls": result.tool_calls,
            "tokens": result.tokens,
            "artifact_paths": self._artifact_paths(result.output),
        })
        if node.get("requires_human") or result.status == "pending":
            # 外部副作用节点在重启后必须重新获得用户确认，不能自动重放。
            record["safe_to_resume"] = False
        record["updated_at"] = _now()
        self._write(records)

    def finish(self, task_id: str, status: str, error: str = "") -> None:
        records = self._read()
        record = next((item for item in records if item.get("task_id") == task_id), None)
        if record is None:
            return
        record["status"] = status
        record["error"] = str(error)[:1000]
        record["updated_at"] = _now()
        self._write(records)

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._read()

    def recovery_candidates(self) -> list[dict[str, Any]]:
        """返回可由 UI 展示的中断任务；是否重试仍需重新规划并重新授权。"""
        return [
            item for item in self._read()
            if item.get("status") in {"interrupted", "failed"}
        ]

    def _mark_interrupted_tasks(self) -> None:
        records = self._read()
        changed = False
        for record in records:
            if record.get("status") != "running":
                continue
            record["status"] = "interrupted"
            record["updated_at"] = _now()
            nodes = record.get("nodes", {})
            unsafe = any(
                isinstance(node, dict)
                and node.get("requires_human")
                and node.get("status") in {"success", "pending"}
                for node in nodes.values()
            ) if isinstance(nodes, dict) else True
            record["safe_to_resume"] = not unsafe
            changed = True
        if changed:
            self._write(records)

    @staticmethod
    def _artifact_paths(output: dict[str, Any]) -> list[str]:
        values = []
        for key in ("artifact_path", "path", "resolved_paths"):
            raw = output.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw if str(item).strip())
            elif raw:
                values.append(str(raw))
        return list(dict.fromkeys(values))[:20]

    def _upsert(self, record: dict[str, Any]) -> None:
        records = self._read()
        records.append(record)
        self._write(records[-200:])

    def _read(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.is_file():
                return []
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                return value if isinstance(value, list) else []
            except (OSError, ValueError):
                return []

    def _write(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
