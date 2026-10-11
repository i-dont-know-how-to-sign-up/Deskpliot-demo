from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


TASK_STATES = frozenset({
    "queued", "running", "retry_wait", "waiting_human", "paused",
    "recovery_required", "compensating", "succeeded", "success", "failed",
    "cancelled", "partial", "interrupted",
})
NODE_STATES = TASK_STATES | {"pending", "blocked", "unknown", "skipped"}
TERMINAL_STATES = frozenset({"succeeded", "success", "failed", "cancelled", "partial"})
SUCCESS_STATES = frozenset({"succeeded", "success"})
RECOVERY_STATES = frozenset({"interrupted", "failed", "recovery_required", "paused", "partial"})


class TaskStoreError(RuntimeError):
    pass


class VersionConflict(TaskStoreError):
    pass


class LeaseConflict(TaskStoreError):
    pass


class ApprovalConflict(TaskStoreError):
    pass


class SerializationError(ValueError):
    pass


_CREDENTIAL_NAMES = frozenset({
    "secret", "secrets", "password", "passwd", "pwd", "credential", "credentials",
    "authorization", "proxy_authorization", "api_key", "apikey", "access_key",
    "secret_key", "private_key", "access_token", "refresh_token", "auth_token",
    "id_token", "cookie", "cookies", "set_cookie", "client_secret", "token",
})
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[\w-]*(?:password|passwd|api[_-]?key|secret|access[_-]?token|"
    r"refresh[_-]?token|auth[_-]?token|credential)[\w-]*)\b[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;\}\]&]+)"
)
_BEARER = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_URL_AUTH = re.compile(r"(https?://)[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        return utc_now()
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def sanitize_text(value: str) -> str:
    value = _ASSIGNMENT.sub(r"\1[REDACTED]", value)
    value = _BEARER.sub(r"\1 [REDACTED]", value)
    return _URL_AUTH.sub(r"\1[REDACTED]@", value)


def _credential_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _CREDENTIAL_NAMES or any(
        normalized.endswith("_" + name) for name in _CREDENTIAL_NAMES
    )


def serializable(value: Any, *, _seen: set[int] | None = None) -> Any:
    """保留恢复所需正文，只移除凭证；不把 closure 或任意对象伪装成字符串。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, Enum):
        return serializable(value.value)
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("Non-finite numbers cannot be persisted")
        return value
    if isinstance(value, Path):
        return sanitize_text(str(value))
    if isinstance(value, datetime):
        return timestamp(value)
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        raise SerializationError("Circular task data cannot be persisted")
    seen.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            value = {item.name: getattr(value, item.name) for item in fields(value) if item.name != "handler"}
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SerializationError("Task data keys must be strings")
                if not _credential_key(key):
                    result[key] = serializable(item, _seen=seen)
            return result
        if isinstance(value, (list, tuple)):
            return [serializable(item, _seen=seen) for item in value]
    finally:
        seen.remove(identity)
    raise SerializationError(f"Unsupported task value: {type(value).__name__}")


def json_data(value: Any) -> str:
    return json.dumps(serializable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(json_data(value).encode("utf-8")).hexdigest()


def plan_data(plan: Any, *, executor_version: str = "1") -> dict[str, Any]:
    if isinstance(plan, Mapping):
        plan = {key: value for key, value in plan.items() if key != "handler"}
        plan = dict(plan)
        plan["steps"] = [
            {key: value for key, value in step.items() if key != "handler"}
            if isinstance(step, Mapping) else step for step in plan.get("steps", [])
        ]
    value = serializable(plan)
    if not isinstance(value, dict):
        raise SerializationError("A plan must be a mapping or dataclass")
    steps = value.get("steps", [])
    if not isinstance(steps, list):
        raise SerializationError("Plan steps must be a list")
    ids = set()
    for step in steps:
        if not isinstance(step, dict):
            raise SerializationError("A plan node must be a mapping")
        node_id = step.get("step_id") or step.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            raise SerializationError("Plan node IDs must be nonempty and unique")
        ids.add(node_id)
        step["step_id"] = node_id
        step.setdefault("agent", "executor")
        step.setdefault("handler_key", f"{step['agent']}.{node_id}")
        step.setdefault("handler_version", str(executor_version))
        step.setdefault("depends_on", [])
        step.setdefault("arguments", {})
        step.setdefault("allowed_tools", [])
        step.setdefault("requires_human", False)
        read_tools = {"files.read_document", "files.scan_folder", "web.search", "web.research", "knowledge.search"}
        step.setdefault("side_effect", bool(step["requires_human"] or not step.get("retry_safe", True)
                                             or set(step["allowed_tools"]) - read_tools))
        step["side_effect"] = bool(step["side_effect"] or step["requires_human"] or not step.get("retry_safe", True))
        if not isinstance(step["arguments"], dict) or not isinstance(step["depends_on"], list):
            raise SerializationError("Node arguments/dependencies have invalid types")
        if not isinstance(step["allowed_tools"], list) or any(not isinstance(tool, str) for tool in step["allowed_tools"]):
            raise SerializationError("Allowed tools must be a list of strings")
    completed: set[str] = set()
    while len(completed) < len(ids):
        ready = {step["step_id"] for step in steps if set(step["depends_on"]) <= completed}
        if not ready - completed:
            raise SerializationError("Plan dependencies are missing or cyclic")
        completed |= ready
    return value
