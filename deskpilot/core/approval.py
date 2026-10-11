from __future__ import annotations

import secrets
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


def action_fingerprint(kwargs: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
    paths = [str(path) for path in kwargs.get("attachment_paths", [])]
    paths.extend(str(kwargs[key]) for key in ("path", "source", "destination") if kwargs.get(key))
    files = {}
    for path in paths:
        target = Path(path).expanduser()
        data: dict[str, Any] = {"exists": target.exists()}
        if target.is_file():
            hasher = hashlib.sha256()
            with target.open("rb") as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(part)
            data.update(sha256=hasher.hexdigest(), size=target.stat().st_size)
        files[path] = data
    return {"arguments_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "files": files}


@dataclass(frozen=True)
class PendingAction:
    action_id: str
    session_id: str
    tool_name: str
    kwargs: dict[str, Any]
    description: str
    risk_level: str
    reasons: list[str]
    expires_at: float
    task_id: str = ""
    node_id: str = ""
    plan_version: str = "1"
    fingerprint: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = ""
    owner: str = ""
    lease_token: int = 0


class PendingActionStore:
    """在服务端保存待审批参数，前端只能回传不可预测的一次性 ID。"""

    def __init__(self, ttl_seconds: int = 600, db_path: Path | None = None) -> None:
        self.ttl_seconds = max(30, int(ttl_seconds))
        self._items: dict[str, PendingAction] = {}
        self._lock = threading.RLock()
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS desktop_approvals "
                                   "(action_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                                   "task_id TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL)")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.db_path), timeout=15)
        try:
            connection.execute("PRAGMA busy_timeout=15000")
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def public(item: PendingAction) -> dict[str, Any]:
        preview = ""
        if item.tool_name in {"email.send", "email.save_draft"}:
            recipients = item.kwargs.get("to", [])
            recipient_text = ", ".join(map(str, recipients)) if isinstance(recipients, list) else str(recipients)
            body = str(item.kwargs.get("body", ""))
            attachments = item.fingerprint.get("files", {})
            attachment_text = ", ".join(
                f"{path} ({meta.get('size', '?')} bytes, SHA256 {str(meta.get('sha256', 'missing'))[:12]})"
                for path, meta in attachments.items()
            ) or "none"
            preview = (f"To: {recipient_text}\nSubject: {item.kwargs.get('subject', '')}\n"
                       f"Body: {body[:500]}{'...' if len(body) > 500 else ''}\nAttachments: {attachment_text}")
        return {"action_id": item.action_id, "tool_name": item.tool_name,
                "description": item.description, "risk_level": item.risk_level,
                "preview": preview,
                "reasons": item.reasons, "task_id": item.task_id, "node_id": item.node_id,
                "plan_version": item.plan_version, "expired": item.expires_at <= time.time(),
                "expires_in_seconds": max(0, int(item.expires_at - time.time()))}

    def create(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        from ..tasks.models import serializable
        kwargs = dict(action.get("kwargs", {}))
        if serializable(kwargs) != kwargs:
            raise ValueError("审批参数不能包含凭证或不可序列化内容")
        action_id = f"act_{secrets.token_urlsafe(24)}"
        item = PendingAction(
            action_id=action_id,
            session_id=session_id,
            tool_name=str(action.get("tool_name", "")),
            kwargs=kwargs,
            description=str(action.get("description", "需要人工确认的操作")),
            risk_level=str(action.get("risk_level", "high")),
            reasons=[str(reason) for reason in action.get("reasons", [])],
            expires_at=time.time() + self.ttl_seconds,
            task_id=str(action.get("task_id", "")), node_id=str(action.get("node_id", "")),
            plan_version=str(action.get("plan_version", "1")), fingerprint=action_fingerprint(kwargs),
        )
        if not item.tool_name:
            raise ValueError("待审批操作缺少 tool_name")
        with self._lock:
            if self.db_path:
                with self._connection() as connection:
                    connection.execute("INSERT INTO desktop_approvals VALUES (?, ?, ?, 'pending', ?)",
                                       (action_id, session_id, item.task_id, json.dumps(asdict(item), ensure_ascii=False)))
            else:
                self._items[action_id] = item
        return self.public(item)

    def consume(self, action_id: str, session_id: str) -> PendingAction:
        with self._lock:
            item = self.inspect(action_id, session_id, include_expired=True)
            if item is None:
                if self.db_path:
                    with self._connection() as connection:
                        foreign = connection.execute("SELECT 1 FROM desktop_approvals WHERE action_id=? "
                                                     "AND state='pending' AND session_id<>?",
                                                     (str(action_id), session_id)).fetchone()
                else:
                    foreign = self._items.get(str(action_id))
                if foreign:
                    raise PermissionError("审批请求不属于当前会话")
                raise ValueError("审批请求不存在或已经执行")
            if item.expires_at <= time.time():
                raise ValueError("审批已过期，请重新预览并申请审批")
            # 不仅绑定正文/收件人，也绑定附件和目标文件，避免审批期间的版本漂移。
            if action_fingerprint(item.kwargs) != item.fingerprint:
                raise ValueError("审批内容或附件/目标文件已变化，请重新申请审批；未执行操作")
            if self.db_path:
                if item.task_id:
                    from ..tasks.store import TaskStore
                    attempt = TaskStore(self.db_path).claim_desktop_approval(
                        item.action_id, session_id, item.task_id, item.node_id, item.plan_version)
                    item = replace(item, attempt_id=attempt["attempt_id"], owner=attempt.get("owner", ""),
                                   lease_token=int(attempt.get("lease_token", 0)))
                else:
                    with self._connection() as connection:
                        changed = connection.execute("UPDATE desktop_approvals SET state='consumed' "
                                                     "WHERE action_id=? AND session_id=? AND state='pending'",
                                                     (item.action_id, session_id)).rowcount
                        if changed != 1:
                            raise ValueError("审批已被消费，不能重复执行")
            else:
                del self._items[item.action_id]
            return item

    def cancel(self, action_id: str, session_id: str) -> bool:
        with self._lock:
            if self.db_path:
                with self._connection() as connection:
                    return connection.execute("UPDATE desktop_approvals SET state='cancelled' "
                                              "WHERE action_id=? AND session_id=? AND state='pending'",
                                              (str(action_id), session_id)).rowcount == 1
            item = self._items.get(str(action_id))
            if item is None or item.session_id != session_id:
                return False
            del self._items[str(action_id)]
            return True

    def inspect(self, action_id: str, session_id: str, *, include_expired: bool = False) -> PendingAction | None:
        """供进程内评测与审计读取；该方法不通过 UI 或工具 Schema 暴露。"""
        with self._lock:
            if self.db_path:
                with self._connection() as connection:
                    row = connection.execute("SELECT payload FROM desktop_approvals "
                                             "WHERE action_id=? AND session_id=? AND state='pending'",
                                             (str(action_id), session_id)).fetchone()
                item = PendingAction(**json.loads(row[0])) if row else None
            else:
                item = self._items.get(str(action_id))
            if item is None or item.session_id != session_id:
                return None
            return item if include_expired or item.expires_at > time.time() else None

    def list_pending(self, session_id: str = "", task_id: str = "") -> list[PendingAction]:
        with self._lock:
            if self.db_path:
                with self._connection() as connection:
                    rows = connection.execute("SELECT payload FROM desktop_approvals WHERE state='pending'").fetchall()
                items = [PendingAction(**json.loads(row[0])) for row in rows]
            else:
                items = list(self._items.values())
            return [item for item in items if (not session_id or item.session_id == session_id)
                    and (not task_id or item.task_id == task_id)]

    def renew(self, action_id: str, session_id: str) -> dict[str, Any]:
        item = self.inspect(action_id, session_id, include_expired=True)
        if item is None:
            raise ValueError("没有可重新审批的操作")
        if action_fingerprint(item.kwargs) != item.fingerprint:
            self.cancel(action_id, session_id)
            raise ValueError("Approval inputs changed; old approval cancelled. Start the action again for a new preview.")
        result = self.create(session_id, asdict(item))
        self.cancel(action_id, session_id)
        return result
