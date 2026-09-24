from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any


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


class PendingActionStore:
    """在服务端保存待审批参数，前端只能回传不可预测的一次性 ID。"""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = max(30, int(ttl_seconds))
        self._items: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        action_id = f"act_{secrets.token_urlsafe(24)}"
        item = PendingAction(
            action_id=action_id,
            session_id=session_id,
            tool_name=str(action.get("tool_name", "")),
            kwargs=dict(action.get("kwargs", {})),
            description=str(action.get("description", "需要人工确认的操作")),
            risk_level=str(action.get("risk_level", "high")),
            reasons=[str(reason) for reason in action.get("reasons", [])],
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        if not item.tool_name:
            raise ValueError("待审批操作缺少 tool_name")
        with self._lock:
            self._purge_expired_locked()
            self._items[action_id] = item
        return {
            "action_id": action_id,
            "tool_name": item.tool_name,
            "description": item.description,
            "risk_level": item.risk_level,
            "reasons": item.reasons,
            "expires_in_seconds": self.ttl_seconds,
        }

    def consume(self, action_id: str, session_id: str) -> PendingAction:
        with self._lock:
            self._purge_expired_locked()
            item = self._items.pop(str(action_id), None)
        if item is None:
            raise ValueError("审批请求不存在、已过期或已经执行")
        if item.session_id != session_id:
            raise PermissionError("审批请求不属于当前会话")
        return item

    def cancel(self, action_id: str, session_id: str) -> bool:
        with self._lock:
            item = self._items.get(str(action_id))
            if item is None or item.session_id != session_id:
                return False
            del self._items[str(action_id)]
            return True

    def inspect(self, action_id: str, session_id: str) -> PendingAction | None:
        """供进程内评测与审计读取；该方法不通过 UI 或工具 Schema 暴露。"""
        with self._lock:
            self._purge_expired_locked()
            item = self._items.get(str(action_id))
            return item if item is not None and item.session_id == session_id else None

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired:
            del self._items[key]
