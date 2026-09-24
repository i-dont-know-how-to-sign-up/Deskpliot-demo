from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any


class MemoryPolicy:
    """控制记忆抽取收益和不同记忆类型的默认生命周期。"""

    _PERSONAL_SIGNAL = re.compile(
        r"(?:我希望|我偏好|我喜欢|请记住|以后|固定使用|我决定|我们决定|计划|待办|下一步|帮我实现|帮我修复)",
        re.IGNORECASE,
    )
    _MEMORY_STEPS = {
        "execute_approved_action", "write_file", "write_report",
        "request_email_confirmation", "supervisor:commit",
    }

    def should_extract(
        self, user_message: str, *, grounded: bool, steps: list[Any] | None = None,
        pending_action: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        if pending_action:
            return True, "pending_side_effect"
        names = {str(getattr(step, "name", "")) for step in (steps or [])}
        if names.intersection(self._MEMORY_STEPS):
            return True, "tool_or_artifact"
        if self._PERSONAL_SIGNAL.search(user_message):
            return True, "explicit_user_state"
        if grounded and names.intersection({"retrieve_evidence", "read_local_documents", "web_search"}):
            return True, "grounded_result"
        return False, "low_value_direct_turn"

    def default_expires_at(self, scope: str, memory_type: str, tags: list[str] | None = None) -> str | None:
        normalized_tags = {str(tag).casefold() for tag in (tags or [])}
        if memory_type in {"preference", "decision"} or "completed" in normalized_tags or "permanent" in normalized_tags:
            return None
        days = {"task": 30 if scope == "session" else 90, "artifact": 90,
                "fact": 120 if scope == "session" else 365}.get(memory_type)
        return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat() if days else None
