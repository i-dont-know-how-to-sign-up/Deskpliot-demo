from __future__ import annotations

import json
import uuid
import shutil
import re
from pathlib import Path
from typing import Any

from .memory_models import SessionInfo, SessionMessage
from ..core.config import MEMORY_SESSIONS_DIR, ensure_dirs
from ..core.encoding_utils import fix_mojibake
from ..core.models import utc_now


class SessionStore:
    def __init__(self, base_dir: Path = MEMORY_SESSIONS_DIR):
        ensure_dirs()
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, title: str = "新会话") -> SessionInfo:
        stamp = utc_now().replace(":", "").replace("-", "").split(".")[0]
        session_id = f"sess_{stamp}_{uuid.uuid4().hex[:6]}"
        now = utc_now()
        info = SessionInfo(session_id=session_id, title=title, created_at=now, updated_at=now)
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(session_dir / "metadata.json", info.to_dict())
        for name in (
            "messages.jsonl",
            "facts.jsonl",
            "preferences.jsonl",
            "decisions.jsonl",
            "tasks.jsonl",
            "artifacts.jsonl",
        ):
            (session_dir / name).touch(exist_ok=True)
        (session_dir / "summary.md").touch(exist_ok=True)
        return info

    def get_or_create(self, session_id: str | None = None) -> SessionInfo:
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
        sessions = self.list_sessions()
        if sessions:
            return sessions[0]
        return self.create_session()

    def get_session(self, session_id: str) -> SessionInfo | None:
        metadata = self._session_dir(session_id) / "metadata.json"
        if not metadata.exists():
            return None
        return SessionInfo.from_dict(json.loads(metadata.read_text(encoding="utf-8")))

    def list_sessions(self) -> list[SessionInfo]:
        sessions: list[SessionInfo] = []
        for metadata in self.base_dir.glob("*/metadata.json"):
            try:
                sessions.append(SessionInfo.from_dict(json.loads(metadata.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        sessions.sort(key=lambda item: item.pinned, reverse=True)
        return sessions

    def rename_session(self, session_id: str, title: str) -> SessionInfo:
        info = self.get_session(session_id)
        if not info:
            raise FileNotFoundError(f"Session not found: {session_id}")
        title = " ".join(str(title).strip().split())[:80]
        if not title:
            raise ValueError("会话名称不能为空")
        info.title = title
        info.updated_at = utc_now()
        self._write_json(self._session_dir(session_id) / "metadata.json", info.to_dict())
        return info

    def set_pinned(self, session_id: str, pinned: bool | None = None) -> SessionInfo:
        info = self.get_session(session_id)
        if not info:
            raise FileNotFoundError(f"Session not found: {session_id}")
        info.pinned = (not info.pinned) if pinned is None else bool(pinned)
        self._write_json(self._session_dir(session_id) / "metadata.json", info.to_dict())
        return info

    def delete_session(self, session_id: str) -> None:
        if not self.get_session(session_id):
            raise FileNotFoundError(f"Session not found: {session_id}")
        shutil.rmtree(self._session_dir(session_id))

    def export_session(self, session_id: str, output_path: Path) -> Path:
        info = self.get_session(session_id)
        if not info:
            raise FileNotFoundError(f"Session not found: {session_id}")
        lines = [f"# {info.title}", ""]
        for message in self.read_messages(session_id):
            role = "用户" if message.role == "user" else "DeskPilot"
            lines.extend([f"## {role}", "", message.content, ""])
            steps = message.metadata.get("steps", []) if isinstance(message.metadata, dict) else []
            if message.role == "assistant" and isinstance(steps, list) and steps:
                lines.extend(["### Steps", ""])
                for index, step in enumerate(steps, start=1):
                    if not isinstance(step, dict):
                        continue
                    lines.extend([
                        f"{index}. [{step.get('status', '')}] {step.get('name', '')}",
                        f"   {step.get('detail', '')}",
                    ])
                lines.append("")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def append_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> SessionMessage:
        info = self.get_or_create(session_id)
        message = SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex}",
            session_id=info.session_id,
            role=role,
            content=_redact_value(fix_mojibake(content), max_text=12000),
            metadata=_redact_value(metadata or {}),
        )
        self._append_jsonl(self._session_dir(info.session_id) / "messages.jsonl", message.to_dict())
        self._touch_session(info.session_id, content if role == "user" else None)
        return message

    def update_message_metadata(self, session_id: str, message_id: str, metadata: dict) -> None:
        """原子更新单条消息元数据，用于持久化本轮 Steps 和 Evidence。"""
        path = self._session_dir(session_id) / "messages.jsonl"
        rows = self._read_jsonl(path)
        updated = False
        for row in rows:
            if str(row.get("message_id")) == message_id:
                current = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                row["metadata"] = _redact_value({**current, **metadata})
                updated = True
                break
        if not updated:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)

    def append_session_item(self, session_id: str, kind: str, payload: dict) -> None:
        file_map = {
            "fact": "facts.jsonl",
            "preference": "preferences.jsonl",
            "decision": "decisions.jsonl",
            "task": "tasks.jsonl",
            "artifact": "artifacts.jsonl",
        }
        filename = file_map.get(kind)
        if not filename:
            return
        self._append_jsonl(self._session_dir(session_id) / filename, _redact_value(payload))

    def read_messages(self, session_id: str) -> list[SessionMessage]:
        return [
            SessionMessage.from_dict(item)
            for item in self._read_jsonl(self._session_dir(session_id) / "messages.jsonl")
        ]

    def recent_messages(self, session_id: str, limit: int = 8) -> list[SessionMessage]:
        return self.read_messages(session_id)[-limit:]

    def message_count(self, session_id: str) -> int:
        return len(self.read_messages(session_id))

    def read_summary(self, session_id: str) -> str:
        path = self._session_dir(session_id) / "summary.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def write_summary(self, session_id: str, summary: str) -> None:
        path = self._session_dir(session_id) / "summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fix_mojibake(summary).strip() + "\n", encoding="utf-8")
        self._touch_session(session_id)

    def read_compaction_state(self, session_id: str) -> dict:
        """读取滚动摘要 checkpoint；旧会话没有该文件时按尚未压缩处理。"""
        path = self._session_dir(session_id) / "compaction.json"
        if not path.exists():
            return {"message_count": 0, "total_chars": 0, "last_message_id": ""}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"message_count": 0, "total_chars": 0, "last_message_id": ""}
        return {
            "message_count": max(0, int(data.get("message_count", 0))),
            "total_chars": max(0, int(data.get("total_chars", 0))),
            "last_message_id": str(data.get("last_message_id", "")),
            "updated_at": str(data.get("updated_at", "")),
        }

    def write_compaction_state(self, session_id: str, messages: list[SessionMessage]) -> None:
        """摘要成功后记录已覆盖的位置，防止后续每轮重复压缩全部历史。"""
        self._write_json(
            self._session_dir(session_id) / "compaction.json",
            {
                "message_count": len(messages),
                "total_chars": sum(len(message.content) for message in messages),
                "last_message_id": messages[-1].message_id if messages else "",
                "updated_at": utc_now(),
            },
        )

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _touch_session(self, session_id: str, maybe_title: str | None = None) -> None:
        info = self.get_session(session_id)
        if not info:
            return
        info.updated_at = utc_now()
        info.message_count = self.message_count(session_id)
        if maybe_title and info.title == "新会话":
            info.title = self._make_title(maybe_title)
        self._write_json(self._session_dir(session_id) / "metadata.json", info.to_dict())

    def _make_title(self, content: str) -> str:
        title = " ".join(content.strip().split())
        return title[:28] + ("..." if len(title) > 28 else "")

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        items: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|password|passwd|authorization|access[_-]?token|refresh[_-]?token|secret)", re.I)
_INLINE_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|authorization|access[_-]?token|refresh[_-]?token|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)


def _redact_sensitive(text: str) -> str:
    return _INLINE_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(text))


def _redact_value(value: Any, *, max_text: int = 12000) -> Any:
    """递归脱敏持久化数据，并限制工具正文、代码和输出的磁盘体积。"""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            result[key_text] = "[REDACTED]" if _SENSITIVE_KEY.search(key_text) else _redact_value(item, max_text=max_text)
        return result
    if isinstance(value, list):
        return [_redact_value(item, max_text=max_text) for item in value[:200]]
    if isinstance(value, tuple):
        return [_redact_value(item, max_text=max_text) for item in value[:200]]
    if isinstance(value, str):
        redacted = _redact_sensitive(value)
        return redacted if len(redacted) <= max_text else redacted[:max_text] + "\n... [persisted value truncated]"
    return value
