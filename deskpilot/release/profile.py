from __future__ import annotations

import json
import os
import sys
from pathlib import Path


SETTING_NAMES = {
    "DASHSCOPE_BASE_URL", "QWEN_MODEL", "QWEN_EMBEDDING_MODEL",
    "TOOL_SAFE_ROOTS", "ALLOW_CLOUD_IMAGE_UPLOAD", "QWEN_ENABLE_THINKING", "EMAIL_PROVIDER",
    "VISION_MODEL", "VISION_BASE_URL",
    "EMAIL_USERNAME", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST",
    "EMAIL_IMAP_PORT", "EMAIL_SMTP_PORT", "EMAIL_DRAFTS_MAILBOX",
    "EMAIL_SENT_MAILBOX", "SEARCH_PROVIDER", "BROWSER_CHANNEL",
}
SECRET_NAMES = {"DASHSCOPE_API_KEY", "VISION_API_KEY", "EMAIL_PASSWORD", "SEARCH_API_KEY"}


def release_mode() -> bool:
    return bool(getattr(sys, "frozen", False)) or os.getenv("DESKPILOT_RELEASE_MODE") == "1"


def default_home() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "DeskPilot"


def profile_home() -> Path:
    override = os.getenv("DESKPILOT_HOME")
    if override:
        path = Path(override).expanduser()
    else:
        pointer = default_home() / "location.json"
        try:
            path = Path(json.loads(pointer.read_text(encoding="utf-8"))["home"])
        except FileNotFoundError:
            path = default_home()
    if not path.is_absolute():
        raise ValueError("DeskPilot profile location must be absolute")
    return path.resolve()


def set_profile_home(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise ValueError("Data directory must be an absolute path")
    selected = selected.resolve()
    pointer = default_home() / "location.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    _write_json(pointer, {"home": str(selected)})
    os.environ["DESKPILOT_HOME"] = str(selected)
    os.environ["DESKPILOT_DATA_DIR"] = str(selected / "data")
    return selected


def read_settings(home: Path | None = None) -> dict[str, str]:
    path = (home or profile_home()) / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(settings, dict) or any(k not in SETTING_NAMES or not isinstance(v, str) for k, v in settings.items()):
        raise ValueError("Invalid DeskPilot settings file")
    return settings


def save_settings(settings: dict[str, str], home: Path | None = None) -> None:
    if any(k not in SETTING_NAMES or not isinstance(v, str) for k, v in settings.items()):
        raise ValueError("Settings contain an unsupported field or secret")
    target = (home or profile_home()) / "settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, settings)


def configure_runtime() -> Path:
    """Must run before importing core.config or the Qt bridge."""
    home = profile_home()
    if release_mode():
        os.environ["DESKPILOT_RELEASE_MODE"] = "1"
        os.environ["DESKPILOT_DATA_DIR"] = str(home / "data")
        os.environ.setdefault("HF_HOME", str(home / "cache" / "huggingface"))
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(home / "cache" / "paddlex"))
        for name, value in read_settings(home).items():
            os.environ.setdefault(name, value)
        os.environ.setdefault("QWEN_ENABLE_THINKING", "false")
    return home


def _write_json(path: Path, value: dict[str, str]) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".deskpilot-", delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
