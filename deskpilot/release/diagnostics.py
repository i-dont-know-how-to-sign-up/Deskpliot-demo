from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from deskpilot import __version__

from .profile import profile_home, read_settings, release_mode


def collect_diagnostics(home: Path | None = None) -> dict:
    home = home or profile_home()
    data = home / "data"
    settings = read_settings(home)
    databases = []
    for path in sorted(data.rglob("*")) if data.exists() else ():
        if not path.is_file() or path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            continue
        try:
            with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
                integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        except sqlite3.Error:
            integrity = "error"
        databases.append({"component": path.stem, "integrity": "ok" if integrity == "ok" else "error"})
    return {
        "version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "release_mode": release_mode(),
        "configured": (home / "settings.json").exists(),
        "model_key_present": _has_model_key(home),
        "cloud_image_upload_enabled": settings.get("ALLOW_CLOUD_IMAGE_UPLOAD", "false").lower() == "true",
        "code_execution_enabled": False if release_mode() else None,
        "free_disk_bytes": shutil.disk_usage(data if data.exists() else home.parent).free,
        "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("PySide6", "fitz", "PIL")},
        "databases": databases,
    }


def _has_model_key(home: Path) -> bool:
    if os.getenv("DASHSCOPE_API_KEY"):
        return True
    if os.name != "nt":
        return False
    from .secrets import SecretStore

    try:
        return bool(SecretStore(home / "secrets.dpapi").get("DASHSCOPE_API_KEY"))
    except (OSError, RuntimeError, ValueError):
        return False


def export_diagnostics(destination: Path, home: Path | None = None) -> None:
    destination = destination.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only aggregate health metadata is exported, never prompts, paths or credentials.
    destination.write_text(json.dumps(collect_diagnostics(home), ensure_ascii=False, indent=2), encoding="utf-8")
