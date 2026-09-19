from __future__ import annotations

import os
import webbrowser
from pathlib import Path


def get_active_window_title() -> dict[str, str | int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        buffer = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buffer, 512)
        title = buffer.value.strip()
        return {"hwnd": int(hwnd), "title": title}
    except Exception as exc:
        return {"hwnd": 0, "title": "", "error": str(exc)}


def open_path(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    os.startfile(str(path))  # type: ignore[attr-defined]
    return {"ok": True, "path": str(path), "message": "Opened in default system app."}


def open_url(url: str) -> dict[str, object]:
    if not url.startswith(("http://", "https://")):
        raise ValueError("Only http and https URLs are allowed.")
    webbrowser.open_new_tab(url)
    return {"ok": True, "url": url, "message": "Opened in default browser."}
