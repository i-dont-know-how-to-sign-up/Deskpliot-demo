from __future__ import annotations

import os
import sys
from pathlib import Path


def shortcut_path() -> Path:
    if os.name != "nt":
        raise RuntimeError("Autostart is supported for Windows desktop builds only")
    appdata = os.getenv("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is unavailable")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "DeskPilot Worker.lnk"


def autostart_enabled() -> bool:
    return shortcut_path().exists()


def set_autostart(enabled: bool) -> None:
    if not getattr(sys, "frozen", False):
        raise RuntimeError("Autostart is only available in an installed desktop build")
    path = shortcut_path()
    if not enabled:
        if path.exists():
            import win32com.client

            shortcut = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(path))
            if Path(shortcut.TargetPath).resolve() != Path(sys.executable).resolve():
                raise PermissionError("Startup shortcut does not belong to this DeskPilot install")
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("A DeskPilot startup shortcut already exists")
    import win32com.client

    shortcut = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(path))
    shortcut.TargetPath = sys.executable
    shortcut.Arguments = "--background-worker"
    shortcut.WorkingDirectory = str(Path(sys.executable).parent)
    shortcut.Description = "DeskPilot background task worker"
    shortcut.Save()
