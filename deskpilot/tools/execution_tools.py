from __future__ import annotations

import subprocess
import sys
import time
import os
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .permissions import assess_command, require_permission
from ..core.config import ROOT_DIR


def execute_python_code(
    code: str,
    timeout_seconds: int = 10,
    confirm: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    decision = assess_command("python code", operation="python_code_execution")
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "executed": False,
            **permission,
        }

    workdir = Path(cwd).expanduser().resolve(strict=False) if cwd else ROOT_DIR
    timeout = _bounded_timeout(timeout_seconds, default=10, maximum=60)
    started = time.monotonic()
    with NamedTemporaryFile("w", suffix=".py", prefix="deskpilot_exec_", encoding="utf-8", delete=False) as file:
        script_path = Path(file.name)
        file.write(str(code))
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "executed": True,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "permission": decision.to_dict(),
        }
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "ok": completed.returncode == 0,
        "executed": True,
        "timed_out": False,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "permission": decision.to_dict(),
    }


def execute_command(
    command: str | list[str],
    timeout_seconds: int | None = None,
    confirm: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    workdir = Path(cwd).expanduser().resolve(strict=False) if cwd else ROOT_DIR
    decision = assess_command(command, operation="shell_command_execution", cwd=workdir)
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "executed": False,
            "command": command,
            **permission,
        }

    timeout = _bounded_timeout(timeout_seconds, default=_env_timeout("TOOL_COMMAND_TIMEOUT_SECONDS", 30), maximum=120)
    started = time.monotonic()
    try:
        argv, shell_name = _command_argv(command)
        completed = subprocess.run(
            argv,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "executed": True,
            "timed_out": True,
            "command": command,
            "timeout_seconds": timeout,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "permission": decision.to_dict(),
            "shell": shell_name,
        }

    return {
        "ok": completed.returncode == 0,
        "executed": True,
        "timed_out": False,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "permission": decision.to_dict(),
        "shell": shell_name,
    }


def _command_argv(command: str | list[str]) -> tuple[list[str], str]:
    """使用显式平台 shell，避免依赖 subprocess 的隐式 shell 行为。"""
    if not isinstance(command, str):
        return [str(part) for part in command], "direct"
    if platform.system().casefold() == "windows":
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], "powershell"
    return ["/bin/sh", "-c", command], "sh"


def _bounded_timeout(value: int | str, default: int, maximum: int) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = default
    return max(1, min(timeout, maximum))


def _env_timeout(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
