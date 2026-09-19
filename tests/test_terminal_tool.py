from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deskpilot.tools.execution_tools import execute_command
from deskpilot.tools.permissions import assess_command


def test_readonly_whitelist_runs_without_confirmation() -> None:
    command = "Get-ChildItem -Name" if platform.system() == "Windows" else "ls"
    result = execute_command(command, cwd=PROJECT_ROOT, timeout_seconds=5)
    assert result["executed"] is True
    assert result["permission"]["risk_level"] == "low"
    assert result["shell"] in {"powershell", "sh"}


def test_write_unknown_and_composite_commands_require_confirmation() -> None:
    assert assess_command("New-Item test.txt").risk_level == "medium"
    assert assess_command("python -c 'print(1)'").risk_level == "high"
    composite = assess_command("Get-ChildItem | Select-Object -First 1")
    assert composite.requires_confirmation is True
    assert composite.allowed_without_confirmation is False
    outside = assess_command(r"Get-Content C:\Users\Public\note.txt")
    assert outside.risk_level == "medium"
    expanded = assess_command("Get-Content $HOME/.env")
    assert expanded.risk_level == "high"


def test_destructive_command_is_blocked() -> None:
    decision = assess_command("Remove-Item -Recurse important")
    assert decision.blocked is True


def test_command_timeout_returns_structured_result() -> None:
    command = (
        [sys.executable, "-c", "import time; time.sleep(3)"]
    )
    result = execute_command(command, timeout_seconds=1, confirm=True)
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 1


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"terminal tool tests passed: {len(tests)}")
