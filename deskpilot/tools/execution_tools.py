from __future__ import annotations

import subprocess
import sys
import time
import os
import platform
import signal
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .permissions import assess_command, assess_python_code, require_permission
from ..core.config import ROOT_DIR
from ..tasks.control import TaskControlError, get_execution_control


def execute_python_code(
    code: str,
    timeout_seconds: int = 10,
    confirm: bool = False,
    cwd: str | Path | None = None,
    output_limit_bytes: int | None = None,
) -> dict[str, Any]:
    _checkpoint()
    decision = assess_python_code(code)
    permission = require_permission(decision, confirm=confirm)
    if not permission["permitted"]:
        return {
            "ok": True,
            "executed": False,
            **permission,
        }

    workdir = Path(cwd).expanduser().resolve(strict=False) if cwd else ROOT_DIR
    timeout = _bounded_timeout(timeout_seconds, default=10, maximum=60)
    with NamedTemporaryFile("w", suffix=".py", prefix="deskpilot_exec_", encoding="utf-8", delete=False) as file:
        script_path = Path(file.name)
        file.write(str(code))
    try:
        result = _run_process([sys.executable, str(script_path)], workdir, timeout, output_limit_bytes)
        return {**result, "permission": decision.to_dict()}
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass


def execute_command(
    command: str | list[str],
    timeout_seconds: int | None = None,
    confirm: bool = False,
    cwd: str | Path | None = None,
    output_limit_bytes: int | None = None,
) -> dict[str, Any]:
    _checkpoint()
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
    argv, shell_name = _command_argv(command)
    result = _run_process(argv, workdir, timeout, output_limit_bytes)
    return {**result, "command": command, "permission": decision.to_dict(), "shell": shell_name}


def _checkpoint() -> None:
    control = get_execution_control()
    if control is not None:
        control.checkpoint()


def _run_process(argv: list[str], workdir: Path, timeout: float,
                 output_limit_bytes: int | None, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    control = get_execution_control()
    effective_timeout = control.timeout(timeout) if control is not None else timeout
    limit = _output_limit(output_limit_bytes)
    started = time.monotonic()
    windows = os.name == "nt"
    job = _WindowsJob() if windows else None
    process = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    seen = {"stdout": 0, "stderr": 0}
    output_lock = threading.Lock()
    readers: list[threading.Thread] = []
    timed_out = False
    control_error = None

    def drain(pipe: Any, name: str) -> None:
        try:
            while True:
                chunk = pipe.read1(65536)
                if not chunk:
                    return
                with output_lock:
                    seen[name] += len(chunk)
                    available = max(0, limit - sum(len(value) for value in buffers.values()))
                    buffers[name].extend(chunk[:available])
        finally:
            pipe.close()

    try:
        _checkpoint()
        process = subprocess.Popen(
            argv, cwd=str(workdir), stdin=subprocess.DEVNULL, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            start_new_session=not windows,
            creationflags=(0x00000004 | subprocess.CREATE_NO_WINDOW) if windows else 0,
        )
        if job is not None:
            job.assign_and_resume(process)
        for name in buffers:
            reader = threading.Thread(target=drain, args=(getattr(process, name), name),
                                      name=f"deskpilot-process-{name}", daemon=True)
            reader.start()
            readers.append(reader)
        while True:
            _checkpoint()
            if process.poll() is not None:
                break
            if time.monotonic() - started >= effective_timeout:
                _checkpoint()
                timed_out = True
                break
            time.sleep(0.02)
    except TaskControlError as exc:
        control_error = exc
    finally:
        # 父进程正常退出也回收其后台后代，避免继承 pipe 的子进程永久挂住读取。
        try:
            if job is not None:
                job.close()
            elif process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            for reader in readers:
                reader.join(timeout=5)
    if any(reader.is_alive() for reader in readers):
        raise RuntimeError("Process tree cleanup could not close output pipes")
    stdout, stdout_decode_truncated = _decode_output(buffers["stdout"])
    stderr, stderr_decode_truncated = _decode_output(buffers["stderr"])
    result = {
        "ok": not timed_out and control_error is None and process.returncode == 0,
        "executed": True, "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": None if timed_out else process.returncode,
        "timeout_seconds": effective_timeout,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_limit_bytes": limit,
        "output_truncated": (sum(seen.values()) > sum(len(value) for value in buffers.values())
                             or stdout_decode_truncated or stderr_decode_truncated),
        "stdout_bytes": seen["stdout"], "stderr_bytes": seen["stderr"],
    }
    if control_error is not None:
        control_error.result = result
        raise control_error
    _checkpoint()
    return result


def _decode_output(raw: bytearray) -> tuple[str, bool]:
    encoded = raw.decode("utf-8", errors="replace").encode("utf-8")
    # 非法字节的替换字符也占用返回值限额；截断不拆开 UTF-8 字符。
    text = encoded[:len(raw)].decode("utf-8", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n"), len(encoded) > len(raw)


def _output_limit(value: int | None) -> int:
    try:
        limit = int(value if value is not None else os.getenv("TOOL_EXECUTION_OUTPUT_MAX_BYTES", "262144"))
    except (TypeError, ValueError):
        limit = 262144
    return max(1, min(limit, 8 * 1024 * 1024))


class _WindowsJob:
    """挂起启动后绑定 KILL_ON_JOB_CLOSE，消除绑定前派生子进程的竞争窗口。"""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self._ctypes = ctypes
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._kernel.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not self._kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign_and_resume(self, process: subprocess.Popen) -> None:
        ctypes = self._ctypes
        if not self._kernel.AssignProcessToJobObject(self._handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Popen 已关闭主线程句柄；原生 NtResumeProcess 恢复挂起的进程。
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes = [ctypes.c_void_p]
        resume.restype = ctypes.c_long
        status = resume(int(process._handle))
        if status != 0:
            raise OSError(f"NtResumeProcess failed: {status:#x}")

    def close(self) -> None:
        if self._handle:
            if not self._kernel.CloseHandle(self._handle):
                raise self._ctypes.WinError(self._ctypes.get_last_error())
            self._handle = None


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
