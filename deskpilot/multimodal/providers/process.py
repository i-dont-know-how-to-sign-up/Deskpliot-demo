from __future__ import annotations

import subprocess
from pathlib import Path

from ...tasks.control import get_execution_control


def run_model_process(argv: list[str], *, cwd: str, env: dict[str, str], timeout: float,
                      **kwargs) -> subprocess.CompletedProcess[str]:
    """可信模型 worker 共用进程树回收；不是允许 LLM 绕过执行权限的新工具。"""
    if get_execution_control() is None:
        return subprocess.run(argv, cwd=cwd, env=env, timeout=timeout, **kwargs)
    from ...tools.execution_tools import _run_process
    result = _run_process(argv, Path(cwd), timeout, 65536, env=env)
    if result.get("timed_out"):
        raise subprocess.TimeoutExpired(argv, timeout, output=result["stdout"], stderr=result["stderr"])
    return subprocess.CompletedProcess(argv, result["returncode"], result["stdout"], result["stderr"])
