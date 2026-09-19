from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import ROOT_DIR, WORKSPACE_DIR


LOW = "low"
MEDIUM = "medium"
HIGH = "high"
BLOCKED = "blocked"


@dataclass(frozen=True)
class PermissionDecision:
    allowed_without_confirmation: bool
    requires_confirmation: bool
    blocked: bool
    risk_level: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_without_confirmation": self.allowed_without_confirmation,
            "requires_confirmation": self.requires_confirmation,
            "blocked": self.blocked,
            "risk_level": self.risk_level,
            "reasons": self.reasons,
        }


def is_confirmed(confirm: bool | str | int | None) -> bool:
    if isinstance(confirm, bool):
        return confirm
    if isinstance(confirm, int):
        return confirm != 0
    if isinstance(confirm, str):
        return confirm.strip().lower() in {"1", "true", "yes", "y", "on", "confirm", "confirmed"}
    return False


def assess_path_write(
    path: str | Path,
    operation: str = "write",
    additional_safe_roots: list[str | Path] | None = None,
) -> PermissionDecision:
    target = _resolve(path)
    reasons: list[str] = []

    if _is_protected_system_path(target):
        return PermissionDecision(
            allowed_without_confirmation=False,
            requires_confirmation=False,
            blocked=True,
            risk_level=BLOCKED,
            reasons=[
                f"Target is inside a protected system directory: {target}",
                "Protected system paths are blocked by policy.",
            ],
        )

    if _is_safe_root(target, additional_safe_roots=additional_safe_roots):
        risk_level = LOW
        allowed = True
        requires_confirmation = False
        reasons.append("Target is inside a configured safe workspace root.")
        if target.exists():
            risk_level = MEDIUM
            reasons.append("Target already exists; overwrite is a medium-risk operation.")
        return PermissionDecision(
            allowed_without_confirmation=allowed,
            requires_confirmation=requires_confirmation,
            blocked=False,
            risk_level=risk_level,
            reasons=reasons,
        )

    if _is_c_drive_path(target):
        reasons.append("Target is on the C: drive and outside the safe workspace.")
    else:
        reasons.append("Target is outside configured safe workspace roots.")
    reasons.append(f"Operation '{operation}' requires explicit human confirmation.")
    return PermissionDecision(
        allowed_without_confirmation=False,
        requires_confirmation=True,
        blocked=False,
        risk_level=HIGH,
        reasons=reasons,
    )


def assess_command(
    command: str | list[str] | tuple[str, ...], operation: str = "execute", cwd: str | Path | None = None
) -> PermissionDecision:
    command_text = _command_to_text(command)
    blocked_reason = _blocked_command_reason(command_text)
    if blocked_reason:
        return PermissionDecision(
            allowed_without_confirmation=False,
            requires_confirmation=False,
            blocked=True,
            risk_level=BLOCKED,
            reasons=[
                blocked_reason,
                "The command is blocked because it can delete data, change system state, or shut down services.",
            ],
        )
    base = _command_base(command_text)
    external_reason = _external_command_path_reason(command_text, cwd)
    if base in _readonly_command_whitelist() and _contains_shell_operators(command_text):
        return PermissionDecision(False, True, False, HIGH, ["Composite commands, redirection, or variable expansion cannot use the read-only whitelist.", "Explicit human confirmation is required."])
    if base in _readonly_command_whitelist() and external_reason:
        return PermissionDecision(False, True, False, MEDIUM, [external_reason, "Explicit human confirmation is required."])
    if base in _readonly_command_whitelist() and not _contains_shell_operators(command_text):
        return PermissionDecision(
            allowed_without_confirmation=True,
            requires_confirmation=False,
            blocked=False,
            risk_level=LOW,
            reasons=[f"Command '{base}' is in the read-only whitelist."],
        )
    if base in _write_command_allowlist():
        return PermissionDecision(
            allowed_without_confirmation=False,
            requires_confirmation=True,
            blocked=False,
            risk_level=MEDIUM,
            reasons=[f"Command '{base}' can modify files or directories.", "Explicit human confirmation is required."],
        )
    return PermissionDecision(
        allowed_without_confirmation=False,
        requires_confirmation=True,
        blocked=False,
        risk_level=HIGH,
        reasons=[f"Command '{base or command_text}' is not in the low-risk whitelist.", f"{operation} may run arbitrary code or change system state.", "Explicit human confirmation is required."],
    )


def _command_base(command_text: str) -> str:
    """提取 PowerShell、cmd 和 POSIX shell 的首个命令名。"""
    normalized = command_text.strip()
    if not normalized:
        return ""
    # 去掉常见调用运算符和路径引号，仅比较可执行文件名。
    token = re.split(r"\s+", normalized.lstrip("& "), maxsplit=1)[0].strip("'\"")
    token = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return token[:-4] if token.endswith(".exe") else token


def _contains_shell_operators(command_text: str) -> bool:
    # 复合命令可绕过单命令白名单，统一升级为需要人工审核。
    return bool(re.search(r"(?:&&|\|\||[|;><`$%]|\r|\n)", command_text))


def _external_command_path_reason(command_text: str, cwd: str | Path | None) -> str:
    workdir = _resolve(cwd or ROOT_DIR)
    if not _is_safe_root(workdir):
        return f"Working directory is outside configured safe roots: {workdir}"
    # 识别 Windows 和 POSIX 绝对路径；相对路径由受限 cwd 约束。
    raw_paths = re.findall(r"(?:[A-Za-z]:[\\/][^\s'\"]+|(?<!:)\/[^\s'\"]+)", command_text)
    for raw in raw_paths:
        candidate = _resolve(raw.rstrip(",;"))
        if not _is_safe_root(candidate):
            return f"Command references a path outside configured safe roots: {candidate}"
    return ""


def _readonly_command_whitelist() -> set[str]:
    return {
        "pwd", "ls", "dir", "get-childitem", "gci", "get-location",
        "cat", "type", "get-content", "gc", "head", "tail", "wc",
        "rg", "grep", "find", "findstr", "where", "where.exe", "which",
        "tree", "stat", "file", "test-path", "resolve-path",
        "whoami", "hostname", "uname", "systeminfo",
    }


def _write_command_allowlist() -> set[str]:
    return {
        "cp", "copy", "copy-item", "mv", "move", "move-item",
        "mkdir", "md", "new-item", "touch", "set-content", "add-content",
        "out-file", "tee", "rename-item", "ren",
    }


def require_permission(decision: PermissionDecision, confirm: bool | str | int | None = False) -> dict[str, Any]:
    if decision.blocked:
        raise PermissionError("; ".join(decision.reasons))
    confirmed = is_confirmed(confirm)
    if decision.requires_confirmation and not confirmed:
        return {
            "permitted": False,
            "confirmed": False,
            "permission": decision.to_dict(),
            "message": "Permission required. Retry with confirm=True after user approval.",
        }
    return {
        "permitted": True,
        "confirmed": confirmed,
        "permission": decision.to_dict(),
        "message": "Permission granted.",
    }


def safe_roots(additional_safe_roots: list[str | Path] | None = None) -> list[Path]:
    raw = os.getenv("TOOL_SAFE_ROOTS", "")
    roots: list[Path] = []
    if raw.strip():
        roots.extend(_resolve(item) for item in raw.split(";") if item.strip())
    roots.extend([ROOT_DIR, WORKSPACE_DIR])
    if additional_safe_roots:
        roots.extend(_resolve(root) for root in additional_safe_roots)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = _resolve(root)
        key = str(resolved).casefold()
        if key not in seen:
            deduped.append(resolved)
            seen.add(key)
    return deduped


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_safe_root(path: Path, additional_safe_roots: list[str | Path] | None = None) -> bool:
    return any(_is_relative_to(path, root) for root in safe_roots(additional_safe_roots=additional_safe_roots))


def _is_c_drive_path(path: Path) -> bool:
    anchor = path.anchor.casefold()
    return anchor.startswith("c:")


def _is_protected_system_path(path: Path) -> bool:
    protected = [
        Path(os.getenv("SystemRoot", r"C:\Windows")),
        Path(os.getenv("ProgramFiles", r"C:\Program Files")),
        Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]
    return any(_is_relative_to(path, _resolve(root)) for root in protected)


def _command_to_text(command: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def _blocked_command_reason(command_text: str) -> str:
    normalized = re.sub(r"\s+", " ", command_text.strip().casefold())
    patterns = [
        (r"(^|[\s;&|])remove-item([\s;&|]|$)", "PowerShell Remove-Item is blocked."),
        (r"(^|[\s;&|])rm([\s;&|]|$)", "rm is blocked."),
        (r"(^|[\s;&|])del([\s;&|]|$)", "del is blocked."),
        (r"(^|[\s;&|])erase([\s;&|]|$)", "erase is blocked."),
        (r"(^|[\s;&|])rmdir([\s;&|]|$)", "rmdir is blocked."),
        (r"(^|[\s;&|])rd\s+/s([\s;&|]|$)", "recursive rd is blocked."),
        (r"(^|[\s;&|])format([\s;&|]|$)", "format is blocked."),
        (r"(^|[\s;&|])diskpart([\s;&|]|$)", "diskpart is blocked."),
        (r"(^|[\s;&|])shutdown([\s;&|]|$)", "shutdown is blocked."),
        (r"(^|[\s;&|])restart-computer([\s;&|]|$)", "Restart-Computer is blocked."),
        (r"(^|[\s;&|])stop-computer([\s;&|]|$)", "Stop-Computer is blocked."),
        (r"(^|[\s;&|])takeown([\s;&|]|$)", "takeown is blocked."),
        (r"(^|[\s;&|])icacls([\s;&|]|$)", "icacls is blocked."),
        (r"(^|[\s;&|])reg\s+delete([\s;&|]|$)", "registry deletion is blocked."),
        (r"(^|[\s;&|])git\s+reset\s+--hard([\s;&|]|$)", "git reset --hard is blocked."),
        (r"(^|[\s;&|])git\s+clean([\s;&|]|$)", "git clean is blocked."),
    ]
    for pattern, reason in patterns:
        if re.search(pattern, normalized):
            return reason
    return ""
