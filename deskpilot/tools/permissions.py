from __future__ import annotations

import ast
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


def assess_file_move(
    source: str | Path,
    destination: str | Path,
    additional_safe_roots: list[str | Path] | None = None,
) -> PermissionDecision:
    """同时评估移动的删除侧和写入侧；移动操作始终需要人工确认。"""
    source_path = _resolve(source)
    destination_path = _resolve(destination)
    for label, path in (("Source", source_path), ("Destination", destination_path)):
        if _is_protected_system_path(path):
            return PermissionDecision(False, False, True, BLOCKED, [
                f"{label} is inside a protected system directory: {path}",
                "Protected system paths are blocked by policy.",
            ])
    if destination_path.exists():
        return PermissionDecision(False, False, True, BLOCKED, [
            f"Destination already exists: {destination_path}",
            "File moves cannot silently overwrite an existing target.",
        ])

    source_safe = _is_safe_root(source_path, additional_safe_roots)
    destination_safe = _is_safe_root(destination_path, additional_safe_roots)
    risk = MEDIUM if source_safe and destination_safe else HIGH
    reasons = ["Moving a file removes it from the source path and changes local state."]
    if not source_safe:
        reasons.append(f"Source is outside configured safe workspace roots: {source_path}")
    if not destination_safe:
        reasons.append(f"Destination is outside configured safe workspace roots: {destination_path}")
    reasons.append("Explicit human confirmation is required.")
    return PermissionDecision(False, True, False, risk, reasons)


def assess_python_code(code: str) -> PermissionDecision:
    """阻断明显危险的 Python 语法；该检查是加固层，不是进程沙箱。"""
    try:
        tree = ast.parse(str(code))
    except SyntaxError as exc:
        return PermissionDecision(False, False, True, BLOCKED, [f"Python syntax is invalid: {exc.msg}"])

    blocked_modules = {
        "subprocess", "socket", "requests", "httpx", "urllib", "http", "ftplib", "telnetlib",
        "smtplib", "imaplib", "ctypes", "winreg",
    }
    blocked_calls = {
        "eval", "exec", "compile", "__import__",
        "os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.system", "os.popen",
        "shutil.rmtree", "shutil.move",
        "pathlib.path.unlink", "pathlib.path.rmdir", "pathlib.path.rename", "pathlib.path.replace",
    }
    aliases: dict[str, str] = {}
    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".", 1)[0].casefold()
                aliases[item.asname or root] = item.name.casefold()
                if root in blocked_modules:
                    reasons.append(f"Import of '{item.name}' is blocked for in-process Python execution.")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").casefold()
            root = module.split(".", 1)[0]
            for item in node.names:
                aliases[item.asname or item.name] = f"{module}.{item.name}".strip(".")
            if root in blocked_modules:
                reasons.append(f"Import from '{module}' is blocked for in-process Python execution.")
        elif isinstance(node, ast.Call):
            name = _python_call_name(node.func, aliases)
            normalized = name.casefold()
            dangerous_suffixes = (".unlink", ".rmdir", ".rename", ".replace", ".write_text", ".write_bytes")
            if normalized in blocked_calls or normalized.startswith("subprocess.") or normalized.endswith(dangerous_suffixes):
                reasons.append(f"Call to '{name}' is blocked by the Python execution policy.")
            if normalized == "open" and _open_call_can_write(node):
                reasons.append("Opening a file in write/append/create mode is blocked; use a file tool instead.")
    if reasons:
        return PermissionDecision(False, False, True, BLOCKED, list(dict.fromkeys(reasons)))
    return PermissionDecision(False, True, False, HIGH, [
        "Python code can access local process data and change local state.",
        "Static analysis found no explicitly blocked operation, but it is not a security sandbox.",
        "Explicit human confirmation is required.",
    ])


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
    readonly_pipeline = _readonly_pipeline_commands(command_text)
    if readonly_pipeline and not external_reason:
        return PermissionDecision(
            allowed_without_confirmation=True,
            requires_confirmation=False,
            blocked=False,
            risk_level=LOW,
            reasons=[f"Read-only command pipeline: {', '.join(readonly_pipeline)}."],
        )
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
    token = re.split(r"\s+", normalized.lstrip("& ([{"), maxsplit=1)[0].strip("'\"()[]{}")
    token = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    token = re.sub(r"[)\]}]+(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", "", token)
    return token[:-4] if token.endswith(".exe") else token


def _contains_shell_operators(command_text: str) -> bool:
    # 复合命令可绕过单命令白名单，统一升级为需要人工审核。
    return bool(re.search(r"(?:&&|\|\||[|;><`$%]|\r|\n)", command_text))


def _readonly_pipeline_commands(command_text: str) -> list[str]:
    """识别仅由只读命令组成的简单管道。

    括号和末尾的 ``.Count`` 只影响 PowerShell 表达式形态，不改变命令风险；
    仍逐段检查管道，防止只读首命令掩盖后续写入或执行动作。
    """
    if not command_text.strip() or re.search(r"(?:&&|\|\||[;><`$%]|\r|\n)", command_text):
        return []
    segments = [part.strip() for part in command_text.split("|")]
    if not segments:
        return []
    commands: list[str] = []
    whitelist = _readonly_command_whitelist()
    for segment in segments:
        base = _command_base(segment)
        if not base or base not in whitelist:
            return []
        commands.append(base)
    return commands


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
        "measure-object",
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
            "message": "Permission required. Use the in-app approval action; tool arguments cannot grant confirmation.",
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


def _python_call_name(node: ast.expr, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _python_call_name(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else node.attr
    return "<dynamic>"


def _open_call_can_write(node: ast.Call) -> bool:
    mode: object = None
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = keyword.value.value
    return isinstance(mode, str) and any(flag in mode for flag in "wax+")
