from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from deskpilot import __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_candidate_record(root: Path, output: Path, baseline: Path | None = None,
                            executable_path: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve(strict=False)
    lock = root / "requirements-desktop.lock"
    executable = (executable_path or root / "dist" / "DeskPilot" / "DeskPilot.exe").resolve(strict=False)
    installer = root / "dist-installer" / "DeskPilot-0.9.0-beta.3-windows-x64.exe"
    if not lock.exists() or not executable.exists():
        raise FileNotFoundError("Build the desktop executable and pin dependencies first")
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, capture_output=True, text=True, check=False, timeout=5,
    )
    status = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
        cwd=root, capture_output=True, text=True, check=False, timeout=5,
    )
    record = {
        "product": "DeskPilot Windows desktop beta",
        "project_version": __version__,
        "release_tag": "0.9.0-beta.3",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "dependency_lock_sha256": _sha256(lock),
        "executable_path": str(executable),
        "executable_sha256": _sha256(executable),
        "executable_bytes": executable.stat().st_size,
        "installer_sha256": _sha256(installer) if installer.exists() else None,
        "installer_bytes": installer.stat().st_size if installer.exists() else None,
        "baseline": None,
        "release_status": "candidate_not_accepted",
        "release_blockers": [
            "Clean Windows installation and upgrade/rollback not yet verified",
            "Fixed API/VLM baseline and real email smoke must be reviewed by the owner",
            "Windows code/shell sandbox is not implemented; those tools remain disabled",
            "Docker sandbox live escape/resource tests have not passed on the target host",
            "Authorized file/network tool broker and per-task Docker execution approval are not implemented",
        ],
    }
    if baseline is not None:
        baseline = baseline.resolve(strict=True)
        record["baseline"] = {"filename": baseline.name, "sha256": _sha256(baseline)}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
