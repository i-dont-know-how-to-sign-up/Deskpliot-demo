from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class SandboxBroker:
    """Copy caller-provided bytes into a private, read-only container input mount."""

    def __init__(self, root: Path | None = None, *, max_files: int = 8,
                 max_file_bytes: int = 64 * 1024, max_total_bytes: int = 128 * 1024) -> None:
        self.root = root
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    def validate(self, inputs: dict[str, bytes] | None) -> dict[str, bytes]:
        if inputs is None:
            return {}
        if not isinstance(inputs, dict) or len(inputs) > self.max_files:
            raise ValueError("Sandbox input file count exceeds limit")
        total = 0
        validated: dict[str, bytes] = {}
        for name, value in inputs.items():
            if (not isinstance(name, str) or not _NAME.fullmatch(name)
                    or name in {".", ".."} or ".." in name):
                raise ValueError("Invalid sandbox input filename")
            if not isinstance(value, bytes) or len(value) > self.max_file_bytes:
                raise ValueError("Invalid or oversized sandbox input")
            total += len(value)
            if total > self.max_total_bytes:
                raise ValueError("Sandbox input total exceeds limit")
            validated[name] = value
        return validated

    @contextmanager
    def prepare_inputs(self, inputs: dict[str, bytes] | None) -> Iterator[Path]:
        validated = self.validate(inputs)
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="deskpilot_sandbox_", dir=self.root) as temporary:
            directory = Path(temporary) / "inputs"
            directory.mkdir(mode=0o755)
            os.chmod(directory, 0o755)
            for name, value in validated.items():
                target = directory / name
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(value)
                os.chmod(target, 0o444)
            yield directory
