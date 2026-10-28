"""Pin installed desktop dependency closure; run inside the clean build venv."""
from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement


ROOTS = ["PySide6", "PyMuPDF", "pypdf", "Pillow", "requests", "playwright", "mcp", "pywin32", "PyInstaller", "cryptography", "fastapi"]


def freeze() -> list[str]:
    pending = list(ROOTS)
    seen: set[str] = set()
    result = []
    while pending:
        name = pending.pop()
        distribution = metadata.distribution(name)
        normalized = distribution.metadata["Name"].lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(f"{distribution.metadata['Name']}=={distribution.version}")
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return sorted(result, key=str.casefold)


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("Desktop release lock must be generated on Windows")
    path = Path(__file__).resolve().parent.parent / "requirements-desktop.lock"
    path.write_text("# Generated on Windows Python 3.11 by packaging/freeze_desktop.py\n"
                    + "\n".join(freeze()) + "\n", encoding="utf-8")
    print(f"Pinned {len(freeze())} distributions to {path}")
