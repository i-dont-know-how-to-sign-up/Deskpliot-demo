from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deskpilot import __version__

from .profile import profile_home


def save_local_feedback(description: str, home: Path | None = None) -> Path:
    description = description.strip()
    if not description or len(description) > 4000:
        raise ValueError("Feedback must contain 1 to 4000 characters")
    folder = (home or profile_home()) / "feedback"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"feedback-{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump({
            "version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": description, "uploaded": False,
        }, output, ensure_ascii=False, indent=2)
    return path
