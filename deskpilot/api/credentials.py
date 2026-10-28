"""Generate a private API token; persist only its SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path


def create_catalog(path: Path, principal: str, *, allow_sandbox_execute: bool = False) -> str:
    if not principal or len(principal) > 80 or not principal.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Principal must be a short alphanumeric identifier")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    record = {"tokens": [{
        "principal": principal,
        "sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "scopes": ["submit", "read", "control", "approve"] + (["execute"] if allow_sandbox_execute else []),
    }]}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a hashed private API token catalog")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--principal", default="owner")
    parser.add_argument("--allow-sandbox-execute", action="store_true",
                        help="Explicitly grant this token access to Docker sandbox tasks")
    args = parser.parse_args()
    try:
        token = create_catalog(args.output, args.principal,
                               allow_sandbox_execute=args.allow_sandbox_execute)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Token (shown once; keep it outside the repository):", token)
    print("Catalog:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
