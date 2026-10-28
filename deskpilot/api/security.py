from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    principal_id: str
    scopes: frozenset[str]


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class TokenCatalog:
    """Reload hashes on each request so revocation does not require a restart."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise ValueError("API token catalog does not exist")
        self._read()

    def _read(self) -> list[dict]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        rows = value.get("tokens") if isinstance(value, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("API token catalog must contain tokens")
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("principal"), str)
                    or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                    or not isinstance(row.get("scopes"), list)):
                raise ValueError("Invalid API token record")
        return rows

    def authenticate(self, header: str | None, required_scope: str) -> Principal:
        if not header or not header.startswith("Bearer "):
            raise AuthenticationError("Missing bearer token")
        token = header[7:].strip()
        if not token or len(token) > 512:
            raise AuthenticationError("Invalid bearer token")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        match = None
        for row in self._read():
            if secrets.compare_digest(digest, row["sha256"].lower()):
                match = row
        if match is None:
            raise AuthenticationError("Invalid bearer token")
        expiry = match.get("expires_at")
        if expiry:
            until = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
            if until.tzinfo is None or until <= datetime.now(timezone.utc):
                raise AuthenticationError("Expired bearer token")
        scopes = frozenset(str(scope) for scope in match["scopes"])
        if required_scope not in scopes:
            raise AuthorizationError("Insufficient scope")
        return Principal(str(match["principal"]), scopes)
