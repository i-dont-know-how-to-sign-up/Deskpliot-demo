from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from .profile import SECRET_NAMES, profile_home, release_mode


class SecretStore:
    """Windows CurrentUser DPAPI vault; never falls back to plaintext storage."""

    def __init__(
        self, path: Path | None = None,
        protect: Callable[[bytes], bytes] | None = None,
        unprotect: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self.path = path or profile_home() / "secrets.dpapi"
        if protect is None or unprotect is None:
            if os.name != "nt":
                raise RuntimeError("Persistent credential storage is supported on Windows only")
            import win32crypt

            protect = lambda data: win32crypt.CryptProtectData(data, "DeskPilot", None, None, None, 0)
            unprotect = lambda data: win32crypt.CryptUnprotectData(data, None, None, None, 0)[1]
        self._protect = protect
        self._unprotect = unprotect

    def get(self, name: str) -> str:
        self._validate_name(name)
        return self._read().get(name, "")

    def set(self, name: str, value: str) -> None:
        self._validate_name(name)
        values = self._read()
        if value:
            values[name] = value
        else:
            values.pop(name, None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ciphertext = self._protect(json.dumps(values, ensure_ascii=False).encode("utf-8"))
        with tempfile.NamedTemporaryFile("wb", dir=self.path.parent, prefix=".deskpilot-", delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self._unprotect(self.path.read_bytes()).decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "无法解密当前用户的 DeskPilot 凭据文件。请在设置中确认重置凭据；"
                "旧加密文件会保留，但模型 Key 和邮箱授权码都需要重新输入。"
            ) from exc
        if not isinstance(value, dict) or any(k not in SECRET_NAMES or not isinstance(v, str) for k, v in value.items()):
            raise ValueError("Invalid credential vault")
        return value

    def reset(self) -> Path | None:
        """Explicit recovery only; preserve the previous encrypted bytes for inspection."""
        previous = None
        if self.path.exists():
            previous = self.path.with_name(f"{self.path.name}.previous-{uuid.uuid4().hex}")
            self.path.rename(previous)
        self.set("DASHSCOPE_API_KEY", "")
        return previous

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in SECRET_NAMES:
            raise ValueError("Unsupported credential name")


def secret_value(name: str) -> str:
    if name not in SECRET_NAMES:
        return ""
    value = os.getenv(name, "")
    if value or not release_mode():
        return value
    try:
        return SecretStore().get(name)
    except RuntimeError:
        # Keep the settings screen reachable so the user can explicitly reset the vault.
        return ""
