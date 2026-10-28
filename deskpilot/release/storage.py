from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from contextlib import closing
from pathlib import Path, PurePosixPath

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


SCHEMA = 1
MAX_RESTORE_BYTES = 10 * 1024**3
DATABASE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
MAGIC = b"DPBACK1\n"


def _key(passphrase: str, salt: bytes) -> bytes:
    if len(passphrase) < 12:
        raise ValueError("Backup passphrase must contain at least 12 characters")
    return Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(passphrase.encode("utf-8"))


def _encrypt(source: Path, destination: Path, passphrase: str) -> None:
    salt, nonce = os.urandom(16), os.urandom(12)
    cipher = Cipher(algorithms.AES(_key(passphrase, salt)), modes.GCM(nonce)).encryptor()
    with source.open("rb") as input_stream, destination.open("xb") as output:
        output.write(MAGIC + salt + nonce)
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            output.write(cipher.update(block))
        output.write(cipher.finalize())
        output.write(cipher.tag)


def _decrypt(source: Path, destination: Path, passphrase: str) -> None:
    from cryptography.exceptions import InvalidTag

    with source.open("rb") as input_stream:
        if input_stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("Unsupported encrypted backup format")
        salt, nonce = input_stream.read(16), input_stream.read(12)
        size = source.stat().st_size
        if len(salt) != 16 or len(nonce) != 12 or size < len(MAGIC) + 16 + 12 + 16:
            raise ValueError("Truncated backup")
        input_stream.seek(size - 16)
        tag = input_stream.read(16)
        cipher = Cipher(algorithms.AES(_key(passphrase, salt)), modes.GCM(nonce, tag)).decryptor()
        input_stream.seek(len(MAGIC) + 16 + 12)
        remaining = size - len(MAGIC) - 16 - 12 - 16
        try:
            with destination.open("xb") as output:
                while remaining:
                    block = input_stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Truncated backup")
                    output.write(cipher.update(block))
                    remaining -= len(block)
                output.write(cipher.finalize())
        except InvalidTag as exc:
            destination.unlink(missing_ok=True)
            raise ValueError("Wrong passphrase or corrupted backup") from exc


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sqlite(path: Path) -> bool:
    if path.suffix.lower() not in DATABASE_SUFFIXES:
        return False
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def _valid_name(name: str) -> bool:
    value = PurePosixPath(name)
    return bool(name and not name.startswith("/") and "\\" not in name and ":" not in name
                and all(part not in {"", ".", ".."} for part in value.parts)
                and value.as_posix() == name)


def _ensure_worker_stopped(source: Path) -> None:
    ledger = source / "index" / "task_ledger.sqlite3"
    if not ledger.exists():
        return
    try:
        with closing(sqlite3.connect(f"{ledger.as_uri()}?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT expires FROM runtime_worker_lock WHERE id=1").fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return
        raise
    if row is not None and row[0] > time.time():
        raise RuntimeError("Stop DeskPilot and its background worker before backing up or migrating data")


def create_backup(source: Path, archive: Path, passphrase: str) -> dict:
    """Create an offline-friendly snapshot. Close the app for a consistent cross-DB backup."""
    if source.is_symlink():
        raise ValueError("Backup refuses symbolic link source")
    source = source.expanduser().resolve(strict=True)
    archive = archive.expanduser().resolve(strict=False)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Source must be a real data directory")
    if archive == source or source in archive.parents:
        raise ValueError("Backup archive must be outside the source data directory")
    if archive.exists():
        raise FileExistsError(archive)
    if len(passphrase) < 12:
        raise ValueError("Backup passphrase must contain at least 12 characters")
    _ensure_worker_stopped(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "files": []}
    with tempfile.TemporaryDirectory(prefix=".deskpilot-backup-", dir=archive.parent) as temporary:
        temp_root = Path(temporary)
        plain = temp_root / "snapshot.zip"
        with zipfile.ZipFile(plain, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Backup refuses symbolic links: {path}")
                if not path.is_file() or path.name.endswith(("-wal", "-shm", "-journal")):
                    continue
                relative = path.relative_to(source).as_posix()
                # DPAPI secrets belong to the Windows user and cannot be restored elsewhere.
                if path.name == "secrets.dpapi":
                    continue
                if not _valid_name(relative):
                    raise ValueError(f"Unsafe backup entry: {relative}")
                materialized = path
                if _is_sqlite(path):
                    materialized = temp_root / f"db-{len(manifest['files'])}.sqlite3"
                    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
                        with closing(sqlite3.connect(materialized)) as snapshot:
                            connection.backup(snapshot)
                            snapshot.commit()
                output.write(materialized, f"payload/{relative}")
                manifest["files"].append({
                    "path": relative, "size": materialized.stat().st_size,
                    "sha256": _digest_file(materialized),
                })
            output.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        encrypted = temp_root / "encrypted.dpb"
        _encrypt(plain, encrypted, passphrase)
        if os.name == "nt":
            os.rename(encrypted, archive)
        else:
            os.link(encrypted, archive)
    return manifest


def restore_backup(archive: Path, destination: Path, passphrase: str) -> dict:
    """Restore only into a nonexistent directory; never overwrite an active profile."""
    archive = archive.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deskpilot-unseal-", dir=destination.parent) as temporary:
        plain = Path(temporary) / "snapshot.zip"
        _decrypt(archive, plain, passphrase)
        return _restore_plain(plain, destination)


def _restore_plain(archive: Path, destination: Path) -> dict:
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read("manifest.json"))
        if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
            raise ValueError("Unsupported backup manifest")
        entries = manifest["files"]
        if any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
               or not isinstance(item.get("size"), int) or isinstance(item.get("size"), bool)
               or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64
               for item in entries):
            raise ValueError("Invalid backup manifest entry")
        expected = {item["path"] for item in entries}
        if (len(expected) != len(entries) or
                any(not _valid_name(name) or name == "secrets.dpapi" for name in expected) or
                set(source.namelist()) != {f"payload/{name}" for name in expected} | {"manifest.json"}):
            raise ValueError("Invalid or duplicate backup entry")
        if sum(item["size"] for item in entries) > MAX_RESTORE_BYTES:
            raise ValueError("Backup exceeds restore size limit")
        for item in entries:
            if item["size"] < 0 or item["size"] > MAX_RESTORE_BYTES:
                raise ValueError("Invalid backup file size")
            info = source.getinfo(f"payload/{item['path']}")
            if info.is_dir() or info.file_size != item["size"]:
                raise ValueError("Backup metadata mismatch")
        stage = Path(tempfile.mkdtemp(prefix=".deskpilot-restore-", dir=destination.parent))
        try:
            for item in entries:
                target = stage.joinpath(*PurePosixPath(item["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with source.open(f"payload/{item['path']}") as input_stream, target.open("xb") as output:
                    size = 0
                    for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                        size += len(block)
                        if size > item["size"]:
                            raise ValueError("Backup entry expanded beyond declared size")
                        digest.update(block)
                        output.write(block)
                if size != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Backup checksum mismatch: {item['path']}")
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return manifest


def migrate_legacy(source: Path, destination: Path) -> dict:
    """Copy a former project data/ tree without touching the original."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deskpilot-migrate-", dir=destination.parent) as temporary:
        archive = Path(temporary) / "migration.dpb"
        passphrase = os.urandom(32).hex()
        create_backup(source, archive, passphrase)
        return restore_backup(archive, destination, passphrase)
