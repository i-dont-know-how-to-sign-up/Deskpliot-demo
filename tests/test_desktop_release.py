from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest

from deskpilot.release.diagnostics import collect_diagnostics, export_diagnostics
from deskpilot.release.candidate import create_candidate_record
from deskpilot.release.feedback import save_local_feedback
from deskpilot.release.profile import configure_runtime, read_settings, save_settings, set_profile_home
from deskpilot.release.secrets import SecretStore
from deskpilot.release.storage import _encrypt, create_backup, migrate_legacy, restore_backup
from deskpilot.release.models import install_optional_model
from deskpilot.tools.permissions import assess_command, assess_python_code, safe_roots


def test_release_profile_is_external_and_settings_reject_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_DATA_DIR", "")
    monkeypatch.delenv("DESKPILOT_HOME", raising=False)
    home = set_profile_home(tmp_path / "profile")
    save_settings({"QWEN_MODEL": "test-model", "ALLOW_CLOUD_IMAGE_UPLOAD": "false"}, home)
    assert configure_runtime() == home
    assert os.environ["DESKPILOT_DATA_DIR"] == str(home / "data")
    assert read_settings(home)["QWEN_MODEL"] == "test-model"
    with pytest.raises(ValueError):
        save_settings({"DASHSCOPE_API_KEY": "do-not-save-plain-text"}, home)


def test_vault_stores_ciphertext_and_limits_field_names(tmp_path: Path) -> None:
    vault = SecretStore(
        tmp_path / "secrets.dpapi",
        protect=lambda value: bytes(item ^ 0x5A for item in value),
        unprotect=lambda value: bytes(item ^ 0x5A for item in value),
    )
    vault.set("DASHSCOPE_API_KEY", "secret-value-123")
    assert b"secret-value-123" not in vault.path.read_bytes()
    assert vault.get("DASHSCOPE_API_KEY") == "secret-value-123"
    with pytest.raises(ValueError):
        vault.set("ARBITRARY_PASSWORD", "x")


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_release_config_reads_dpapi_and_skips_project_dotenv(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.core import config

    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    SecretStore(tmp_path / "secrets.dpapi").set("DASHSCOPE_API_KEY", "vault-key")
    monkeypatch.setattr(config, "_load_dotenv", lambda *_: pytest.fail("release must not read source .env"))
    assert config.load_config().llm_api_key == "vault-key"


def test_beta_exec_is_blocked_and_source_root_not_safe(monkeypatch) -> None:
    from deskpilot.core.config import ROOT_DIR, WORKSPACE_DIR

    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.delenv("TOOL_SAFE_ROOTS", raising=False)
    assert assess_python_code("print(1)").blocked
    assert assess_command("ls").blocked
    assert ROOT_DIR not in safe_roots()
    assert WORKSPACE_DIR in safe_roots()
    from deskpilot.tools.tool_registry import build_default_tool_registry

    result = build_default_tool_registry(None).call_approved("code.execute_python", code="print('never run')")
    assert not result.ok or not (result.output or {}).get("executed", False)


def test_encrypted_backup_restores_sqlite_snapshot_without_overwriting(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "index").mkdir(parents=True)
    (source / "workspace" / "docs").mkdir(parents=True)
    (source / "workspace" / "docs" / "note.md").write_text("private example", encoding="utf-8")
    database = source / "index" / "memory.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample(value TEXT)")
        connection.execute("INSERT INTO sample VALUES('remember')")
    archive = tmp_path / "release.dpb"
    manifest = create_backup(source, archive, "long-passphrase-123")
    assert b"private example" not in archive.read_bytes()
    assert archive.read_bytes().startswith(b"DPBACK1\n")
    assert len(manifest["files"]) == 2
    target = tmp_path / "restored"
    with pytest.raises(ValueError):
        restore_backup(archive, target, "wrong-passphrase")
    assert not target.exists()
    restore_backup(archive, target, "long-passphrase-123")
    assert (target / "workspace" / "docs" / "note.md").read_text(encoding="utf-8") == "private example"
    with sqlite3.connect(target / "index" / "memory.sqlite3") as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == ("remember",)
    with pytest.raises(FileExistsError):
        restore_backup(archive, target, "long-passphrase-123")


def test_backup_excludes_nonportable_dpapi_secrets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "note.txt").write_text("keep", encoding="utf-8")
    (source / "secrets.dpapi").write_bytes(b"user-bound-ciphertext")
    archive = tmp_path / "backup.dpb"
    manifest = create_backup(source, archive, "long-passphrase-123")
    assert [item["path"] for item in manifest["files"]] == ["note.txt"]
    target = tmp_path / "restored"
    restore_backup(archive, target, "long-passphrase-123")
    assert (target / "note.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "secrets.dpapi").exists()


def test_restore_rejects_zip_traversal_before_writing(tmp_path: Path) -> None:
    plain = tmp_path / "malicious.zip"
    with zipfile.ZipFile(plain, "w") as output:
        output.writestr("../outside.txt", "bad")
        output.writestr("manifest.json", json.dumps({
            "schema": 1,
            "files": [{"path": "../outside.txt", "size": 3, "sha256": "0" * 64}],
        }))
    archive = tmp_path / "malicious.dpb"
    _encrypt(plain, archive, "long-passphrase-123")
    with pytest.raises(ValueError):
        restore_backup(archive, tmp_path / "restored", "long-passphrase-123")
    assert not (tmp_path / "outside.txt").exists()


def test_backup_refuses_live_worker_lease(tmp_path: Path) -> None:
    source = tmp_path / "data"
    (source / "index").mkdir(parents=True)
    with sqlite3.connect(source / "index" / "task_ledger.sqlite3") as connection:
        connection.execute("CREATE TABLE runtime_worker_lock(id INTEGER PRIMARY KEY, owner TEXT, expires REAL)")
        connection.execute("INSERT INTO runtime_worker_lock VALUES(1, 'worker', 99999999999)")
    with pytest.raises(RuntimeError, match="Stop DeskPilot"):
        create_backup(source, tmp_path / "backup.dpb", "long-passphrase-123")
    assert not (tmp_path / "backup.dpb").exists()


def test_migration_leaves_original_and_diagnostics_exclude_private_data(tmp_path: Path) -> None:
    legacy = tmp_path / "old" / "data"
    legacy.mkdir(parents=True)
    (legacy / "note.txt").write_text("PRIVATE_MAIL_BODY", encoding="utf-8")
    home = tmp_path / "new"
    migrate_legacy(legacy, home / "data")
    assert (legacy / "note.txt").exists()
    assert (home / "data" / "note.txt").read_text(encoding="utf-8") == "PRIVATE_MAIL_BODY"
    save_settings({"QWEN_MODEL": "test"}, home)
    exported = tmp_path / "diagnostics.json"
    export_diagnostics(exported, home)
    assert "PRIVATE_MAIL_BODY" not in exported.read_text(encoding="utf-8")
    assert "note.txt" not in exported.read_text(encoding="utf-8")
    assert collect_diagnostics(home)["configured"]
    feedback = save_local_feedback("Task did not finish", home)
    assert json.loads(feedback.read_text(encoding="utf-8"))["uploaded"] is False


def test_candidate_manifest_is_not_a_release_claim(tmp_path: Path) -> None:
    (tmp_path / "requirements-desktop.lock").write_text("package==1\n", encoding="utf-8")
    executable = tmp_path / "dist" / "DeskPilot" / "DeskPilot.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test executable")
    report = create_candidate_record(tmp_path, tmp_path / "candidate.json")
    assert report["release_status"] == "candidate_not_accepted"
    assert report["release_tag"] == "0.9.0-beta.3"
    assert report["executable_path"] == str(executable.resolve())
    assert report["baseline"] is None
    assert len(report["release_blockers"]) >= 3
    assert any("Docker sandbox live" in item for item in report["release_blockers"])
    with pytest.raises(FileExistsError):
        create_candidate_record(tmp_path, tmp_path / "candidate.json")


def test_candidate_manifest_can_fingerprint_versioned_build(tmp_path: Path) -> None:
    (tmp_path / "requirements-desktop.lock").write_text("package==1\n", encoding="utf-8")
    old_executable = tmp_path / "dist" / "DeskPilot" / "DeskPilot.exe"
    old_executable.parent.mkdir(parents=True)
    old_executable.write_bytes(b"old build")
    current_executable = tmp_path / "dist" / "beta2" / "DeskPilot" / "DeskPilot.exe"
    current_executable.parent.mkdir(parents=True)
    current_executable.write_bytes(b"new build")

    report = create_candidate_record(tmp_path, tmp_path / "candidate.json",
                                     executable_path=current_executable)
    assert report["executable_path"] == str(current_executable.resolve())
    assert report["executable_sha256"] == hashlib.sha256(b"new build").hexdigest()


def test_optional_model_name_is_allowlisted(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        install_optional_model("arbitrary-repo", tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_email_provider_reads_vault_without_network(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.mcp.email_mcp import EmailMCPService, ImapSmtpEmailProvider

    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path))
    monkeypatch.setenv("EMAIL_PROVIDER", "qq")
    monkeypatch.setenv("EMAIL_USERNAME", "test@example.com")
    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    SecretStore(tmp_path / "secrets.dpapi").set("EMAIL_PASSWORD", "mail-authorization-code")
    provider = EmailMCPService().provider
    assert isinstance(provider, ImapSmtpEmailProvider)
    assert provider.password == "mail-authorization-code"


@pytest.mark.skipif(os.name != "nt", reason="Windows first-run uses DPAPI")
def test_first_run_dialog_writes_profile_before_agent_import(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("DESKPILOT_DATA_DIR", "")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLineEdit
    from deskpilot.release.onboarding import run_first_run

    application = QApplication.instance() or QApplication([])

    def complete() -> None:
        dialog = next(item for item in application.topLevelWidgets()
                      if isinstance(item, QDialog) and "首次设置" in item.windowTitle())
        key = next(item for item in dialog.findChildren(QLineEdit) if item.accessibleName() == "模型 API Key")
        key.setText("dummy-key-for-ui-test")
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).click()

    QTimer.singleShot(100, complete)
    assert run_first_run()
    assert read_settings(tmp_path / "profile")["QWEN_MODEL"] == "qwen3.7-plus"
    assert SecretStore(tmp_path / "profile" / "secrets.dpapi").get("DASHSCOPE_API_KEY") == "dummy-key-for-ui-test"
