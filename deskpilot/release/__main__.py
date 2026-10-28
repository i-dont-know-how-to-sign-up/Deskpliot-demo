from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import collect_diagnostics, export_diagnostics
from .profile import configure_runtime, profile_home, save_settings, set_profile_home
from .secrets import SecretStore
from .storage import create_backup, migrate_legacy, restore_backup
from .autostart import autostart_enabled, set_autostart
from .candidate import create_candidate_record
from .models import install_optional_model


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskPilot desktop release maintenance")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    location = sub.add_parser("set-home")
    location.add_argument("path", type=Path)
    setup = sub.add_parser("setup")
    setup.add_argument("--model", default="qwen3.7-plus")
    setup.add_argument("--workspace", type=Path)
    setup.add_argument("--api-key-stdin", action="store_true")
    backup = sub.add_parser("backup")
    backup.add_argument("archive", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("destination", type=Path)
    migrate = sub.add_parser("migrate")
    migrate.add_argument("legacy_data", type=Path)
    migrate.add_argument("destination", type=Path)
    export = sub.add_parser("diagnostics")
    export.add_argument("destination", type=Path)
    auto = sub.add_parser("autostart")
    auto.add_argument("action", choices=("status", "enable", "disable"))
    candidate = sub.add_parser("candidate")
    candidate.add_argument("output", type=Path)
    candidate.add_argument("--baseline", type=Path)
    candidate.add_argument("--executable", type=Path, help="Exact packaged EXE to fingerprint")
    model = sub.add_parser("install-model")
    model.add_argument("kind", choices=("siglip2", "ocr"))
    args = parser.parse_args()
    if args.command == "set-home":
        print(set_profile_home(args.path))
        return 0
    home = configure_runtime()
    if args.command == "setup":
        settings = {"QWEN_MODEL": args.model, "ALLOW_CLOUD_IMAGE_UPLOAD": "false"}
        if args.workspace:
            settings["TOOL_SAFE_ROOTS"] = str(args.workspace.expanduser().resolve(strict=True))
        save_settings(settings, home)
        if args.api_key_stdin:
            import sys

            SecretStore(home / "secrets.dpapi").set("DASHSCOPE_API_KEY", sys.stdin.readline().rstrip("\r\n"))
        print("Settings saved; restart DeskPilot to apply changes")
    elif args.command == "doctor":
        print(json.dumps(collect_diagnostics(home), ensure_ascii=False, indent=2))
    elif args.command == "backup":
        from getpass import getpass

        password = getpass("Backup passphrase (12+ characters): ")
        if password != getpass("Repeat passphrase: "):
            raise SystemExit("Passphrases do not match")
        print(json.dumps(create_backup(home / "data", args.archive, password), ensure_ascii=False))
    elif args.command == "restore":
        from getpass import getpass

        print(json.dumps(restore_backup(args.archive, args.destination, getpass("Backup passphrase: ")), ensure_ascii=False))
    elif args.command == "migrate":
        print(json.dumps(migrate_legacy(args.legacy_data, args.destination), ensure_ascii=False))
    elif args.command == "diagnostics":
        export_diagnostics(args.destination, home)
        print(args.destination)
    elif args.command == "autostart":
        if args.action != "status":
            set_autostart(args.action == "enable")
        print("enabled" if autostart_enabled() else "disabled")
    elif args.command == "candidate":
        print(json.dumps(create_candidate_record(Path.cwd(), args.output, args.baseline,
                                                 executable_path=args.executable),
                         ensure_ascii=False, indent=2))
    elif args.command == "install-model":
        print(install_optional_model(args.kind, home))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
