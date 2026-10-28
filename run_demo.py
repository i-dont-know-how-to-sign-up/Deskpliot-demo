from __future__ import annotations

import runpy
import sys


def _main() -> int:
    from deskpilot.release.profile import configure_runtime, release_mode

    configure_runtime()
    if getattr(sys, "frozen", False) and len(sys.argv) >= 3 and sys.argv[1] == "-m":
        allowed = {
            "deskpilot.multimodal.providers.ocr_worker",
            "deskpilot.multimodal.providers.vision_embedding_worker",
            "deskpilot.tasks.worker",
            "deskpilot.api",
        }
        if sys.argv[2] not in allowed:
            raise SystemExit("Unsupported worker module")
        module = sys.argv[2]
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__")
        return 0
    if "--background-worker" in sys.argv:
        sys.argv = [sys.argv[0]]
        runpy.run_module("deskpilot.tasks.worker", run_name="__main__")
        return 0
    if "--maintenance" in sys.argv:
        from deskpilot.release.maintenance import run_maintenance

        return run_maintenance()
    if len(sys.argv) > 1 and sys.argv[1] in {
        "doctor", "set-home", "setup", "backup", "restore", "migrate", "diagnostics", "autostart", "candidate", "install-model",
    }:
        sys.argv = ["deskpilot.release", *sys.argv[1:]]
        runpy.run_module("deskpilot.release.__main__", run_name="__main__")
        return 0
    if release_mode() and "--smoke-ui" not in sys.argv:
        from deskpilot.release.onboarding import run_first_run

        if not run_first_run():
            return 0
        configure_runtime()
    from deskpilot.qt_app import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
