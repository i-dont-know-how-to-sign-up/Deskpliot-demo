"""End-to-end smoke check of the packaged, opt-in Docker sandbox API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _docker_image_id() -> str:
    engine = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    if engine.stdout.strip() != "linux":
        raise RuntimeError("Docker Desktop must use the Linux engine")
    image = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", "python:3.11-alpine"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    if not image.startswith("sha256:") or len(image) != 71:
        raise RuntimeError("Local python:3.11-alpine image did not resolve to a SHA-256 ID")
    return image


def _request(url: str, *, token: str = "", payload: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = secrets.token_hex(16)
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=5,
    ) as response:
        return json.load(response)


def _wait_for_ready(url: str, processes: list[subprocess.Popen], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("Packaged API or worker exited before becoming ready")
        try:
            if _request(url + "/health/ready").get("status") == "ready":
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError("Packaged API and worker did not become ready")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the packaged Docker sandbox API")
    parser.add_argument("--exe", type=Path, required=True, help="Newly packaged DeskPilot.exe")
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    image_id = _docker_image_id()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    with tempfile.TemporaryDirectory(prefix="deskpilot_packaged_sandbox_") as temp:
        home = Path(temp)
        index = home / "index" / "index.json"
        index.parent.mkdir()
        token = secrets.token_urlsafe(32)
        catalog = home / "tokens.json"
        catalog.write_text(json.dumps({"tokens": [{
            "principal": "smoke", "sha256": hashlib.sha256(token.encode()).hexdigest(),
            "scopes": ["execute", "read"],
        }]}), encoding="utf-8")
        host_secret = home / "synthetic_host_secret.txt"
        host_secret.write_text("synthetic-host-only", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "DESKPILOT_HOME": str(home),
            "DESKPILOT_ENABLE_DOCKER_SANDBOX": "1",
            "DESKPILOT_DOCKER_IMAGE": image_id,
        })
        kwargs = {"env": env, "stdin": subprocess.DEVNULL,
                  "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
        processes: list[subprocess.Popen] = []
        with (home / "worker.log").open("wb") as worker_log, (home / "api.log").open("wb") as api_log:
            try:
                worker = subprocess.Popen(
                    [str(exe), "-m", "deskpilot.tasks.worker", "--api", "--index", str(index)],
                    stdout=worker_log, stderr=subprocess.STDOUT, **kwargs,
                )
                processes.append(worker)
                api = subprocess.Popen(
                    [str(exe), "-m", "deskpilot.api", "--index", str(index),
                     "--tokens", str(catalog), "--host", "127.0.0.1", "--port", str(port)],
                    stdout=api_log, stderr=subprocess.STDOUT, **kwargs,
                )
                processes.append(api)
                _wait_for_ready(url, processes, 90)
                code = (
                    "import os, platform\n"
                    "from pathlib import Path\n"
                    "print('PLATFORM=' + platform.system())\n"
                    "print('UID=' + str(os.getuid()))\n"
                    "print('INPUT=' + Path('/inputs/note.txt').read_text())\n"
                    f"host_path = {str(host_secret)!r}\n"
                    "print('HOST_FILE_PRESENT=' + str(Path(host_path).exists()))\n"
                )
                compile(code, "<packaged-sandbox-smoke>", "exec")
                submitted = _request(url + "/v1/sandbox/tasks", token=token,
                                     payload={"code": code, "inputs": {"note.txt": "sandbox-fixture"}})
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    result = _request(url + "/v1/tasks/" + submitted["task_id"], token=token)
                    if result["status"] in {"succeeded", "failed", "cancelled", "partial"}:
                        break
                    time.sleep(0.4)
                else:
                    raise TimeoutError("Sandbox task did not finish")
                assert result["status"] == "succeeded", result
                sandbox = result["sandbox"]
                assert sandbox and sandbox["sandbox"] == "docker" and sandbox["ok"] is True, result
                answer = result["answer"] or ""
                for expected in ("PLATFORM=Linux", "UID=65534", "INPUT=sandbox-fixture",
                                 "HOST_FILE_PRESENT=False"):
                    assert expected in answer, result
                print("PASS: packaged EXE API + worker executed code in Docker")
                print("Image:", image_id)
                print("Task status:", result["status"])
                print("Sandbox:", sandbox["sandbox"], "returncode:", sandbox["returncode"])
                print(answer.strip())
            finally:
                for process in reversed(processes):
                    _stop(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
