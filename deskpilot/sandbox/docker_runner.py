from __future__ import annotations

import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .broker import SandboxBroker


_IMAGE = re.compile(r"(?:[A-Za-z0-9._/-]+@)?sha256:[0-9a-f]{64}\Z")


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandboxRunner:
    """Run untrusted Python in a pinned, offline Linux container; never on the host."""

    def __init__(self, image: str, docker_binary: str = "docker", timeout_seconds: float = 10,
                 memory_limit: str = "256m", cpu_limit: str = "1", pids_limit: int = 32,
                 broker: SandboxBroker | None = None, max_output_bytes: int = 64 * 1024) -> None:
        if not isinstance(image, str) or not _IMAGE.fullmatch(image):
            raise ValueError("Docker image must be pinned by sha256 digest or local image ID")
        if not 0 < timeout_seconds <= 60 or not 1 <= pids_limit <= 128:
            raise ValueError("Invalid sandbox resource limit")
        if not re.fullmatch(r"[1-9][0-9]{1,3}m", memory_limit):
            raise ValueError("Invalid Docker memory limit")
        if not re.fullmatch(r"(?:0\.[1-9][0-9]?|[1-4])", cpu_limit):
            raise ValueError("Invalid Docker CPU limit")
        if not 1024 <= max_output_bytes <= 1024 * 1024:
            raise ValueError("Invalid sandbox output limit")
        self.image = image
        self.docker_binary = docker_binary
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.broker = broker or SandboxBroker()
        self.max_output_bytes = max_output_bytes

    @staticmethod
    def _docker_env(config_dir: Path) -> dict[str, str]:
        # Ignore user Docker contexts and inherited remote endpoints.
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"}
        env = {key: value for key, value in os.environ.items() if key.upper() in keep}
        env["DOCKER_CONFIG"] = str(config_dir)
        host = os.getenv("DESKPILOT_DOCKER_HOST", "")
        allowed = ({"npipe:////./pipe/docker_engine", "npipe:////./pipe/dockerDesktopLinuxEngine"}
                   if os.name == "nt" else {"unix:///var/run/docker.sock"})
        if host and host not in allowed:
            raise ValueError("Docker host must be a local approved socket")
        env["DOCKER_HOST"] = host or ("npipe:////./pipe/docker_engine" if os.name == "nt"
                                      else "unix:///var/run/docker.sock")
        return env

    def _check_engine(self, env: dict[str, str]) -> None:
        try:
            result = subprocess.run(
                [self.docker_binary, "info", "--format", "{{.OSType}}"],
                capture_output=True, text=True, timeout=5, check=False, env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable("Docker daemon is unavailable") from exc
        if result.returncode != 0 or result.stdout.strip() != "linux":
            raise SandboxUnavailable("A running Linux Docker engine is required")
        try:
            image = subprocess.run(
                [self.docker_binary, "image", "inspect", "--format", "{{.Id}}", self.image],
                capture_output=True, text=True, timeout=5, check=False, env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable("Pinned Docker image is unavailable") from exc
        if image.returncode != 0 or not _IMAGE.fullmatch(image.stdout.strip()):
            raise SandboxUnavailable("Pinned Docker image is not installed locally")

    def _argv(self, name: str, inputs: Path) -> list[str]:
        return [
            self.docker_binary, "run", "--rm", "--pull", "never", "--name", name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges=true", "--pids-limit", str(self.pids_limit),
            "--memory", self.memory_limit, "--memory-swap", self.memory_limit,
            "--cpus", self.cpu_limit, "--user", "65534:65534", "--init",
            "--ipc", "private",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
            "--ulimit", "nofile=64:64", "--workdir", "/tmp",
            "--mount", f"type=bind,source={inputs},target=/inputs,readonly",
            "--interactive", "--entrypoint", "python", self.image, "-I", "-",
        ]

    def _remove_container(self, name: str, env: dict[str, str]) -> bool:
        try:
            result = subprocess.run([self.docker_binary, "rm", "--force", name],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=5, check=False, env=env)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run_python(self, code: str, inputs: dict[str, bytes] | None = None) -> dict:
        try:
            code_bytes = code.encode("utf-8") if isinstance(code, str) else b""
        except UnicodeError as exc:
            raise ValueError("Sandbox code must be valid UTF-8") from exc
        if not isinstance(code, str) or not code.strip() or b"\x00" in code_bytes or len(code_bytes) > 8192:
            raise ValueError("Sandbox code must be 1-8192 UTF-8 bytes")
        validated = self.broker.validate(inputs)
        name = "deskpilot-sbx-" + secrets.token_hex(12)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="deskpilot_docker_cfg_") as config_dir, \
                self.broker.prepare_inputs(validated) as input_dir:
            env = self._docker_env(Path(config_dir))
            self._check_engine(env)
            try:
                process = subprocess.Popen(
                    self._argv(name, input_dir), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=env,
                )
            except OSError as exc:
                raise SandboxUnavailable("Docker execution could not start") from exc
            streams: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
            exceeded = threading.Event()
            lock = threading.Lock()
            total = 0

            def drain(label: str, pipe) -> None:
                nonlocal total
                while chunk := pipe.read1(4096):
                    with lock:
                        remaining = max(0, self.max_output_bytes - total)
                        streams[label].extend(chunk[:remaining])
                        total += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            exceeded.set()

            def feed() -> None:
                try:
                    assert process.stdin is not None
                    process.stdin.write(code_bytes)
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            readers = [threading.Thread(target=drain, args=(label, pipe), daemon=True)
                       for label, pipe in (("stdout", process.stdout), ("stderr", process.stderr))]
            for reader in readers:
                reader.start()
            threading.Thread(target=feed, daemon=True).start()
            timed_out = False
            cleanup_confirmed = True
            try:
                deadline = started + self.timeout_seconds
                while process.poll() is None:
                    if exceeded.is_set() or time.monotonic() >= deadline:
                        timed_out = not exceeded.is_set()
                        cleanup_confirmed = self._remove_container(name, env)
                        if process.poll() is None:
                            process.kill()
                        break
                    time.sleep(0.02)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
            finally:
                if process.poll() is None:
                    cleanup_confirmed = self._remove_container(name, env) and cleanup_confirmed
                    process.kill()
                for reader in readers:
                    reader.join(timeout=2)
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            stdout = streams["stdout"].decode("utf-8", errors="replace")
            stderr = streams["stderr"].decode("utf-8", errors="replace")
            return {
                "ok": process.returncode == 0 and not timed_out and not exceeded.is_set(),
                "stdout": stdout, "stderr": stderr, "returncode": process.returncode,
                "timed_out": timed_out, "output_limited": exceeded.is_set(),
                "cleanup_confirmed": cleanup_confirmed,
                "duration_seconds": round(time.monotonic() - started, 3), "sandbox": "docker",
            }
