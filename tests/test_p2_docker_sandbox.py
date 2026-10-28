"""Security contract for the opt-in Docker Python sandbox.

The mocked tests inspect the command sent to Docker. The real tests require an
explicit opt-in and a Python image that is already present locally; they never
pull images or access real host secrets.
"""

from __future__ import annotations

import os
import io
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from deskpilot.sandbox.docker_runner import DockerSandboxRunner, SandboxUnavailable


IMAGE = "sha256:" + "a" * 64
DEFAULT_LOCAL_IMAGE = "python:3.11-alpine"


def _docker_option(argv: list[str], name: str) -> str | None:
    for position, value in enumerate(argv):
        if value == name and position + 1 < len(argv):
            return argv[position + 1]
        if value.startswith(name + "="):
            return value[len(name) + 1:]
    return None


def _run_argv(calls: list[list[str]]) -> list[str]:
    matches = [argv for argv in calls if "run" in argv]
    assert len(matches) == 1, f"Expected exactly one docker run, got {calls!r}"
    argv = matches[0]
    assert isinstance(argv, list), "Docker must be invoked with an argument vector"
    return argv


class _FakeDockerProcess:
    def __init__(self, owner: "_FakeDocker", argv: list[str], *, text: bool = False) -> None:
        self.owner = owner
        self.argv = argv
        self.text = text
        self.returncode = None if owner.timeout and "run" in argv else 0
        self.pid = 12345
        self.killed = False
        self.stdin = io.BytesIO()
        self.stdout = _FakePipe(b"sandbox-ok\n")
        self.stderr = _FakePipe()

    def communicate(self, input=None, timeout=None):
        if self.owner.timeout and "run" in self.argv and not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout or 1)
        self.returncode = 0
        if self.text:
            return "sandbox-ok\n", ""
        return b"sandbox-ok\n", b""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if not self.killed else -9
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    terminate = kill

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        if self.returncode is None:
            self.wait()


class _FakeDocker:
    def __init__(self, *, timeout: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.timeout = timeout
        self.processes: list[_FakeDockerProcess] = []
        self.broker_files: dict[str, bytes] | None = None

    def run(self, args, *unused, **kwargs):
        assert not unused
        assert kwargs.get("shell") is not True
        assert isinstance(args, (list, tuple))
        argv = [str(part) for part in args]
        self.calls.append(argv)
        if self.timeout and "run" in argv:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 1))
        if "info" in argv:
            value = "linux\n"
        elif "inspect" in argv:
            value = IMAGE + "\n"
        else:
            value = "sandbox-ok\n"
        output = value if kwargs.get("text") or kwargs.get("encoding") else value.encode()
        error = "" if isinstance(output, str) else b""
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr=error)

    def popen(self, args, *unused, **kwargs):
        assert not unused
        assert kwargs.get("shell") is not True
        assert isinstance(args, (list, tuple))
        argv = [str(part) for part in args]
        self.calls.append(argv)
        mount = _docker_option(argv, "--mount")
        if mount:
            parts = dict(part.split("=", 1) for part in mount.split(",") if "=" in part)
            source = parts.get("source") or parts.get("src")
            if source:
                self.broker_files = {entry.name: entry.read_bytes() for entry in Path(source).iterdir()}
        process = _FakeDockerProcess(self, argv, text=bool(kwargs.get("text") or kwargs.get("encoding")))
        self.processes.append(process)
        return process


class _FakePipe(io.BytesIO):
    def read1(self, size=-1):
        return self.read(size)


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> _FakeDocker:
    fake = _FakeDocker()
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    original_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: "docker" if name == "docker" else original_which(name, *args, **kwargs))
    return fake


def _runner(**overrides) -> DockerSandboxRunner:
    settings = dict(
        image=IMAGE,
        docker_binary="docker",
        timeout_seconds=10,
        memory_limit="256m",
        cpu_limit="1",
        pids_limit=32,
    )
    settings.update(overrides)
    return DockerSandboxRunner(**settings)


def _assert_rejected(call, fake: _FakeDocker) -> None:
    before = len([argv for argv in fake.calls if "run" in argv])
    try:
        result = call()
    except (TypeError, ValueError):
        pass
    else:
        assert isinstance(result, dict) and result.get("ok") is False, result
    after = len([argv for argv in fake.calls if "run" in argv])
    assert after == before, "Invalid input must be rejected before docker run"


def test_docker_command_uses_isolation_and_resource_limits(fake_docker: _FakeDocker) -> None:
    result = _runner().run_python("print('sandbox-ok')", inputs={"allowed.txt": b"fixture"})
    assert result["ok"] is True
    argv = _run_argv(fake_docker.calls)
    assert _docker_option(argv, "--network") == "none"
    assert "--read-only" in argv
    security = _docker_option(argv, "--security-opt") or ""
    assert "no-new-privileges" in security
    assert (_docker_option(argv, "--cap-drop") or "").upper() == "ALL"
    assert _docker_option(argv, "--pids-limit") == "32"
    assert _docker_option(argv, "--memory") == "256m"
    assert _docker_option(argv, "--cpus") == "1"
    assert _docker_option(argv, "--pull") == "never"
    assert "--rm" in argv
    user = _docker_option(argv, "--user")
    assert user and user.split(":", 1)[0] not in {"0", "root"}
    assert "--privileged" not in argv
    assert "--pid=host" not in argv
    assert "--pid" not in argv
    assert "--uts" not in argv
    assert "--ipc=host" not in argv
    assert "--userns=host" not in argv
    assert "--env" not in argv and "-e" not in argv


def test_only_private_read_only_broker_inputs_may_be_host_mounted(fake_docker: _FakeDocker) -> None:
    _runner().run_python("print('ok')", inputs={"allowed.txt": b"fixture"})
    argv = _run_argv(fake_docker.calls)
    mounts: list[tuple[str, bool]] = []
    for position, arg in enumerate(argv):
        if arg == "--mount" or arg.startswith("--mount="):
            value = argv[position + 1] if arg == "--mount" else arg.split("=", 1)[1]
            parts = dict(part.split("=", 1) for part in value.split(",") if "=" in part)
            mounts.append((parts.get("src") or parts.get("source") or "", "readonly" in value or "ro" in value.split(",")))
        if arg in {"-v", "--volume"} or arg.startswith("--volume="):
            value = argv[position + 1] if arg in {"-v", "--volume"} else arg.split("=", 1)[1]
            source, _destination, mode = value.rsplit(":", 2)
            mounts.append((source, "ro" in mode.split(",")))
    assert len(mounts) <= 1, f"Unexpected host bind mounts: {mounts!r}"
    for source, read_only in mounts:
        assert read_only, "Broker inputs must be mounted read-only"
        path = Path(source).resolve()
        assert path.is_relative_to(Path(tempfile.gettempdir()).resolve())
        assert path != Path.cwd().resolve()
        assert path != Path.home().resolve()
        assert path.name == "inputs" and path.parent.name.startswith("deskpilot_sandbox_")
    assert fake_docker.broker_files == {"allowed.txt": b"fixture"}


@pytest.mark.parametrize("image", ["", "-v", "python:3.11 --privileged", "$(whoami)", "repo/image\n--network=host"])
def test_invalid_image_is_rejected_before_execution(fake_docker: _FakeDocker, image: str) -> None:
    _assert_rejected(lambda: _runner(image=image).run_python("print(1)"), fake_docker)


@pytest.mark.parametrize("code", [None, b"print(1)", "", " \n\t ", "print(1)\x00", "x" * 8193])
def test_invalid_code_is_rejected_before_execution(fake_docker: _FakeDocker, code) -> None:
    _assert_rejected(lambda: _runner().run_python(code), fake_docker)


@pytest.mark.parametrize("name", ["", ".", "..", "../escape.txt", "nested/../../escape.txt", "/etc/passwd", "..\\escape.txt", "C:\\secret.txt", "bad\x00name"])
def test_invalid_input_name_is_rejected_before_execution(fake_docker: _FakeDocker, name: str) -> None:
    _assert_rejected(lambda: _runner().run_python("print(1)", inputs={name: b"fixture"}), fake_docker)


def test_non_bytes_input_is_rejected_before_execution(fake_docker: _FakeDocker) -> None:
    _assert_rejected(lambda: _runner().run_python("print(1)", inputs={"file.txt": "not bytes"}), fake_docker)


def test_timeout_marks_result_and_cleans_container(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDocker(timeout=True)
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: "docker" if name == "docker" else None)
    result = _runner(timeout_seconds=0.1).run_python("import time; time.sleep(30)")
    assert result["timed_out"] is True
    assert result["ok"] is False
    assert any("run" in argv for argv in fake.calls)
    run = _run_argv(fake.calls)
    container_name = _docker_option(run, "--name")
    assert container_name
    assert any(("rm" in argv or "kill" in argv) and container_name in argv
               for argv in fake.calls if "run" not in argv), (
        "Timing out a Docker CLI is insufficient: the container must be explicitly killed/removed"
    )


def test_unavailable_daemon_does_not_start_any_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def unavailable(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="daemon unavailable")

    def forbidden_popen(*args, **kwargs):
        pytest.fail("No process may start after Docker daemon check fails")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    with pytest.raises(SandboxUnavailable, match="Linux Docker engine"):
        _runner().run_python("print('must not run')")
    assert len(calls) == 1 and "info" in calls[0]


def test_remote_docker_endpoint_is_rejected(monkeypatch: pytest.MonkeyPatch, fake_docker: _FakeDocker) -> None:
    monkeypatch.setenv("DESKPILOT_DOCKER_HOST", "tcp://example.invalid:2375")
    with pytest.raises(ValueError, match="local approved socket"):
        _runner().run_python("print('must not run')")
    assert fake_docker.calls == []


@pytest.fixture
def real_docker() -> tuple[str, str]:
    if os.environ.get("DESKPILOT_RUN_DOCKER_SECURITY_TESTS") != "1":
        pytest.skip("Set DESKPILOT_RUN_DOCKER_SECURITY_TESTS=1 for live Docker security tests")
    binary = shutil.which("docker")
    if not binary:
        pytest.fail("Docker CLI is unavailable; real sandbox acceptance cannot run")
    image = os.environ.get("DESKPILOT_TEST_DOCKER_IMAGE", DEFAULT_LOCAL_IMAGE)
    with tempfile.TemporaryDirectory(prefix="deskpilot_docker_test_cfg_") as config:
        env = DockerSandboxRunner._docker_env(Path(config))
        try:
            daemon = subprocess.run([binary, "info", "--format", "{{.OSType}}"],
                                    capture_output=True, text=True, timeout=5, env=env)
            inspect = subprocess.run([binary, "image", "inspect", "--format", "{{.Id}}", image],
                                     capture_output=True, text=True, timeout=5, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            pytest.fail(f"Docker daemon unavailable: {exc}")
    if daemon.returncode or daemon.stdout.strip() != "linux":
        pytest.fail("A running Linux Docker engine is required for real sandbox acceptance")
    if inspect.returncode:
        pytest.fail(f"Local Python image {image!r} is unavailable; no pull attempted")
    image_id = inspect.stdout.strip()
    if not image_id.startswith("sha256:"):
        pytest.fail(f"Local image {image!r} did not resolve to a pinned sha256 image ID")
    return binary, image_id


def _real_runner(real_docker: tuple[str, str], *, timeout_seconds: float = 5) -> DockerSandboxRunner:
    binary, image = real_docker
    return DockerSandboxRunner(image=image, docker_binary=binary, timeout_seconds=timeout_seconds)


def test_real_docker_cannot_read_synthetic_host_secret(real_docker, tmp_path: Path, monkeypatch) -> None:
    marker = "synthetic-secret-do-not-leak-" + tmp_path.name
    secret = tmp_path / "synthetic_host_secret.txt"
    secret.write_text(marker, encoding="utf-8")
    monkeypatch.setenv("DESKPILOT_SYNTHETIC_HOST_SECRET", marker)
    code = (
        "import os, pathlib\n"
        f"p = pathlib.Path({str(secret)!r})\n"
        "try:\n print('HOST_FILE=' + p.read_text())\n"
        "except (OSError, ValueError):\n print('HOST_FILE=denied')\n"
        "print('HOST_ENV=' + os.environ.get('DESKPILOT_SYNTHETIC_HOST_SECRET', 'denied'))\n"
    )
    result = _real_runner(real_docker).run_python(code, inputs={"allowed.txt": b"fixture"})
    assert result["ok"] is True, result
    assert marker not in result["stdout"]
    assert "HOST_FILE=denied" in result["stdout"]
    assert "HOST_ENV=denied" in result["stdout"]


def test_real_docker_rejects_broker_path_traversal(real_docker, tmp_path: Path) -> None:
    secret = tmp_path / "synthetic_host_secret.txt"
    secret.write_bytes(b"synthetic-host-only")
    runner = _real_runner(real_docker)
    with pytest.raises((TypeError, ValueError)):
        runner.run_python("print(1)", inputs={"../synthetic_host_secret.txt": secret.read_bytes()})
    result = runner.run_python(
        "from pathlib import Path\n"
        "try:\n print('TRAVERSAL=' + Path('../synthetic_host_secret.txt').read_text())\n"
        "except OSError:\n print('TRAVERSAL=denied')\n",
        inputs={"allowed.txt": b"fixture"},
    )
    assert result["ok"] is True, result
    assert "TRAVERSAL=denied" in result["stdout"]


def test_real_docker_has_no_dns_or_http_egress(real_docker) -> None:
    code = (
        "import socket, urllib.request\n"
        "try:\n socket.getaddrinfo('example.com', 80); print('DNS=allowed')\n"
        "except OSError:\n print('DNS=denied')\n"
        "try:\n urllib.request.urlopen('http://1.1.1.1/', timeout=1); print('HTTP=allowed')\n"
        "except Exception:\n print('HTTP=denied')\n"
    )
    result = _real_runner(real_docker, timeout_seconds=15).run_python(code)
    assert result["ok"] is True, result
    assert "DNS=denied" in result["stdout"]
    assert "HTTP=denied" in result["stdout"]


def test_real_docker_caps_output_and_cleans_timed_out_child(real_docker) -> None:
    runner = _real_runner(real_docker, timeout_seconds=2)
    noisy = runner.run_python("import sys; sys.stdout.write('X' * 2000000)")
    assert len(noisy["stdout"]) <= 1_048_576, "Unbounded output can exhaust the API worker"

    binary, image = real_docker
    def containers() -> set[str]:
        found = subprocess.run(
            [binary, "ps", "-aq", "--filter", f"ancestor={image}"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return set(found.stdout.splitlines())

    before = containers()
    started = time.monotonic()
    result = runner.run_python(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "time.sleep(30)\n"
    )
    assert result["timed_out"] is True, result
    assert time.monotonic() - started < 10
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and containers() - before:
        time.sleep(0.2)
    assert not containers() - before, "Timed-out container or child was left running"


def test_real_docker_enforces_child_process_limit(real_docker) -> None:
    code = (
        "import subprocess, sys\n"
        "children = []\n"
        "try:\n"
        " for _ in range(40):\n"
        "  try: children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']))\n"
        "  except OSError: break\n"
        " print('CHILD_COUNT=' + str(len(children)))\n"
        "finally:\n"
        " for child in children: child.terminate()\n"
        " for child in children: child.wait()\n"
    )
    result = _real_runner(real_docker, timeout_seconds=8).run_python(code)
    assert result["ok"] is True, result
    count = int(next(line.split("=", 1)[1] for line in result["stdout"].splitlines() if line.startswith("CHILD_COUNT=")))
    assert count < 40, "pids_limit=32 did not stop child process creation"


def test_real_docker_does_not_share_tmp_between_tasks(real_docker) -> None:
    first = _real_runner(real_docker).run_python(
        "from pathlib import Path\nPath('/tmp/task_one_only').write_text('synthetic-private')\nprint('written')\n"
    )
    assert first["ok"] is True, first
    second = _real_runner(real_docker).run_python(
        "from pathlib import Path\nprint('PRESENT=' + str(Path('/tmp/task_one_only').exists()))\n"
    )
    assert second["ok"] is True, second
    assert "PRESENT=False" in second["stdout"]


def test_real_docker_limits_tmpfs_and_memory(real_docker) -> None:
    runner = _real_runner(real_docker, timeout_seconds=6)
    disk = runner.run_python(
        "try:\n"
        " with open('/tmp/too_large', 'wb') as f: f.write(b'x' * (40 * 1024 * 1024))\n"
        " print('DISK=unlimited')\n"
        "except OSError:\n print('DISK=limited')\n"
    )
    assert "DISK=limited" in disk["stdout"] and "DISK=unlimited" not in disk["stdout"], disk
    memory = runner.run_python(
        "items = []\n"
        "try:\n"
        " while True: items.append(bytearray(16 * 1024 * 1024))\n"
        "except MemoryError:\n print('MEMORY=limited')\n"
    )
    assert not memory["timed_out"], memory
    assert memory["returncode"] != 0 or "MEMORY=limited" in memory["stdout"], memory
