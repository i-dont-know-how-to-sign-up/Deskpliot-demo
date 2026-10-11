from __future__ import annotations

import asyncio
import ctypes
import io
import json
import os
import smtplib
import ssl
import sys
import threading
import time
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import load_config
from deskpilot.mcp.email_mcp import ImapSmtpEmailProvider
from deskpilot.tasks.control import (
    ExecutionControl, TaskBudgetExceeded, TaskCancelled, TaskControlError,
    TaskDeadlineExceeded, TaskPaused, execution_scope, get_execution_control,
)
from deskpilot.tasks.retry import ErrorCategory, RetryPolicy, UnknownSideEffect, classify_error
from deskpilot.tools.execution_tools import _run_process, execute_command, execute_python_code


def client(*, fallback: bool = False) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(replace(load_config(), llm_api_key="test",
                                          embedding_api_key="test", allow_local_fallback=fallback))


class Response:
    def __init__(self, value=None, *, lines=None):
        self.value = value if value is not None else {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        self.lines = lines or []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def read(self, *_args):
        return json.dumps(self.value).encode()

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def test_scope_restores_and_does_not_leak_to_other_tasks():
    first = ExecutionControl(task_id="first")
    second = ExecutionControl(task_id="second")

    async def run(control):
        with execution_scope(control, node_id="node", attempt_id="attempt"):
            await asyncio.sleep(0)
            assert get_execution_control() is control
            control.record_tool("example")

    async def both():
        await asyncio.gather(run(first), run(second))

    asyncio.run(both())
    assert get_execution_control() is None
    assert first.events[0]["task_id"] == "first"
    assert second.events[0]["task_id"] == "second"
    with execution_scope(first, node_id="outer"):
        with execution_scope(first, node_id="inner", attempt_id="retry"):
            assert first.attribution()["node_id"] == "inner"
        assert first.attribution()["node_id"] == "outer"
        with execution_scope(None):
            assert get_execution_control() is None
        assert get_execution_control() is first


@pytest.mark.parametrize("method,error", [("request_cancel", TaskCancelled), ("request_pause", TaskPaused)])
def test_wait_is_interruptible(method, error):
    control = ExecutionControl()
    trigger = threading.Timer(0.05, getattr(control, method))
    trigger.start()
    started = time.monotonic()
    try:
        with pytest.raises(error):
            control.wait(10)
    finally:
        trigger.join()
    assert time.monotonic() - started < 0.5
    if error is TaskPaused:
        control.resume()
        control.checkpoint()


def test_deadline_limits_timeout_and_wait():
    control = ExecutionControl(deadline=time.monotonic() + 0.08)
    assert 0 < control.timeout(60) <= 0.08
    with pytest.raises(TaskDeadlineExceeded):
        control.wait(10)


def test_shared_tool_attempt_token_budgets_are_checked_before_dispatch():
    control = ExecutionControl(max_tool_calls=1, max_attempts=1, max_tokens=10)
    control.record_tool("first")
    with pytest.raises(TaskBudgetExceeded):
        control.record_tool("second")
    sequence = control.record_attempt(reserved_tokens=8)
    assert control.remaining_tokens() == 2
    control.record_usage({"total_tokens": 9}, api_attempt=sequence)
    assert control.remaining_tokens() == 1
    with pytest.raises(TaskBudgetExceeded):
        control.record_attempt()
    assert control.usage["api_attempts"] == 1


def test_node_budgets_and_attribution_share_one_control():
    control = ExecutionControl(task_id="t", max_node_attempts=1)
    for node in ("a", "b"):
        with execution_scope(control, node_id=node, attempt_id="1"):
            control.record_attempt()
            with pytest.raises(TaskBudgetExceeded):
                control.record_attempt()
    assert [event["node_id"] for event in control.events] == ["a", "b"]
    assert control.usage["api_attempts"] == 2


@pytest.mark.parametrize("error,category", [
    (urllib.error.URLError("temporary"), ErrorCategory.TRANSIENT),
    (TimeoutError(), ErrorCategory.TRANSIENT),
    (urllib.error.HTTPError("url", 429, "rate", {}, None), ErrorCategory.TRANSIENT),
    (urllib.error.HTTPError("url", 401, "auth", {}, None), ErrorCategory.AUTH),
    (urllib.error.HTTPError("url", 400, "bad", {}, None), ErrorCategory.INVALID_INPUT),
    (PermissionError(), ErrorCategory.AUTH), (ValueError(), ErrorCategory.INVALID_INPUT),
    (TaskPaused(), ErrorCategory.CONTROL), (UnknownSideEffect(), ErrorCategory.UNKNOWN_SIDE_EFFECT),
    (urllib.error.URLError(ssl.SSLCertVerificationError()), ErrorCategory.PERMANENT),
    (smtplib.SMTPAuthenticationError(535, b"bad credentials"), ErrorCategory.AUTH),
])
def test_retry_classification(error, category):
    assert classify_error(error) == category
    assert RetryPolicy().allows(error, 1) is (category == ErrorCategory.TRANSIENT)
    assert not RetryPolicy().allows(error, 3)
    assert not RetryPolicy().allows(error, 1, idempotent=False)


def test_retry_budget_is_shared_across_layers_and_nodes():
    policy = RetryPolicy(max_task_retries=2, max_node_retries=1, base_delay=0, jitter=0)
    control = ExecutionControl(retry_policy=policy)
    error = TimeoutError()
    with execution_scope(control, node_id="a"):
        assert policy.allows(error, 1, control=control)
        assert not policy.allows(error, 1, control=control)
    with execution_scope(control, node_id="b"):
        assert policy.allows(error, 1, control=control)
    with execution_scope(control, node_id="c"):
        assert not policy.allows(error, 1, control=control)
    assert control.usage["retries"] == 2


def test_api_retries_count_each_failure_and_server_usage_once(monkeypatch):
    responses = iter([urllib.error.URLError("temporary"),
                      urllib.error.HTTPError("url", 503, "busy", {}, io.BytesIO(b'{}')), Response()])
    timeouts = []

    def open_url(_request, timeout):
        timeouts.append(timeout)
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    api = client()
    control = ExecutionControl(task_id="t", deadline=time.monotonic() + 2,
                               retry_policy=RetryPolicy(base_delay=0, jitter=0))
    with execution_scope(control, node_id="n", attempt_id="a"):
        assert api.chat([{"role": "user", "content": "test"}]) == "ok"
    assert len(timeouts) == 3
    assert all(0 < timeout <= 2 for timeout in timeouts)
    assert control.usage["api_attempts"] == 3
    assert control.usage["failed_attempts"] == 2
    assert control.usage["unknown_usage_calls"] == 2
    assert control.usage["total_tokens"] == api.token_usage["total_tokens"] == 7
    assert len(api.usage_events) == 3
    assert [event["api_attempt"] for event in api.usage_events] == [1, 2, 3]
    assert all(event["node_id"] == "n" for event in api.usage_events)


def test_api_reserves_and_caps_output_before_dispatch(monkeypatch):
    captured = []

    def open_url(request, timeout):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    control = ExecutionControl(max_tokens=100)
    with execution_scope(control):
        client().chat([{"role": "user", "content": "test"}], max_tokens=200)
    assert 0 < captured[0]["max_tokens"] < 100
    assert control.usage["reserved_tokens"] == 0
    exhausted = ExecutionControl(max_tokens=1)
    with execution_scope(exhausted), pytest.raises(TaskBudgetExceeded):
        client(fallback=True).chat([{"role": "user", "content": "test"}])
    assert len(captured) == 1


def test_api_http_failure_usage_is_not_lost(monkeypatch):
    body = io.BytesIO(b'{"usage":{"total_tokens":11}}')
    error = urllib.error.HTTPError("url", 400, "bad", {}, body)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(error))
    control = ExecutionControl()
    api = client()
    with execution_scope(control), pytest.raises(urllib.error.HTTPError):
        api.chat([])
    assert control.usage["total_tokens"] == 11
    assert control.usage["failed_attempts"] == 1
    assert len(api.usage_events) == 1


def test_api_attempt_budget_and_retry_after_are_bounded(monkeypatch):
    calls = []

    def open_url(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("url", 429, "rate", {"Retry-After": "100"}, io.BytesIO(b'{}'))

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    control = ExecutionControl()
    with execution_scope(control), pytest.raises(RuntimeError):
        client().chat([])
    assert len(calls) == 1
    assert control.usage["retries"] == 0
    assert control.usage["failed_attempts"] == 1


def test_api_attempt_budget_blocks_extra_dispatch(monkeypatch):
    calls = []

    def open_url(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("temporary")

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    control = ExecutionControl(max_attempts=2, retry_policy=RetryPolicy(base_delay=0, jitter=0))
    with execution_scope(control), pytest.raises(TaskBudgetExceeded):
        client(fallback=True).chat([])
    assert len(calls) == control.usage["api_attempts"] == control.usage["failed_attempts"] == 2


@pytest.mark.parametrize("kind", ["chat", "embed"])
def test_pause_is_not_swallowed_by_api_fallback(monkeypatch, kind):
    control = ExecutionControl()

    def open_url(*args, **kwargs):
        control.request_pause()
        raise urllib.error.URLError("interrupted")

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    api = client(fallback=True)
    with execution_scope(control), pytest.raises(TaskPaused):
        api.chat([]) if kind == "chat" else api.embed(["test"])
    assert control.usage["failed_attempts"] == 1
    assert len(api.usage_events) == 1


def test_api_retry_wait_observes_deadline(monkeypatch):
    calls = []

    def open_url(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("temporary")

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    control = ExecutionControl(deadline=time.monotonic() + 0.05,
                               retry_policy=RetryPolicy(base_delay=1, jitter=0))
    with execution_scope(control), pytest.raises(TaskDeadlineExceeded):
        client(fallback=True).chat([])
    assert len(calls) == 1


def test_stream_disconnect_is_not_retried_or_concatenated(monkeypatch):
    response = Response(lines=[b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                               b'data: {"usage":{"total_tokens":9},"choices":[]}\n'])
    calls = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: calls.append(1) or response)
    api = client()
    events = []
    control = ExecutionControl(task_id="t")
    with execution_scope(control, node_id="n", attempt_id="a"), api.event_stream(events.append), api.stage("Answer", stream=True):
        with pytest.raises(ConnectionError):
            api.chat([])
    assert len(calls) == 1
    assert control.usage["api_attempts"] == control.usage["failed_attempts"] == 1
    assert control.usage["total_tokens"] == 9
    assert len(api.usage_events) == 1
    tokens = [event for event in events if event["type"] == "token"]
    assert [event["text"] for event in tokens] == ["partial"]
    assert tokens[0]["attempt_id"] == "a"
    assert any(event["type"] == "stream_interrupted" for event in events)


def test_stream_connection_pause_cannot_be_swallowed(monkeypatch):
    control = ExecutionControl()

    def open_url(*args, **kwargs):
        control.request_pause()
        raise urllib.error.URLError("interrupted")

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    api = client(fallback=True)
    with execution_scope(control), api.event_stream(lambda event: None), api.stage("Answer", stream=True):
        with pytest.raises(TaskPaused):
            api.chat([])
    assert control.usage["failed_attempts"] == 1


def test_blocked_http_read_is_closed_on_cancel(monkeypatch):
    reading = threading.Event()
    released = threading.Event()

    class Blocked(Response):
        def read(self):
            reading.set()
            assert released.wait(2)
            raise OSError("closed")

        def close(self):
            released.set()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Blocked())
    control = ExecutionControl()

    def cancel():
        assert reading.wait(2)
        control.request_cancel()

    trigger = threading.Thread(target=cancel)
    trigger.start()
    try:
        with execution_scope(control), pytest.raises(TaskCancelled):
            client(fallback=True).chat([])
    finally:
        trigger.join(timeout=2)
    assert control.usage["failed_attempts"] == 1


def test_execution_outputs_are_bounded_and_legacy_calls_still_work(tmp_path):
    command = [sys.executable, "-c", "import sys; sys.stdout.write('x'*200000); sys.stderr.write('y'*200000)"]
    result = execute_command(command, cwd=tmp_path, confirm=True, output_limit_bytes=2048)
    assert result["ok"]
    assert result["output_truncated"]
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= 2048
    assert result["stdout_bytes"] == result["stderr_bytes"] == 200000
    assert execute_python_code("print('python-ok')", cwd=tmp_path, confirm=True)["stdout"].strip() == "python-ok"


def test_invalid_utf8_cannot_expand_beyond_output_quota(tmp_path):
    result = execute_command([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff'*4096)"],
                             cwd=tmp_path, confirm=True, output_limit_bytes=128)
    assert result["ok"] and result["output_truncated"]
    assert result["output_limit_bytes"] == 128
    assert len(result["stdout"].encode("utf-8")) <= 128


def test_budget_reservation_is_atomic_between_workers():
    control = ExecutionControl(max_tokens=10)
    barrier = threading.Barrier(2)
    outcomes = []

    def reserve(node):
        with execution_scope(control, node_id=node):
            barrier.wait()
            try:
                control.record_attempt(reserved_tokens=8)
                outcomes.append("reserved")
            except TaskBudgetExceeded:
                outcomes.append("blocked")

    workers = [threading.Thread(target=reserve, args=(str(index),)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    assert sorted(outcomes) == ["blocked", "reserved"]
    assert control.usage["api_attempts"] == 1


def _alive(pid: int) -> bool:
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    status = Path(f"/proc/{pid}/stat")
    if status.exists() and status.read_text().split()[2] == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("signal_name,error", [("request_cancel", TaskCancelled),
                                               ("request_pause", TaskPaused), ("deadline", TaskDeadlineExceeded)])
def test_cancel_pause_deadline_reap_real_process_tree(tmp_path, signal_name, error):
    marker = tmp_path / "pids.txt"
    grandchild_marker = tmp_path / "grandchild.txt"
    child_code = ("import subprocess,sys,time\n"
                  "grandchild=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
                  f"open({str(grandchild_marker)!r},'w').write(str(grandchild.pid))\ntime.sleep(30)")
    code = (
        "import subprocess,sys,time,os\n"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"while not os.path.exists({str(grandchild_marker)!r}): time.sleep(0.01)\n"
        "print('ready',flush=True)\n"
        f"open({str(marker)!r},'w').write(str(os.getpid())+' '+str(child.pid)+' '+open({str(grandchild_marker)!r}).read())\n"
        "time.sleep(30)\n"
    )
    control = ExecutionControl()

    def interrupt():
        end = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < end:
            time.sleep(0.01)
        if signal_name == "deadline":
            control.deadline = time.monotonic() - 1
        else:
            getattr(control, signal_name)()

    trigger = threading.Thread(target=interrupt)
    trigger.start()
    try:
        with execution_scope(control), pytest.raises(error) as caught:
            _run_process([sys.executable, "-c", code], tmp_path, 10, 2048)
    finally:
        trigger.join(timeout=4)
    assert marker.exists()
    pids = [int(value) for value in marker.read_text().split()]
    end = time.monotonic() + 2
    while any(_alive(pid) for pid in pids) and time.monotonic() < end:
        time.sleep(0.02)
    assert all(not _alive(pid) for pid in pids)
    assert "ready" in caught.value.result["stdout"]


def test_tool_timeout_reaps_descendants_even_without_control(tmp_path):
    marker = tmp_path / "child.txt"
    code = ("import subprocess,sys,time\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
            f"open({str(marker)!r},'w').write(str(child.pid))\ntime.sleep(30)")
    result = _run_process([sys.executable, "-c", code], tmp_path, 0.3, 2048)
    assert result["timed_out"]
    assert result["returncode"] is None
    end = time.monotonic() + 2
    while _alive(int(marker.read_text())) and time.monotonic() < end:
        time.sleep(0.02)
    assert not _alive(int(marker.read_text()))


def provider():
    return ImapSmtpEmailProvider({"imap_host": "imap.example.com", "smtp_host": "smtp.example.com",
                                  "username": "me@example.com", "password": "secret"})


class SMTP:
    def __init__(self, *, error=None, quit_error=False):
        self.error = error
        self.quit_error = quit_error
        self.sent = 0
        self.closed = False

    def login(self, *args):
        pass

    def starttls(self, **kwargs):
        pass

    def send_message(self, message):
        self.sent += 1
        if self.error:
            raise self.error
        return {"refused@example.com": (550, b"rejected")}

    def close(self):
        self.closed = True
        if self.quit_error:
            raise OSError("close failed after acceptance")


@pytest.mark.parametrize("port", [465, 587])
def test_smtp_timeout_and_copy_failure_preserve_accepted_delivery(monkeypatch, port):
    smtp = SMTP(quit_error=True)
    timeouts = []

    def connect(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return smtp

    monkeypatch.setattr("smtplib.SMTP_SSL", connect)
    monkeypatch.setattr("smtplib.SMTP", connect)
    mail = provider()
    mail.smtp_port = port
    monkeypatch.setattr(mail, "_append_to_mailbox", lambda *args: (_ for _ in ()).throw(OSError("IMAP failed")))
    with execution_scope(ExecutionControl(deadline=time.monotonic() + 1)):
        result = mail.send(["a@example.com"], "subject", "body")
    assert 0 < timeouts[0] <= 1
    assert smtp.sent == 1 and smtp.closed
    assert result["status"] == "sent"
    assert not result["sent_copy_saved"]
    assert result["refused_recipients"]
    assert result["message_id"]


def test_smtp_lost_response_is_unknown_and_never_retryable(monkeypatch):
    smtp = SMTP(error=TimeoutError("accepted but response lost"))
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *args, **kwargs: smtp)
    with pytest.raises(UnknownSideEffect) as caught:
        provider().send(["a@example.com"], "subject", "body")
    assert smtp.sent == 1
    assert caught.value.state == "unknown"
    assert caught.value.message_id
    assert not RetryPolicy().allows(caught.value, 1)


def test_smtp_pause_after_acceptance_is_not_swallowed(monkeypatch):
    smtp = SMTP()
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *args, **kwargs: smtp)
    mail = provider()
    monkeypatch.setattr(mail, "_append_to_mailbox", lambda *args: (_ for _ in ()).throw(TaskPaused()))
    with pytest.raises(TaskPaused) as caught:
        mail.send(["a@example.com"], "subject", "body")
    assert smtp.sent == 1
    assert caught.value.side_effect_state == "committed"
    assert caught.value.result["status"] == "sent"


def test_smtp_expired_deadline_does_not_connect(monkeypatch):
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *args, **kwargs: pytest.fail("must not dispatch"))
    with execution_scope(ExecutionControl(deadline=time.monotonic() - 1)), pytest.raises(TaskDeadlineExceeded):
        provider().send(["a@example.com"], "subject", "body")
