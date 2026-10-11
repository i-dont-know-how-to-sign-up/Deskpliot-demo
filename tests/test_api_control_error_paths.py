from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.multimodal.providers.vision_llm import VisionLLMProvider
from deskpilot.tasks.control import (
    ExecutionControl, TaskCancelled, TaskDeadlineExceeded, TaskPaused, execution_scope,
)
from deskpilot.tasks.retry import RetryPolicy


@pytest.fixture(autouse=True)
def _memory_only_with_joined_workers(monkeypatch):
    workers = []
    start = threading.Thread.start

    def tracked_start(worker, *args, **kwargs):
        if worker.name == "deskpilot-http-control":
            workers.append(worker)
        return start(worker, *args, **kwargs)

    network_guard = Mock(
        side_effect=AssertionError("Real network access is disabled in API control tests")
    )
    monkeypatch.setattr(threading.Thread, "start", tracked_start)
    monkeypatch.setattr("urllib.request.urlopen", network_guard)
    yield workers
    network_guard.assert_not_called()
    assert all(not worker.is_alive() for worker in workers), "HTTP control worker outlived its request"


def _client(*, fallback: bool = True) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(SimpleNamespace(
        llm_api_key="fixture", llm_base_url="https://example.invalid", llm_model="fixture",
        embedding_api_key="fixture", embedding_base_url="https://example.invalid",
        embedding_model="fixture", qwen_enable_thinking=None, allow_local_fallback=fallback,
    ))


def _vision() -> VisionLLMProvider:
    provider = VisionLLMProvider.__new__(VisionLLMProvider)
    provider.allow_cloud = True
    provider.api_key = "fixture"
    provider.model = "fixture"
    provider.base_url = "https://example.invalid"
    provider.timeout = 17
    provider.last_usage = {}
    return provider


def _invoke(kind: str, provider):
    if kind == "vision":
        return provider.answer("fixture", [])
    if kind == "embedding":
        return provider.embed(["fixture"])
    if kind == "stream":
        with provider.event_stream(lambda event: None), provider.stage("Answer", stream=True):
            return provider.chat([])
    return provider.chat([])


def _assert_settled(control: ExecutionControl, *, tokens: int, reported: bool) -> None:
    usage = control.snapshot_usage()
    assert usage["api_attempts"] == usage["failed_attempts"] == 1
    assert usage["total_tokens"] == tokens
    assert usage["reported_calls"] == int(reported)
    assert usage["unknown_usage_calls"] == int(not reported)
    assert usage["reserved_tokens"] == usage["retries"] == 0
    events = [event for event in control.events if event["kind"] == "usage"]
    assert len(events) == 1
    assert events[0]["success"] is False
    assert events[0]["node_id"] == "node"
    assert events[0]["attempt_id"] == "attempt"


@pytest.mark.parametrize("kind", ["chat", "embedding", "stream", "vision"])
@pytest.mark.parametrize("signal,error_type", [
    ("pause", TaskPaused), ("cancel", TaskCancelled), ("deadline", TaskDeadlineExceeded),
])
def test_http_error_usage_survives_control_signal_before_exit(monkeypatch, kind, signal, error_type):
    control = ExecutionControl(max_tokens=10000)

    class ErrorBody(io.BytesIO):
        def read(self, *args):
            data = super().read(*args)
            if signal == "deadline":
                control.deadline = time.monotonic() - 1
            else:
                getattr(control, "request_" + signal)()
            return data

    body = ErrorBody(json.dumps({"usage": {"total_tokens": 13}}).encode())
    error = urllib.error.HTTPError("url", 429, "rate", {"Retry-After": "100"}, body)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _vision() if kind == "vision" else _client()
    with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(error_type):
        _invoke(kind, provider)
    assert opened.call_count == 1
    assert body.closed
    _assert_settled(control, tokens=13, reported=True)
    if kind != "vision":
        assert len(provider.usage_events) == 1
        assert provider.token_usage["total_tokens"] == 13


@pytest.mark.parametrize("kind", ["chat", "embedding", "stream", "vision"])
@pytest.mark.parametrize("signal,error_type", [
    ("pause", TaskPaused), ("cancel", TaskCancelled), ("deadline", TaskDeadlineExceeded),
])
def test_blocked_http_error_body_is_closed_by_control(monkeypatch, kind, signal, error_type):
    control = ExecutionControl(max_tokens=10000)
    reading = threading.Event()
    released = threading.Event()

    class BlockedBody(io.BytesIO):
        def read(self, *args):
            reading.set()
            if not released.wait(1.5):
                raise AssertionError("Error body was not closed by execution control")
            raise OSError("connection closed")

        def close(self):
            released.set()
            super().close()

    body = BlockedBody(b"{}")
    error = urllib.error.HTTPError("url", 503, "busy", {}, body)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _vision() if kind == "vision" else _client()

    def interrupt():
        if reading.wait(1):
            if signal == "deadline":
                control.deadline = time.monotonic() + 0.025
            else:
                getattr(control, "request_" + signal)()

    trigger = threading.Thread(target=interrupt)
    trigger.start()
    started = time.monotonic()
    try:
        with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(error_type):
            _invoke(kind, provider)
    finally:
        trigger.join(timeout=2)
    assert not trigger.is_alive()
    assert time.monotonic() - started < 1
    assert reading.is_set() and released.is_set() and body.closed
    assert opened.call_count == 1
    _assert_settled(control, tokens=0, reported=False)


@pytest.mark.parametrize("signal,error_type", [
    ("pause", TaskPaused), ("cancel", TaskCancelled), ("deadline", TaskDeadlineExceeded),
])
def test_vision_success_response_usage_is_saved_before_exit_checkpoint(monkeypatch, signal, error_type):
    control = ExecutionControl(max_tokens=10000)

    class Response(io.BytesIO):
        def read(self, *args):
            data = super().read(*args)
            if signal == "deadline":
                control.deadline = time.monotonic() - 1
            else:
                getattr(control, "request_" + signal)()
            return data

    body = Response(json.dumps({"choices": [{"message": {"content": "ok"}}],
                                "usage": {"total_tokens": 12}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=body))
    provider = _vision()
    with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(error_type):
        provider.answer("fixture", [])
    assert provider.last_usage["total_tokens"] == 12
    _assert_settled(control, tokens=12, reported=True)


@pytest.mark.parametrize("kind", ["chat", "embedding", "stream", "vision"])
def test_http_failure_usage_is_settled_once_without_control_signal(monkeypatch, kind):
    control = ExecutionControl(max_tokens=10000)
    body = io.BytesIO(json.dumps({"usage": {"total_tokens": 13}}).encode())
    error = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _vision() if kind == "vision" else _client(fallback=False)
    with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(urllib.error.HTTPError):
        _invoke(kind, provider)
    assert opened.call_count == 1
    assert body.closed
    _assert_settled(control, tokens=13, reported=True)


@pytest.mark.parametrize("signal,error_type", [
    ("none", urllib.error.HTTPError), ("pause", TaskPaused),
    ("cancel", TaskCancelled), ("deadline", TaskDeadlineExceeded),
    ("invalid_json", urllib.error.HTTPError), ("read_error", urllib.error.HTTPError),
])
def test_http_error_worker_exits_and_closes_handler(
    monkeypatch, _memory_only_with_joined_workers, signal, error_type,
):
    control = ExecutionControl()

    class ErrorBody(io.BytesIO):
        def read(self, *args):
            if signal == "read_error":
                raise OSError("fixture read failed")
            data = super().read(*args)
            if signal == "deadline":
                control.deadline = time.monotonic() - 1
            elif signal in {"pause", "cancel"}:
                getattr(control, "request_" + signal)()
            return data

    body = ErrorBody(b"invalid" if signal == "invalid_json" else b"{}")
    error = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    error.close = Mock(wraps=error.close)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    with execution_scope(control), pytest.raises(error_type):
        _client(fallback=False).chat([])
    assert opened.call_count == 1
    error.close.assert_called()
    assert body.closed
    assert len(_memory_only_with_joined_workers) == 1
    assert not _memory_only_with_joined_workers[0].is_alive()


@pytest.mark.parametrize("kind", ["chat", "embedding"])
def test_fallback_rechecks_control_after_model_error(monkeypatch, kind):
    control = ExecutionControl()
    provider = _client()

    def fail(*args):
        control.request_pause()
        raise RuntimeError("adapter failed while pause was requested")

    monkeypatch.setattr(provider, "_post_json", fail)
    with execution_scope(control), pytest.raises(TaskPaused):
        _invoke(kind, provider)


def test_existing_timeout_contract_is_unchanged(monkeypatch):
    seen = []

    def open_url(request, timeout):
        seen.append(timeout)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}],
                                      "usage": {"total_tokens": 12}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    assert _client().chat([]) == "ok"
    assert _vision().answer("fixture", []) == "ok"
    assert seen == [60, 17]
    control = ExecutionControl(deadline=time.monotonic() + 1, max_tokens=10000)
    with execution_scope(control):
        assert _vision().answer("fixture", []) == "ok"
    assert 0 < seen[-1] <= 1
    assert control.usage["api_attempts"] == control.usage["reported_calls"] == 1
    assert control.usage["reserved_tokens"] == 0


def test_http_error_retry_keeps_both_attempts_and_usage(monkeypatch):
    control = ExecutionControl(max_tokens=10000, retry_policy=RetryPolicy(base_delay=0, jitter=0))
    body = io.BytesIO(json.dumps({"usage": {"total_tokens": 3}}).encode())
    error = urllib.error.HTTPError("url", 503, "busy", {}, body)
    success = io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}],
                                   "usage": {"total_tokens": 7}}).encode())
    opened = Mock(side_effect=[error, success])
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _client(fallback=False)
    with execution_scope(control):
        assert provider.chat([]) == "ok"
    assert body.closed and success.closed
    assert opened.call_count == control.usage["api_attempts"] == 2
    assert control.usage["failed_attempts"] == control.usage["retries"] == 1
    assert control.usage["total_tokens"] == 10
    assert control.usage["reported_calls"] == 2
    assert control.usage["reserved_tokens"] == 0
    assert len(provider.usage_events) == 2


def test_http_error_wrapper_shutdown_reaches_underlying_socket(monkeypatch):
    control = ExecutionControl()
    reading = threading.Event()
    released = threading.Event()

    class Body(io.BytesIO):
        def read(self, *args):
            reading.set()
            if not released.wait(1.5):
                raise AssertionError("Socket shutdown did not interrupt the read")
            raise OSError("socket shut down")

        def close(self):
            released.set()
            super().close()

    connection = SimpleNamespace(shutdown=Mock(side_effect=lambda how: released.set()))
    body = Body(b"{}")
    body.fp = SimpleNamespace(raw=SimpleNamespace(_sock=connection))
    error = urllib.error.HTTPError("url", 503, "busy", {}, body)
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=error))

    def cancel():
        if reading.wait(1):
            control.request_cancel()

    trigger = threading.Thread(target=cancel)
    trigger.start()
    try:
        with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(TaskCancelled):
            _client().chat([])
    finally:
        trigger.join(timeout=2)
    connection.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    _assert_settled(control, tokens=0, reported=False)


@pytest.mark.parametrize("kind", ["chat", "embedding", "stream", "vision"])
def test_invalid_error_usage_cannot_replace_control_exception(monkeypatch, kind):
    control = ExecutionControl(max_tokens=10000)

    class Body(io.BytesIO):
        def read(self, *args):
            data = super().read(*args)
            control.request_pause()
            return data

    body = Body(json.dumps({"usage": {"total_tokens": "invalid"}}).encode())
    error = urllib.error.HTTPError("url", 429, "rate", {}, body)
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=error))
    provider = _vision() if kind == "vision" else _client()
    with execution_scope(control, node_id="node", attempt_id="attempt"), pytest.raises(TaskPaused):
        _invoke(kind, provider)
    _assert_settled(control, tokens=0, reported=False)


def test_no_context_long_retry_after_keeps_existing_fallback(monkeypatch):
    body = io.BytesIO(json.dumps({"usage": {"total_tokens": 13}}).encode())
    error = urllib.error.HTTPError("url", 429, "rate", {"Retry-After": "100"}, body)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _client()
    assert provider.chat([]) == ""
    assert opened.call_count == 1
    assert opened.call_args.kwargs["timeout"] == 60
    assert body.closed
    assert len(provider.usage_events) == 1
    assert provider.usage_events[0]["success"] is False
    assert provider.token_usage["total_tokens"] == 13


def test_no_context_vision_http_failure_retains_usage_and_timeout(monkeypatch):
    body = io.BytesIO(json.dumps({"usage": {"total_tokens": 13}}).encode())
    error = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    opened = Mock(side_effect=error)
    monkeypatch.setattr("urllib.request.urlopen", opened)
    provider = _vision()
    with pytest.raises(urllib.error.HTTPError):
        provider.answer("fixture", [])
    assert opened.call_count == 1
    assert opened.call_args.kwargs["timeout"] == 17
    assert body.closed
    assert provider.last_usage["total_tokens"] == 13
