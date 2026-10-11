from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from ..core.config import load_config
from ..tasks.control import TaskControlError, get_execution_control


_STAGE: ContextVar[str] = ContextVar("jev_stage", default="Jev")
_MODES = {"off", "shadow", "gated"}


@dataclass(frozen=True)
class JevConfig:
    mode: str = "off"
    api_key: str = ""
    model: str = "typesafe/jev-1.13"
    endpoint: str = "https://openrouter.ai/api/alpha/decisions"
    timeout_seconds: float = 8.0
    route_confidence: float = 0.92
    route_veto_probability: float = 0.2
    passage_threshold: float = 0.2
    max_passages: int = 8
    max_passage_chars: int = 1600
    allow_document_text: bool = False

    @classmethod
    def from_env(cls) -> "JevConfig":
        load_config()  # Reads .env without overriding explicit process environment variables.
        mode = os.getenv("JEV_MODE", "off").strip().lower()
        if mode not in _MODES:
            raise ValueError("JEV_MODE must be off, shadow or gated")

        def number(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                return max(minimum, min(float(os.getenv(name, str(default))), maximum))
            except ValueError:
                return default

        return cls(
            mode=mode,
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            model=os.getenv("JEV_MODEL", "typesafe/jev-1.13").strip(),
            endpoint=os.getenv("JEV_ENDPOINT", "https://openrouter.ai/api/alpha/decisions").strip(),
            timeout_seconds=number("JEV_TIMEOUT_SECONDS", 8.0, 1.0, 30.0),
            route_confidence=number("JEV_ROUTE_CONFIDENCE", 0.92, 0.5, 1.0),
            route_veto_probability=number("JEV_ROUTE_VETO_PROBABILITY", 0.2, 0.0, 0.5),
            passage_threshold=number("JEV_PASSAGE_THRESHOLD", 0.2, 0.0, 1.0),
            max_passages=int(number("JEV_MAX_PASSAGES", 8, 1, 16)),
            max_passage_chars=int(number("JEV_MAX_PASSAGE_CHARS", 1600, 200, 3000)),
            allow_document_text=os.getenv("JEV_ALLOW_DOCUMENT_TEXT", "false").strip().lower() in
            {"1", "true", "yes", "on"},
        )


class JevClient:
    """OpenRouter alpha Decisions transport; input text is never written to usage logs."""

    def __init__(self, config: JevConfig | None = None) -> None:
        self.config = config or JevConfig.from_env()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}
        self.usage_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.last_error = ""

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        token = _STAGE.set(name)
        try:
            yield
        finally:
            _STAGE.reset(token)

    def judge(self, state: dict[str, Any] | str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not self.config.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if not self.config.endpoint.startswith("https://"):
            raise ValueError("JEV_ENDPOINT must use HTTPS")
        payload = {"model": self.config.model, "state": state, "questions": questions}
        control = get_execution_control()
        attempt = control.record_attempt("jev_decision") if control is not None else None
        started = time.perf_counter()
        error = ""
        response: dict[str, Any] = {}
        try:
            response = self._post(payload)
            if control is not None:
                control.checkpoint()
            answers = response.get("answers")
            if not isinstance(answers, dict) or not all(isinstance(answers.get(key), dict) for key in questions):
                raise ValueError("Jev response is missing typed answers")
            for key, question in questions.items():
                answer = answers[key]
                if answer.get("type") != question.get("type"):
                    raise ValueError(f"Jev answer type mismatch: {key}")
            self.last_error = ""
            return answers
        except TaskControlError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_error = error
            raise
        finally:
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            reported = isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int)
            prompt = int(usage.get("input_tokens", 0)) if reported else 0
            completion = int(usage.get("output_tokens", 0)) if reported else 0
            event = {"stage": _STAGE.get(), "kind": "decision", "model": self.config.model,
                     "prompt_tokens": prompt, "completion_tokens": completion,
                     "total_tokens": prompt + completion, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                     "success": not error, "usage_reported": reported, "error": error[:180]}
            with self._lock:
                self.usage_events.append(event)
                if reported:
                    self.token_usage["prompt_tokens"] += prompt
                    self.token_usage["completion_tokens"] += completion
                    self.token_usage["total_tokens"] += prompt + completion
                    self.token_usage["reported_calls"] += 1
            if control is not None and attempt is not None:
                control.record_usage(usage, usage_reported=reported, success=not error,
                                     error=error, api_attempt=attempt, model=self.config.model)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}",
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            control = get_execution_control()
            timeout = control.timeout(self.config.timeout_seconds) if control is not None else self.config.timeout_seconds
            with urllib.request.urlopen(request, timeout=timeout) as stream:
                return json.load(stream)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter Jev HTTP {exc.code}") from exc


def noul(answers: dict[str, dict[str, Any]], name: str) -> float:
    value = answers[name].get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"Invalid Jev noul answer: {name}")
    return float(value)
