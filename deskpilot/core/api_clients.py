from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Iterator

from .config import ModelConfig
from .encoding_utils import fix_mojibake, strip_qwen_thinking


_CALL_STAGE: ContextVar[str] = ContextVar("deskpilot_call_stage", default="Unattributed")
_STREAM_VISIBLE: ContextVar[bool] = ContextVar("deskpilot_stream_visible", default=False)
_EVENT_CALLBACK: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "deskpilot_event_callback", default=None
)


@dataclass(frozen=True)
class UsageEvent:
    """一次模型调用的真实 usage 与延迟记录。"""

    stage: str
    kind: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    usage_reported: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "success": self.success,
            "usage_reported": self.usage_reported,
            "error": self.error,
        }


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.last_error = ""
        # 只记录服务端 usage 字段返回的真实 token，不用字符数做不准确估算。
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reported_calls": 0,
        }
        self.usage_events: list[dict[str, Any]] = []
        self._usage_lock = threading.Lock()

    @contextmanager
    def stage(self, name: str, *, stream: bool = False) -> Iterator[None]:
        """为当前调用链设置归因阶段；ContextVar 可隔离并发线程/任务。"""
        stage_token = _CALL_STAGE.set(name.strip() or "Unattributed")
        stream_token = _STREAM_VISIBLE.set(bool(stream))
        try:
            yield
        finally:
            _STREAM_VISIBLE.reset(stream_token)
            _CALL_STAGE.reset(stage_token)

    @contextmanager
    def event_stream(self, callback: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
        """绑定本轮事件消费者，供桌面端接收 token 和模型阶段状态。"""
        token = _EVENT_CALLBACK.set(callback)
        try:
            yield
        finally:
            _EVENT_CALLBACK.reset(token)

    def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        stage = _CALL_STAGE.get()
        if stage == "Unattributed":
            stage = "Embedding"
        self.last_error = ""
        if not self.config.embedding_api_key:
            self.last_error = "EMBEDDING_API_KEY is not configured"
            if self.config.allow_local_fallback:
                return [local_hash_embedding(text) for text in texts]
            raise RuntimeError("EMBEDDING_API_KEY is not configured")

        url = self.config.embedding_base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.config.embedding_model, "input": texts}
        try:
            response = self._post_json(url, payload, self.config.embedding_api_key)
            self._record_call(response, stage, "embedding", self.config.embedding_model, started)
            items = sorted(response["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in items]
        except Exception as exc:
            self.last_error = str(exc)
            self._record_call({}, stage, "embedding", self.config.embedding_model, started, error=str(exc))
            if self.config.allow_local_fallback:
                return [local_hash_embedding(text) for text in texts]
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        started = time.perf_counter()
        stage = _CALL_STAGE.get()
        self.last_error = ""
        if not self.config.llm_api_key:
            self.last_error = "LLM_API_KEY is not configured"
            if self.config.allow_local_fallback:
                return ""
            raise RuntimeError("LLM_API_KEY is not configured")

        url = self.config.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max(1, int(max_tokens))
        if self.config.qwen_enable_thinking is not None and _is_dashscope_qwen(self.config):
            payload["enable_thinking"] = self.config.qwen_enable_thinking
        callback = _EVENT_CALLBACK.get()
        visible_stream = bool(callback and _STREAM_VISIBLE.get())
        self._emit({"type": "model_start", "stage": stage, "kind": "chat", "visible": visible_stream})
        try:
            if visible_stream:
                content, response = self._post_stream_json(url, payload, self.config.llm_api_key)
            else:
                response = self._post_json(url, payload, self.config.llm_api_key)
                message = response["choices"][0]["message"]
                content = message.get("content") or ""
            result = strip_qwen_thinking(fix_mojibake(str(content))).strip()
            self._record_call(response, stage, "chat", self.config.llm_model, started)
            self._emit({"type": "model_end", "stage": stage, "kind": "chat", "ok": True})
            return result
        except Exception as exc:
            self.last_error = str(exc)
            self._record_call({}, stage, "chat", self.config.llm_model, started, error=str(exc))
            self._emit({"type": "model_end", "stage": stage, "kind": "chat", "ok": False, "error": str(exc)})
            if self.config.allow_local_fallback:
                return ""
            raise

    def _post_stream_json(self, url: str, payload: dict, api_key: str) -> tuple[str, dict]:
        """消费 OpenAI-compatible SSE；最终 usage 仍只信任服务端返回值。"""
        stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        data = json.dumps(stream_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Accept-Charset": "utf-8",
            },
            method="POST",
        )
        pieces: list[str] = []
        final_usage: dict[str, Any] = {}
        visible_text = ""
        stage = _CALL_STAGE.get()
        with urllib.request.urlopen(request, timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if not value or value == "[DONE]":
                    continue
                try:
                    item = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(item.get("usage"), dict):
                    final_usage = item["usage"]
                choices = item.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    text = fix_mojibake(str(content))
                    pieces.append(text)
                    if stage == "Router":
                        # Router 输出结构化 JSON；聊天区只显示 direct_response，不能泄漏内部路由字段。
                        extracted = _partial_json_string_field("".join(pieces), "direct_response")
                        delta = extracted[len(visible_text):] if extracted.startswith(visible_text) else ""
                        visible_text = extracted
                    else:
                        delta = text
                    if delta:
                        self._emit({"type": "token", "stage": stage, "text": delta})
        return "".join(pieces), {"usage": final_usage}

    def _post_json(self, url: str, payload: dict, api_key: str) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Charset": "utf-8",
            },
            method="POST",
        )
        attempts = _bounded_env_int("LLM_API_MAX_ATTEMPTS", 3, 1, 5)
        base_delay = _bounded_env_float("LLM_API_RETRY_BASE_SECONDS", 0.5, 0.05, 5.0)
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8-sig"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                if exc.code not in {429, 502, 503, 504} or attempt >= attempts:
                    raise RuntimeError(f"API request failed: {exc.code} {body}") from exc
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
                _sleep_before_retry(attempt, base_delay, retry_after)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= attempts:
                    raise RuntimeError(f"API request failed after {attempts} attempts: {exc}") from exc
                _sleep_before_retry(attempt, base_delay)
        raise RuntimeError("API request failed without a response")

    def _record_call(
        self,
        response: dict,
        stage: str,
        kind: str,
        model: str,
        started: float,
        *,
        error: str = "",
    ) -> None:
        usage = response.get("usage")
        usage_reported = isinstance(usage, dict) and bool(usage)
        usage = usage if isinstance(usage, dict) else {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        try:
            prompt_tokens = int(prompt or 0)
            completion_tokens = int(completion or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        except (TypeError, ValueError):
            prompt_tokens = completion_tokens = total_tokens = 0
            usage_reported = False
        event = UsageEvent(
            stage=stage,
            kind=kind,
            model=model,
            prompt_tokens=max(0, prompt_tokens),
            completion_tokens=max(0, completion_tokens),
            total_tokens=max(0, total_tokens),
            latency_ms=(time.perf_counter() - started) * 1000,
            success=not error,
            usage_reported=usage_reported,
            error=error[:500],
        ).to_dict()
        with self._usage_lock:
            if usage_reported:
                self.token_usage["prompt_tokens"] += event["prompt_tokens"]
                self.token_usage["completion_tokens"] += event["completion_tokens"]
                self.token_usage["total_tokens"] += event["total_tokens"]
                self.token_usage["reported_calls"] += 1
            self.usage_events.append(event)

    def usage_by_stage(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        with self._usage_lock:
            events = list(self.usage_events)
        for event in events:
            item = result.setdefault(event["stage"], {
                "calls": 0, "reported_calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0, "latency_total_ms": 0.0,
            })
            item["calls"] += 1
            item["reported_calls"] += int(event["usage_reported"])
            item["prompt_tokens"] += int(event["prompt_tokens"])
            item["completion_tokens"] += int(event["completion_tokens"])
            item["total_tokens"] += int(event["total_tokens"])
            item["latency_total_ms"] += float(event["latency_ms"])
        return result

    @staticmethod
    def _emit(event: dict[str, Any]) -> None:
        callback = _EVENT_CALLBACK.get()
        if callback is not None:
            callback(dict(event))


def local_hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    for token in tokens:
        digest = sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return dot / (left_norm * right_norm)


def _is_dashscope_qwen(config: ModelConfig) -> bool:
    base_url = config.llm_base_url.lower()
    model = config.llm_model.lower()
    return "dashscope" in base_url or model.startswith("qwen")


def _retry_after_seconds(value: str | None) -> float | None:
    try:
        return max(0.0, min(float(value), 10.0)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _sleep_before_retry(attempt: int, base_delay: float, retry_after: float | None = None) -> None:
    delay = retry_after if retry_after is not None else base_delay * (2 ** (attempt - 1))
    time.sleep(min(10.0, delay + random.uniform(0.0, base_delay * 0.25)))


def _partial_json_string_field(value: str, field: str) -> str:
    """从尚未闭合的 JSON 中安全提取字符串字段，用于 Router 单调用流式直答。"""
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', value)
    if not match:
        return ""
    index = match.end()
    result: list[str] = []
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}
    while index < len(value):
        char = value[index]
        if char == '"':
            break
        if char != "\\":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(value):
            break
        escaped = value[index + 1]
        if escaped == "u":
            digits = value[index + 2:index + 6]
            if len(digits) < 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                break
            result.append(chr(int(digits, 16)))
            index += 6
            continue
        result.append(escapes.get(escaped, escaped))
        index += 2
    return "".join(result)
