from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from hashlib import sha256

from .config import ModelConfig
from .encoding_utils import fix_mojibake, strip_qwen_thinking


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

    def embed(self, texts: list[str]) -> list[list[float]]:
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
            self._record_usage(response)
            items = sorted(response["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in items]
        except Exception as exc:
            self.last_error = str(exc)
            if self.config.allow_local_fallback:
                return [local_hash_embedding(text) for text in texts]
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
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
        try:
            response = self._post_json(url, payload, self.config.llm_api_key)
            self._record_usage(response)
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            return strip_qwen_thinking(fix_mojibake(str(content))).strip()
        except Exception as exc:
            self.last_error = str(exc)
            if self.config.allow_local_fallback:
                return ""
            raise

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
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8-sig"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: {exc.code} {body}") from exc

    def _record_usage(self, response: dict) -> None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        try:
            prompt_tokens = int(prompt or 0)
            completion_tokens = int(completion or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        except (TypeError, ValueError):
            return
        self.token_usage["prompt_tokens"] += max(0, prompt_tokens)
        self.token_usage["completion_tokens"] += max(0, completion_tokens)
        self.token_usage["total_tokens"] += max(0, total_tokens)
        self.token_usage["reported_calls"] += 1


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
