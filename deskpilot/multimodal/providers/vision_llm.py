from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ...core.config import _load_dotenv, ROOT_DIR


class VisionLLMProvider:
    """OpenAI-compatible 视觉消息适配器；Base64 只存在于瞬时请求体中。"""

    def __init__(self) -> None:
        _load_dotenv(ROOT_DIR / ".env")
        self.provider = os.getenv("VISION_PROVIDER", "openai_compatible").strip().casefold()
        self.model = os.getenv("VISION_MODEL", "qwen-vl-max-latest").strip()
        self.base_url = (os.getenv("VISION_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or
                         "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.api_key = os.getenv("VISION_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        self.allow_cloud = os.getenv("ALLOW_CLOUD_IMAGE_UPLOAD", "false").casefold() in {"1", "true", "yes", "on"}
        self.timeout = max(10, min(int(os.getenv("VISION_TIMEOUT_SECONDS", "90")), 300))
        self.last_usage: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return self.provider not in {"disabled", "none"} and bool(self.api_key)

    def answer(
        self, question: str, image_paths: list[Path], *, max_tokens: int = 1200,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        self.last_usage = {}
        if not self.allow_cloud:
            raise PermissionError("云端图片上传未启用，请在 .env 设置 ALLOW_CLOUD_IMAGE_UPLOAD=true")
        if not self.api_key:
            raise RuntimeError("未配置 VISION_API_KEY 或 DASHSCOPE_API_KEY")
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(question, len(image_paths))}]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": max(1, max_tokens),
        }
        if event_callback:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if event_callback else "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if event_callback:
                pieces: list[str] = []
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
                        self.last_usage = self._normalize_usage(item["usage"])
                    choices = item.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content") or ""
                    if isinstance(token, list):
                        token = "".join(
                            str(part.get("text", "")) for part in token if isinstance(part, dict)
                        )
                    if token:
                        pieces.append(str(token))
                        event_callback({"type": "token", "stage": "Vision Answer", "text": str(token)})
                return "".join(pieces).strip()
            result = json.loads(response.read().decode("utf-8-sig"))
        if isinstance(result.get("usage"), dict):
            self.last_usage = self._normalize_usage(result["usage"])
        value = result["choices"][0]["message"].get("content") or ""
        if isinstance(value, list):
            value = "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
        return str(value).strip()

    @staticmethod
    def _normalize_usage(value: dict[str, Any]) -> dict[str, int]:
        prompt = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
        completion = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
        total = int(value.get("total_tokens", prompt + completion) or prompt + completion)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "reported_calls": 1,
        }

    @staticmethod
    def _prompt(question: str, count: int) -> str:
        return (
            f"你是 DeskPilot 的视觉问答助手。用户提供了 {count} 张图片，依次编号为图片1到图片{count}。\n"
            "只陈述图片中可核验的内容；涉及文字和数字时逐字核对，不确定就明确说明。"
            "回答中的每个视觉结论必须使用 [图片N] 标注来源。\n"
            f"用户问题：{question.strip() or '请描述这些图片。'}"
        )
