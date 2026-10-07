from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any

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

    @property
    def available(self) -> bool:
        return self.provider not in {"disabled", "none"} and bool(self.api_key)

    def answer(self, question: str, image_paths: list[Path], *, max_tokens: int = 1200) -> str:
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
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8-sig"))
        value = result["choices"][0]["message"].get("content") or ""
        if isinstance(value, list):
            value = "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
        return str(value).strip()

    @staticmethod
    def _prompt(question: str, count: int) -> str:
        return (
            f"你是 DeskPilot 的视觉问答助手。用户提供了 {count} 张图片，依次编号为图片1到图片{count}。\n"
            "只陈述图片中可核验的内容；涉及文字和数字时逐字核对，不确定就明确说明。"
            "回答中的每个视觉结论必须使用 [图片N] 标注来源。\n"
            f"用户问题：{question.strip() or '请描述这些图片。'}"
        )
