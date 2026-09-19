from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .budget import estimate_tokens


@dataclass
class TokenCount:
    tokens: int
    provider: str
    exact: bool


class ModelTokenizer:
    """按模型选择本地 tokenizer；不可用时显式退回保守估算。"""

    def __init__(self, model: str, provider: str = "auto") -> None:
        self.model = model
        self.provider = "heuristic"
        self.exact = False
        self._tokenizer: Any = None
        requested = provider.strip().lower()
        openai_family = model.casefold().startswith(("gpt-", "o1", "o3", "o4"))
        if requested == "auto" and not openai_family:
            self._try_transformers()
        if self._tokenizer is None and requested in {"auto", "tiktoken"}:
            self._try_tiktoken()
        if self._tokenizer is None and requested in {"auto", "transformers"}:
            self._try_transformers()

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.provider.startswith("tiktoken"):
            return len(self._tokenizer.encode(text))
        if self.provider == "transformers":
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return estimate_tokens(text)

    def inspect(self, text: str) -> TokenCount:
        return TokenCount(self.count(text), self.provider, self.exact)

    def _try_tiktoken(self) -> None:
        try:
            import tiktoken  # type: ignore

            try:
                tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # OpenAI-compatible 服务可能使用自定义模型名，cl100k 仅作为近似值。
                tokenizer = tiktoken.get_encoding("cl100k_base")
                self._tokenizer = tokenizer
                self.provider = "tiktoken_approx"
                self.exact = False
                return
            self._tokenizer = tokenizer
            self.provider = "tiktoken"
            self.exact = True
        except (ImportError, OSError, ValueError):
            return

    def _try_transformers(self) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore

            # 禁止隐式联网下载大模型 tokenizer，只复用本机已有缓存。
            self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
            self.provider = "transformers"
            self.exact = True
        except (ImportError, OSError, ValueError):
            return
