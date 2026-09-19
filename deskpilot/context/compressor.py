from __future__ import annotations

import re

from .budget import estimate_tokens, trim_to_token_budget
from .models import ContextPacket


class ContextCompressor:
    def __init__(self, tool_output_tokens: int = 450):
        self.tool_output_tokens = tool_output_tokens

    def compress(self, packet: ContextPacket, max_tokens: int | None = None) -> ContextPacket:
        content = self._normalize(packet.content)
        limit = max_tokens or packet.estimated_tokens or estimate_tokens(content)
        if packet.kind == "tool_output":
            limit = min(limit, self.tool_output_tokens)
            content = self._compress_tool_output(content, limit)
        elif estimate_tokens(content) > limit:
            content = trim_to_token_budget(content, limit)
        packet.content = content
        packet.estimated_tokens = estimate_tokens(content)
        return packet

    def _normalize(self, text: str) -> str:
        lines: list[str] = []
        previous = None
        for raw in text.replace("\r\n", "\n").splitlines():
            line = re.sub(r"[ \t]+", " ", raw).strip()
            if not line or line == previous:
                continue
            lines.append(line)
            previous = line
        return "\n".join(lines)

    def _compress_tool_output(self, text: str, limit: int) -> str:
        if estimate_tokens(text) <= limit:
            return text
        # 工具错误通常位于输出尾部，保留头部状态和尾部错误比只截取开头更有诊断价值。
        head = trim_to_token_budget(text, max(32, limit // 3))
        tail = trim_to_token_budget(text, max(32, limit - estimate_tokens(head) - 8), keep_tail=True)
        return f"{head}\n[中间输出已压缩]\n{tail}"
