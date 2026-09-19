from __future__ import annotations

from .memory_models import MemoryItem, SessionMessage
from .session_store import SessionStore
from ..core.api_clients import OpenAICompatibleClient
from ..core.config import load_config


class MemoryCompactor:
    def __init__(self, session_store: SessionStore, max_messages: int = 30, max_chars: int = 12000):
        self.session_store = session_store
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.client = OpenAICompatibleClient(load_config())

    def compact_if_needed(
        self,
        session_id: str,
        memories: list[MemoryItem] | None = None,
        force: bool = False,
    ) -> tuple[bool, str]:
        messages = self.session_store.read_messages(session_id)
        total_chars = sum(len(message.content) for message in messages)
        checkpoint = self.session_store.read_compaction_state(session_id)
        new_message_count = max(0, len(messages) - int(checkpoint.get("message_count", 0)))
        new_chars = max(0, total_chars - int(checkpoint.get("total_chars", 0)))
        if not force and new_message_count < self.max_messages and new_chars < self.max_chars:
            return False, self.session_store.read_summary(session_id)

        # 已有摘要承接历史，只把 checkpoint 之后的新消息送入下一轮摘要。
        start = min(int(checkpoint.get("message_count", 0)), len(messages))
        summary = self._summarize(session_id, messages[start:], memories or [])
        self.session_store.write_summary(session_id, summary)
        self.session_store.write_compaction_state(session_id, messages)
        return True, summary

    def _summarize(self, session_id: str, messages: list[SessionMessage], memories: list[MemoryItem]) -> str:
        existing_summary = self.session_store.read_summary(session_id)
        transcript = "\n".join(f"{message.role}: {message.content[:800]}" for message in messages[-40:])
        memory_lines = "\n".join(f"- [{item.memory_type}] {item.content}" for item in memories[:20])
        prompt = (
            "请为 DeskPilot 当前会话生成滚动摘要，保留项目目标、关键决策、用户偏好、"
            "未完成待办、重要文件和当前上下文。\n"
            "摘要要短，使用中文 Markdown，控制在 800 字以内。不要写入 API Key、密码或 token。\n\n"
            f"已有摘要：\n{existing_summary or '无'}\n\n"
            f"结构化记忆：\n{memory_lines or '无'}\n\n"
            f"checkpoint 后新增对话：\n{transcript or '无'}"
        )
        response = self.client.chat(
            [
                {"role": "system", "content": "你是会话压缩器，只输出 Markdown 摘要。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        if response:
            return response.strip()
        return self._fallback_summary(existing_summary, messages, memories)

    def _fallback_summary(
        self,
        existing_summary: str,
        messages: list[SessionMessage],
        memories: list[MemoryItem],
    ) -> str:
        lines = ["# 会话摘要", ""]
        if existing_summary:
            lines.extend(["## 之前摘要", existing_summary[:1200], ""])
        important = [item for item in memories if item.memory_type in {"decision", "preference", "task"}]
        if important:
            lines.append("## 关键记忆")
            for item in important[:12]:
                lines.append(f"- [{item.memory_type}] {item.content}")
            lines.append("")
        lines.append("## 最近上下文")
        for message in messages[-12:]:
            content = " ".join(message.content.split())
            if len(content) > 160:
                content = content[:160] + "..."
            lines.append(f"- {message.role}: {content}")
        return "\n".join(lines).strip()
