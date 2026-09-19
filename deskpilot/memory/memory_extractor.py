from __future__ import annotations

import json
import re

from .memory_models import MemoryItem
from ..core.api_clients import OpenAICompatibleClient
from ..core.config import load_config
from ..core.encoding_utils import fix_mojibake


class MemoryExtractor:
    def __init__(self) -> None:
        self.client = OpenAICompatibleClient(load_config())

    def extract(
        self,
        user_message: str,
        assistant_message: str,
        session_id: str,
        source_message_ids: list[str],
        *,
        grounded: bool = False,
    ) -> list[dict]:
        llm_items = self._extract_with_llm(user_message, assistant_message, grounded=grounded)
        if llm_items:
            items = [
                {
                    **item,
                    "source_session_id": session_id,
                    "source_message_ids": source_message_ids,
                }
                for item in llm_items
                if not self._is_sensitive(item.get("content", ""))
            ]
            return [self._apply_grounding_policy(item, grounded=grounded) for item in items]
        return self._extract_with_rules(user_message, assistant_message, session_id, source_message_ids)

    def _extract_with_llm(self, user_message: str, assistant_message: str, *, grounded: bool) -> list[dict]:
        prompt = (
            "你是 DeskPilot 的会话记忆抽取器。请从本轮对话中抽取值得长期或会话内保留的记忆。\n"
            "只抽取明确、有用、可复用的信息。不要保存 API Key、密码、token、隐私正文或临时玩笑。\n"
            "如果用户明确说不要记住、只是举例、临时假设、开玩笑或忽略这句话，请返回空数组。\n"
            "记忆类型只能是 fact、preference、decision、task、artifact。\n"
            "scope 只能是 session、workspace、user。用户稳定偏好用 user，项目阶段/技术决策用 workspace，临时事实用 session。\n"
            "只输出 JSON 数组，每项格式："
            "{\"scope\":\"workspace|user|session\",\"memory_type\":\"fact|preference|decision|task|artifact\","
            "\"content\":\"一句中文记忆\",\"confidence\":0.0,\"tags\":[\"tag\"]}\n\n"
            "fact 只能抽取用户明确陈述的事实；不要把助手自行生成的知识性回答当作用户事实。\n"
            f"本轮助手回答是否有外部证据支持：{'是' if grounded else '否'}。"
            "无外部证据时，禁止从助手回答抽取 fact。\n"
            f"用户消息：{user_message}\n\n助手回答：{assistant_message[:1200]}"
        )
        response = self.client.chat(
            [
                {"role": "system", "content": "你只输出合法 JSON，不输出解释。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        if not response:
            return []
        match = re.search(r"\[.*\]", response, flags=re.DOTALL)
        if not match:
            return []
        try:
            raw_items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        items: list[dict] = []
        for raw in raw_items if isinstance(raw_items, list) else []:
            normalized = self._normalize_item(raw)
            if normalized:
                items.append(normalized)
        return items[:6]

    def _apply_grounding_policy(self, item: dict, *, grounded: bool) -> dict:
        """无证据事实只能作为低置信度会话候选，不能污染工作区长期记忆。"""
        if grounded or item.get("memory_type") != "fact":
            return item
        tags = [str(tag) for tag in item.get("tags", [])]
        if "unverified" not in tags:
            tags.append("unverified")
        return {
            **item,
            "scope": "session",
            "confidence": min(float(item.get("confidence", 0.0)), 0.45),
            "status": "pending",
            "tags": tags[:5],
        }

    def _extract_with_rules(
        self,
        user_message: str,
        assistant_message: str,
        session_id: str,
        source_message_ids: list[str],
    ) -> list[dict]:
        text = fix_mojibake(user_message).strip()
        items: list[dict] = []
        lowered = text.lower()
        if self._is_sensitive(text) or self._should_not_remember(text):
            return []

        if any(marker in text for marker in ("我希望", "我喜欢", "偏好", "以后都", "回答风格", "优先使用")):
            explicit = any(marker in text for marker in ("记住", "以后都", "固定", "总是"))
            items.append(
                self._make_item(
                    "user",
                    "preference",
                    f"用户偏好：{self._compact(text)}",
                    0.86 if explicit else 0.72,
                    ["preference"],
                    session_id,
                    source_message_ids,
                    "active" if explicit else "pending",
                )
            )

        if any(marker in text for marker in ("决定", "打算", "计划", "第二阶段", "下一阶段", "优先", "顺延")):
            explicit = any(marker in text for marker in ("决定", "确认", "改成", "调整为"))
            items.append(
                self._make_item(
                    "workspace",
                    "decision",
                    f"项目决策：{self._compact(text)}",
                    0.9 if explicit else 0.78,
                    ["decision", "project_plan"],
                    session_id,
                    source_message_ids,
                    "active",
                )
            )

        if any(marker in text for marker in ("需要", "帮我", "实现", "修复", "开发", "测试方案", "验证")):
            items.append(
                self._make_item(
                    "session",
                    "task",
                    f"待办：{self._compact(text)}",
                    0.66,
                    ["task"],
                    session_id,
                    source_message_ids,
                    "active",
                )
            )

        path_matches = re.findall(r"[\w./\\:-]+\.md|[\w./\\:-]+\.py|data[/\\][\w./\\-]+", assistant_message)
        for path in path_matches[:3]:
            items.append(
                self._make_item(
                    "session",
                    "artifact",
                    f"产物或相关文件：{path}",
                    0.62,
                    ["artifact"],
                    session_id,
                    source_message_ids,
                    "active",
                )
            )

        if not items and len(lowered) > 12:
            if any(marker in text for marker in ("项目", "DeskPilot", "Agent", "RAG", "记忆系统")):
                items.append(
                    self._make_item(
                        "session",
                        "fact",
                        f"会话事实：{self._compact(text)}",
                        0.55,
                        ["fact"],
                        session_id,
                        source_message_ids,
                        "pending",
                    )
                )
        return items[:6]

    def _normalize_item(self, raw: object) -> dict | None:
        if not isinstance(raw, dict):
            return None
        scope = str(raw.get("scope", "session")).strip()
        memory_type = str(raw.get("memory_type", raw.get("type", "fact"))).strip()
        content = fix_mojibake(str(raw.get("content", ""))).strip()
        if scope not in {"session", "workspace", "user"}:
            scope = "session"
        if memory_type not in {"fact", "preference", "decision", "task", "artifact"}:
            memory_type = "fact"
        if not content or self._is_sensitive(content):
            return None
        try:
            confidence = float(raw.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        return {
            "scope": scope,
            "memory_type": memory_type,
            "content": content,
            "confidence": max(0.0, min(confidence, 1.0)),
            "tags": [str(tag) for tag in tags[:5]],
            "status": "pending" if confidence < 0.6 else str(raw.get("status", "active")),
        }

    def _make_item(
        self,
        scope: str,
        memory_type: str,
        content: str,
        confidence: float,
        tags: list[str],
        session_id: str,
        source_message_ids: list[str],
        status: str = "active",
    ) -> dict:
        return {
            "scope": scope,
            "memory_type": memory_type,
            "content": content,
            "confidence": confidence,
            "tags": tags,
            "source_session_id": session_id,
            "source_message_ids": source_message_ids,
            "status": status,
        }

    def _compact(self, text: str, limit: int = 120) -> str:
        text = " ".join(text.strip().split())
        return text[:limit] + ("..." if len(text) > limit else "")

    def _is_sensitive(self, content: str) -> bool:
        lowered = content.lower()
        if any(marker in lowered for marker in ("api_key", "apikey", "secret", "password", "token=", "bearer ")):
            return True
        return bool(re.search(r"sk-[a-zA-Z0-9]{12,}", content))

    def _should_not_remember(self, content: str) -> bool:
        markers = (
            "不要记住",
            "别记住",
            "无需记住",
            "不用记住",
            "只是举例",
            "仅举例",
            "临时假设",
            "假设一下",
            "开玩笑",
            "忽略这句话",
        )
        return any(marker in content for marker in markers)
