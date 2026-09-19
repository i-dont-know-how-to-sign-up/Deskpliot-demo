from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .schemas import AgentResult


ToolCaller = Callable[..., Any]


@dataclass
class KnowledgeAgent:
    """统一网页和本地文档资料获取，输出证据包。"""

    tool_call: ToolCaller
    allowed_tools: tuple[str, ...] = ("web.search", "web.read_page", "files.read_document")

    def search(self, query: str, limit: int = 5) -> AgentResult:
        result = self.tool_call("web.search", query=query, limit=limit)
        if not getattr(result, "ok", False):
            return AgentResult("knowledge", "failed", error=str(getattr(result, "error", "搜索失败")), tool_calls=1)
        output = getattr(result, "output", None)
        return AgentResult("knowledge", "success", {"evidence": output if isinstance(output, list) else []}, tool_calls=1)


@dataclass
class CommunicationAgent:
    """根据证据生成邮件或文档草稿，但不执行外部副作用。"""

    llm_call: Callable[..., str]
    tool_call: ToolCaller | None = None

    def compose_email(self, subject: str, request: str, evidence: Any = None, template: str = "") -> AgentResult:
        prompt = (
            f"邮件主题：{subject}\n用户要求：{request}\n"
            f"证据：{json.dumps(evidence or [], ensure_ascii=False)}"
        )
        if template:
            prompt += f"\n请按以下模板生成：\n{template}"
        body = self.llm_call(prompt)
        if not body:
            return AgentResult("communication", "failed", error="无法生成邮件正文")
        return AgentResult("communication", "success", {"subject": subject, "body": body, "requires_human": True})


@dataclass
class ReflectionAgent:
    """在高可靠、低实时性任务中进行最终质量检查。"""

    llm_call: Callable[..., str] | None = None

    def review(self, output: str, evidence: Any = None, *, quality: int = 1, latency: int = 1) -> AgentResult:
        if quality < 4 or latency > 2:
            return AgentResult("reflection", "skipped", {"reason": "质量要求或实时性条件不满足"})
        if self.llm_call is None:
            return AgentResult("reflection", "failed", error="Reflection Agent 未配置 LLM")
        response = self.llm_call(
            f"检查结果是否被证据支持，只输出 approved=true/false 及问题：\n"
            f"结果：{output}\n证据：{json.dumps(evidence or [], ensure_ascii=False)}"
        )
        normalized = response.casefold().replace(" ", "")
        approved = "approved=true" in normalized or '"approved":true' in normalized
        return AgentResult("reflection", "success" if approved else "failed", {"approved": approved, "review": response})

