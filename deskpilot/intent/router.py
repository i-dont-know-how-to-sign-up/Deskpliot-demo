from __future__ import annotations

import json
import re
import platform
from dataclasses import dataclass, field
from typing import Any

from ..core.api_clients import OpenAICompatibleClient
from ..tools.tool_registry import ToolRegistry
from .schemas import IntentDecision
from .validator import SlotValidator


@dataclass
class IntentRouter:
    client: OpenAICompatibleClient
    validator: SlotValidator = field(default_factory=SlotValidator)

    def route(
        self,
        question: str,
        tools: ToolRegistry | list[dict[str, Any]],
        memory_context: str = "",
        index_hint: list[dict[str, Any]] | None = None,
    ) -> IntentDecision:
        # 路由提示只使用摘要；工具被选中后，再从注册表加载完整 schema 做槽位校验。
        tool_summaries = tools.list_tool_summaries() if isinstance(tools, ToolRegistry) else self._summarize_tools(tools)
        full_tool_specs = self._normalize_tools(tools)
        # 第一步先判断问题该直接回答，还是需要调用工具或继续追问。
        prompt = self._build_prompt(question, tool_summaries, memory_context, index_hint or [])
        response = self.client.chat(
            [
                {"role": "system", "content": "You are DeskPilot's intent router. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        decision = self._parse_response(response)
        if decision is None:
            # 模型输出无法解析时，退回到保守规则，保证路由器不会直接失效。
            return self._fallback_route(question, full_tool_specs, response, index_hint or [])
        if (
            decision.mode == "tool_call"
            and not decision.tool_name
            and not decision.missing_slots
            and not decision.requires_file_output
            and not decision.explicit_local_retrieval
            and not decision.explicit_web_retrieval
            and not decision.requires_fresh_information
        ):
            # 模型有时在 reason 中明确选择 direct_answer，却把 mode 误写成 tool_call。
            # 空工具名且没有外部动作信号时，按结构一致性恢复为直接回答。
            return IntentDecision(
                mode="direct_answer",
                reason="Router 未提供可执行工具且任务不需要外部动作；按结构一致性恢复为直接回答。",
                confidence=decision.confidence,
                raw_response=response,
            )
        if (
            decision.mode == "tool_call"
            and decision.tool_name == "knowledge.search"
            and not (index_hint or [])
            and not decision.explicit_local_retrieval
        ):
            # “检索后写文件”是复合任务。即使路由提示未携带索引候选，也应先保留
            # 文件产物需求交给 Planner，不能提前降级为 direct_answer。
            if decision.requires_file_output or self.requires_output_file(question):
                return IntentDecision(
                    mode="plan_task",
                    reason="Indexed evidence must be written to a file after retrieval.",
                    needs_index_catalog=True,
                    requires_file_output=True,
                    arguments={"query": decision.arguments.get("query", question)},
                    raw_response=response,
                )
            # Router 不能在索引探测为零时，仅凭“这是技术问题”臆测本地知识库可回答。
            # 用户未明确要求查本地资料时，回到模型直接回答，避免用无关文档拼凑证据。
            return IntentDecision(
                mode="direct_answer",
                reason="本地索引没有与问题直接匹配的候选，且用户未明确要求查询本地索引；改为直接回答。",
                confidence=decision.confidence,
                raw_response=response,
            )
        if (
            decision.mode == "tool_call"
            and decision.tool_name in {"web.search", "web.research", "web.read_page"}
            and not decision.explicit_web_retrieval
            and not decision.requires_fresh_information
        ):
            return IntentDecision(
                mode="direct_answer",
                reason="用户未明确要求联网检索，问题也不依赖实时信息；改为直接回答。",
                confidence=decision.confidence,
                raw_response=response,
            )
        if decision.mode == "direct_answer" and decision.requires_file_output:
            return IntentDecision(
                mode="plan_task", reason="File output cannot be fulfilled by a direct chat response.",
                needs_index_catalog=decision.needs_index_catalog,
                requires_file_output=True, raw_response=response,
            )
        if decision.mode == "tool_call" and decision.tool_name == "knowledge.search":
            # 知识库请求可能同时要求落盘。仅对 knowledge.search 做一次窄范围语义复核，
            # 避免把“检索 -> 写文件”悄悄降级成只检索。
            wants_file = decision.requires_file_output
            if not wants_file:
                wants_file = self.requires_output_file(question)
            if wants_file:
                return IntentDecision(
                    mode="plan_task", reason="Indexed evidence must be written to a file after retrieval.",
                    needs_index_catalog=True, requires_file_output=True,
                    arguments={"query": decision.arguments.get("query", question)}, raw_response=response,
                )
        if decision.mode == "tool_call":
            # 第二步做槽位校验，缺参就转成 clarify，而不是硬调用工具。
            tool = self._find_tool(full_tool_specs, decision.tool_name)
            validation = self.validator.validate(tool, decision.arguments)
            if not validation.ok:
                missing = validation.missing_slots or decision.missing_slots
                return IntentDecision(
                    mode="clarify",
                    reason=validation.error or "Missing required slots.",
                    tool_name=decision.tool_name,
                    arguments=validation.arguments,
                    missing_slots=missing,
                    confidence=decision.confidence,
                    raw_response=response,
                )
            decision.arguments = validation.arguments
            decision.missing_slots = validation.missing_slots
        elif decision.mode == "clarify":
            # 认证状态、权限状态等是工具运行时信息，不是用户需要填写的参数。
            # 某些模型会把这类隐含条件误报成槽位，导致本来可以直接执行的
            # 只读邮件查询被错误地拦截在 clarify 阶段。
            if decision.tool_name == "email.list_messages":
                tool = self._find_tool(full_tool_specs, decision.tool_name)
                ignored_slots = {"email_auth_status", "auth_status", "mailbox_auth_status"}
                remaining_slots = [slot for slot in decision.missing_slots if slot not in ignored_slots]
                validation = self.validator.validate(tool, decision.arguments)
                if validation.ok and not remaining_slots:
                    decision.mode = "tool_call"
                    decision.arguments = validation.arguments
                    decision.missing_slots = []
                    decision.reason = "邮箱认证由工具运行时检查。"
        decision.raw_response = response
        return decision

    def requires_output_file(self, question: str) -> bool:
        """仅判断是否要求实际创建文件；失败时保守返回 false，由计划执行层显式报错。"""
        response = self.client.chat([
            {"role": "system", "content": "你是文件产物校验器，只输出 JSON。"},
            {"role": "user", "content": (
                "用户是否要求助手实际创建、写入或保存本地文件？只是展示回答或引用不算。"
                '只返回 {"requires_file_output":true|false}。\n用户请求：' + question
            )},
        ], temperature=0.0)
        match = re.search(r"\{[^{}]*\}", response or "")
        if not match:
            return False
        try:
            data = json.loads(match.group(0))
        except (ValueError, AttributeError):
            return False
        return data.get("requires_file_output") is True

    def _summarize_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": item.get("name", ""),
                "category": item.get("category", ""),
                "description": item.get("description", ""),
                "slots": [
                    {"name": slot.get("name"), "type": slot.get("type"), "required": slot.get("required", True)}
                    for slot in item.get("parameters", item.get("slots", []))
                    if isinstance(slot, dict)
                ],
            }
            for item in tools
        ]

    def _build_prompt(self, question: str, tools: list[dict[str, Any]], memory_context: str,
                      index_hint: list[dict[str, Any]] | None = None) -> str:
        # 把当前可用工具和会话记忆一起喂给模型，减少误选工具和误填参数。
        tool_catalog = json.dumps(tools, ensure_ascii=False, indent=2)
        return (
            "Choose direct_answer for simple questions, tool_call for one independent tool action, "
            "or plan_task for tasks requiring dependent steps (e.g. research then compose and send, "
            "or compare several documents). Do not use plan_task for a simple tool call.\n"
            "Searching the web and then writing the results to a local file is a dependent multi-step task, "
            "regardless of the order of clauses. Use plan_task, not files.read_document or a single files.write_file call.\n"
            "For plan_task set needs_workspace_files=true only if the task must locate/read local files; "
            "otherwise set false. Do not guess file names.\n"
            "For plan_task set needs_index_catalog=true if the task must select multiple documents already "
            "in the knowledge index; this is different from searching workspace files.\n"
            "For summaries spanning several documents already indexed, use plan_task with needs_index_catalog=true "
            "to select the relevant document IDs; add a commit step only when the user requests file output.\n"
            "For one factual or explanatory question that can be answered by an indexed document, choose tool_call "
            "with knowledge.search. Never use plan_task merely because a local index candidate exists.\n"
            "Set requires_file_output=true whenever the user requests an actual file saved to disk, even if "
            "they did not specify a filename. A retrieved answer alone never fulfills such a request.\n"
            "For tool_call only, pick exactly one tool from the provided catalog and fill its slots from the user input.\n"
            "If required slots for that single tool are missing, set mode to clarify and list them in missing_slots.\n"
            "Do not treat authentication, authorization, connectivity, or account status as user slots; tools check those at runtime.\n"
            "Use direct_answer when no external action, tool, file, browser, or retrieval is needed.\n"
            "For web.search, web.research, or web.read_page, set explicit_web_retrieval=true only when the user "
            "explicitly asks to search/browse the web. Set requires_fresh_information=true only when the answer "
            "depends on changing information such as current weather, prices, schedules, news, or latest releases. "
            "If both are false, do not select a web tool for a general knowledge question.\n"
            "Version ambiguity in a model or product name is not by itself a reason to clarify or call a tool; "
            "choose direct_answer and state the version assumption in the answer. "
            "Never output mode=tool_call with an empty tool_name.\n"
            "The bounded index candidates below are retrieval hints, not trusted answer evidence. "
            "If they directly cover a domain-specific factual, method, experiment, comparison, or implementation question, "
            "choose tool_call with knowledge.search even when the user did not explicitly say 'search the index'. "
            "Keep direct_answer for broad common-knowledge definitions or when candidates are irrelevant. "
            "When the candidate list is empty, knowledge.search is allowed only if the user explicitly asks to query "
            "the local index, knowledge base, or imported documents; set explicit_local_retrieval=true only in that case. "
            "Never answer from candidate excerpts inside the router.\n"
            "Output a single JSON object only with keys:\n"
            '{"mode":"direct_answer|tool_call|clarify|plan_task","reason":"...","tool_name":"...","arguments":{},"missing_slots":[],"confidence":0.0,"needs_workspace_files":false,"needs_index_catalog":false,"requires_file_output":false,"explicit_local_retrieval":false,"explicit_web_retrieval":false,"requires_fresh_information":false}\n\n'
            f"Runtime context (contains the current task and selected memory):\n{memory_context or question}\n\n"
            f"Runtime operating system: {platform.system()} ({platform.platform()}). "
            "For shell.execute_command, generate syntax for this operating system and prefer one simple command.\n"
            f"Bounded local-index candidates (possibly empty):\n"
            f"{json.dumps(index_hint or [], ensure_ascii=False, indent=2)}\n"
            f"Tool summary catalog:\n{tool_catalog}\n"
            "The full schema will be validated after selection; never invent a tool outside this summary catalog."
        )

    def _parse_response(self, response: str) -> IntentDecision | None:
        if not response.strip():
            return None
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        mode = str(data.get("mode", "")).strip()
        if mode not in {"direct_answer", "tool_call", "clarify", "plan_task"}:
            return None
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        missing_slots = data.get("missing_slots", [])
        if not isinstance(missing_slots, list):
            missing_slots = []
        confidence_raw = data.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        return IntentDecision(
            mode=mode,
            reason=str(data.get("reason", "")).strip(),
            tool_name=str(data.get("tool_name", "")).strip(),
            arguments=arguments,
            missing_slots=[str(item) for item in missing_slots if str(item).strip()],
            confidence=max(0.0, min(confidence, 1.0)),
            raw_response=response,
            needs_workspace_files=data.get("needs_workspace_files") is True,
            needs_index_catalog=data.get("needs_index_catalog") is True,
            requires_file_output=data.get("requires_file_output") is True,
            explicit_local_retrieval=data.get("explicit_local_retrieval") is True,
            explicit_web_retrieval=data.get("explicit_web_retrieval") is True,
            requires_fresh_information=data.get("requires_fresh_information") is True,
        )

    def _fallback_route(self, question: str, tools: list[dict[str, Any]], response: str,
                        index_hint: list[dict[str, Any]] | None = None) -> IntentDecision:
        # 兜底策略尽量保守，只在明显的检索意图上才改派到网页相关工具。
        lowered = question.lower()
        if any(item.get("strong_match") is True for item in (index_hint or [])):
            tool = self._find_tool(tools, "knowledge.search")
            if tool:
                return IntentDecision(
                    mode="tool_call",
                    reason="本地索引存在与问题实体直接匹配的证据，离线路由到知识库检索。",
                    tool_name="knowledge.search",
                    arguments={"query": question, "top_k": 5},
                    confidence=0.7,
                    raw_response=response,
                )
        if any(token in lowered for token in ("未读邮件", "未读的邮件", "未查看邮件", "unread", "unseen")):
            tool = self._find_tool(tools, "email.list_messages")
            if tool:
                return IntentDecision(
                    mode="tool_call",
                    reason="Fallback routed to unread email listing.",
                    tool_name="email.list_messages",
                    arguments={"limit": 20, "unread_only": True},
                    confidence=0.75,
                    raw_response=response,
                )
        if any(token in lowered for token in ("搜索", "调研", "网页", "联网", "web", "browser")):
            tool = self._find_tool(tools, "web.research") or self._find_tool(tools, "web.search")
            if tool:
                return IntentDecision(
                    mode="tool_call",
                    reason="Fallback routed to web tool.",
                    tool_name=str(tool["name"]),
                    arguments={"topic": question, "query": question, "max_results": 5, "limit": 5},
                    confidence=0.25,
                    raw_response=response,
                )
        return IntentDecision(
            mode="direct_answer",
            reason="Fallback to direct answer.",
            confidence=0.1,
            raw_response=response,
        )

    def _normalize_tools(self, tools: ToolRegistry | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(tools, ToolRegistry):
            return tools.list_tools()
        return list(tools)

    def _find_tool(self, tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for tool in tools:
            if str(tool.get("name", "")) == name:
                return tool
        return None
