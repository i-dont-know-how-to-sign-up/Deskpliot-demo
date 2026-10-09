from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.contracts import SOURCE_TOOLS, ToolNames
from .schemas import TaskPlan, TaskPlanStep
from .supervisor import SupervisorAgent, SupervisorResult
from .task_ledger import TaskLedger


@dataclass
class PlanExecution:
    supervisor: SupervisorResult
    tool_name: str = ""
    tool_result: Any = None
    call_arguments: dict[str, Any] | None = None


class PlanExecutor:
    """把 Planner 节点绑定到受控 handler，并交给 Supervisor 执行 DAG。"""

    def __init__(
        self,
        *,
        tool_call: Callable[..., Any],
        research: Callable[[str], Any],
        llm_call: Callable[[str], str],
        supervisor: SupervisorAgent | None = None,
        task_ledger: TaskLedger | None = None,
    ) -> None:
        self.tool_call = tool_call
        self.research = research
        self.llm_call = llm_call
        self.supervisor = supervisor or SupervisorAgent()
        self.task_ledger = task_ledger

    def execute_bound_plan(
        self,
        plan_preview: dict[str, Any],
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        *,
        initial_values: dict[str, Any] | None = None,
        parallel: bool = False,
    ) -> SupervisorResult:
        """执行 Planner DAG，并按节点 ID 注入受控领域 handler。

        Planner 只描述依赖、Agent 和允许工具，不直接携带可执行函数。Core 在
        执行前完成绑定，因此同一 Runtime 可以承载网页、文件、索引和邮件任务，
        也便于评测替换外部依赖。
        """
        raw_steps = plan_preview.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            return SupervisorResult("failed", dict(initial_values or {}), [], "计划没有可执行节点")
        steps: list[TaskPlanStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            step_id = str(raw.get("id") or raw.get("step_id") or "").strip()
            if not step_id:
                return SupervisorResult("failed", dict(initial_values or {}), [], "计划节点缺少 ID")
            handler = handlers.get(step_id)
            if handler is None:
                return SupervisorResult(
                    "failed", dict(initial_values or {}), [], f"计划节点未绑定执行器：{step_id}",
                )
            steps.append(TaskPlanStep(
                step_id=step_id,
                agent=str(raw.get("agent") or "executor"),
                description=str(raw.get("description") or step_id),
                depends_on=[str(value) for value in raw.get("depends_on", [])],
                allowed_tools=[str(value) for value in raw.get("allowed_tools", [])],
                requires_human=bool(raw.get("requires_human", False)),
                arguments=dict(raw.get("arguments", {})) if isinstance(raw.get("arguments"), dict) else {},
                handler=handler,
                max_retries=max(0, int(raw.get("max_retries", 1) or 0)),
            ))
        plan = TaskPlan(
            goal=str(plan_preview.get("goal") or "执行复杂任务"),
            route="multi_agent",
            steps=steps,
            complexity_score=int(plan_preview.get("score", 0) or 0),
        )
        task_id = self.task_ledger.start(plan) if self.task_ledger else ""
        result = self.supervisor.execute(
            plan,
            initial_values=initial_values,
            parallel=parallel,
            on_result=(lambda item: self.task_ledger.update_node(task_id, item)) if self.task_ledger else None,
        )
        if self.task_ledger:
            self.task_ledger.finish(task_id, result.status, result.error)
            result.values["task_id"] = task_id
        return result

    def execute_email_plan(
        self,
        plan_preview: dict[str, Any],
        request: dict[str, Any],
    ) -> PlanExecution:
        plan = self._email_plan_from_preview(plan_preview, request)
        task_id = self.task_ledger.start(plan) if self.task_ledger else ""
        result = self.supervisor.execute(
            plan,
            initial_values={"request": dict(request)},
            parallel=False,
            on_result=(lambda item: self.task_ledger.update_node(task_id, item)) if self.task_ledger else None,
        )
        if self.task_ledger:
            self.task_ledger.finish(task_id, result.status, result.error)
            result.values["task_id"] = task_id
        commit = next((item for item in result.results if item.output.get("tool_name") in {
            ToolNames.EMAIL_SEND, ToolNames.EMAIL_SAVE_DRAFT,
        }), None)
        if commit is None:
            return PlanExecution(result)
        return PlanExecution(
            result,
            tool_name=str(commit.output.get("tool_name", "")),
            tool_result=commit.output.get("tool_result"),
            call_arguments=dict(commit.output.get("call_arguments", {})),
        )

    def _email_plan_from_preview(
        self, plan_preview: dict[str, Any], request: dict[str, Any],
    ) -> TaskPlan:
        raw_steps = plan_preview.get("steps", [])
        normalized_steps = [item for item in raw_steps if isinstance(item, dict)]
        email_tool = str(request.get("email_tool") or ToolNames.EMAIL_SEND)
        commit_raw = next((item for item in normalized_steps if email_tool in item.get("allowed_tools", [])), {})
        source_raw = [item for item in normalized_steps if SOURCE_TOOLS.intersection(item.get("allowed_tools", []))]
        if not source_raw:
            source_raw = [{"id": "knowledge", "agent": "knowledge", "allowed_tools": [ToolNames.WEB_RESEARCH],
                           "arguments": {"query": request.get("request", "")}}]
        steps: list[TaskPlanStep] = []
        source_ids: list[str] = []
        for index, raw in enumerate(source_raw, start=1):
            step_id = str(raw.get("id") or raw.get("step_id") or f"knowledge_{index}")
            source_ids.append(step_id)
            steps.append(TaskPlanStep(
                step_id, "knowledge", str(raw.get("description", "获取邮件所需资料")),
                depends_on=[str(item) for item in raw.get("depends_on", [])],
                allowed_tools=[str(item) for item in raw.get("allowed_tools", [])],
                arguments=dict(raw.get("arguments", {})),
                handler=self._knowledge_handler(request, raw), max_retries=1,
            ))
        communication_raw = next((
            item for item in normalized_steps
            if item is not commit_raw and item.get("agent") == "communication" and not item.get("allowed_tools")
        ), {})
        communication_id = str(communication_raw.get("id") or communication_raw.get("step_id") or "communication")
        steps.append(TaskPlanStep(
            communication_id, "communication",
            str(communication_raw.get("description", "根据资料生成邮件正文")),
            depends_on=source_ids, handler=self._communication_handler(request), max_retries=1,
        ))
        commit_id = str(commit_raw.get("id") or commit_raw.get("step_id") or "commit")
        steps.append(TaskPlanStep(
            commit_id, "communication", str(commit_raw.get("description", "准备邮件提交")),
            depends_on=[communication_id], allowed_tools=[email_tool], requires_human=True,
            arguments=dict(commit_raw.get("arguments", {})),
            handler=self._commit_handler(request, email_tool), max_retries=0,
        ))
        return TaskPlan(
            goal=str(plan_preview.get("goal") or request.get("request") or "准备邮件"),
            route="multi_agent", steps=steps,
            complexity_score=int(plan_preview.get("score", 8) or 8),
        )

    def _knowledge_handler(self, request: dict[str, Any], raw: dict[str, Any]):
        tools = [str(item) for item in raw.get("allowed_tools", [])]
        arguments = dict(raw.get("arguments", {}))
        query = str(arguments.get("query") or request.get("request", "")).strip()
        attach_report = bool(request.get("attach_report", False))

        def handler(_values: dict[str, Any]) -> dict[str, Any]:
            if ToolNames.FILES_READ_DOCUMENT in tools:
                paths = arguments.get("paths") or arguments.get("path") or request.get("attachment_paths") or []
                if isinstance(paths, str):
                    paths = [paths]
                documents: list[str] = []
                resolved_paths: list[str] = []
                for path in paths:
                    result = self.tool_call(ToolNames.FILES_READ_DOCUMENT, path=str(path))
                    if not getattr(result, "ok", False):
                        raise RuntimeError(str(getattr(result, "error", "本地文档读取失败")))
                    output = getattr(result, "output", {})
                    if isinstance(output, dict):
                        documents.append(str(output.get("content", "")))
                        resolved_paths.append(str(output.get("path") or path))
                if not documents:
                    raise RuntimeError("本地文档读取节点没有提供可用路径")
                return {"evidence": "\n\n".join(documents), "resolved_paths": resolved_paths, "_tool_calls": len(documents)}
            if ToolNames.WEB_RESEARCH in tools or (attach_report and ToolNames.WEB_RESEARCH in tools):
                result = self.research(query)
                report = str(getattr(result, "report", ""))
                if not report.strip():
                    raise RuntimeError("网页调研没有生成可用报告")
                return {
                    "research_report": report,
                    "artifact_path": str(getattr(result, "artifact_path", "")),
                    "_tool_calls": 1,
                }
            tool_name = ToolNames.WEB_SEARCH if ToolNames.WEB_SEARCH in tools else tools[0]
            call_arguments = {"query": query, "limit": 5}
            if tool_name == ToolNames.KNOWLEDGE_SEARCH:
                call_arguments = {"query": query, "top_k": 5}
            result = self.tool_call(tool_name, **call_arguments)
            if not getattr(result, "ok", False):
                raise RuntimeError(str(getattr(result, "error", "资料获取失败")))
            output = getattr(result, "output", None)
            return {"evidence": output if output is not None else [], "_tool_calls": 1}

        return handler

    def _communication_handler(self, request: dict[str, Any]):
        def handler(values: dict[str, Any]) -> dict[str, Any]:
            evidence = values.get("research_report", values.get("evidence", []))
            prompt = (
                f"邮件主题：{request.get('subject', '')}\n用户要求：{request.get('request', '')}\n"
                f"请仅根据以下资料撰写简洁中文邮件正文，不得编造：\n{evidence}"
            )
            body = self.llm_call(prompt).strip()
            if not body:
                body = str(evidence)[:3000].strip()
            if not body:
                raise RuntimeError("无法根据资料生成邮件正文")
            return {
                "body": body,
                "artifact_path": values.get("artifact_path", ""),
                "resolved_paths": values.get("resolved_paths", []),
            }

        return handler

    def _commit_handler(self, request: dict[str, Any], email_tool: str):
        def handler(values: dict[str, Any]) -> dict[str, Any]:
            raw_attachments = request.get("attachment_paths") or []
            if isinstance(raw_attachments, str):
                raw_attachments = [raw_attachments]
            attachments = [str(item) for item in raw_attachments if str(item).strip()]
            resolved_paths = values.get("resolved_paths", [])
            if isinstance(resolved_paths, list) and resolved_paths:
                attachments = [str(item) for item in resolved_paths if str(item).strip()]
            artifact = str(values.get("artifact_path", "")).strip()
            if request.get("attach_report"):
                if not artifact or not Path(artifact).is_file():
                    raise RuntimeError("计划要求附件，但调研节点没有生成可用文件")
                attachments.append(artifact)
            missing = [path for path in attachments if not Path(path).is_file()]
            if missing:
                raise RuntimeError(f"邮件附件不存在：{missing[0]}")
            recipients = request.get("to", [])
            if isinstance(recipients, str):
                recipients = [recipients]
            call_arguments = {
                "to": [str(item) for item in recipients],
                "subject": str(request["subject"]),
                "body": str(values.get("body", "")),
                "attachment_paths": attachments,
            }
            result = self.tool_call(email_tool, **call_arguments)
            if not getattr(result, "ok", False):
                raise RuntimeError(str(getattr(result, "error", "邮件工具调用失败")))
            output = getattr(result, "output", None)
            permission = output.get("permission", {}) if isinstance(output, dict) else {}
            waiting = bool(permission.get("requires_confirmation") and not permission.get("blocked"))
            return {
                "tool_name": email_tool,
                "tool_result": result,
                "call_arguments": call_arguments,
                "_waiting_human": waiting,
                "_tool_calls": 1,
            }

        return handler
