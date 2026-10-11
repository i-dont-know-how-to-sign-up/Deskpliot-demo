from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from ..core.contracts import SOURCE_TOOLS, ToolNames
from ..tasks.control import ExecutionControl, get_execution_control
from .schemas import AgentResult, TaskPlan, TaskPlanStep, persist_output, restore_output
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
        self.handler_registry: dict[str, tuple[str, Callable[..., Any]]] = {}
        self.register_handler("email.knowledge", lambda binding: self._knowledge_handler(binding["request"], binding["raw"]))
        self.register_handler("email.compose", lambda binding: self._communication_handler(binding["request"]))
        self.register_handler("email.commit", lambda binding: self._commit_handler(binding["request"], binding["email_tool"]))

    def register_handler(
        self, handler_key: str, factory: Callable[[dict[str, Any]], Callable[..., Any]], *, version: str = "1",
    ) -> None:
        """领域注册由应用代码完成；factory 接收可序列化 binding 并重建 closure。"""
        if not handler_key or not callable(factory):
            raise ValueError("执行器注册名和 factory 无效")
        self.handler_registry[handler_key] = (str(version), factory)

    def _bind_registered(self, step: TaskPlanStep, context: dict[str, Any]) -> None:
        entry = self.handler_registry.get(step.handler_key)
        if entry is None or entry[0] != step.handler_version:
            raise ValueError(f"恢复执行器未注册或版本变化：{step.handler_key or step.step_id}")
        binding = dict(step.handler_binding)
        binding.setdefault("recovery_context", context)
        binding.setdefault("arguments", dict(step.arguments))
        step.handler = entry[1](binding)
        if not callable(step.handler):
            raise ValueError(f"恢复执行器未返回 handler：{step.handler_key}")

    def execute_bound_plan(
        self,
        plan_preview: dict[str, Any],
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        *,
        initial_values: dict[str, Any] | None = None,
        parallel: bool = False,
        handler_keys: dict[str, str] | None = None,
        recovery_context: dict[str, Any] | None = None,
        task_id: str = "",
        session_id: str = "",
        resume_task_id: str = "",
        control: Any = None,
        approved_steps: Iterable[str] = (),
        prepare_steps: Iterable[str] = (),
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
        preparers = set(prepare_steps)
        for raw in raw_steps:
            if not isinstance(raw, dict):
                return SupervisorResult("failed", dict(initial_values or {}), [], "计划节点格式无效")
            step_id = str(raw.get("id") or raw.get("step_id") or "").strip()
            if not step_id:
                return SupervisorResult("failed", dict(initial_values or {}), [], "计划节点缺少 ID")
            handler = handlers.get(step_id)
            key = str((handler_keys or {}).get(step_id) or raw.get("handler_key") or "")
            if handler is None and not key:
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
                handler_key=key,
                handler_version=str(raw.get("handler_version") or "1"),
                handler_binding=dict(raw.get("handler_binding") or {}),
                retry_safe=bool(raw.get("retry_safe", True)),
                approval_mode="prepare" if step_id in preparers else "gate",
            ))
            if handler is None:
                try:
                    self._bind_registered(steps[-1], recovery_context or {})
                except ValueError as exc:
                    return SupervisorResult("recovery_required", dict(initial_values or {}), [], str(exc))
        plan = TaskPlan(
            goal=str(plan_preview.get("goal") or "执行复杂任务"),
            route="multi_agent",
            steps=steps,
            complexity_score=int(plan_preview.get("score", 0) or 0),
            max_agents=int(plan_preview.get("max_agents", 3)),
            max_tool_calls=int(plan_preview.get("max_tool_calls", 6)),
            max_total_tokens=int(plan_preview.get("max_total_tokens", 12000)),
            max_duration_seconds=float(plan_preview.get("max_duration_seconds", 120)),
            max_total_attempts=int(plan_preview.get("max_total_attempts", 12)),
            plan_version=str(plan_preview.get("plan_version") or "1"),
            metadata={**dict(plan_preview.get("metadata") or {}), "session_id": session_id,
                      "recovery_context": dict(recovery_context or {}), "initial_values": dict(initial_values or {})},
        )
        return self._execute(plan, initial_values or {}, parallel=parallel, task_id=resume_task_id or task_id,
                             control=control, approved_steps=approved_steps)

    def resume(
        self, task_id: str, *, control: Any = None, approved_steps: Iterable[str] = (),
    ) -> SupervisorResult:
        if self.task_ledger is None:
            return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复需要持久任务账本")
        descriptor = self.task_ledger.load_plan(task_id)
        if not descriptor or not descriptor.get("steps"):
            return SupervisorResult("recovery_required", {"task_id": task_id}, [], "任务没有可恢复计划")
        plan = TaskPlan.from_dict(descriptor)
        record = self.task_ledger.get_task(task_id) or {}
        saved = self._saved_results(task_id, record)
        for step in plan.steps:
            if record.get("nodes", {}).get(step.step_id, {}).get("status") in {"success", "succeeded"}:
                continue
            try:
                self._bind_registered(step, plan.metadata.get("recovery_context", {}))
            except ValueError as exc:
                return self._recovery_result(task_id, str(exc), saved)
        return self._execute(plan, {}, task_id=task_id, control=control, approved_steps=approved_steps)

    @staticmethod
    def _recovery_result(task_id: str, error: str, saved: list[AgentResult]) -> SupervisorResult:
        values: dict[str, Any] = {}
        for item in saved:
            if item.status in {"success", "succeeded"}:
                values.update(item.output)
        values["task_id"] = task_id
        return SupervisorResult("recovery_required", values, saved, error)

    def _saved_results(self, task_id: str, record: dict[str, Any]) -> list[AgentResult]:
        items = self.task_ledger.completed_results(task_id)
        if isinstance(items, dict):
            items = list(items.values())
        results = [AgentResult.from_dict(item) if isinstance(item, dict) else item for item in items]
        for item in results:
            item.output = restore_output(item.output)
        # 等待输出也必须恢复，否则重开审批时会丢失正文、附件和工具预览。
        known = {item.step_id for item in results}
        for node_id, node in record.get("nodes", {}).items():
            if node_id not in known and node.get("status") in {"pending", "waiting_human"} and node.get("output_ref"):
                output = self.task_ledger.load_output(task_id, node_id)
                results.append(AgentResult(str(node.get("agent", "executor")), "pending", restore_output(output or {}),
                                           step_id=node_id, checks=dict(node.get("checks") or {})))
        return results

    def _execute(
        self, plan: TaskPlan, initial_values: dict[str, Any], *, parallel: bool = False,
        task_id: str = "", control: Any = None, approved_steps: Iterable[str] = (),
    ) -> SupervisorResult:
        control = control if control is not None else get_execution_control()
        outer_task_id = str(getattr(control, "task_id", "") or "")
        if task_id and outer_task_id and task_id != outer_task_id:
            return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复任务与控制上下文不一致")
        task_id = task_id or outer_task_id
        owned = bool(outer_task_id)
        guard = dict(getattr(control, "ledger_guard", {}) or {})
        saved: list[AgentResult] = []
        counts: dict[str, int] = {}
        usage: dict[str, int] = {}
        ledger = self.task_ledger
        if task_id and ledger is None:
            return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复需要持久任务账本")
        # 计划预算只统计执行节点的增量；外层 ExecutionControl 继续约束整项任务。
        # 起点随计划持久化，恢复时不能通过重建控制器重置节点预算。
        if control is not None and "_execution_usage_baseline" not in plan.metadata:
            snapshot = control.snapshot_usage()
            plan.metadata["_execution_usage_baseline"] = {
                "tool_calls": int(snapshot.get("tool_calls", 0)),
                "total_tokens": int(snapshot.get("total_tokens", 0)),
            }
        if ledger is not None:
            if task_id:
                record = ledger.get_task(task_id)
                if record is None:
                    return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复任务不存在")
                descriptor = ledger.load_plan(task_id)
                if not descriptor or not descriptor.get("steps"):
                    ledger.save_plan(task_id, plan, **guard)
                    ledger.save_input(task_id, initial_values, **guard)
                else:
                    old_plan = TaskPlan.from_dict(descriptor)
                    def identity(item: TaskPlanStep) -> Any:
                        data = item.to_dict()
                        return {key: data[key] for key in (
                            "step_id", "depends_on", "allowed_tools", "requires_human", "arguments",
                            "handler_key", "handler_version", "handler_binding", "approval_mode", "retry_safe",
                        )}
                    if [identity(item) for item in old_plan.steps] != [identity(item) for item in plan.steps]:
                        return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复计划绑定或输入发生变化")
                    if old_plan.plan_version != plan.plan_version:
                        return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复计划版本发生变化")
                    if old_plan.metadata.get("recovery_context", {}) != plan.metadata.get("recovery_context", {}):
                        return SupervisorResult("recovery_required", {"task_id": task_id}, [], "恢复领域上下文发生变化")
                    initial_values = restore_output(ledger.load_initial_values(task_id) or {})
                    saved = self._saved_results(task_id, record)
                    success_ids = {node_id for node_id, node in record.get("nodes", {}).items()
                                   if node.get("status") in {"success", "succeeded"}}
                    if success_ids - {item.step_id for item in saved}:
                        return self._recovery_result(task_id, "已完成节点缺少有效检查点，不能自动重放", saved)
                    counts = {node_id: int(node.get("attempt_count", node.get("attempt", 0)) or 0)
                              for node_id, node in record.get("nodes", {}).items()}
                    usage = {"tool_calls": sum(int(node.get("tool_calls", 0)) for node in record.get("nodes", {}).values()),
                             "total_tokens": sum(int(node.get("tokens", 0)) for node in record.get("nodes", {}).values())}
                    unsafe = [node_id for node_id, node in record.get("nodes", {}).items()
                              if node.get("status") in {"unknown", "recovery_required"}
                              or (node.get("status") in {"running", "interrupted"} and
                                  (node.get("requires_human") or node.get("side_effect")))]
                    if unsafe:
                        return self._recovery_result(task_id, f"副作用节点需核对，禁止重放：{', '.join(unsafe)}", saved)
                    # 使用持久预算，调用者不能通过重建 preview 增加恢复额度。
                    handlers = {step.step_id: step.handler for step in plan.steps}
                    plan = old_plan
                    for step in plan.steps:
                        step.handler = handlers.get(step.step_id)
                if not owned:
                    ledger.resume(task_id, **guard)
            else:
                task_id = ledger.start(plan, initial_values=initial_values,
                                       session_id=str(plan.metadata.get("session_id") or ""),
                                       plan_version=plan.plan_version)
            if control is None:
                control = ExecutionControl(
                    task_id=task_id, deadline=time.monotonic() + plan.max_duration_seconds,
                    max_tool_calls=plan.max_tool_calls, max_tokens=plan.max_total_tokens,
                    max_attempts=plan.max_total_attempts, retry_policy=self.supervisor.retry_policy,
                )
                control.usage.update(usage)
            result = self.supervisor.execute(
                plan, initial_values=initial_values, parallel=parallel, completed_results=saved,
                attempt_counts=counts, initial_usage=usage, approved_steps=approved_steps, control=control,
                on_attempt=(lambda step, attempt: ledger.start_attempt(task_id, step.step_id, attempt, **guard)),
                on_result=(lambda item: ledger.update_node(
                    task_id, replace(item, output=persist_output(item.output),
                                     status="waiting_human" if item.status == "pending" else item.status),
                    record_attempt=bool(item.attempt_id or item.attempt), **guard,
                )),
            )
            if not owned:
                ledger.finish(task_id, result.status, result.error, **guard)
        else:
            result = self.supervisor.execute(plan, initial_values=initial_values, parallel=parallel,
                                             control=control, approved_steps=approved_steps)
        result.values["task_id"] = task_id
        return result

    def execute_email_plan(
        self,
        plan_preview: dict[str, Any],
        request: dict[str, Any],
        *, session_id: str = "", resume_task_id: str = "", control: Any = None,
    ) -> PlanExecution:
        plan = self._email_plan_from_preview(plan_preview, request)
        plan.metadata.update({"session_id": session_id, "initial_values": {"request": dict(request)}})
        result = self._execute(plan, {"request": dict(request)}, task_id=resume_task_id, control=control)
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
            raw = dict(raw)
            # 附件由用户显式指定时，Planner 的文件候选不能改写为其他来源。
            if ToolNames.FILES_READ_DOCUMENT in raw.get("allowed_tools", []) and request.get("attachment_paths"):
                paths = request["attachment_paths"]
                raw["arguments"] = {**dict(raw.get("arguments") or {}),
                                    "paths": [paths] if isinstance(paths, str) else list(paths)}
            step_id = str(raw.get("id") or raw.get("step_id") or f"knowledge_{index}")
            source_ids.append(step_id)
            steps.append(TaskPlanStep(
                step_id, "knowledge", str(raw.get("description", "获取邮件所需资料")),
                depends_on=[str(item) for item in raw.get("depends_on", [])],
                allowed_tools=[str(item) for item in raw.get("allowed_tools", [])],
                arguments=dict(raw.get("arguments", {})),
                handler=self._knowledge_handler(request, raw), max_retries=1,
                handler_key="email.knowledge", handler_binding={"request": dict(request), "raw": dict(raw)},
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
            handler_key="email.compose", handler_binding={"request": dict(request)},
        ))
        commit_id = str(commit_raw.get("id") or commit_raw.get("step_id") or "commit")
        steps.append(TaskPlanStep(
            commit_id, "communication", str(commit_raw.get("description", "准备邮件提交")),
            depends_on=[communication_id], allowed_tools=[email_tool], requires_human=True,
            arguments=dict(commit_raw.get("arguments", {})),
            handler=self._commit_handler(request, email_tool), max_retries=0,
            handler_key="email.commit", handler_binding={"request": dict(request), "email_tool": email_tool},
            retry_safe=False, approval_mode="prepare",
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
            if not attachments and isinstance(resolved_paths, list) and resolved_paths:
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
