from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import COMMIT_TOOLS, SOURCE_TOOLS, ToolNames
from ..intent.schemas import IntentDecision


PlanBuilder = Callable[..., dict[str, Any]]


@dataclass
class PlanRepairResult:
    """计划修复后的结果和可直接写入 Steps 的审计事件。"""

    plan: dict[str, Any]
    events: list[str] = field(default_factory=list)


class PlanRepairService:
    """对 Planner 结果执行有界、确定顺序且每种最多一次的修复。"""

    def __init__(self, plan_builder: PlanBuilder) -> None:
        self.plan_builder = plan_builder

    def repair(
        self,
        *,
        question: str,
        planner_context: str,
        decision: IntentDecision,
        plan: dict[str, Any],
        has_index_documents: bool,
        workspace_files: list[str] | None = None,
    ) -> PlanRepairResult:
        current = plan
        events: list[str] = []
        tools = self._tools(current)

        # 策略 1：Planner 已经选择知识库，但 Router 未预告目录时，只补送一次有限目录。
        if (
            ToolNames.KNOWLEDGE_SEARCH in tools
            and not decision.needs_index_catalog
            and has_index_documents
        ):
            current = self.plan_builder(
                question,
                planner_context + "\n已选用 knowledge.search，请从索引目录挑选实际文档 ID。",
                include_workspace_files=decision.needs_workspace_files,
                include_index_catalog=True,
            )
            events.append("计划选择了知识库检索，已补送有限索引目录并重规划一次。")
            tools = self._tools(current)

        # 策略 2：Planner 必须满足 Router 声明的工具契约，且不得自行增加文件副作用。
        required_tools = set(decision.required_tools)
        unexpected_file_commit = (
            ToolNames.FILES_WRITE_FILE in tools and not decision.requires_file_output
        )
        contract_incomplete = bool(required_tools - tools) or not tools or unexpected_file_commit
        if contract_incomplete:
            feedback_parts = [
                "上一次计划不满足执行契约。",
                f"必须包含的注册工具：{sorted(required_tools)}。",
                "至少一个节点必须选择可执行工具，不得只返回分析或 communication 空节点。",
            ]
            if unexpected_file_commit:
                feedback_parts.append("用户没有要求生成文件，删除 files.write_file 及其提交节点。")
            repaired = self.plan_builder(
                question,
                planner_context + "\n" + "".join(feedback_parts),
                include_workspace_files=(
                    decision.needs_workspace_files
                    or ToolNames.FILES_READ_DOCUMENT in required_tools
                ),
                include_index_catalog=(
                    decision.needs_index_catalog
                    or ToolNames.KNOWLEDGE_SEARCH in required_tools
                ),
            )
            repaired_tools = self._tools(repaired)
            repaired_unexpected_file = (
                ToolNames.FILES_WRITE_FILE in repaired_tools and not decision.requires_file_output
            )
            if (
                repaired.get("valid")
                and repaired_tools
                and required_tools.issubset(repaired_tools)
                and not repaired_unexpected_file
            ):
                current = repaired
                tools = repaired_tools
                events.append("原计划未满足 Router 工具契约，已按必需工具有界重规划一次。")

        # 策略 3：文件产物必须同时具备资料源和写入节点；缺失时只重规划一次。
        if decision.requires_file_output and not self._has_file_pipeline(current, tools):
            feedback = (
                "上一次计划无法执行："
                + str(current.get("error") or "缺少资料获取或文件写入节点")
                + "。请重新输出完整 JSON 计划：先由 knowledge 检索资料，再由 commit 使用 "
                "files.write_file 写入；写入节点必须标记人工确认。如源为已索引文档使用 "
                "knowledge.search，如需联网使用 web.search。请给出纯检索主题及用户指定的文件路径，"
                "不要虚构检索结果。"
            )
            repaired = self.plan_builder(
                question,
                planner_context + "\n" + feedback,
                include_workspace_files=decision.needs_workspace_files,
                include_index_catalog=(
                    decision.needs_index_catalog or ToolNames.KNOWLEDGE_SEARCH in tools
                ),
            )
            repaired_tools = self._tools(repaired)
            if repaired.get("valid") and self._has_file_pipeline(repaired, repaired_tools):
                current = repaired
                tools = repaired_tools
                events.append("原计划缺少可执行依赖，已重规划检索与文件提交节点。")

        # 有界重规划仍不满足 Router 契约时，不再退回原错误计划。这里只根据
        # 结构化工具契约、已有参数和有限工作区候选编译最小 DAG。
        required_tools = list(dict.fromkeys(decision.required_tools))
        tools = self._tools(current)
        if required_tools and not set(required_tools).issubset(tools):
            current = self._compile_contract_plan(
                question=question,
                decision=decision,
                required_tools=required_tools,
                workspace_files=workspace_files or [],
                previous=current,
            )
            events.append("LLM 重规划仍未满足工具契约，已编译最小可执行 DAG。")

        return PlanRepairResult(plan=current, events=events)

    @staticmethod
    def _compile_contract_plan(
        *,
        question: str,
        decision: IntentDecision,
        required_tools: list[str],
        workspace_files: list[str],
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        """把 Router 的工具契约编译为最小计划，不推断新的业务意图。"""
        previous_arguments: dict[str, dict[str, Any]] = {}
        shared_arguments: dict[str, Any] = dict(decision.arguments)
        for step in previous.get("steps", []):
            if not isinstance(step, dict):
                continue
            arguments = step.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            shared_arguments.update(arguments)
            for tool in step.get("allowed_tools", []):
                previous_arguments.setdefault(str(tool), {}).update(arguments)

        explicit_files = [
            path for path in workspace_files
            if path.casefold() in question.casefold()
            or path.replace("\\", "/").split("/")[-1].casefold() in question.casefold()
        ]
        selected_files = explicit_files or workspace_files[:1]
        steps: list[dict[str, Any]] = []
        source_ids: list[str] = []
        for index, tool in enumerate((item for item in required_tools if item in SOURCE_TOOLS), start=1):
            arguments = dict(previous_arguments.get(tool, {}))
            if tool in {ToolNames.WEB_SEARCH, ToolNames.WEB_RESEARCH, ToolNames.KNOWLEDGE_SEARCH}:
                arguments.setdefault("query", str(decision.arguments.get("query") or question))
            elif tool == ToolNames.FILES_READ_DOCUMENT and selected_files:
                arguments.setdefault("paths", selected_files)
            step_id = f"source_{index}"
            source_ids.append(step_id)
            steps.append({
                "id": step_id,
                "agent": "knowledge",
                "depends_on": [],
                "allowed_tools": [tool],
                "requires_human": False,
                "arguments": arguments,
            })

        dependency_ids = list(source_ids)
        for index, tool in enumerate((item for item in required_tools if item in COMMIT_TOOLS), start=1):
            arguments = dict(shared_arguments)
            arguments.update(previous_arguments.get(tool, {}))
            if tool in {ToolNames.EMAIL_SEND, ToolNames.EMAIL_SAVE_DRAFT} and selected_files:
                arguments.setdefault("attachment_paths", selected_files)
            step_id = f"commit_{index}"
            steps.append({
                "id": step_id,
                "agent": "communication",
                "depends_on": list(dependency_ids),
                "allowed_tools": [tool],
                "requires_human": True,
                "arguments": arguments,
            })
            dependency_ids = [step_id]

        return {
            "route": "multi_agent" if len(steps) > 1 else "single_agent",
            "score": max(3, len(steps) * 3),
            "reason": "根据 Router 工具契约编译的最小执行计划。",
            "valid": bool(steps),
            "error": "" if steps else "Router 工具契约为空",
            "steps": steps,
        }

    @staticmethod
    def _tools(plan: dict[str, Any]) -> set[str]:
        return {
            str(tool)
            for step in plan.get("steps", [])
            if isinstance(step, dict)
            for tool in step.get("allowed_tools", [])
        }

    @staticmethod
    def _has_file_pipeline(plan: dict[str, Any], tools: set[str]) -> bool:
        return bool(
            plan.get("valid")
            and ToolNames.FILES_WRITE_FILE in tools
            and SOURCE_TOOLS.intersection(tools)
        )
