from __future__ import annotations

from deskpilot.core.contracts import StepStatus, ToolNames
from deskpilot.core.handlers import DirectAnswerHandler
from deskpilot.core.plan_repair import PlanRepairService
from deskpilot.core.runtime import PlanAndExecuteRuntime, PlanStep, SequentialExecution
from deskpilot.intent.schemas import IntentDecision
from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
from deskpilot.multi_agent.supervisor import SupervisorAgent


def test_sequential_runtime_has_unambiguous_result_type() -> None:
    result = PlanAndExecuteRuntime().run([PlanStep("one", lambda _values: {"ok": True})])

    assert isinstance(result, SequentialExecution)
    assert result.values["ok"] is True
    assert all(str(item.status) == StepStatus.SUCCESS for item in result.agent_steps)


def test_supervisor_defaults_to_deterministic_sequential_execution() -> None:
    calls: list[str] = []
    plan = TaskPlan(
        goal="ordered",
        route="multi_agent",
        steps=[
            TaskPlanStep("a", "knowledge", "a", handler=lambda _values: calls.append("a") or {}),
            TaskPlanStep("b", "knowledge", "b", handler=lambda _values: calls.append("b") or {}),
        ],
    )

    result = SupervisorAgent().execute(plan)

    assert result.status == StepStatus.SUCCESS
    assert calls == ["a", "b"]


def test_plan_repair_runs_each_strategy_at_most_once() -> None:
    calls: list[dict] = []

    def build(_question: str, _context: str, **kwargs):
        calls.append(kwargs)
        return {
            "valid": True,
            "steps": [
                {"id": "source", "allowed_tools": [ToolNames.KNOWLEDGE_SEARCH]},
                {"id": "commit", "allowed_tools": [ToolNames.FILES_WRITE_FILE]},
            ],
        }

    service = PlanRepairService(build)
    decision = IntentDecision(
        mode="plan_task", reason="fixture", requires_file_output=True,
        needs_index_catalog=False, needs_workspace_files=False,
    )
    result = service.repair(
        question="生成报告", planner_context="context", decision=decision,
        plan={"valid": False, "steps": [{"allowed_tools": [ToolNames.KNOWLEDGE_SEARCH]}]},
        has_index_documents=True,
    )

    # 补目录后的计划已经完整，因此不会再执行第二次文件依赖重规划。
    assert len(calls) == 1
    assert len(result.events) == 1
    assert ToolNames.FILES_WRITE_FILE in PlanRepairService._tools(result.plan)


def test_plan_repair_replans_incomplete_file_pipeline_once() -> None:
    calls = 0

    def build(_question: str, _context: str, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "valid": True,
            "steps": [
                {"allowed_tools": [ToolNames.WEB_SEARCH]},
                {"allowed_tools": [ToolNames.FILES_WRITE_FILE]},
            ],
        }

    result = PlanRepairService(build).repair(
        question="搜索后写入", planner_context="context",
        decision=IntentDecision(
            mode="plan_task", reason="fixture", requires_file_output=True,
            needs_index_catalog=False, needs_workspace_files=False,
        ),
        plan={"valid": True, "steps": [{"allowed_tools": [ToolNames.FILES_WRITE_FILE]}]},
        has_index_documents=False,
    )

    assert calls == 1
    assert result.events == ["原计划缺少可执行依赖，已重规划检索与文件提交节点。"]


def test_plan_repair_enforces_router_required_tool_contract() -> None:
    calls = 0

    def build(_question: str, context: str, **_kwargs):
        nonlocal calls
        calls += 1
        assert "web.search" in context
        return {
            "valid": True,
            "steps": [{"id": "source", "allowed_tools": [ToolNames.WEB_SEARCH]}],
        }

    result = PlanRepairService(build).repair(
        question="搜索资料", planner_context="context",
        decision=IntentDecision(
            mode="plan_task", required_tools=[ToolNames.WEB_SEARCH],
            explicit_web_retrieval=True,
        ),
        plan={"valid": True, "steps": [{"id": "compose", "allowed_tools": []}]},
        has_index_documents=False,
    )

    assert calls == 1
    assert PlanRepairService._tools(result.plan) == {ToolNames.WEB_SEARCH}
    assert result.events == ["原计划未满足 Router 工具契约，已按必需工具有界重规划一次。"]


def test_plan_repair_removes_unrequested_file_side_effect() -> None:
    def build(_question: str, _context: str, **_kwargs):
        return {
            "valid": True,
            "steps": [{"id": "source", "allowed_tools": [ToolNames.WEB_RESEARCH]}],
        }

    result = PlanRepairService(build).repair(
        question="调研并回答", planner_context="context",
        decision=IntentDecision(
            mode="plan_task", required_tools=[ToolNames.WEB_RESEARCH],
            explicit_web_retrieval=True, requires_file_output=False,
        ),
        plan={"valid": True, "steps": [
            {"id": "source", "allowed_tools": [ToolNames.WEB_RESEARCH]},
            {"id": "commit", "allowed_tools": [ToolNames.FILES_WRITE_FILE]},
        ]},
        has_index_documents=False,
    )

    assert PlanRepairService._tools(result.plan) == {ToolNames.WEB_RESEARCH}


def test_plan_repair_compiles_minimal_dag_after_replan_still_violates_contract() -> None:
    def stubborn_builder(*args, **kwargs):
        return {
            "valid": True,
            "steps": [{"id": "wrong", "allowed_tools": [ToolNames.FILES_WRITE_FILE],
                       "arguments": {}}],
        }

    result = PlanRepairService(stubborn_builder).repair(
        question="读取 note.txt 并作为附件发送",
        planner_context="context",
        decision=IntentDecision(
            mode="plan_task",
            needs_workspace_files=True,
            required_tools=[ToolNames.FILES_READ_DOCUMENT, ToolNames.EMAIL_SEND],
        ),
        plan=stubborn_builder(),
        has_index_documents=False,
        workspace_files=["note.txt", "other.txt"],
    )

    assert PlanRepairService._tools(result.plan) == {
        ToolNames.FILES_READ_DOCUMENT, ToolNames.EMAIL_SEND,
    }
    assert result.plan["steps"][0]["arguments"]["paths"] == ["note.txt"]
    assert result.plan["steps"][1]["arguments"]["attachment_paths"] == ["note.txt"]
    assert result.plan["steps"][1]["depends_on"] == ["source_1"]
    assert any("最小可执行 DAG" in event for event in result.events)


def test_plan_repair_compiles_web_to_file_contract() -> None:
    service = PlanRepairService(lambda *args, **kwargs: {
        "valid": True,
        "steps": [{"id": "wrong", "allowed_tools": [ToolNames.FILES_READ_DOCUMENT]}],
    })
    result = service.repair(
        question="搜索上海明天天气并保存为 weather.md",
        planner_context="context",
        decision=IntentDecision(
            mode="plan_task",
            requires_file_output=True,
            required_tools=[ToolNames.WEB_SEARCH, ToolNames.FILES_WRITE_FILE],
            arguments={"query": "上海明天天气"},
        ),
        plan={"valid": True, "steps": []},
        has_index_documents=False,
    )

    assert [step["allowed_tools"] for step in result.plan["steps"]] == [
        [ToolNames.WEB_SEARCH], [ToolNames.FILES_WRITE_FILE],
    ]
    assert result.plan["steps"][1]["depends_on"] == ["source_1"]


def test_direct_answer_handler_owns_fallback_and_review_policy() -> None:
    class Context:
        output_budget = 200

    handler = DirectAnswerHandler(
        answer=lambda _question, _context: "共三项：\n1. A\n2. B\n3. C",
        fallback=lambda _question, _context: "fallback",
        needs_review=lambda _answer: True,
        review=lambda _question, answer, budget: answer + f"\nreviewed={budget}",
        api_error=lambda: "",
    )

    result = handler.handle("列举", Context())

    assert result.used_llm is True
    assert "reviewed=200" in result.answer
    assert [step.name for step in result.steps] == ["review_enumeration", "direct_answer"]
