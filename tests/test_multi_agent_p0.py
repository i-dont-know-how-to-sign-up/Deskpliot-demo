from __future__ import annotations

import tempfile
from pathlib import Path
import sys

# 支持从项目根目录直接执行本文件：python tests/test_multi_agent_p0.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.multi_agent import MultiAgentRouter, PlannerAgent, SupervisorAgent
from deskpilot.multi_agent.agents import CommunicationAgent, KnowledgeAgent, ReflectionAgent
from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
from deskpilot.rag.vector_index import DocumentIndex


def test_planner_creates_ordered_multi_agent_plan_for_research_email() -> None:
    plan = PlannerAgent().build_plan("搜索明天上海天气并发送邮件")
    assert plan.route == "multi_agent"
    assert [step.step_id for step in plan.steps] == ["knowledge", "communication", "commit"]
    assert plan.steps[-1].requires_human is True
    assert PlannerAgent().validate(plan)[0] is True


def test_router_keeps_simple_question_single_agent() -> None:
    decision = MultiAgentRouter().route("什么是 RAG")
    assert decision.route == "single_agent"


def test_planner_prefers_structured_llm_plan_over_keyword_matching() -> None:
    response = '{"goal":"准备周报并发给负责人","route":"multi_agent","complexity_score":9,"steps":[' \
        '{"step_id":"knowledge","agent":"knowledge","description":"读取周报资料","depends_on":[],"allowed_tools":["files.read_document"]},' \
        '{"step_id":"communication","agent":"communication","description":"生成邮件","depends_on":["knowledge"],"allowed_tools":["email.create_reply_draft"]},' \
        '{"step_id":"commit","agent":"communication","description":"发送邮件","depends_on":["communication"],"allowed_tools":["email.send"],"requires_human":true}]}'
    planner = PlannerAgent(llm_call=lambda prompt: response)
    plan = planner.build_plan("请把本周资料整理好发给负责人")
    assert [step.step_id for step in plan.steps] == ["knowledge", "communication", "commit"]
    assert plan.steps[-1].requires_human is True
    assert planner.validate(plan)[0] is True


def test_planner_rejects_cycle_and_unapproved_side_effect() -> None:
    planner = PlannerAgent()
    cycle = TaskPlan("bad", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", ["b"]),
        TaskPlanStep("b", "communication", "b", ["a"]),
    ])
    assert planner.validate(cycle)[0] is False
    unsafe = TaskPlan("unsafe", "single_agent", [
        TaskPlanStep("send", "communication", "send", allowed_tools=["email.send"])
    ])
    assert planner.validate(unsafe)[0] is False


def test_supervisor_runs_ordered_steps_and_stops_for_human() -> None:
    plan = TaskPlan("demo", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=lambda values: {"evidence": "ok"}),
        TaskPlanStep("b", "communication", "b", ["a"], requires_human=True),
    ])
    result = SupervisorAgent().execute(plan)
    assert result.status == "waiting_human"
    assert result.values["evidence"] == "ok"


def test_specialist_agents_have_bounded_roles() -> None:
    class Result:
        ok = True
        output = [{"title": "source"}]
        error = ""

    knowledge = KnowledgeAgent(lambda name, **kwargs: Result())
    assert knowledge.search("query").output["evidence"]
    communication = CommunicationAgent(lambda prompt: "draft")
    draft = communication.compose_email("subject", "request")
    assert draft.output["requires_human"] is True
    reflection = ReflectionAgent(lambda prompt: '{"approved": true}')
    assert reflection.review("result", quality=4, latency=1).output["approved"] is True
    assert reflection.review("result", quality=1, latency=1).status == "skipped"


def test_document_agent_exposes_planner_preview() -> None:
    with tempfile.TemporaryDirectory(prefix="deskpilot_multi_agent_") as tmp:
        agent = DocumentQAAgent(DocumentIndex(Path(tmp) / "index.json"))
        preview = agent.plan_task("搜索明天上海天气并发送邮件")
        assert preview["valid"] is True
        assert preview["route"] == "multi_agent"
        assert preview["steps"][0]["agent"] == "knowledge"


def test_supervisor_retries_failed_step_and_runs_independent_steps() -> None:
    attempts = {"count": 0}

    def flaky(values: dict[str, object]) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return {"recovered": True}

    plan = TaskPlan("p1", "multi_agent", [
        TaskPlanStep("a", "knowledge", "flaky", handler=flaky, max_retries=1),
        TaskPlanStep("b", "knowledge", "independent", handler=lambda values: {"parallel": True}),
    ])
    result = SupervisorAgent().execute(plan)
    assert result.status == "success"
    assert attempts["count"] == 2
    assert result.values["recovered"] is True
    assert result.values["parallel"] is True


def test_planned_email_accepts_recipient_array_and_fills_explicit_subject() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "captured"

    agent._answer_research_email_request = capture
    result = agent._answer_planned_task(
        question="搜索上海天气并发送给 a@example.com，标题为test5",
        plan_preview={"steps": [{"arguments": {"to": ["a@example.com"], "query": "上海天气"}}]},
        steps=[], session_id="s", user_message_id="m", memory_context=object(),
    )
    assert result == "captured"
    assert captured["request"]["to"] == "a@example.com"
    assert captured["request"]["subject"] == "test5"


def test_planned_email_preserves_attachment_requirement() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "captured"

    agent._answer_research_email_request = capture
    result = agent._answer_planned_task(
        question="调研 Agentic RL，整理成文档并作为附件发送给 a@example.com，标题为 agentic RL",
        plan_preview={
            "steps": [
                {
                    "arguments": {
                        "to": ["a@example.com"],
                        "subject": "agentic RL",
                        "request": "Agentic RL 最新论文",
                        "attach_report": True,
                    }
                }
            ]
        },
        steps=[], session_id="s", user_message_id="m", memory_context=object(),
    )
    assert result == "captured"
    assert captured["request"]["attach_report"] is True


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"multi agent tests passed: {len(tests)}")
