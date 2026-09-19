from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.core.models import Chunk, Document
from deskpilot.intent.schemas import IntentDecision
from deskpilot.intent.router import IntentRouter
from deskpilot.multi_agent.schemas import TaskPlan
from deskpilot.rag.vector_index import DocumentIndex


def _unexpected_plan(*args, **kwargs):
    raise AssertionError("Simple requests must not call Planner")


def test_simple_requests_skip_planner() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(DocumentIndex(Path(folder) / "index.json"))
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent, "plan_task", side_effect=_unexpected_plan
        ), patch.object(agent.memory_extractor, "extract", return_value=[]):
            with patch.object(agent.intent_router, "route", return_value=IntentDecision(mode="direct_answer")), patch.object(
                agent, "_direct_answer", return_value="RAG 是检索增强生成。"
            ):
                result = agent.answer("什么是 RAG？请用三句话回答。")
                assert "RAG" in result.answer
                assert not any(step.name == "planner_agent" for step in result.steps)
            with patch.object(agent.intent_router, "route", return_value=IntentDecision(
                mode="tool_call", tool_name="web.search", arguments={"query": "RAG"}
            )), patch.object(agent, "_handle_intent_tool_call", return_value="tool dispatched"):
                assert agent.answer("搜索 RAG") == "tool dispatched"


def test_document_candidates_are_bounded() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        agent = DocumentQAAgent.__new__(DocumentQAAgent)
        agent.workspace_root = root
        (root / ".conda" / "pkgs").mkdir(parents=True)
        (root / ".conda" / "pkgs" / "secret.md").write_text("hidden", encoding="utf-8")
        (root / "开发计划文档.md").write_text("plan", encoding="utf-8")
        for i in range(100):
            (root / f"other_{i}.md").write_text("x", encoding="utf-8")
        candidates = agent._workspace_document_candidates("阅读开发计划文档")
        assert candidates[0] == "开发计划文档.md"
        assert len(candidates) <= 60
        assert all(".conda" not in name for name in candidates)
        assert len(json.dumps(candidates, ensure_ascii=False)) <= 6002


def test_planner_receives_candidates_only_when_requested() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "项目说明.md").write_text("demo", encoding="utf-8")
        agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
        agent.workspace_root = root
        with patch.object(agent.multi_agent_router.planner, "build_plan", return_value=TaskPlan("demo", "single_agent", [])) as build:
            agent.plan_task("搜索天气")
            assert build.call_args.kwargs["workspace_files"] == []
            agent.plan_task("读取项目说明", include_workspace_files=True)
            assert build.call_args.kwargs["workspace_files"] == ["项目说明.md"]


def test_complex_document_request_uses_planner() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
        agent.workspace_root = root
        (root / "a.md").write_text("a", encoding="utf-8")
        (root / "b.md").write_text("b", encoding="utf-8")
        decision = IntentDecision(mode="plan_task", needs_workspace_files=True)
        preview = {"route": "single_agent", "score": 4, "valid": True, "steps": [
            {"id": "knowledge", "arguments": {"paths": ["a.md", "b.md"]}}
        ]}
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.intent_router, "route", return_value=decision
        ), patch.object(agent, "plan_task", return_value=preview) as plan, patch.object(
            agent, "_answer_local_documents", return_value="compared"
        ):
            assert agent.answer("比较两个文档") == "compared"
            assert plan.call_args.kwargs["include_workspace_files"] is True


def test_router_understands_planning_decision() -> None:
    decision = IntentRouter._parse_response(
        IntentRouter.__new__(IntentRouter),
        '{"mode":"plan_task","needs_workspace_files":true,"reason":"compare documents"}',
    )
    assert decision is not None and decision.mode == "plan_task"
    assert decision.needs_workspace_files is True


def test_complex_email_plan_does_not_list_files() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(DocumentIndex(Path(folder) / "index.json"))
        decision = IntentDecision(mode="plan_task", needs_workspace_files=False)
        preview = {"route": "multi_agent", "score": 9, "valid": True, "steps": [
            {"id": "commit", "allowed_tools": ["email.send"], "requires_human": True, "arguments": {}}
        ]}
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.intent_router, "route", return_value=decision
        ) as route, patch.object(agent, "plan_task", return_value=preview) as plan, patch.object(
            agent, "_answer_planned_task", return_value="planned"
        ) as execute:
            assert agent.answer("先搜索资料，再整理成邮件并发送") == "planned"
            assert route.call_count == 1
            assert plan.call_args.kwargs["include_workspace_files"] is False
            assert execute.call_count == 1


def test_single_index_question_recovers_from_empty_plan() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        index = DocumentIndex(root / "index.json")
        content = "ATLAS-503 出现后，客户端采用指数退避，最多重试三次，并保留 checkpoint。"
        document = Document("atlas", str(root / "atlas.md"), "atlas.md", "md", content)
        chunk = Chunk("atlas_1", "atlas", content, "atlas.md#故障恢复", 1,
                      metadata={"heading_path": ["故障恢复"]})
        chunk.embedding = local_hash_embedding(content)
        index.documents[document.doc_id] = document
        index.chunks[chunk.chunk_id] = chunk
        agent = DocumentQAAgent(index)
        decision = IntentDecision(mode="plan_task", needs_index_catalog=True, arguments={"query": "ATLAS-503 恢复"})
        preview = {"route": "single_agent", "score": 2, "valid": True, "steps": []}
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.intent_router, "route", return_value=decision
        ), patch.object(agent, "plan_task", return_value=preview), patch.object(
            agent.client, "chat", return_value=""
        ), patch.object(index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]), patch.object(
            agent.memory_extractor, "extract", return_value=[]
        ):
            result = agent.answer("ATLAS-503 出现后客户端应该如何恢复？")
        assert result.evidences
        assert any(step.name == "repair_plan" for step in result.steps)
        assert any(step.name == "retrieve_evidence" for step in result.steps)
        assert "任务规划缺少" not in result.answer


def test_multi_document_collection_is_not_downgraded_to_simple_search() -> None:
    decision = IntentDecision(mode="plan_task", needs_index_catalog=True)
    preview = {"steps": [{"allowed_tools": ["knowledge.search"], "arguments": {
        "query": "比较论文", "scope": "collection", "doc_ids": ["one", "two"]
    }}]}
    hint = [{"doc_id": "one"}, {"doc_id": "two"}]
    assert DocumentQAAgent._planned_simple_index_query("比较论文", preview, decision, hint) is None


if __name__ == "__main__":
    test_simple_requests_skip_planner()
    test_document_candidates_are_bounded()
    test_planner_receives_candidates_only_when_requested()
    test_complex_document_request_uses_planner()
    test_router_understands_planning_decision()
    test_complex_email_plan_does_not_list_files()
    test_single_index_question_recovers_from_empty_plan()
    test_multi_document_collection_is_not_downgraded_to_simple_search()
    print("Planner gate tests passed.")
