from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.core.models import Chunk, Document
from deskpilot.intent.schemas import IntentDecision
from deskpilot.multi_agent.schemas import TaskPlan
from deskpilot.multi_agent.planner import PlannerAgent
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.write_tools import write_file
from deskpilot.memory.session_store import SessionStore


def make_index(root: Path) -> DocumentIndex:
    index = DocumentIndex(root / "index.json")
    for doc_id, title, content, count in (
        ("one", "BLIP multimodal.pdf", "BLIP aligns image and text representations.", 5),
        ("two", "BLIP-2 multimodal.pdf", "BLIP-2 connects frozen visual and language models.", 2),
        ("other", "Weather.pdf", "Tomorrow is rainy.", 1),
    ):
        index.documents[doc_id] = Document(doc_id, str(root / title), title, "pdf", content)
        for n in range(count):
            chunk = Chunk(f"{doc_id}_{n}", doc_id, content, f"{title}#chunk-{n}", n)
            chunk.embedding = local_hash_embedding(content)
            index.chunks[chunk.chunk_id] = chunk
    return index


def test_mixed_dimensions_use_compatible_query_vectors() -> None:
    chunk = Chunk("c", "d", "multimodal image text", "paper", 0)
    chunk.embedding = local_hash_embedding(chunk.text)
    assert DocumentIndex._vector_score(chunk, [1.0] * 1024, local_hash_embedding(chunk.text)) > 0.99
    chunk.embedding = [1.0] * 1024
    assert DocumentIndex._vector_score(chunk, [1.0] * 384, local_hash_embedding(chunk.text)) == 0


def test_collection_retrieval_covers_selected_papers() -> None:
    with tempfile.TemporaryDirectory() as folder:
        index = make_index(Path(folder))
        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]):
            found = index.search_collection("multimodal papers", doc_ids=["one", "two"])
        assert {e.doc_id for e in found} == {"one", "two"}
        assert len(found) <= 4


def test_report_writes_cited_content_not_just_chat_text() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        index = make_index(root)
        agent = DocumentQAAgent(index)
        agent.workspace_root = root
        captured = {}

        def call(tool, **kwargs):
            captured.update(kwargs)
            return type("Result", (), {"ok": True, "output": {"written": True}, "error": ""})()

        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]), patch.object(
            agent.client, "chat", return_value="# 技术报告\n\nBLIP 对齐图文 [1]，BLIP-2 连接预训练模型 [3]。"
        ), patch.object(agent.tool_registry, "call", side_effect=call), patch.object(
            agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs
        ):
            result = agent._answer_index_report(
                "总结多模态论文并输出报告", {"query": "multimodal", "doc_ids": ["one", "two"], "write_report": True},
                [], "session", "message", object(),
            )
        assert result["pending_action"] is None
        assert captured["path"] == str(root / "技术报告.md")
        assert "## 参考来源" in captured["content"]
        assert "BLIP multimodal.pdf" in captured["content"]
        assert "Weather.pdf" not in captured["content"]
        assert "已生成技术报告" in result["answer"]


def test_complex_indexed_request_goes_through_plan() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(make_index(Path(folder)))
        decision = IntentDecision(mode="plan_task", needs_index_catalog=True)
        preview = {"route": "multi_agent", "score": 8, "valid": True, "steps": [
            {"id": "knowledge", "allowed_tools": ["knowledge.search"], "arguments": {"query": "multimodal", "doc_ids": ["one", "two"]}},
            {"id": "commit", "allowed_tools": ["files.write_file"], "requires_human": True, "arguments": {}},
        ]}
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.intent_router, "route", return_value=decision
        ), patch.object(agent, "plan_task", return_value=preview) as plan, patch.object(
            agent, "_answer_index_report", return_value="report called"
        ) as report:
            assert agent.answer("总结索引论文并写技术报告") == "report called"
            assert plan.call_args.kwargs["include_workspace_files"] is False
            assert plan.call_args.kwargs["include_index_catalog"] is True
            assert report.call_args.args[1]["write_report"] is True


def test_planner_gets_index_titles_without_workspace_scan() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(make_index(Path(folder)))
        agent.workspace_root = Path(folder)
        with patch.object(agent.multi_agent_router.planner, "build_plan", return_value=TaskPlan("summary", "single_agent", [])) as build:
            agent.plan_task("总结索引中的多模态论文", include_index_catalog=True)
        assert build.call_args.kwargs["workspace_files"] == []
        titles = {doc["title"] for doc in build.call_args.kwargs["indexed_documents"]}
        assert "BLIP multimodal.pdf" in titles
        assert "Weather.pdf" in titles  # 目录由 Planner 决定筛选，不能预先按关键词丢失候选。


def test_planner_parses_index_report_plan() -> None:
    response = ('{"goal":"report","route":"multi_agent","complexity_score":8,"steps":['
                '{"step_id":"knowledge","agent":"knowledge","allowed_tools":["knowledge.search"],'
                '"arguments":{"query":"multimodal","doc_ids":["one","two"]}},'
                '{"step_id":"commit","agent":"communication","depends_on":["knowledge"],'
                '"allowed_tools":["files.write_file"],"arguments":{"path":""}}]}')
    prompts = []
    planner = PlannerAgent(llm_call=lambda prompt: (prompts.append(prompt), response)[1])
    plan = planner.build_plan("总结索引论文并写报告", indexed_documents=[{"doc_id": "one", "title": "BLIP.pdf"}])
    assert planner.validate(plan)[0]
    assert plan.steps[0].arguments["doc_ids"] == ["one", "two"]
    assert plan.steps[-1].requires_human
    assert "BLIP.pdf" in prompts[0]


def test_report_is_actually_saved_in_safe_workspace() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        index = make_index(root)
        agent = DocumentQAAgent(index)
        agent.workspace_root = root

        def write(tool, **kwargs):
            assert tool == "files.write_file"
            output = write_file(**kwargs, safe_roots=[root])
            return type("Result", (), {"ok": True, "output": output, "error": ""})()

        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]), patch.object(
            agent.client, "chat", return_value="# 报告\n\n图文模型 [1]。"
        ), patch.object(agent.tool_registry, "call", side_effect=write), patch.object(
            agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs
        ):
            result = agent._answer_index_report(
                "总结多模态论文并保存到当前目录下", {"query": "multimodal", "doc_ids": ["one", "two"],
                                                "write_report": True, "path": "当前目录下/别处.md"},
                [], "session", "message", object(),
            )
        saved = root / "技术报告.md"
        assert saved.exists()
        assert "## 参考来源" in saved.read_text(encoding="utf-8")
        assert "已生成技术报告" in result["answer"]


def test_report_outside_workspace_requires_confirmation() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        index = make_index(root)
        agent = DocumentQAAgent(index)
        agent.workspace_root = root
        external = root.parent / "deskpilot_report_outside.md"

        def write(tool, **kwargs):
            output = write_file(**kwargs, safe_roots=[root])
            return type("Result", (), {"ok": True, "output": output, "error": ""})()

        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]), patch.object(
            agent.client, "chat", return_value="# 报告\n\n证据 [1]。"
        ), patch.object(agent.tool_registry, "call", side_effect=write), patch.object(
            agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs
        ):
            result = agent._answer_index_report(
                "汇总索引论文并保存", {"query": "multimodal", "doc_ids": ["one"],
                               "write_report": True, "path": str(external)},
                [], "session", "message", object(),
            )
        assert result["pending_action"]["tool_name"] == "files.write_file"
        assert not external.exists()


def test_document_coverage_samples_different_positions() -> None:
    with tempfile.TemporaryDirectory() as folder:
        index = make_index(Path(folder))
        doc = index.documents["one"]
        for position in range(5, 80):
            text = f"{doc.title} section {position} explains visual language details."
            chunk = Chunk(f"one_{position}", "one", text, f"{doc.title}#chunk-{position}", position)
            chunk.embedding = local_hash_embedding(text)
            index.chunks[chunk.chunk_id] = chunk
        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]):
            found = index.search_collection("multimodal image text", doc_ids=["one"], chunks_per_doc=6)
        positions = sorted(index.chunks[e.chunk_id].position for e in found)
        assert len(found) == 6
        assert positions[-1] - positions[0] >= 40


def test_collection_does_not_treat_bibliography_as_paper_body() -> None:
    with tempfile.TemporaryDirectory() as folder:
        index = make_index(Path(folder))
        body = Chunk("body", "one", "Multimodal method and experiments.", "paper#body", 10)
        body.embedding = local_hash_embedding(body.text)
        heading = Chunk("references", "one", "Conclusion\nREFERENCES\n[1] Other paper.", "paper#references", 20)
        heading.embedding = local_hash_embedding(heading.text)
        citation = Chunk("citation", "one", "[2] Multimodal attacks from another paper.", "paper#citation", 30)
        citation.embedding = local_hash_embedding(citation.text)
        index.chunks.update({chunk.chunk_id: chunk for chunk in (body, heading, citation)})
        with patch.object(index.client, "embed", return_value=[[1.0] * 1024]):
            found = index.search_collection("Multimodal attacks", doc_ids=["one"], chunks_per_doc=6)
        assert "body" in {item.chunk_id for item in found}
        assert "references" not in {item.chunk_id for item in found}
        assert "citation" not in {item.chunk_id for item in found}


def test_router_repairs_single_search_for_file_request() -> None:
    class Client:
        def __init__(self):
            self.responses = iter([
                '{"mode":"tool_call","tool_name":"knowledge.search","arguments":{"query":"多模态论文"}}',
                '{"requires_file_output":true}',
            ])

        def chat(self, messages, temperature=0):
            return next(self.responses)

    from deskpilot.intent.router import IntentRouter
    question = "请总结当前索引中的多模态论文并将技术报告保存到当前目录"
    decision = IntentRouter(Client()).route(question, [{"name": "knowledge.search", "parameters": [
        {"name": "query", "type": "string", "required": True}]}])
    assert decision.mode == "plan_task"
    assert decision.needs_index_catalog and decision.requires_file_output


def test_output_requirement_accepts_fenced_json() -> None:
    from deskpilot.intent.router import IntentRouter
    class Client:
        def chat(self, messages, temperature=0):
            return '```json\n{"requires_file_output":true}\n```'

    assert IntentRouter(Client()).requires_output_file("将报告保存到当前目录")


def test_missing_commit_is_repaired_instead_of_returning_five_rag_chunks() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(make_index(Path(folder)))
        decision = IntentDecision(mode="plan_task", needs_index_catalog=True, requires_file_output=True)
        preview = {"route": "single_agent", "score": 3, "valid": True, "steps": [
            {"id": "knowledge", "allowed_tools": ["knowledge.search"],
             "arguments": {"query": "多模态论文", "doc_ids": ["one", "two"]}}
        ]}
        with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.intent_router, "route", return_value=decision
        ), patch.object(agent, "plan_task", return_value=preview), patch.object(
            agent, "_answer_index_report", return_value="report workflow"
        ) as report, patch.object(agent, "_answer_rag_request", side_effect=AssertionError("RAG fallback is forbidden")):
            assert agent.answer("请总结一下当前索引中的多模态论文，并输出一份技术报告放在当前目录下") == "report workflow"
        assert report.call_args.args[1]["write_report"] is True


def test_missing_document_ids_are_selected_from_catalog() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(make_index(Path(folder)))
        with patch.object(agent.client, "chat", return_value='{"doc_ids":["one","two","not-in-index"]}'):
            assert agent._select_index_doc_ids("multimodal paper") == ["one", "two"]


def test_answer_recovers_search_only_route_and_saves_report() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        index = make_index(root)
        agent = DocumentQAAgent(index)
        agent.workspace_root = root
        agent.session_store = SessionStore(root / "sessions")
        del agent.usage_tracker
        question = "请总结当前索引中的多模态论文，并输出一份技术报告放在当前目录下，报告中需要有引用来源"

        def chat(messages, **kwargs):
            role = messages[0]["content"]
            if "intent router" in role:
                return '{"mode":"tool_call","tool_name":"knowledge.search","arguments":{"query":"multimodal papers"}}'
            if "文件产物校验器" in role:
                return '{"requires_file_output":true}'
            if "任务规划器" in role:
                return ('{"goal":"report","route":"single_agent","complexity_score":4,"steps":['
                        '{"step_id":"knowledge","agent":"knowledge","allowed_tools":["knowledge.search"],'
                        '"arguments":{"query":"multimodal papers","doc_ids":["one","two"]}}]}')
            if "技术报告" in role:
                return "# 多模态论文技术报告\n\nBLIP 的方法见 [1]，BLIP-2 的方法见 [7]。"
            raise AssertionError(f"Unexpected model request: {role}")

        def call(name, **kwargs):
            assert name == "files.write_file"
            output = write_file(**kwargs, safe_roots=[root])
            return type("Result", (), {"ok": True, "output": output, "error": ""})()

        with patch.object(agent.client, "chat", side_effect=chat), patch.object(
            index.client, "embed", side_effect=lambda items: [local_hash_embedding(value) for value in items]
        ), patch.object(agent.memory_store, "search", return_value=[]), patch.object(
            agent.memory_store, "list_memories", return_value=[]
        ), patch.object(agent.memory_extractor, "extract", return_value=[]), patch.object(
            agent.memory_compactor, "compact_if_needed", return_value=(False, "")
        ), patch.object(agent.tool_registry, "call", side_effect=call):
            result = agent.answer(question)
        saved = root / "技术报告.md"
        assert saved.exists()
        assert "## 参考来源" in saved.read_text(encoding="utf-8")
        assert "BLIP-2 multimodal.pdf" in saved.read_text(encoding="utf-8")
        assert "已生成技术报告" in result.answer
        assert any(step.name == "repair_plan" for step in result.steps)
        assert any(step.name == "write_report" and step.status == "success" for step in result.steps)


if __name__ == "__main__":
    test_mixed_dimensions_use_compatible_query_vectors()
    test_collection_retrieval_covers_selected_papers()
    test_report_writes_cited_content_not_just_chat_text()
    test_complex_indexed_request_goes_through_plan()
    test_planner_gets_index_titles_without_workspace_scan()
    test_planner_parses_index_report_plan()
    test_report_is_actually_saved_in_safe_workspace()
    test_report_outside_workspace_requires_confirmation()
    test_document_coverage_samples_different_positions()
    test_collection_does_not_treat_bibliography_as_paper_body()
    test_router_repairs_single_search_for_file_request()
    test_output_requirement_accepts_fenced_json()
    test_missing_commit_is_repaired_instead_of_returning_five_rag_chunks()
    test_missing_document_ids_are_selected_from_catalog()
    test_answer_recovers_search_only_route_and_saves_report()
    print("Indexed report tests passed.")
