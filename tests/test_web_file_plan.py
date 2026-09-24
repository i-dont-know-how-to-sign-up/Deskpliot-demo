from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.intent.schemas import IntentDecision
from deskpilot.intent.router import IntentRouter
from deskpilot.multi_agent.planner import PlannerAgent
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.write_tools import write_file


QUERIES = [
    "搜索‘上海明天的天气’，然后在桌面创建一个weather.txt文件，把结果写入到整个文件中",
    "在桌面创建一个weather.txt文件，然后搜索‘上海明天的天气’，把搜索结果写入到weather.txt中",
]


def preview() -> dict:
    return {"valid": True, "route": "multi_agent", "score": 8, "steps": [
        {"id": "knowledge", "allowed_tools": ["web.search"], "arguments": {"query": "上海明天的天气"}},
        {"id": "commit", "allowed_tools": ["files.write_file"], "requires_human": True,
         "arguments": {"path": "weather.txt"}},
    ]}


def test_both_word_orders_dispatch_search_then_write() -> None:
    for question in QUERIES:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
            desktop = root / "Desktop"
            desktop.mkdir()
            agent.workspace_root = root
            calls = []

            def call(name, **kwargs):
                calls.append((name, kwargs))
                if name == "web.search":
                    output = [{"title": "上海天气", "url": "https://example.org/weather", "snippet": "明日预报"}]
                else:
                    output = {"written": False, "permission": {
                        "requires_confirmation": True, "risk_level": "high", "reasons": ["outside workspace"]}}
                return type("Result", (), {"ok": True, "output": output, "error": ""})()

            with patch.object(agent, "_desktop_directory", return_value=desktop), patch.object(
                agent.tool_registry, "call", side_effect=call
            ), patch.object(agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs):
                plan = preview()
                plan["steps"][1]["arguments"]["path"] = str(root / "weather.txt")
                request = agent._planned_web_file_request(plan)
                result = agent._answer_web_file_request(question, request, [], "s", "m", object())
            assert [name for name, _ in calls] == ["web.search", "files.write_file"]
            assert calls[0][1]["query"] == "上海明天的天气"
            assert calls[1][1]["path"] == str(desktop / "weather.txt")
            assert "https://example.org/weather" in calls[1][1]["content"]
            assert "到整个文件中" not in calls[1][1]["content"]
            assert result["pending_action"]["kwargs"]["content"] == calls[1][1]["content"]
            assert not (root / "weather.txt").exists()


def test_empty_search_does_not_write() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(DocumentIndex(Path(folder) / "index.json"))
        calls = []

        def call(name, **kwargs):
            calls.append(name)
            return type("Result", (), {"ok": True, "output": [], "error": ""})()

        with patch.object(agent, "_desktop_directory", return_value=Path(folder)), patch.object(
            agent.tool_registry, "call", side_effect=call
        ), patch.object(agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs):
            result = agent._answer_web_file_request(QUERIES[0], agent._planned_web_file_request(preview()),
                                                    [], "s", "m", object())
        assert calls == ["web.search"]
        assert "未写入" in result["answer"]


def test_actual_file_write_stays_pending_until_approval() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        workspace = root / "workspace"
        desktop = root / "Desktop"
        workspace.mkdir()
        desktop.mkdir()
        agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
        agent.workspace_root = workspace

        def call(name, **kwargs):
            if name == "web.search":
                output = [{"title": "上海天气", "url": "https://example.org/weather", "snippet": "明日预报"}]
            else:
                output = write_file(**kwargs, safe_roots=[workspace])
            return type("Result", (), {"ok": True, "output": output, "error": ""})()

        with patch.object(agent, "_desktop_directory", return_value=desktop), patch.object(
            agent.tool_registry, "call", side_effect=call
        ), patch.object(agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs):
            response = agent._answer_web_file_request(QUERIES[0], agent._planned_web_file_request(preview()),
                                                      [], "s", "m", object())
        target = desktop / "weather.txt"
        assert response["pending_action"]["tool_name"] == "files.write_file"
        assert not target.exists()
        output = write_file(**response["pending_action"]["kwargs"], safe_roots=[workspace], confirm=True)
        assert output["written"]
        assert "明日预报" in target.read_text(encoding="utf-8")


def test_answer_routes_composite_before_document_resolution() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(DocumentIndex(Path(folder) / "index.json"))
        for question in QUERIES:
            with patch.object(agent.memory_store, "search", return_value=[]), patch.object(
                agent.intent_router, "route", return_value=IntentDecision(mode="plan_task")
            ), patch.object(agent, "plan_task", return_value=preview()), patch.object(
                agent, "_answer_web_file_request", return_value="dispatched"
            ) as execute, patch.object(agent, "_answer_local_documents", side_effect=AssertionError("wrong route")):
                assert agent.answer(question) == "dispatched"
                assert execute.call_count == 1


def test_single_write_refuses_destination_phrase_as_content() -> None:
    with tempfile.TemporaryDirectory() as folder:
        agent = DocumentQAAgent(DocumentIndex(Path(folder) / "index.json"))
        with patch.object(agent.tool_registry, "call", side_effect=AssertionError("should not write")), patch.object(
            agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs
        ):
            result = agent._answer_file_write(
                QUERIES[0], {"path": str(Path(folder) / "weather.txt"), "content": "到整个文件中"},
                [], "s", "m", object(),
            )
        assert "未写入文件" in result["answer"]
        assert not (Path(folder) / "weather.txt").exists()


def test_existing_desktop_file_is_not_overwritten() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        agent = DocumentQAAgent(DocumentIndex(root / "index.json"))
        desktop = root / "Desktop"
        desktop.mkdir()
        target = desktop / "weather.txt"
        target.write_text("original", encoding="utf-8")
        with patch.object(agent, "_desktop_directory", return_value=desktop), patch.object(
            agent.tool_registry, "call", side_effect=AssertionError("should stop before tools")
        ), patch.object(agent, "_finalize_answer", side_effect=lambda **kwargs: kwargs):
            result = agent._answer_web_file_request(QUERIES[0], agent._planned_web_file_request(preview()),
                                                    [], "s", "m", object())
        assert "未覆盖" in result["answer"]
        assert target.read_text(encoding="utf-8") == "original"


def test_semantic_router_and_planner_accept_web_file_dependencies() -> None:
    class Client:
        def chat(self, messages, temperature=0.0):
            return '{"mode":"plan_task","reason":"dependent operations","needs_workspace_files":false}'

    decision = IntentRouter(Client()).route(QUERIES[0], [])
    assert decision.mode == "plan_task"
    assert decision.needs_workspace_files is False
    response = ('{"goal":"search and save","route":"multi_agent","complexity_score":8,"steps":['
                '{"step_id":"knowledge","agent":"knowledge","allowed_tools":["web.search"],'
                '"arguments":{"query":"上海明天的天气"}},'
                '{"step_id":"commit","agent":"communication","depends_on":["knowledge"],'
                '"allowed_tools":["files.write_file"],"arguments":{"path":"weather.txt"}}]}')
    planner = PlannerAgent(llm_call=lambda prompt: response)
    for question in QUERIES:
        plan = planner.build_plan(question)
        assert planner.validate(plan)[0]
        agent = DocumentQAAgent.__new__(DocumentQAAgent)
        steps = [{"id": s.step_id, "allowed_tools": s.allowed_tools, "arguments": s.arguments} for s in plan.steps]
        assert agent._planned_web_file_request({"steps": steps}) == {
            "query": "上海明天的天气", "path": "weather.txt"
        }


if __name__ == "__main__":
    test_both_word_orders_dispatch_search_then_write()
    test_empty_search_does_not_write()
    test_actual_file_write_stays_pending_until_approval()
    test_answer_routes_composite_before_document_resolution()
    test_single_write_refuses_destination_phrase_as_content()
    test_existing_desktop_file_is_not_overwritten()
    test_semantic_router_and_planner_accept_web_file_dependencies()
    print("Web-to-file plan tests passed.")
