from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import AgentStep
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.rag.web_research import ResearchResult, SearchResult, WebPage
from deskpilot.tools.file_tools import (
    apply_file_move,
    build_organization_plan,
    classify_folder,
    resolve_document_path,
    scan_folder,
)
from deskpilot.tools.permissions import assess_command, assess_path_write
from deskpilot.tools.tool_registry import build_default_tool_registry


def force_local_fallback() -> None:
    for name in (
        "DASHSCOPE_API_KEY",
        "LLM_API_KEY",
        "EMBEDDING_API_KEY",
        "SEARCH_API_KEY",
        "BING_SEARCH_API_KEY",
        "TAVILY_API_KEY",
    ):
        os.environ[name] = ""
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"
    os.environ["MEMORY_VECTOR_PROVIDER"] = "sqlite"
    os.environ["SEARCH_PROVIDER"] = "duckduckgo"


class FakeSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(title="LangGraph", url="https://example.com/langgraph", snippet="LangGraph docs"),
            SearchResult(title="AutoGen", url="https://example.com/autogen", snippet="AutoGen docs"),
        ][:limit]

    def fetch_page(self, url: str) -> WebPage:
        return WebPage(url=url, title="Example", text="Example page body " * 20)


class FakeWebResearchAgent:
    def __init__(self) -> None:
        self.search_client = FakeSearchClient()

    def research(self, topic: str, max_results: int | None = None):
        return type(
            "ResearchOutput",
            (),
            {
                "report": f"# {topic}\n\nmock report",
            },
        )()


class FakeFullWebResearchAgent:
    def research(self, topic: str, max_results: int | None = None) -> ResearchResult:
        return ResearchResult(
            topic=topic,
            report=f"# Web report\n\nTopic: {topic}\n\n2026 FIFA World Cup mock facts.",
            evidences=[],
            steps=[AgentStep("web_search", "success", "mock search")],
            sources=[],
            artifact_path="mock_report.md",
            used_llm=False,
        )


def test_registry_exposes_expected_tools(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    registry = build_default_tool_registry(index, FakeWebResearchAgent(), workspace_root=base)
    tool_names = {tool["name"] for tool in registry.list_tools()}

    assert {
        "web.search",
        "web.read_page",
        "web.research",
        "files.scan_folder",
        "files.classify_folder",
        "files.plan_organization",
        "files.apply_move",
        "files.build_index",
        "files.read_document",
        "files.resolve_document",
        "files.write_file",
        "files.write_markdown",
        "files.write_text",
        "files.write_docx",
        "files.write_pdf",
        "permissions.assess_path_write",
        "permissions.assess_command",
        "code.execute_python",
        "shell.execute_command",
        "desktop.get_active_window_title",
        "desktop.open_path",
        "desktop.open_url",
        "files.workspace_root",
    }.issubset(tool_names)
    knowledge = next(tool for tool in registry.list_tools() if tool["name"] == "knowledge.search")
    assert any(parameter["name"] == "context_expansion" for parameter in knowledge["parameters"])


def test_web_tools_and_file_tools_call_paths(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    registry = build_default_tool_registry(index, FakeWebResearchAgent(), workspace_root=base)

    web_search = registry.call("web.search", query="LangGraph AutoGen CrewAI", limit=2)
    assert web_search.ok
    assert len(web_search.output) == 2

    page = registry.call("web.read_page", url="https://example.com/langgraph")
    assert page.ok
    assert page.output["title"] == "Example"

    report = registry.call("web.research", topic="LangGraph AutoGen CrewAI 鍖哄埆", max_results=2)
    assert report.ok
    assert "mock report" in report.output

    folder = base / "downloads"
    folder.mkdir()
    (folder / "浼氳绾_2024.pdf").write_text("meeting notes", encoding="utf-8")
    (folder / "椤圭洰璁″垝.docx").write_text("project plan", encoding="utf-8")
    (folder / "readme.txt").write_text("plain note", encoding="utf-8")

    scan_result = registry.call("files.scan_folder", folder=folder)
    assert scan_result.ok
    assert len(scan_result.output) == 3

    plan_result = registry.call("files.plan_organization", folder=folder, target_root=base / "_organized")
    assert plan_result.ok
    assert plan_result.output["move_count"] == 3

    move_source = folder / "readme.txt"
    move_destination = base / "_organized" / "notes" / "readme.txt"
    move_result = registry.call("files.apply_move", source=move_source, destination=move_destination, confirm=False)
    assert move_result.ok
    assert move_result.output["confirmed"] is False
    assert move_source.exists()

    spoofed = registry.call("files.apply_move", source=move_source, destination=move_destination, confirm=True)
    assert spoofed.ok and spoofed.output["confirmed"] is False
    assert move_source.exists()
    confirm_result = registry.call_approved("files.apply_move", source=move_source, destination=move_destination)
    assert confirm_result.ok
    assert move_destination.exists()


def test_file_tools_direct_helpers(base: Path) -> None:
    folder = base / "files"
    folder.mkdir()
    (folder / "\u62a5\u544a_\u8c03\u7814.pdf").write_text("report", encoding="utf-8")
    (folder / "notes.txt").write_text("notes", encoding="utf-8")

    items = scan_folder(folder)
    assert len(items) == 2

    classification = classify_folder(folder)
    assert classification["total_files"] == 2
    assert classification["categories"]["research"] == 1

    plan = build_organization_plan(folder, base / "organized")
    assert plan["move_count"] == 2


def test_document_agent_lists_tools(base: Path) -> None:
    force_local_fallback()
    index = DocumentIndex(base / "index" / "index.json")
    agent = DocumentQAAgent(index)
    tools = {tool["name"] for tool in agent.list_tools()}

    assert "web.search" in tools
    assert "files.scan_folder" in tools
    assert "desktop.get_active_window_title" in tools


def test_document_agent_answers_local_file_question(base: Path) -> None:
    force_local_fallback()
    document_path = base / "\u9700\u6c42\u5206\u6790\u6587\u6863.md"
    document_path.write_text(
        "# \u9700\u6c42\u5206\u6790\\n\\n- \u76ee\u6807\uff1a\u529e\u516c\u52a9\u624b Agent\u3002\\n- \u80fd\u529b\uff1a\u9605\u8bfb\u6587\u6863\u3001\u7f51\u9875\u641c\u7d22\u3001\u6587\u4ef6\u6574\u7406\u3002\\n",
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        result = agent.answer("\u6253\u5f00\u5f53\u524d\u76ee\u5f55\u4e0b\u7684\u9700\u6c42\u5206\u6790\u6587\u6863.md\uff0c\u7136\u540e\u603b\u7ed3\u5176\u4e2d\u7684\u5185\u5bb9")
    finally:
        os.chdir(previous_cwd)

    assert "Agent" in result.answer or "\u672c\u5730\u6587\u6863" in result.answer
    assert any(step.name == "route_local_document" for step in result.steps)
    assert any(step.name == "read_local_document" for step in result.steps)


def test_document_agent_handles_open_action_prefix(base: Path) -> None:
    force_local_fallback()
    document_path = base / "\u6280\u672f\u8def\u7ebf\u6587\u6863.md"
    document_path.write_text(
        "# \u6280\u672f\u8def\u7ebf\\n\\n- \u7b2c\u4e00\u90e8\u5206\uff1a\u5de5\u5177\u6ce8\u518c\u3002\\n- \u7b2c\u4e8c\u90e8\u5206\uff1a\u7f51\u9875\u641c\u7d22\u3002\\n- \u7b2c\u4e09\u90e8\u5206\uff1a\u6587\u4ef6\u6574\u7406\u3002\\n",
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        result = agent.answer("\u6253\u5f00\u6280\u672f\u8def\u7ebf\u6587\u6863.md\uff0c\u7136\u540e\u603b\u7ed3\u5176\u4e2d\u7684\u5185\u5bb9")
    finally:
        os.chdir(previous_cwd)

    assert "\u5de5\u5177\u6ce8\u518c" in result.answer or "\u6280\u672f\u8def\u7ebf" in result.answer
    assert any(step.name == "route_local_document" for step in result.steps)
    assert any(step.name == "resolve_local_document" for step in result.steps)


def test_resolve_document_path_strips_action_words(base: Path) -> None:
    document_path = base / "\u6280\u672f\u8def\u7ebf\u6587\u6863.md"
    document_path.write_text("# \u6280\u672f\u8def\u7ebf\\n\\n\u5185\u5bb9", encoding="utf-8")

    resolved = resolve_document_path("\u6253\u5f00\u6280\u672f\u8def\u7ebf\u6587\u6863.md", search_roots=[base])

    assert resolved == document_path.resolve()


def test_write_tools_and_permission_paths(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    registry = build_default_tool_registry(index, FakeWebResearchAgent(), workspace_root=base)

    markdown_path = base / "outputs" / "demo.md"
    write_result = registry.call("files.write_markdown", path=markdown_path, content="# Demo\n\nhello")
    assert write_result.ok
    assert write_result.output["written"] is True
    assert markdown_path.read_text(encoding="utf-8").startswith("# Demo")

    no_overwrite = registry.call("files.write_markdown", path=markdown_path, content="new")
    assert no_overwrite.ok
    assert no_overwrite.output["written"] is False
    assert "overwrite=True" in no_overwrite.output["message"]

    overwrite = registry.call("files.write_markdown", path=markdown_path, content="new", overwrite=True)
    assert overwrite.ok
    assert overwrite.output["written"] is True
    assert markdown_path.read_text(encoding="utf-8") == "new"

    docx_path = base / "outputs" / "demo.docx"
    docx_result = registry.call("files.write_docx", path=docx_path, content="hello docx")
    assert docx_result.ok
    assert docx_result.output["written"] is True
    assert docx_path.read_bytes().startswith(b"PK")

    pdf_path = base / "outputs" / "demo.pdf"
    pdf_result = registry.call("files.write_pdf", path=pdf_path, content="hello pdf")
    assert pdf_result.ok
    assert pdf_result.output["written"] is True
    assert pdf_path.read_bytes().startswith(b"%PDF")

    risky = assess_path_write("C:/Temp/deskpilot_risky.md")
    assert risky.risk_level in {"high", "blocked"}
    if risky.risk_level == "high":
        assert risky.requires_confirmation


def test_execution_tools_require_confirmation_and_block_dangerous_commands(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    registry = build_default_tool_registry(index, FakeWebResearchAgent(), workspace_root=base)

    dry_python = registry.call("code.execute_python", code="print('nope')")
    assert dry_python.ok
    assert dry_python.output["executed"] is False
    assert dry_python.output["permission"]["requires_confirmation"] is True

    spoofed_python = registry.call("code.execute_python", code="print(1 + 1)", confirm=True)
    assert spoofed_python.output["executed"] is False
    run_python = registry.call_approved("code.execute_python", code="print(1 + 1)")
    assert run_python.ok
    assert run_python.output["executed"] is True
    assert run_python.output["stdout"].strip() == "2"

    dry_shell = registry.call("shell.execute_command", command=[sys.executable, "-c", "print('shell-ok')"])
    assert dry_shell.ok
    assert dry_shell.output["executed"] is False

    run_shell = registry.call_approved(
        "shell.execute_command",
        command=[sys.executable, "-c", "print('shell-ok')"],
    )
    assert run_shell.ok
    assert run_shell.output["executed"] is True
    assert run_shell.output["stdout"].strip() == "shell-ok"

    blocked = registry.call_approved("shell.execute_command", command="git reset --hard")
    assert not blocked.ok

    decision = assess_command("Remove-Item -Recurse C:/Temp/demo")
    assert decision.blocked


def test_execution_tools_ignore_extra_kwargs_during_approval_replay(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    registry = build_default_tool_registry(index, FakeWebResearchAgent(), workspace_root=base)

    result = registry.call_approved(
        "code.execute_python",
        code="print(1 + 1)",
        command="ignored-by-schema",
    )

    assert result.ok
    assert result.output["executed"] is True
    assert result.output["stdout"].strip() == "2"


def test_document_agent_creates_file_in_current_directory(base: Path) -> None:
    force_local_fallback()
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        result = agent.answer("\u5e2e\u6211\u5728\u5f53\u524d\u76ee\u5f55\u4e0b\u521b\u5efa\u4e00\u4e2atest.txt")
    finally:
        os.chdir(previous_cwd)

    created = base / "test.txt"
    assert created.exists()
    assert created.read_text(encoding="utf-8") == ""
    assert "\u5df2\u5199\u5165\u6587\u4ef6" in result.answer
    assert any(step.name == "route_file_write" for step in result.steps)
    assert any(step.name == "plan_task" for step in result.steps)
    assert any(step.name == "write_file" and step.status == "success" for step in result.steps)
    assert not any(step.name == "route_local_document" for step in result.steps)


def test_document_agent_keeps_absolute_directory_hint_for_file_creation(base: Path) -> None:
    force_local_fallback()
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        parsed = agent._extract_file_write_request(
            "\u5728C:\\Users\\TestUser\\Desktop\u4e0b\u521b\u5efatest1.docx"
        )
        result = agent.answer("\u5728C:\\Users\\TestUser\\Desktop\u4e0b\u521b\u5efatest1.docx")
    finally:
        os.chdir(previous_cwd)

    assert parsed is not None
    assert str(parsed["path"]).lower() == r"c:\users\testuser\desktop\test1.docx"
    assert not (base / "test1.docx").exists()
    assert r"C:\Users\TestUser\Desktop\test1.docx" in result.answer
    assert result.pending_action is not None
    assert result.pending_action["tool_name"] == "files.write_file"
    assert result.pending_action["action_id"].startswith("act_")
    assert "kwargs" not in result.pending_action
    assert result.pending_action["tool_name"] == "files.write_file"
    assert "\u4eba\u5de5\u786e\u8ba4" in result.answer or "Permission required" in result.answer
    assert any(step.name == "route_file_write" for step in result.steps)
    assert any(step.name == "plan_task" for step in result.steps)


def test_document_agent_writes_content_to_existing_file(base: Path) -> None:
    force_local_fallback()
    target = base / "test.txt"
    target.write_text("", encoding="utf-8")
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        result = agent.answer("\u5728test.txt\u4e2d\u5199\u51652026\u5e74\u4e16\u754c\u676f\u7684\u8d5b\u7a0b")
    finally:
        os.chdir(previous_cwd)

    assert target.read_text(encoding="utf-8") == "\u0032\u0030\u0032\u0036\u5e74\u4e16\u754c\u676f\u7684\u8d5b\u7a0b"
    assert "\u5df2\u5199\u5165\u6587\u4ef6" in result.answer
    assert any(step.name == "route_file_write" for step in result.steps)
    assert any(step.name == "plan_task" for step in result.steps)
    assert any(step.name == "write_file" and step.status == "success" for step in result.steps)
    assert not any(step.name == "route_local_document" for step in result.steps)


def test_document_agent_generates_public_domain_full_text_before_writing(base: Path) -> None:
    force_local_fallback()
    target = base / "test.txt"
    target.write_text("", encoding="utf-8")
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        generated = "静夜思\n\n李白\n\n床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。"
        with patch.object(agent.client, "chat", return_value=generated):
            result = agent.answer(
                "\u5728test.txt\u4e2d\u5199\u5165\u674e\u767d\u7684\u9759\u591c\u601d\u5168\u6587"
            )
    finally:
        os.chdir(previous_cwd)

    content = target.read_text(encoding="utf-8")
    assert "\u9759\u591c\u601d" in content
    assert "\u674e\u767d" in content
    assert "\u5e8a\u524d\u660e\u6708\u5149" in content
    assert "\u4f4e\u5934\u601d\u6545\u4e61" in content
    assert content != "\u674e\u767d\u7684\u9759\u591c\u601d\u5168\u6587"
    assert any(step.name == "generate_file_content" for step in result.steps)
    assert any(step.name == "plan_task" for step in result.steps)
    assert any(step.name == "write_file" and step.status == "success" for step in result.steps)


def test_document_agent_executes_python_test_and_reports_output(base: Path) -> None:
    force_local_fallback()
    tests_dir = base / "tests"
    tests_dir.mkdir()
    script = tests_dir / "test_sample.py"
    script.write_text("print('sample test passed')\n", encoding="utf-8")
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        result = agent.answer("\u6267\u884ctests/test_sample.py\u5e76\u544a\u8bc9\u6211\u6267\u884c\u7ed3\u679c")
    finally:
        os.chdir(previous_cwd)

    assert result.pending_action and result.pending_action["action_id"].startswith("act_")
    approved = agent.approve_pending_action(result.pending_action, session_id=result.session_id)
    assert "returncode" in approved.answer
    assert "sample test passed" in approved.answer
    assert any(step.name == "route_test_execution" for step in result.steps)
    assert any(step.name == "plan_task" for step in result.steps)
    assert any(step.name == "request_test_confirmation" for step in result.steps)
    assert any(step.name == "execute_approved_action" and step.status == "success" for step in approved.steps)
    assert not any(step.name == "retrieve_evidence" for step in result.steps)


def test_document_agent_routes_web_search_from_chat(base: Path) -> None:
    force_local_fallback()
    previous_cwd = Path.cwd()
    os.chdir(base)
    try:
        index = DocumentIndex(base / "index" / "index.json")
        agent = DocumentQAAgent(index)
        agent.web_research_agent = FakeFullWebResearchAgent()
        result = agent.answer("\u5e2e\u6211\u5728\u7f51\u4e0a\u641c\u7d222026\u4e16\u754c\u676f\u7684\u76f8\u5173\u4fe1\u606f")
    finally:
        os.chdir(previous_cwd)

    assert "2026 FIFA World Cup mock facts" in result.answer
    assert "mock_report.md" in result.answer
    assert any(step.name == "route_web_research" for step in result.steps)
    assert any(step.name == "web_search" and step.status == "success" for step in result.steps)
    assert not any(step.name == "direct_answer" for step in result.steps)


def main() -> None:
    force_local_fallback()
    with tempfile.TemporaryDirectory(prefix="deskpilot_tool_registry_test_") as tmp:
        base = Path(tmp)
        test_registry_exposes_expected_tools(base)
        test_web_tools_and_file_tools_call_paths(base)
        test_file_tools_direct_helpers(base)
        test_document_agent_lists_tools(base)
        test_write_tools_and_permission_paths(base)
        test_execution_tools_require_confirmation_and_block_dangerous_commands(base)
        test_document_agent_creates_file_in_current_directory(base)
        test_document_agent_keeps_absolute_directory_hint_for_file_creation(base)
        test_document_agent_writes_content_to_existing_file(base)
        test_document_agent_generates_public_domain_full_text_before_writing(base)
        test_document_agent_executes_python_test_and_reports_output(base)
        test_document_agent_routes_web_search_from_chat(base)
    print("Tool registry tests passed.")


if __name__ == "__main__":
    main()
