from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.context import ContextBuilder
from deskpilot.memory.memory_compactor import MemoryCompactor
from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.rag.web_research import (
    SearchResult,
    WebPage,
    WebResearchAgent,
    WebSearchClient,
    clean_html,
    extract_report_title,
    extract_main_text,
    make_report_slug,
    make_browser_search_url,
    normalize_research_topic,
    normalize_search_result_url,
    parse_duckduckgo_results,
    parse_playwright_search_links,
)


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
            SearchResult(
                title="DeskPilot Memory",
                url="https://example.com/memory",
                snippet="DeskPilot 第二阶段实现会话记忆系统。",
            ),
            SearchResult(
                title="DeskPilot Web Research",
                url="https://example.com/research",
                snippet="DeskPilot 第三阶段实现网页搜索和调研报告。",
            ),
        ][:limit]

    def fetch_page(self, url: str) -> WebPage:
        if "memory" in url:
            return WebPage(
                url=url,
                title="DeskPilot Memory",
                text=(
                    "DeskPilot 第二阶段实现会话记忆系统。"
                    "该系统保存原始消息、结构化记忆、滚动摘要和向量检索索引。"
                    "这些能力支持长会话和跨会话项目跟进。"
                )
                * 4,
            )
        return WebPage(
            url=url,
            title="DeskPilot Web Research",
            text=(
                "DeskPilot 第三阶段实现网页搜索和调研报告。"
                "该阶段会搜索网页、抓取正文、建立索引、检索证据并保存 Markdown 报告。"
                "报告会带有来源和可核对的证据片段。"
            )
            * 4,
        )


def test_extract_main_text_removes_noise() -> None:
    html_text = """
    <html><head><title>测试页面</title><style>.x{}</style></head>
    <body><nav>导航</nav><article><h1>标题</h1><p>这是主要正文。</p><p>第二段内容。</p></article>
    <script>alert(1)</script></body></html>
    """
    text = extract_main_text(html_text)

    assert "这是主要正文" in text
    assert "第二段内容" in text
    assert "alert" not in text
    assert "导航" not in text


def test_parse_duckduckgo_results() -> None:
    html_text = """
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A <b>Result</b></a>
    <a class="result__snippet">Snippet <b>one</b></a>
    """
    results = parse_duckduckgo_results(html_text, 3)

    assert len(results) == 1
    assert results[0].url == "https://example.com/a"
    assert results[0].title == "A Result"
    assert results[0].snippet == "Snippet one"


def test_playwright_search_link_cleaning() -> None:
    raw_links = [
        {"title": "Images", "url": "https://www.bing.com/images/search?q=agent", "snippet": "noise"},
        {"title": "", "url": "https://example.com/empty-title", "snippet": "noise"},
        {"title": "LangGraph Docs", "url": "https://langchain-ai.github.io/langgraph/", "snippet": "LangGraph docs"},
        {"title": "LangGraph Docs Duplicate", "url": "https://langchain-ai.github.io/langgraph/#overview", "snippet": "duplicate"},
        {
            "title": "AutoGen",
            "url": "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fmicrosoft.github.io%2Fautogen%2F",
            "snippet": "AutoGen docs",
        },
        {"title": "Icon", "url": "https://example.com/logo.png", "snippet": "image"},
    ]

    results = parse_playwright_search_links(raw_links, 5, "bing")

    assert [item.url for item in results] == [
        "https://langchain-ai.github.io/langgraph/",
        "https://microsoft.github.io/autogen/",
    ]
    assert results[0].title == "LangGraph Docs"


def test_browser_search_url_and_normalize() -> None:
    assert make_browser_search_url("LangGraph AutoGen", "bing").startswith("https://www.bing.com/search?")
    assert make_browser_search_url("LangGraph AutoGen", "duckduckgo").startswith("https://duckduckgo.com/?")
    assert normalize_search_result_url("//example.com/a#top") == "https://example.com/a"


def test_playwright_missing_dependency_has_clear_error() -> None:
    if importlib.util.find_spec("playwright") is not None:
        return
    previous_provider = os.environ.get("SEARCH_PROVIDER")
    os.environ["SEARCH_PROVIDER"] = "playwright"
    try:
        client = WebSearchClient()
        try:
            client.search("LangGraph AutoGen CrewAI 区别", 1)
        except RuntimeError as exc:
            assert "未安装 Playwright" in str(exc)
        else:
            raise AssertionError("Expected missing Playwright RuntimeError.")
    finally:
        if previous_provider is None:
            os.environ.pop("SEARCH_PROVIDER", None)
        else:
            os.environ["SEARCH_PROVIDER"] = previous_provider


def test_web_research_agent_with_fake_client(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    agent = WebResearchAgent(index, search_client=FakeSearchClient(), reports_dir=base / "reports")

    result = agent.research("DeskPilot 第三阶段 网页调研", max_results=2)

    assert "DeskPilot 第三阶段 网页调研" in result.report
    assert result.artifact_path.endswith(".md")
    assert Path(result.artifact_path).exists()
    assert result.sources
    assert result.evidences
    assert "## 参考来源" in result.report
    assert "https://example.com/memory" in result.report
    assert "访问日期:" in result.report
    assert index.stats()["documents"] == 2
    assert any(step.name == "save_report" for step in result.steps)


def test_document_agent_research_integration(base: Path) -> None:
    index = DocumentIndex(base / "index" / "index.json")
    agent = DocumentQAAgent(index)
    agent.session_store = SessionStore(base / "sessions")
    agent.memory_store = MemoryStore(base / "indexes" / "memory.sqlite", base / "workspace", vector_provider="sqlite")
    agent.memory_extractor = MemoryExtractor()
    agent.memory_compactor = MemoryCompactor(agent.session_store, max_messages=100, max_chars=100000)
    agent.context_assembler = ContextBuilder()
    agent.web_research_agent = WebResearchAgent(index, search_client=FakeSearchClient(), reports_dir=base / "reports")

    session = agent.session_store.create_session("网页调研测试")
    result = agent.research("DeskPilot 第三阶段 网页调研", session_id=session.session_id, max_results=2)

    assert result.session_id == session.session_id
    assert Path(result.artifact_path).exists()
    assert agent.session_store.message_count(session.session_id) == 2
    memories = agent.memory_store.search("网页调研报告在哪里", session_id=session.session_id, top_k=5)
    assert any("网页调研报告" in item.content for item in memories)


def test_research_topic_and_filename_ignore_delivery_instructions(base: Path) -> None:
    raw = (
        "在网上搜索一些agentic RL相关的最新论文，整理成一个文档，"
        "作为附件发送给recipient@example.com"
    )
    topic = normalize_research_topic(raw)
    assert topic == "agentic RL相关的最新论文"
    assert normalize_research_topic("调研方法与问卷设计") == "调研方法与问卷设计"

    agent = WebResearchAgent(
        DocumentIndex(base / "short_name_index.json"),
        search_client=FakeSearchClient(),
        reports_dir=base / "short_reports",
    )
    path = agent._save_report(raw, "# Agentic RL 最新研究进展调研报告\n\n正文")
    assert path.name.endswith("_Agentic_RL_最新研究进展.md")
    assert "在网上搜索" not in path.name
    assert "作为附件" not in path.name
    assert len(path.name) <= 64
    assert extract_report_title("# 短标题\n\n正文") == "短标题"
    assert len(make_report_slug("这是一个非常非常非常非常非常非常长的报告标题")) <= 28


def test_llm_reference_section_is_replaced_and_invalid_citations_removed(base: Path) -> None:
    class FakeReportClient:
        def chat(self, messages, temperature=0.2):
            return (
                "# 测试报告\n\n有效判断 [1]，越界判断 [9]。\n\n"
                "## 参考来源\n\n1. 模型虚构的来源"
            )

    agent = WebResearchAgent(
        DocumentIndex(base / "citation_index.json"),
        search_client=FakeSearchClient(),
        reports_dir=base / "citation_reports",
    )
    agent.client = FakeReportClient()
    report, used_llm = agent._generate_report(
        "Agentic RL",
        [SearchResult("Real paper", "https://example.com/paper", "abstract")],
        [],
    )
    assert used_llm is True
    assert "有效判断 [1]" in report
    assert "[9]" not in report
    assert "模型虚构的来源" not in report
    assert report.count("## 参考来源") == 1
    assert "[Real paper](https://example.com/paper)" in report
    assert "URL: https://example.com/paper" in report


def main() -> None:
    force_local_fallback()
    assert clean_html("A <b>bold</b> text") == "A bold text"
    with tempfile.TemporaryDirectory(prefix="deskpilot_web_research_test_") as tmp:
        base = Path(tmp)
        test_extract_main_text_removes_noise()
        test_parse_duckduckgo_results()
        test_playwright_search_link_cleaning()
        test_browser_search_url_and_normalize()
        test_playwright_missing_dependency_has_clear_error()
        test_web_research_agent_with_fake_client(base)
        test_document_agent_research_integration(base)
        test_research_topic_and_filename_ignore_delivery_instructions(base)
        test_llm_reference_section_is_replaced_and_invalid_citations_removed(base)
    print("Web research tests passed.")


if __name__ == "__main__":
    main()
