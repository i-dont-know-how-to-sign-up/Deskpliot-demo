from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.context import AdaptiveBudgetManager, ContextBuilder, UsageCostTracker
from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import AgentStep
from deskpilot.core.models import Chunk, Document, Evidence
from deskpilot.intent import IntentRouter
from deskpilot.memory.memory_models import SessionMessage
from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
from deskpilot.multi_agent.supervisor import SupervisorAgent
from deskpilot.rag.query_optimizer import QueryOptimizer
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.tool_registry import ToolParameter, ToolRegistry, ToolSpec


def configure_offline() -> None:
    for name in ("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"):
        os.environ[name] = ""
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"


def test_runtime_packets_and_role_budgets() -> None:
    builder = ContextBuilder(model="qwen3.7-plus")
    base = builder.assemble(
        "比较两个方案并给出结论",
        "工作区规则",
        "会话摘要",
        [SessionMessage("m1", "s", "user", "之前选择方案 A")],
        [],
    )
    evidence = Evidence("c1", "d1", "方案文档.md", "方案 B 的成本更低。", 0.91)
    answer = builder.for_role(
        base,
        "answer",
        complexity=8,
        planner_state={"route": "multi_agent", "score": 8},
        tool_outputs=[{"ok": True, "result": "完成"}],
        evidences=[evidence],
    )

    kinds = {packet.kind for packet in answer.packets}
    assert {"task", "planner_state", "tool_output", "evidence"} <= kinds
    assert answer.role == "answer"
    assert answer.output_budget > 0
    assert answer.estimated_tokens <= answer.input_budget
    assert answer.quality["tokenizer_provider"] in {"heuristic", "tiktoken", "tiktoken_approx", "transformers"}


def test_adaptive_budget_grows_with_complexity() -> None:
    manager = AdaptiveBudgetManager()
    simple = manager.allocate("answer", 1)
    complex_task = manager.allocate("answer", 9)
    assert complex_task.input_tokens > simple.input_tokens
    assert complex_task.output_tokens > simple.output_tokens
    assert manager.allocate("router", 5).input_tokens < manager.allocate("answer", 5).input_tokens


def test_tool_catalog_uses_progressive_disclosure() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "demo.write", "demo", "Write demo content.",
        [ToolParameter("content", "string", True, "Very long internal slot description", "secret-default")],
        lambda content: content,
    ))
    summary = registry.list_tool_summaries()[0]
    full = registry.get_tool_spec("demo.write").to_dict()  # type: ignore[union-attr]
    assert "parameters" not in summary
    assert summary["slots"][0] == {"name": "content", "type": "string", "required": True}
    assert full["parameters"][0]["default"] == "secret-default"


def test_subagent_receives_only_direct_dependency_outputs() -> None:
    observed: dict[str, dict] = {}

    def capture(name: str):
        def handler(values: dict) -> dict:
            observed[name] = dict(values)
            return {name: True}
        return handler

    plan = TaskPlan("isolation", "multi_agent", [
        TaskPlanStep("a", "knowledge", "a", handler=lambda values: {"evidence_a": "A"}),
        TaskPlanStep("b", "knowledge", "b", handler=lambda values: {"evidence_b": "B"}),
        TaskPlanStep("c", "communication", "c", depends_on=["a"], handler=capture("c")),
    ])
    result = SupervisorAgent().execute(plan, initial_values={"task": "demo"})
    assert result.status == "success"
    assert observed["c"]["task"] == "demo"
    assert observed["c"]["evidence_a"] == "A"
    assert "evidence_b" not in observed["c"]


def test_query_rewrite_and_mmr_diversity(base: Path) -> None:
    rewritten = QueryOptimizer().rewrite("请帮我搜索 上下文工程 token预算 并给出回答")
    assert rewritten.method == "deterministic"
    assert not rewritten.rewritten.startswith("请帮我搜索")

    index = DocumentIndex(base / "index.json")

    class FakeEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index.client = FakeEmbeddingClient()  # type: ignore[assignment]
    index.chunks = {
        "a": Chunk("a", "doc_a", "token预算用于限制上下文成本", "A.md", 0, embedding=[1.0, 0.0]),
        "b": Chunk("b", "doc_a", "token预算用于限制上下文成本", "A.md", 1, embedding=[1.0, 0.0]),
        "c": Chunk("c", "doc_b", "上下文成本还需要监控利用率", "B.md", 0, embedding=[0.6, 0.8]),
    }
    # 本用例专门看 P0 MMR；P1 默认路径在 RRF 后不再提前执行强 MMR。
    results = index._legacy_search("上下文 token预算 成本", top_k=2)
    assert len(results) == 2
    assert len({item.doc_id for item in results}) == 2


def test_usage_tracker_reports_percentiles(base: Path) -> None:
    tracker = UsageCostTracker(base / "usage.jsonl")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}
    for total in (100, 200, 500):
        before = tracker.snapshot(usage)
        usage = {
            "prompt_tokens": usage["prompt_tokens"] + total - 20,
            "completion_tokens": usage["completion_tokens"] + 20,
            "total_tokens": usage["total_tokens"] + total,
            "reported_calls": usage["reported_calls"] + 1,
        }
        tracker.record_delta(before, usage, "rag", "demo-model")
    stats = tracker.stats("rag")
    assert stats == {"samples": 3, "p50_total_tokens": 200, "p95_total_tokens": 500}


def test_agent_rag_path_uses_evidence_context(base: Path) -> None:
    index = DocumentIndex(base / "rag_index.json")

    class FakeClient:
        config = type("Config", (), {"llm_api_key": "", "llm_model": "fake"})()
        last_error = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def chat(self, messages, temperature=0.2, max_tokens=None):
            return ""

    index.client = FakeClient()  # type: ignore[assignment]
    index.documents = {"doc": Document("doc", "context.md", "context.md", "md", "", {})}
    index.chunks = {
        "e1": Chunk("e1", "doc", "P3 使用自适应预算控制成本。", "context.md", 0, embedding=[1.0, 0.0])
    }
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.index = index
    agent.client = FakeClient()
    agent.query_optimizer = QueryOptimizer()
    agent.query_analyzer = QueryAnalyzer()
    agent.context_assembler = ContextBuilder(model="fake")
    agent._finalize_answer = lambda **kwargs: kwargs  # type: ignore[method-assign]
    context = agent.context_assembler.assemble("P3 如何控制成本", "规则", "", [], [])
    steps: list[AgentStep] = []
    result = agent._answer_rag_request("P3 如何控制成本", "P3 成本", 3, steps, "s", "m", context)

    assert any(step.name == "rewrite_query" for step in steps)
    assert any(step.name == "evidence_context" for step in steps)
    assert "### Evidence" in result["memory_context"].text
    assert result["evidences"][0].source_label == "context.md"


def main() -> None:
    configure_offline()
    test_runtime_packets_and_role_budgets()
    test_adaptive_budget_grows_with_complexity()
    test_tool_catalog_uses_progressive_disclosure()
    test_subagent_receives_only_direct_dependency_outputs()
    with tempfile.TemporaryDirectory(prefix="deskpilot_context_p23_") as raw:
        base = Path(raw)
        test_query_rewrite_and_mmr_diversity(base)
        test_usage_tracker_reports_percentiles(base)
        test_agent_rag_path_uses_evidence_context(base)
    print("Context engineering P2/P3 tests passed: 7")


if __name__ == "__main__":
    main()
