from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.context import ContextBuilder
from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import AgentStep, Chunk, Document, Evidence
from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.memory.memory_models import MemoryItem, SessionMessage
from deskpilot.rag.query_optimizer import QueryOptimizer
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.vector_index import DocumentIndex


def test_mixed_language_workflow_query_is_expanded_for_english_documents() -> None:
    rewritten = QueryOptimizer().rewrite("recognize anything 的工作流程是什么样的")
    assert rewritten.method == "deterministic_multilingual"
    assert "workflow" in rewritten.rewritten
    assert "architecture" in rewritten.rewritten


def test_explicit_document_title_limits_retrieval_to_that_document(base: Path) -> None:
    index = DocumentIndex(base / "title_filter.json")

    class FakeEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index.client = FakeEmbeddingClient()  # type: ignore[assignment]
    index.documents = {
        "ram": Document("ram", "ram.pdf", "recognize_anything.pdf", "pdf", "", {}),
        "blip": Document("blip", "blip.pdf", "blip.pdf", "pdf", "", {}),
    }
    index.chunks = {
        "ram-1": Chunk("ram-1", "ram", "RAM model architecture workflow", "recognize_anything.pdf#page-3", 1,
                       embedding=[1.0, 0.0]),
        "blip-1": Chunk("blip-1", "blip", "recognize image model workflow", "blip.pdf#page-1", 1,
                        embedding=[1.0, 0.0]),
    }

    results = index.search("recognize anything workflow architecture", top_k=5)
    assert results
    assert all(item.doc_id == "ram" for item in results)


def test_short_and_hyphenated_titles_are_resolved_as_distinct_documents(base: Path) -> None:
    index = DocumentIndex(base / "comparison_titles.json")
    index.documents = {
        "blip": Document("blip", "blip.pdf", "blip.pdf", "pdf", "", {}),
        "blip2": Document("blip2", "blip-2.pdf", "blip-2.pdf", "pdf", "", {}),
        "fixture": Document("fixture", "multimodal_blip.md", "multimodal_blip.md", "md", "", {}),
    }

    matched = index.matching_title_doc_ids("BLIP 和 BLIP2 的区别是什么？")

    assert matched == {"blip", "blip2"}


def test_named_document_comparison_uses_balanced_evidence(base: Path) -> None:
    index = DocumentIndex(base / "balanced_comparison.json")

    class FakeClient:
        config = type("Config", (), {"llm_api_key": "fake", "llm_model": "fake"})()
        last_error = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def chat(self, messages, temperature=0.2, max_tokens=None):
            return "BLIP 使用统一的编码器-解码器结构与合成字幕过滤机制。[1]\n\nBLIP-2 使用冻结模型和 Q-Former 进行连接。[3]"

    fake = FakeClient()
    index.client = fake  # type: ignore[assignment]
    index.documents = {
        "blip": Document("blip", "blip.pdf", "blip.pdf", "pdf", "", {}),
        "blip2": Document("blip2", "blip-2.pdf", "blip-2.pdf", "pdf", "", {}),
    }
    index.chunks = {
        "b1": Chunk("b1", "blip", "BLIP unified encoder decoder architecture and caption filtering.",
                    "blip.pdf#page-2", 1, embedding=[1.0, 0.0]),
        "b2": Chunk("b2", "blip", "BLIP trains image text matching and language modeling objectives.",
                    "blip.pdf#page-3", 2, embedding=[0.9, 0.1]),
        "b21": Chunk("b21", "blip2", "BLIP-2 uses frozen image encoders and frozen language models.",
                     "blip-2.pdf#page-1", 1, embedding=[1.0, 0.0]),
        "b22": Chunk("b22", "blip2", "BLIP-2 bridges modalities through a lightweight Q-Former.",
                     "blip-2.pdf#page-3", 2, embedding=[0.9, 0.1]),
    }
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.index = index
    agent.client = fake
    agent.query_optimizer = QueryOptimizer()
    agent.query_analyzer = QueryAnalyzer()
    agent.context_assembler = ContextBuilder(model="fake")
    agent._finalize_answer = lambda **kwargs: kwargs  # type: ignore[method-assign]
    context = agent.context_assembler.assemble("BLIP 和 BLIP2 的区别是什么？", "", "", [], [])
    steps: list[AgentStep] = []

    result = agent._answer_rag_request(
        "BLIP 和 BLIP2 的区别是什么？", "BLIP architecture Q-Former",
        5, steps, "s", "u", context,
    )

    assert {item.doc_id for item in result["evidences"]} == {"blip", "blip2"}
    retrieval = next(step for step in steps if step.name == "retrieve_evidence")
    assert "strategy=hybrid_rrf" in retrieval.detail
    rewrite = next(step for step in steps if step.name == "rewrite_query")
    assert "training objectives" in rewrite.detail
    assert "Q-Former" not in rewrite.detail


def test_reference_section_does_not_displace_document_body(base: Path) -> None:
    index = DocumentIndex(base / "reference_filter.json")

    class FakeEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index.client = FakeEmbeddingClient()  # type: ignore[assignment]
    index.documents = {"paper": Document("paper", "paper.pdf", "paper.pdf", "pdf", "", {})}
    index.chunks = {
        "body-1": Chunk("body-1", "paper", "model architecture", "paper.pdf#page-1", 1, embedding=[1.0, 0.0]),
        "body-2": Chunk("body-2", "paper", "training workflow", "paper.pdf#page-2", 2, embedding=[1.0, 0.0]),
        "refs": Chunk("refs", "paper", "References [1] model architecture workflow", "paper.pdf#page-3", 3,
                      embedding=[1.0, 0.0]),
    }

    results = index.search("paper model architecture workflow", top_k=5)
    assert results
    assert all(item.chunk_id != "refs" for item in results)


def test_route_hint_ignores_nested_reference_titles_and_acronym_substrings(base: Path) -> None:
    index = DocumentIndex(base / "reference_hint.json")
    index.documents = {
        "paper": Document("paper", "paper.md", "paper.md", "md", "", {}),
        "other": Document("other", "other.pdf", "other.pdf", "pdf", "", {}),
    }
    index.chunks = {
        "ref": Chunk(
            "ref", "paper", "[1] Agentic RL: unrelated reference title.",
            "paper.md#Paper/References", 3,
            metadata={"heading_path": ["Paper", "References"]}, embedding=[1.0, 0.0],
        ),
        "body": Chunk(
            "body", "other", "The world model reads a URL for ordinary vision-language training.",
            "other.pdf#page-1", 1, embedding=[1.0, 0.0],
        ),
    }

    assert index.retrieval_hint("什么是 Agentic RL？") == []


def test_route_hint_does_not_treat_incidental_short_acronym_as_topic(base: Path) -> None:
    index = DocumentIndex(base / "acronym_hint.json")
    index.documents = {
        "survey": Document("survey", "mllm_survey.pdf", "mllm_survey.pdf", "pdf", "", {}),
    }
    index.chunks = {
        "survey-1": Chunk(
            "survey-1", "survey",
            "A survey of multimodal models mentions GPT-4V before discussing the general MLLM architecture.",
            "mllm_survey.pdf#page-1", 1, embedding=[1.0, 0.0],
        ),
    }

    assert index.retrieval_hint("GPT 的工作流程是什么样的？") == []


def test_direct_answer_prompt_forbids_unverified_project_implementation_claims() -> None:
    class FakeClient:
        config = type("Config", (), {"llm_api_key": "fake"})()
        last_error = ""

        def __init__(self) -> None:
            self.messages = []

        def chat(self, messages, temperature=0.2, max_tokens=None):
            self.messages = messages
            return "Agentic RL 是将智能体决策过程与强化学习结合的方法。"

    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.client = FakeClient()
    context = ContextBuilder(model="fake").assemble("什么是 Agentic RL？", "", "", [], [])

    answer = agent._direct_answer("什么是 Agentic RL？", context)
    system_prompt = agent.client.messages[0]["content"]

    assert "Agentic RL" in answer
    assert "只回答用户询问的主题" in system_prompt
    assert "不得声称 DeskPilot 使用了" in system_prompt


def test_evidence_context_excludes_fact_memory_and_assistant_history() -> None:
    builder = ContextBuilder(model="fake")
    memories = [
        MemoryItem("fact-1", "workspace", "fact", "RAM 使用 Q-Former。", confidence=0.95, retrieval_score=0.9),
        MemoryItem("pref-1", "user", "preference", "用户偏好中文回答。", confidence=0.9, retrieval_score=0.9),
    ]
    context = builder.assemble(
        "RAM 的架构是什么？",
        "工作区规则",
        "旧摘要声称 RAM 使用 BERT。",
        [
            SessionMessage("u1", "s1", "user", "请简洁回答。"),
            SessionMessage("a1", "s1", "assistant", "RAM 使用 Q-Former。"),
        ],
        memories,
    )
    grounded = builder.for_role(
        context,
        "answer",
        evidences=[Evidence("e1", "doc", "ram.pdf#page-3", "RAM 使用图像编码器和识别解码器。", 0.9)],
        evidence_grounded=True,
    )

    assert "RAM 使用 Q-Former" not in grounded.text
    assert "旧摘要声称" not in grounded.text
    assert "用户偏好中文回答" in grounded.text
    assert "ram.pdf#page-3" in grounded.text


def test_unverified_assistant_fact_cannot_become_workspace_fact() -> None:
    extractor = MemoryExtractor()
    extractor._extract_with_llm = lambda *args, **kwargs: [{  # type: ignore[method-assign]
        "scope": "workspace",
        "memory_type": "fact",
        "content": "RAM 使用 Q-Former。",
        "confidence": 0.95,
        "tags": ["ram"],
        "status": "active",
    }]

    item = extractor.extract("RAM 是什么？", "RAM 使用 Q-Former。", "s1", ["u1", "a1"])[0]

    assert item["scope"] == "session"
    assert item["status"] == "pending"
    assert item["confidence"] <= 0.45
    assert "unverified" in item["tags"]


def test_citation_validator_rejects_memory_source_and_uncovered_claims() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    evidences = [Evidence("e1", "doc", "ram.pdf#page-3", "RAM 由三个模块构成。", 0.9)]
    invalid = "根据记忆，RAM 使用 Q-Former。[来源: Retrieved Memories]\n\n1. 使用 BERT 标签编码。"
    valid = "RAM 包含图像编码器、图像标签识别解码器和文本生成编码器-解码器。[1]"

    ok, reasons = agent._validate_grounded_answer(invalid, evidences)
    assert not ok
    assert any("Retrieved Memories" in reason for reason in reasons)
    assert agent._validate_grounded_answer(valid, evidences)[0]
    assert "[1] ram.pdf#page-3" in agent._append_source_list(valid, evidences)


def test_citation_validator_ignores_markdown_headings_and_lead_ins() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    evidences = [Evidence("e1", "doc", "ram.pdf#page-3", "RAM has three modules.", 0.9)]
    answer = (
        "根据提供的证据，Recognize Anything Model (RAM) 的工作流程和架构主要包含以下环节：\n\n"
        "**1. 整体架构组成**\n"
        "RAM 由图像编码器、识别解码器和文本生成模块组成 [1]。"
    )
    assert agent._validate_grounded_answer(answer, evidences)[0]


def test_offline_excerpt_uses_relevant_sentence_instead_of_chunk_prefix() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.query_optimizer = QueryOptimizer()
    agent.query_analyzer = QueryAnalyzer()
    text = (
        "This unrelated introduction describes training cost and resources. "
        "2.1. Model Architecture. RAM consists of an image encoder and an image-tag recognition decoder. "
        "The final paragraph discusses limitations."
    )
    excerpt = agent._relevant_evidence_excerpt("RAM 的工作流程和架构是什么？", text)
    assert "image encoder" in excerpt
    assert "unrelated introduction" not in excerpt


def test_offline_fallback_drops_low_relevance_evidence() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.query_optimizer = QueryOptimizer()
    evidences = [
        Evidence("e1", "ram", "ram.pdf#page-3", "Model Architecture uses an image encoder.", 0.9),
        Evidence("e2", "ram", "ram.pdf#page-5", "Unrelated benchmark table values.", 0.2),
    ]
    answer = agent._fallback_answer("RAM 的工作流程是什么？", evidences)
    assert "[1]" in answer
    assert "[2]" not in answer
    assert "Unrelated benchmark" not in answer


def test_evidence_guard_rejects_unrelated_named_entity_results() -> None:
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    unrelated = [Evidence(
        "blip", "blip", "blip-2.pdf#page-13",
        "Q-Former extracts image features for a frozen language model.", 0.2,
    )]
    covered, entities = agent._evidence_covers_query_entities("qwenVL 的工作流程是什么？", unrelated)
    assert not covered
    assert "qwenVL" in entities

    related = [Evidence(
        "qwen", "qwen", "qwen-vl.pdf#page-3",
        "Qwen-VL contains a visual receptor, adapter and language model.", 0.8,
    )]
    assert agent._evidence_covers_query_entities("qwenVL 的工作流程是什么？", related)[0]


def test_rag_answer_repairs_invalid_citations_and_records_steps(base: Path) -> None:
    index = DocumentIndex(base / "rag.json")

    class FakeClient:
        config = type("Config", (), {"llm_api_key": "fake", "llm_model": "fake"})()
        last_error = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def chat(self, messages, temperature=0.2, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return "RAM 使用 Q-Former。[来源: Retrieved Memories]"
            return "RAM 使用图像编码器和图像标签识别解码器处理图像标签。[1]"

    fake = FakeClient()
    index.client = fake  # type: ignore[assignment]
    index.documents = {"ram": Document("ram", "ram.pdf", "ram.pdf", "pdf", "", {})}
    index.chunks = {
        "e1": Chunk(
            "e1",
            "ram",
            "RAM consists of an image encoder and an image-tag recognition decoder.",
            "recognize_anything.pdf#page-3",
            1,
            embedding=[1.0, 0.0],
        )
    }
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.index = index
    agent.client = fake
    agent.query_optimizer = QueryOptimizer()
    agent.query_analyzer = QueryAnalyzer()
    agent.context_assembler = ContextBuilder(model="fake")
    agent._finalize_answer = lambda **kwargs: kwargs  # type: ignore[method-assign]
    context = agent.context_assembler.assemble("RAM 的工作流程是什么？", "", "", [], [])
    steps: list[AgentStep] = []

    result = agent._answer_rag_request("RAM 的工作流程是什么？", "RAM 工作流程", 3, steps, "s", "u", context)
    names = [step.name for step in steps]

    assert "retrieve_evidence" in names
    assert "evidence_context" in names
    assert "validate_citations" in names
    assert "repair_citations" in names
    assert "attach_citations" in names
    assert "Retrieved Memories" not in result["answer"]
    assert "recognize_anything.pdf#page-3" in result["answer"]


def main() -> None:
    test_mixed_language_workflow_query_is_expanded_for_english_documents()
    test_evidence_context_excludes_fact_memory_and_assistant_history()
    test_unverified_assistant_fact_cannot_become_workspace_fact()
    test_citation_validator_rejects_memory_source_and_uncovered_claims()
    test_citation_validator_ignores_markdown_headings_and_lead_ins()
    test_offline_excerpt_uses_relevant_sentence_instead_of_chunk_prefix()
    test_offline_fallback_drops_low_relevance_evidence()
    test_evidence_guard_rejects_unrelated_named_entity_results()
    with tempfile.TemporaryDirectory(prefix="deskpilot_rag_grounding_") as raw:
        base = Path(raw)
        test_explicit_document_title_limits_retrieval_to_that_document(base)
        test_short_and_hyphenated_titles_are_resolved_as_distinct_documents(base)
        test_named_document_comparison_uses_balanced_evidence(base)
        test_reference_section_does_not_displace_document_body(base)
        test_route_hint_ignores_nested_reference_titles_and_acronym_substrings(base)
        test_rag_answer_repairs_invalid_citations_and_records_steps(base)
        test_route_hint_does_not_treat_incidental_short_acronym_as_topic(base)
    test_direct_answer_prompt_forbids_unverified_project_implementation_claims()
    print("RAG grounding tests passed: 16")


if __name__ == "__main__":
    main()
