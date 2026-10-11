from __future__ import annotations

from pathlib import Path

import pytest

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.decision.jev import JevClient, JevConfig
from deskpilot.intent.router import IntentRouter
from deskpilot.rag.retrieval import RetrievalFilters, RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


def response(questions, *, route="direct_answer", relevant=1.0):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "choice":
            answers[key] = {"type": "choice", "choice": route, "confidence": 0.99,
                            "probabilities": {route: 0.99}}
        else:
            answers[key] = {"type": "noul", "noul": relevant if key.startswith(("relevant_", "support_")) else 0.01}
    return {"model": "typesafe/jev-1.13", "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 4}}


class FakeAnswerClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return "RAG 是检索增强生成。"


def test_jev_transport_uses_openrouter_typed_payload_without_logging_text(monkeypatch) -> None:
    client = JevClient(JevConfig(mode="shadow", api_key="fake"))
    sent = []
    def post(payload):
        sent.append(payload)
        return response(payload["questions"])
    monkeypatch.setattr(client, "_post", post)
    answers = client.judge("private input", {"flag": {"type": "noul", "instructions": "true?"}})
    assert answers["flag"]["noul"] == 0.01
    assert sent[0]["model"] == "typesafe/jev-1.13"
    assert sent[0]["state"] == "private input"
    assert client.token_usage["total_tokens"] == 16
    assert "private input" not in str(client.usage_events)


def test_jev_route_shadow_and_gated_shortcut(monkeypatch) -> None:
    answer_client = FakeAnswerClient()
    jev = JevClient(JevConfig(mode="shadow", api_key="fake"))
    monkeypatch.setattr(jev, "_post", lambda payload: response(payload["questions"]))
    router = IntentRouter(answer_client, decision_client=jev)
    assert router._jev_direct_gate("什么是 RAG？", []) is None
    assert answer_client.calls == 0
    jev.config = JevConfig(mode="gated", api_key="fake")
    decision = router._jev_direct_gate("什么是 RAG？", [])
    assert decision and decision.mode == "direct_answer" and not decision.direct_response
    assert answer_client.calls == 0
    assert router.last_jev_decision["used_shortcut"]


def test_jev_direct_shortcut_tolerates_small_web_uncertainty(monkeypatch) -> None:
    answer_client = FakeAnswerClient()
    jev = JevClient(JevConfig(mode="gated", api_key="fake"))
    def post(payload):
        result = response(payload["questions"])
        result["answers"]["needs_fresh_web"]["noul"] = 0.11
        return result
    monkeypatch.setattr(jev, "_post", post)
    router = IntentRouter(answer_client, decision_client=jev)
    decision = router.route("什么是 RAG？", [])
    assert decision.mode == "direct_answer"
    assert router.last_jev_decision["used_shortcut"] is True
    assert answer_client.calls == 0


def test_jev_route_does_not_shortcut_actions_or_indexed_questions(monkeypatch) -> None:
    jev = JevClient(JevConfig(mode="gated", api_key="fake"))
    monkeypatch.setattr(jev, "_post", lambda payload: response(payload["questions"], route="multi_step"))
    router = IntentRouter(FakeAnswerClient(), decision_client=jev)
    assert router._jev_direct_gate("写报告并发邮件", []) is None
    monkeypatch.setattr(jev, "_post", lambda payload: response(payload["questions"]))
    assert router._jev_direct_gate("总结已索引文档", [{"doc_id": "doc"}]) is None


def test_jev_bad_response_falls_back_without_tool_execution(monkeypatch) -> None:
    jev = JevClient(JevConfig(mode="gated", api_key="fake"))
    monkeypatch.setattr(jev, "_post", lambda payload: {"answers": {}})
    router = IntentRouter(FakeAnswerClient(), decision_client=jev)
    assert router._jev_direct_gate("什么是 RAG？", []) is None
    assert router.last_jev_decision["fallback"] == "ValueError"


def test_jev_router_skips_mail_content_before_network(monkeypatch) -> None:
    jev = JevClient(JevConfig(mode="gated", api_key="fake"))
    monkeypatch.setattr(jev, "_post", lambda payload: pytest.fail("private mail input was sent"))
    router = IntentRouter(FakeAnswerClient(), decision_client=jev)
    assert router._jev_direct_gate("写邮件给自己，正文是私人内容", []) is None
    assert router.last_jev_decision["skipped"] == "sensitive_input"


def test_jev_document_text_requires_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.core.models import Document
    index = DocumentIndex(tmp_path / "index.json")
    index.client.embed = lambda values: [local_hash_embedding(value) for value in values]
    index._configured_embedding_space = lambda: "local-hash-v1:384"
    index._embedding_space = "local-hash-v1:384"
    index.add_document(Document("doc", str(tmp_path / "note.md"), "note.md", "md", "private evidence"))
    jev = JevClient(JevConfig(mode="gated", api_key="fake"))
    monkeypatch.setattr(jev, "_post", lambda payload: pytest.fail("document text was sent"))
    index.hybrid_retriever.decision_client = jev
    index.search_hybrid(RetrievalRequest("private evidence", ("private evidence",), top_k=2))
    events = index.last_trace.to_dict()["events"]
    assert any(event["stage"] == "jev_evidence" and event["detail"].get("skipped") for event in events)


def test_jev_rag_filters_only_when_gated_and_preserves_explicit_document(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.core.models import Document
    index = DocumentIndex(tmp_path / "index.json")
    index.client.embed = lambda values: [local_hash_embedding(value) for value in values]
    index._configured_embedding_space = lambda: "local-hash-v1:384"
    index._embedding_space = "local-hash-v1:384"
    document = Document(doc_id="doc", title="notes.md", path=str(tmp_path / "notes.md"),
                        source_type="md", content="NEBULA-712 needs a new lease after failover.")
    index.add_document(document)
    jev = JevClient(JevConfig(mode="shadow", api_key="fake", passage_threshold=0.5, allow_document_text=True))
    monkeypatch.setattr(jev, "_post", lambda payload: response(payload["questions"], relevant=0.01))
    index.hybrid_retriever.decision_client = jev
    request = RetrievalRequest("NEBULA-712", ("NEBULA-712",), top_k=2)
    assert index.search_hybrid(request)
    assert any(event["stage"] == "jev_evidence" for event in index.last_trace.to_dict()["events"])
    jev.config = JevConfig(mode="gated", api_key="fake", passage_threshold=0.5, allow_document_text=True)
    assert not index.search_hybrid(request)
    protected = RetrievalRequest("NEBULA-712", ("NEBULA-712",), top_k=2,
                                 filters=RetrievalFilters(doc_ids=("doc",)))
    assert index.search_hybrid(protected)
