from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import AgentStep, Chunk, Document
from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.memory.policy import MemoryPolicy
from deskpilot.multimodal.providers.vision_llm import VisionLLMProvider
from deskpilot.multimodal.service import MultimodalService
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.vector_index import DocumentIndex


def test_self_report_in_web_request_is_user_fact_without_extra_llm(monkeypatch) -> None:
    question = "我是一个agent开发工程师，帮我在网上搜索一下这个岗位需要的技术栈有哪些"
    assert MemoryPolicy().should_extract(
        question, grounded=False, steps=[AgentStep("web_search", "success", "ok")],
    ) == (True, "explicit_user_state")
    extractor = MemoryExtractor()
    monkeypatch.setattr(extractor.client, "chat", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Self-report should not need another LLM call")
    ))
    items = extractor.extract(question, "网页搜索结果", "session", ["user", "assistant"])
    assert len(items) == 1
    assert items[0]["scope"] == "user" and items[0]["memory_type"] == "fact"
    assert items[0]["status"] == "active"
    assert "agent开发工程师" in items[0]["content"]


def test_blip2_title_and_pdf_reference_boundary(tmp_path: Path) -> None:
    index = DocumentIndex(tmp_path / "index.json")
    index.documents = {
        "blip": Document("blip", "blip.pdf", "blip.pdf", "pdf", ""),
        "blip2": Document("blip2", "blip-2.pdf", "blip-2.pdf", "pdf", ""),
    }
    index.chunks = {
        "body": Chunk("body", "blip2", "Bootstrap Vision-Language Representation Learning", "blip-2.pdf#page-3", 14),
        "refs": Chunk("refs", "blip2", "BLIP-2 title References Agrawal, H., Anderson, P.", "blip-2.pdf#page-9", 42),
        "appendix": Chunk("appendix", "blip2", "Fine-tuning parameters", "blip-2.pdf#page-12", 58),
    }
    assert index.matching_title_doc_ids("blip-2是怎么预训练的") == {"blip2"}
    assert index.matching_title_doc_ids("比较blip和blip-2") == {"blip", "blip2"}
    assert index._reference_start_positions()["blip2"] == 42


def test_missing_named_document_does_not_answer_from_unrelated_papers(tmp_path: Path) -> None:
    index = DocumentIndex(tmp_path / "index.json")
    index.documents = {"blip": Document("blip", "blip.pdf", "blip.pdf", "pdf", "BLIP retrieval")}
    agent = DocumentQAAgent.__new__(DocumentQAAgent)
    agent.index = index
    agent.query_analyzer = QueryAnalyzer()
    agent._finalize_answer = lambda **kwargs: kwargs
    result = agent._answer_rag_request(
        "根据索引的知识库回答：多模态检索测试说明包含哪些内容", "多模态检索测试说明", 5,
        [], "session", "message", SimpleNamespace(text="", complexity=1),
    )
    assert "没有名为“多模态检索测试说明”的文档" in result["answer"]
    assert result["evidences"] == []
    assert any(step.name == "index_document_check" for step in result["steps"])


def test_vision_403_reports_provider_code_without_exposing_key(monkeypatch) -> None:
    monkeypatch.setenv("VISION_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("ALLOW_CLOUD_IMAGE_UPLOAD", "true")
    provider = VisionLLMProvider()
    error = urllib.error.HTTPError("https://example.invalid", 403, "Forbidden", {}, io.BytesIO())
    error.usage_response = {"error": {"code": "AccessDenied", "message": "secret body"}}
    monkeypatch.setattr(provider, "_read_answer", lambda *_args: (_ for _ in ()).throw(error))
    try:
        provider.answer("describe", [])
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected the provider to reject the request")
    assert "HTTP 403" in message and "AccessDenied" in message
    assert "synthetic-secret-key" not in message and "secret body" not in message


def test_vision_403_is_persisted_with_a_failed_step(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (16, 16), "red").save(image_path, format="PNG")
    agent = DocumentQAAgent(DocumentIndex(tmp_path / "index.json"))
    agent.multimodal = MultimodalService(tmp_path / "multimodal")
    agent.multimodal.vision_llm.allow_cloud = True
    asset = agent.multimodal.add_chat_attachment(image_path)
    monkeypatch.setattr(agent.visual_memory, "search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(agent.multimodal, "answer", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("视觉模型请求被拒绝（HTTP 403，模型 fixture）。请检查授权。")
    ))
    result = agent.answer_multimodal("描述图像", [asset.asset_id])
    assert "HTTP 403" in result.answer
    assert any(step.name == "vision_answer" and step.status == "failed" for step in result.steps)
    assert agent.session_store.read_messages(result.session_id)[-1].content == result.answer
    agent.task_service.close()


def test_email_subject_only_does_not_include_body(tmp_path: Path) -> None:
    agent = DocumentQAAgent(DocumentIndex(tmp_path / "index.json"))
    agent._finalize_answer = lambda **kwargs: kwargs
    output = {"count": 2, "messages": [
        {"subject": "First", "body": "private body one"},
        {"subject": "Second", "body": "private body two"},
    ]}
    result = agent._finalize_generic_tool_result(
        question="帮我查看最近的两封未读邮件的标题", tool_name="email.list_messages",
        tool_result=SimpleNamespace(ok=True, output=output, error=""), call_arguments={"limit": 2},
        steps=[], session_id="session", user_message_id="user",
        memory_context=agent.context_assembler.assemble("test", "", "", [], []),
    )
    assert "First" in result["answer"] and "Second" in result["answer"]
    assert "private body" not in result["answer"]
