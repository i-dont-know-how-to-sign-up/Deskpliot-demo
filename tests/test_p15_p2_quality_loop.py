from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from deskpilot.context import (
    ContextBuilder,
    context_precision_recall,
    lost_in_middle_metrics,
    summary_fact_consistency,
)
from deskpilot.core.api_clients import OpenAICompatibleClient, _partial_json_string_field
from deskpilot.core.config import load_config
from deskpilot.intent.router import IntentRouter
from deskpilot.rag.reranker import SentenceTransformersCrossEncoderReranker
from deskpilot.rag.web_research import SearchResult, WebCache, WebSearchClient
from eval.run_eval import aggregate_stage_usage


class FakeApiClient(OpenAICompatibleClient):
    def _post_json(self, url: str, payload: dict, api_key: str) -> dict:
        return {
            "choices": [{"message": {"content": "普通回答"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    def _post_stream_json(self, url: str, payload: dict, api_key: str) -> tuple[str, dict]:
        self._emit({"type": "token", "stage": "Answer", "text": "流式"})
        self._emit({"type": "token", "stage": "Answer", "text": "回答"})
        return "流式回答", {
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        }


def test_stage_usage_and_stream_events(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    client = FakeApiClient(load_config())
    events: list[dict] = []
    with client.event_stream(events.append), client.stage("Answer", stream=True):
        assert client.chat([{"role": "user", "content": "test"}]) == "流式回答"
    assert "".join(item.get("text", "") for item in events if item["type"] == "token") == "流式回答"
    assert client.token_usage["total_tokens"] == 17
    assert client.usage_events[0]["stage"] == "Answer"
    summary = aggregate_stage_usage([{"status": "passed", "usage_events": client.usage_events}])
    assert summary["Answer"]["calls"] == 1
    assert summary["Answer"]["total_tokens"] == 17


def test_router_can_answer_directly_in_one_call() -> None:
    class Client:
        calls = 0

        @contextmanager
        def stage(self, name: str, **kwargs):
            yield

        def chat(self, messages, temperature=0.0, max_tokens=None):
            self.calls += 1
            return json.dumps({
                "mode": "direct_answer",
                "reason": "simple",
                "direct_response": "RAG 是检索增强生成。",
                "confidence": 0.99,
                "knowledge_scope": "general",
            }, ensure_ascii=False)

    client = Client()
    decision = IntentRouter(client).route("什么是 RAG？", [])  # type: ignore[arg-type]
    assert client.calls == 1
    assert decision.mode == "direct_answer"
    assert decision.direct_response == "RAG 是检索增强生成。"


def test_partial_router_json_only_exposes_direct_response() -> None:
    value = '{"mode":"direct_answer","reason":"internal","direct_response":"RAG 是检索\\u589e强'
    assert _partial_json_string_field(value, "direct_response") == "RAG 是检索增强"
    assert "internal" not in _partial_json_string_field(value, "direct_response")


def test_context_p4_metrics_and_role_filtering() -> None:
    builder = ContextBuilder()
    context = builder.assemble(
        "解释 RAG",
        "项目使用邮件 SMTP，与当前问题无关。",
        "之前决定使用蓝色 UI。",
        [],
        [],
    )
    router = builder.for_role(context, "router")
    assert "SMTP" not in router.text
    assert router.quality["role_filtered_packets"] >= 1
    assert any(item["reason"] == "role_filtered" for item in router.attribution)
    assert context_precision_recall(["task", "noise"], ["task", "evidence"]) == {
        "selected": 2, "relevant": 2, "true_positive": 1,
        "context_precision": 0.5, "context_recall": 0.5,
    }
    consistency = summary_fact_consistency("RAG 使用检索。不存在量子邮箱。", ["RAG 使用检索增强生成。"])
    assert consistency["consistency"] < 1.0
    middle = lost_in_middle_metrics(["a", "important", "c"], ["important"])
    assert middle["middle_retained"] is True


def test_web_provider_fallback_and_cache(base: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDER", "playwright")
    monkeypatch.setenv("SEARCH_PROVIDER_FALLBACKS", "duckduckgo")
    client = WebSearchClient()
    client.cache = WebCache(base / "cache", ttl_seconds=3600)
    calls = {"playwright": 0, "duckduckgo": 0}

    def fail(query: str, limit: int):
        calls["playwright"] += 1
        raise TimeoutError("browser timeout")

    def succeed(query: str, limit: int):
        calls["duckduckgo"] += 1
        return [SearchResult("DeskPilot", "https://example.org", "cached")]

    client._search_playwright = fail  # type: ignore[method-assign]
    client._search_duckduckgo = succeed  # type: ignore[method-assign]
    assert client.search("agent eval", 1)[0].title == "DeskPilot"
    assert client.last_provider == "duckduckgo"
    assert client.last_errors
    assert client.search("agent eval", 1)[0].title == "DeskPilot"
    assert client.last_cache_hit is True
    assert calls == {"playwright": 2, "duckduckgo": 1}


def test_cross_encoder_is_lazy_and_reports_local_size(base: Path) -> None:
    model_dir = base / "reranker"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"1234")
    reranker = SentenceTransformersCrossEncoderReranker(str(model_dir))
    metadata = reranker.metadata()
    assert metadata["lazy_loaded"] is False
    assert metadata["disk_bytes"] == 4
