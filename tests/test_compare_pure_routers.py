from __future__ import annotations

from types import SimpleNamespace

from eval.compare_pure_routers import classify_jev, classify_qwen, run


class FakeJev:
    config = SimpleNamespace(model="fake-jev")

    def __init__(self) -> None:
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}
        self.questions = []

    def judge(self, state, questions):
        self.questions.append((state, questions))
        self.token_usage = {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11, "reported_calls": 1}
        return {"route": {"type": "choice", "choice": "direct_answer"}}


class FakeQwen:
    config = SimpleNamespace(llm_model="fake-qwen")

    def __init__(self) -> None:
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_calls": 0}
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        self.token_usage = {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11, "reported_calls": 1}
        return '{"route":"direct_answer"}'


def test_pure_routers_call_only_their_own_client() -> None:
    jev, qwen = FakeJev(), FakeQwen()
    assert classify_jev(jev, "什么是 RAG？") == "direct_answer"
    assert len(jev.questions) == 1 and not qwen.messages
    assert list(jev.questions[0][1]) == ["route"]
    assert classify_qwen(qwen, "什么是 RAG？") == "direct_answer"
    assert len(qwen.messages) == 1


def test_pure_router_report_has_paired_latency_accuracy_and_usage() -> None:
    jev, qwen = FakeJev(), FakeQwen()
    report = run([("direct_1", "什么是 RAG？", "direct_answer")], 1, jev, qwen)
    assert {row["router"] for row in report["rows"]} == {"jev", "qwen"}
    assert all(row["correct"] and not row["error"] for row in report["rows"])
    assert report["summary"]["jev"]["reported_input_tokens"] == 9
    assert report["summary"]["qwen"]["reported_input_tokens"] == 8
    assert report["comparable"] is True


def test_pure_router_report_stops_after_jev_access_denied() -> None:
    class DeniedJev(FakeJev):
        def judge(self, state, questions):
            raise RuntimeError("OpenRouter Jev HTTP 403")

    report = run([("a", "A?", "direct_answer"), ("b", "B?", "direct_answer")], 1,
                 DeniedJev(), FakeQwen())
    assert len(report["rows"]) == 2
    assert report["comparable"] is False
    assert "HTTP 403" in report["blocked_reason"]
