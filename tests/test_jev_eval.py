from __future__ import annotations

from pathlib import Path

import pytest

from deskpilot.decision.jev import JevClient, JevConfig
from eval.compare_jev import compare, intervention_counts
from eval.run_jev_judge import DATASET, _normalize_score, judge_jev, judge_llm, load_cases, score_metrics


def test_jev_judge_dataset_is_balanced_and_labelled() -> None:
    cases = load_cases(DATASET)
    assert len(cases) >= 8
    for source in ("rag", "web"):
        selected = [case for case in cases if case["source_type"] == source]
        assert selected
        for label in ("relevant", "faithful"):
            assert {0.0, 1.0}.issubset({case["expected"][label] for case in selected}) or (
                min(case["expected"][label] for case in selected) == 0.0 and
                max(case["expected"][label] for case in selected) >= 0.75)


def test_jev_judge_scores_and_continuous_metrics(monkeypatch) -> None:
    client = JevClient(JevConfig(api_key="fake"))
    def post(payload):
        return {"answers": {"relevant": {"type": "score", "score": 3.0},
                            "faithful": {"type": "score", "score": 1.0}},
                "usage": {"input_tokens": 4, "output_tokens": 2}}
    monkeypatch.setattr(client, "_post", post)
    prediction = judge_jev(client, load_cases(DATASET)[0])
    assert prediction == {"relevant": 0.75, "faithful": 0.25}
    rows = [{"prediction": prediction, "expected": {"relevant": 1.0, "faithful": 1.0}},
            {"prediction": {"relevant": 0.0, "faithful": 0.0},
             "expected": {"relevant": 0.0, "faithful": 0.0}}]
    assert score_metrics(rows, "relevant")["mae"] == 0.125
    assert score_metrics(rows, "faithful")["mae"] == 0.375


def test_jev_judge_accepts_probability_weighted_fractional_scores(monkeypatch) -> None:
    client = JevClient(JevConfig(api_key="fake"))
    monkeypatch.setattr(client, "_post", lambda payload: {
        "answers": {"relevant": {"type": "score", "score": 3.91,
                                 "probabilities": {"0": 0, "1": 0.01, "2": 0.01, "3": 0.05, "4": 0.93}},
                    "faithful": {"type": "score", "score": 3.86}},
    })
    assert judge_jev(client, load_cases(DATASET)[0]) == {
        "relevant": 0.9775, "faithful": 0.965,
    }


@pytest.mark.parametrize("bad", [True, None, -0.1, 4.1, float("nan"), float("inf")])
def test_jev_judge_rejects_invalid_scores(bad) -> None:
    with pytest.raises(ValueError, match="score"):
        _normalize_score(bad, allow_fractional=True)


def test_llm_judge_rejects_binary_output() -> None:
    class FakeLLM:
        def chat(self, *args, **kwargs):
            return '{"relevant": true, "faithful": false}'
    with pytest.raises(ValueError, match="score"):
        judge_llm(FakeLLM(), load_cases(DATASET)[0])


def test_llm_judge_normalizes_five_level_scores() -> None:
    class FakeLLM:
        def chat(self, *args, **kwargs):
            return '{"relevant": 2, "faithful": 3}'
    assert judge_llm(FakeLLM(), load_cases(DATASET)[0]) == {
        "relevant": 0.5, "faithful": 0.75,
    }


def test_jev_comparison_requires_paired_36_cases() -> None:
    with pytest.raises(ValueError, match="same 36"):
        compare([{"case_id": "a"}], [{"case_id": "b"}])


def test_jev_comparison_reports_regressions_and_observed_f1() -> None:
    base = [{"case_id": f"case-{number}", "subset": "DirectQA", "status": "passed",
             "metrics": {"f1": 0.8 if number == 0 else None},
             "score": 1, "latency_ms": 100, "token_usage": {"total_tokens": 10}}
            for number in range(36)]
    gated = [{**item, "status": "failed" if item["case_id"] == "case-0" else "passed",
              "metrics": {"f1": 0.4 if item["case_id"] == "case-0" else None}}
             for item in base]
    report = compare(base, gated)
    assert report["regressed_case_ids"] == ["case-0"]
    assert report["rows"][0]["off_answer_f1"] == 0.8
    assert report["rows"][0]["gated_answer_f1"] == 0.4


def test_jev_comparison_counts_actual_shortcuts_and_rag_filtering() -> None:
    details = [
        {"name": "jev_router", "detail": '{"mode":"gated","used_shortcut":true}'},
        {"name": "retrieval_trace", "detail": '{"events":[{"stage":"jev_evidence","input_count":12,"output_count":5,"detail":{"assessed":8}}]}'},
    ]
    assert intervention_counts([{"step_details": details}]) == {
        "router_shortcuts": 1, "router_fallbacks": 0,
        "rag_evidence_filtered": 7, "rag_evidence_calls": 1,
    }
