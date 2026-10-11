import json

import pytest

from eval.run_reliability_eval import summarize_contracts, _percentile95, load_cases


def test_reliability_report_does_not_invent_missing_observations() -> None:
    summary = summarize_contracts([{"scenario": "unknown_no_replay", "passed": False}])
    assert summary["checkpoint_reuse_rate"] is None
    assert summary["cancel_p95_seconds"] is None
    assert summary["unknown_effect_cases"] == 0
    assert summary["retry_attempt_max"] is None


def test_reliability_report_uses_only_applicable_denominators() -> None:
    summary = summarize_contracts([
        {"scenario": "pdf_pause_resume", "checkpoint_eligible": 2, "checkpoint_reused": 1},
        {"scenario": "unknown_no_replay", "automatic_replays": 0},
        {"scenario": "bounded_retry", "attempts": 3},
        {"scenario": "cancel_process", "cancel_latency_seconds": 0.2},
    ])
    assert summary["checkpoint_reuse_rate"] == 0.5
    assert summary["unknown_effect_cases"] == 1
    assert summary["unknown_automatic_replays"] == 0
    assert summary["cancel_p95_seconds"] == 0.2
    assert summary["retry_attempt_max"] == 3
    assert _percentile95([0.1] * 19 + [100]) == 0.1


@pytest.mark.parametrize("identifier", ["../outside", "C:/outside", "valid", ""])
def test_dataset_ids_cannot_escape_temporary_workspace_or_collide(identifier: str) -> None:
    cases = [{"id": identifier, "scenario": "deadline", "seed": 0}] * 2
    with pytest.raises(ValueError):
        load_cases("\n".join(json.dumps(case) for case in cases).encode())
