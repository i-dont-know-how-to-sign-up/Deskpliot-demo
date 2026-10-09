from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import load_config
from eval.api_health import (
    circuit_breaker_results,
    classify_api_error,
    is_infrastructure_error,
    run_api_preflight,
)
from eval import run_eval


def test_api_preflight_checks_llm_and_embedding_without_fallback() -> None:
    config = load_config()
    with (
        patch.object(OpenAICompatibleClient, "chat", return_value="OK") as chat,
        patch.object(OpenAICompatibleClient, "embed", return_value=[[0.1, 0.2]]) as embed,
    ):
        result = run_api_preflight(config)

    assert result["ok"] is True
    assert [item["name"] for item in result["checks"]] == ["llm", "embedding"]
    assert chat.call_args.kwargs["max_tokens"] == 8
    assert embed.call_count == 1


def test_api_preflight_reports_connectivity_failure_and_stops_before_embedding() -> None:
    config = load_config()
    with (
        patch.object(
            OpenAICompatibleClient,
            "chat",
            side_effect=RuntimeError("API request failed: <urlopen error [WinError 10061] refused>"),
        ),
        patch.object(OpenAICompatibleClient, "embed") as embed,
    ):
        result = run_api_preflight(config)

    assert result["ok"] is False
    assert result["stage"] == "llm"
    assert result["category"] == "connectivity"
    assert embed.call_count == 0


def test_api_error_classification_distinguishes_business_and_infrastructure_errors() -> None:
    assert classify_api_error("HTTP 401 Unauthorized") == "authentication"
    assert classify_api_error("HTTP 429 rate limit") == "rate_limit"
    assert is_infrastructure_error({"status": "failed", "error": "timeout"}) is False
    assert is_infrastructure_error({"status": "error", "error": "assertion mismatch"}) is False
    assert is_infrastructure_error({"status": "error", "error": "connection reset"}) is True


def test_circuit_breaker_marks_remaining_cases_as_skipped() -> None:
    results = circuit_breaker_results(
        [
            {"id": "case-1", "subset": "DirectQA", "difficulty": "simple"},
            {"id": "case-2", "subset": "RAG-DocQA", "difficulty": "complex"},
        ],
        cause="connection refused",
    )

    assert [item["status"] for item in results] == ["skipped", "skipped"]
    assert results[0]["reason"].startswith("api_circuit_open:")
    assert results[1]["difficulty"] == "complex"


def _write_eval_dataset(path: Path, count: int = 4) -> None:
    cases = [
        {
            "id": f"case-{index}",
            "subset": "DirectQA",
            "input": "test",
            "expected": {},
            "runtime": {"requires_llm": True},
        }
        for index in range(1, count + 1)
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in cases) + "\n",
        encoding="utf-8",
    )


def test_run_eval_preflight_failure_skips_all_cases(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_eval_dataset(dataset)
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "run_eval",
        "--dataset", str(dataset),
        "--count", "4",
        "--mode", "api",
        "--output", str(output),
        "--report", str(tmp_path / "report.md"),
        "--metadata", str(tmp_path / "metadata.json"),
        "--no-progress",
    ])
    failed = {
        "ok": False,
        "stage": "llm",
        "category": "connectivity",
        "error": "connection refused",
        "guidance": "check network",
        "checks": [],
    }
    with (
        patch.object(run_eval, "run_api_preflight", return_value=failed),
        patch.object(run_eval, "run_case") as run_case,
    ):
        exit_code = run_eval.main()

    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 2
    assert run_case.call_count == 0
    assert len(results) == 4
    assert all(item["status"] == "skipped" for item in results)


def test_run_eval_opens_circuit_after_three_infrastructure_errors(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_eval_dataset(dataset)
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "run_eval",
        "--dataset", str(dataset),
        "--count", "4",
        "--mode", "api",
        "--output", str(output),
        "--report", str(tmp_path / "report.md"),
        "--metadata", str(tmp_path / "metadata.json"),
        "--no-progress",
    ])
    error_result = {
        "case_id": "placeholder",
        "subset": "DirectQA",
        "status": "error",
        "score": 0.0,
        "error": "API request failed: connection reset",
    }
    with (
        patch.object(run_eval, "run_api_preflight", return_value={"ok": True, "checks": []}),
        patch.object(run_eval, "run_case", side_effect=lambda case, mode: {
            **error_result,
            "case_id": case["id"],
        }) as run_case,
    ):
        exit_code = run_eval.main()

    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 1
    assert run_case.call_count == 3
    assert [item["status"] for item in results] == ["error", "error", "error", "skipped"]


if __name__ == "__main__":
    test_api_preflight_checks_llm_and_embedding_without_fallback()
    test_api_preflight_reports_connectivity_failure_and_stops_before_embedding()
    test_api_error_classification_distinguishes_business_and_infrastructure_errors()
    test_circuit_breaker_marks_remaining_cases_as_skipped()
