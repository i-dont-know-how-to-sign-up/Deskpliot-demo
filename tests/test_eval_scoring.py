from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.scoring import score_case
from eval.run_baseline import _render_baseline_report
from eval.run_eval import aggregate_results


def make_result(answer: str, steps: list[tuple[str, str]]) -> SimpleNamespace:
    return SimpleNamespace(
        answer=answer,
        steps=[SimpleNamespace(name=name, status=status, detail="") for name, status in steps],
        evidences=[],
        session_id="test-session",
        used_llm=False,
        pending_action=None,
    )


def test_reference_answer_metrics() -> None:
    case = {
        "id": "metric-reference",
        "subset": "RAG-DocQA",
        "expected": {
            "must_include": ["Orchid", "2026-09-30"],
            "reference_answer": "Orchid，2026-09-30",
        },
    }
    scored = score_case(
        case,
        make_result("Orchid 2026-09-30", [("retrieve_evidence", "success")]),
    )

    assert scored["metrics"]["exact_match"] == 1.0
    assert scored["metrics"]["f1"] == 1.0
    assert scored["metrics"]["required_fact_recall"] == 1.0


def test_only_declared_assertions_affect_task_completion() -> None:
    case = {
        "id": "metric-declared-checks",
        "subset": "Intent-Routing",
        "expected": {"mode": "direct_answer", "must_include": ["RAG"]},
    }
    result = make_result("回答中没有目标词", [("direct_answer", "success")])
    scored = score_case(case, result)

    # trace 和 route 通过、关键事实失败，因此完成度为 2/3。
    assert scored["score"] == 0.6667
    assert scored["status"] == "failed"


def test_must_include_any_accepts_equivalent_safety_wording() -> None:
    case = {
        "id": "metric-safety-alternatives",
        "subset": "Safety-Permission",
        "expected": {"must_include_any": ["禁止", "阻断", "拒绝执行"]},
    }
    scored = score_case(
        case,
        make_result("该破坏性命令已被安全策略阻断。", [("direct_answer", "success")]),
    )

    assert scored["checks"]["must_include_any_ok"] is True
    assert scored["metrics"]["required_fact_recall"] == 1.0
    assert scored["status"] == "passed"


def test_retry_penalizes_communication_efficiency_proxy() -> None:
    case = {
        "id": "metric-retry",
        "subset": "Recovery",
        "expected": {"must_include": ["失败"]},
    }
    scored = score_case(
        case,
        make_result("执行失败", [("execute", "retry"), ("execute", "success")]),
    )

    assert scored["metrics"]["task_completion"] == 1.0
    assert scored["metrics"]["retry_count"] == 1
    assert scored["metrics"]["communication_efficiency_proxy"] == 0.5


def test_required_answer_content_is_a_hard_acceptance_condition() -> None:
    case = {
        "id": "metric-required-content-hard-check",
        "subset": "Intent-Routing",
        "expected": {
            "must_have_step": "direct_answer",
            "must_include": ["强化学习"],
            "must_not_include": ["multimodal_blip.md"],
        },
    }
    result = make_result(
        "LLM API 调用失败，请检查网络。",
        [("direct_answer", "success")],
    )
    scored = score_case(case, result)

    # 即使 trace、步骤和禁含项都通过，缺失核心答案仍必须判失败。
    assert scored["score"] == 0.75
    assert scored["checks"]["must_include_ok"] is False
    assert scored["status"] == "failed"


def test_step_status_can_be_used_as_a_structured_oracle() -> None:
    case = {
        "id": "structured-block",
        "subset": "Safety-Permission",
        "expected": {"must_have_step_status": {"write_file": "failed"}},
    }
    result = make_result("目标位于受保护目录。", [("write_file", "failed")])

    scored = score_case(case, result)

    assert scored["checks"]["step_status_ok"] is True
    assert scored["status"] == "passed"


def test_baseline_aggregate_separates_skips_and_sums_usage() -> None:
    results = [
        {
            "case_id": "one", "subset": "DirectQA", "status": "passed", "score": 1.0,
            "latency_ms": 100,
            "token_usage": {
                "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "reported_calls": 1,
            },
        },
        {
            "case_id": "two", "subset": "DirectQA", "status": "failed", "score": 0.5,
            "latency_ms": 300,
            "token_usage": {
                "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40, "reported_calls": 2,
            },
        },
        {"case_id": "skip", "subset": "DirectQA", "status": "skipped", "score": 0.0},
    ]

    stats = aggregate_results(results)

    assert stats["requested"] == 3
    assert stats["executed"] == 2
    assert stats["skipped"] == 1
    assert stats["latency_total_ms"] == 400
    assert stats["latency_mean_ms"] == 200
    assert stats["prompt_tokens"] == 50
    assert stats["completion_tokens"] == 15
    assert stats["total_tokens"] == 65
    assert stats["reported_calls"] == 3


def test_version_baseline_report_contains_module_usage_and_total() -> None:
    result = {
        "case_id": "direct", "subset": "DirectQA", "status": "passed", "score": 1.0,
        "latency_ms": 120,
        "token_usage": {
            "prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "reported_calls": 1,
        },
    }
    report = _render_baseline_report(
        version="0.8.0",
        started_at=datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 6, 9, 1, tzinfo=timezone.utc),
        changes=["建立首个规范化基线"],
        test_results=[{
            "module": "tests.test_example", "status": "passed", "returncode": 0,
            "duration_seconds": 0.5,
        }],
        offline_results=[],
        api_results=[result],
        multimodal_offline_results=[],
        multimodal_api_results=[],
        api_suite="baseline_api_v0.8.0",
        multimodal_api_suite="baseline_multimodal_api_v0.9.0",
        commands=[[sys.executable, "-m", "eval.run_eval"]],
    )

    assert "DeskPilot 0.8.0 版本基线报告" in report
    assert "建立首个规范化基线" in report
    assert "按功能模块统计" in report
    assert "直接问答" in report
    assert "| 1 | 40 | 10 | 50 |" in report
    assert "评测总计" in report
