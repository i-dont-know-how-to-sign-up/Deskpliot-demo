from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.scoring import score_case


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
