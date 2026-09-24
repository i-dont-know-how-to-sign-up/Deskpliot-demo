from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_rag_p2_eval import load_cases, run_case


def test_p2_eval_runs_offline_and_exposes_metrics() -> None:
    case = load_cases(ROOT / "eval" / "dataset" / "rag_p2_cases.jsonl")[0]
    result = run_case(case, "offline", "lexical", "sentence_window")
    assert result["status"] == "passed"
    assert result["source_recall"] == 1.0
    assert 0.0 <= result["precision_at_k"] <= 1.0
    assert result["context_tokens"] > 0
    assert result["effective_provider"] == "lexical"


def main() -> None:
    test_p2_eval_runs_offline_and_exposes_metrics()
    print("RAG P2 eval tests passed: 1")


if __name__ == "__main__":
    main()
