from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_rag_p1_eval import DEFAULT_DATASET, load_cases, run_case


def test_p1_dataset_is_unique_and_reproducible() -> None:
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) == 8
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        for relative in case.get("corpus", []):
            assert (ROOT / relative).is_file()


def test_exact_bm25_case_passes_offline() -> None:
    case = next(item for item in load_cases(DEFAULT_DATASET) if item["id"] == "rag_p1_001")
    result = run_case(case, "offline", "bm25")
    assert result["status"] == "passed"
    assert result["trace"]["mode"] == "hybrid"


def test_hybrid_fills_dense_recall_gap() -> None:
    case = next(item for item in load_cases(DEFAULT_DATASET) if item["id"] == "rag_p1_003")
    dense = run_case(case, "offline", "dense")
    hybrid = run_case(case, "offline", "hybrid")
    assert dense["recall"] < hybrid["recall"]
    assert hybrid["status"] == "passed"


def main() -> None:
    test_p1_dataset_is_unique_and_reproducible()
    test_exact_bm25_case_passes_offline()
    test_hybrid_fills_dense_recall_gap()
    print("RAG P1 eval tests passed: 3")


if __name__ == "__main__":
    main()
