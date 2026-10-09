from __future__ import annotations

import json
from pathlib import Path

from eval.run_eval import DATASET, load_cases
from eval.run_multimodal_eval import DATASET as MULTIMODAL_DATASET
from eval.run_multimodal_eval import load_cases as load_multimodal_cases


ROOT = Path(__file__).resolve().parents[1]


def test_v090_api_suite_has_three_difficulties_per_main_subset() -> None:
    suite = json.loads((ROOT / "eval" / "suites" / "baseline_api_v0.9.0.json").read_text(encoding="utf-8"))
    rows = {item["id"]: item for item in load_cases(DATASET)}
    declarations = {item["id"]: item["difficulty"] for item in suite["cases"]}
    grouped: dict[str, list[str]] = {}
    for case_id in suite["case_ids"]:
        grouped.setdefault(rows[case_id]["subset"], []).append(declarations[case_id])

    assert len(suite["case_ids"]) == 36
    assert len(grouped) == 12
    assert all(sorted(values) == ["complex", "medium", "simple"] for values in grouped.values())
    assert all(not rows[case_id].get("runtime", {}).get("requires_network") for case_id in suite["case_ids"])


def test_v090_multimodal_api_suite_has_three_difficulties() -> None:
    suite = json.loads(
        (ROOT / "eval" / "suites" / "baseline_multimodal_api_v0.9.0.json").read_text(encoding="utf-8")
    )
    rows = {item["id"]: item for item in load_multimodal_cases()}

    assert len(rows) == 22
    assert [rows[case_id]["difficulty"] for case_id in suite["case_ids"]] == [
        "simple", "medium", "complex",
    ]
    assert all(rows[case_id]["runtime"]["requires_vlm"] for case_id in suite["case_ids"])
    assert MULTIMODAL_DATASET.is_file()
