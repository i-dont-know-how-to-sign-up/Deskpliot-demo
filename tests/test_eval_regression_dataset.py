from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.run_eval import DATASET, load_cases, run_case
from eval.scoring import score_case


class RegressionDatasetTests(unittest.TestCase):
    def test_cases_are_unique_and_fixtures_exist(self) -> None:
        cases = load_cases(DATASET)
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(cases), 140)
        regression = [case for case in cases if case["id"].startswith("reg_index_report_") or case["id"].startswith("reg_web_file_")]
        self.assertEqual(len(regression), 6)
        for case in regression:
            for relative in case.get("setup", {}).get("index_files", []):
                self.assertTrue((DATASET.parents[2] / relative).is_file(), relative)
            self.assertEqual(run_case(case, "offline")["status"], "skipped")

    def test_missing_artifact_cannot_pass_on_answer_alone(self) -> None:
        case = {"id": "check", "subset": "RAG-DocQA", "expected": {
            "must_include": ["BLIP"], "artifact": {"path": "report.md", "must_include": ["来源"]},
        }}
        result = SimpleNamespace(answer="BLIP 来源", steps=[SimpleNamespace(name="write_report")])
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            self.assertEqual(score_case(case, result, workspace=workspace)["status"], "failed")
            (workspace / "report.md").write_text("BLIP 来源", encoding="utf-8")
            self.assertEqual(score_case(case, result, workspace=workspace)["status"], "passed")

    def test_step_order_and_document_coverage_are_hard_constraints(self) -> None:
        case = {"id": "check", "subset": "RAG-DocQA", "expected": {
            "ordered_steps": ["retrieve_index_collection", "write_report"],
            "min_evidence_documents": 2,
        }}
        result = SimpleNamespace(answer="ok", steps=[SimpleNamespace(name="write_report"),
            SimpleNamespace(name="retrieve_index_collection")], evidences=[SimpleNamespace(doc_id="one")])
        self.assertEqual(score_case(case, result)["status"], "failed")
        result.steps.reverse()
        result.evidences.append(SimpleNamespace(doc_id="two"))
        self.assertEqual(score_case(case, result)["status"], "passed")

    def test_desktop_target_and_confirmation_are_required(self) -> None:
        case = {"id": "desktop", "subset": "Multi-Agent", "expected": {
            "ordered_steps": ["web_search", "write_file"],
            "requires_confirmation": True, "pending_path_suffix": "Desktop/weather.txt",
        }}
        result = SimpleNamespace(answer="done", steps=[SimpleNamespace(name="web_search"),
            SimpleNamespace(name="write_file")], pending_action=None)
        self.assertEqual(score_case(case, result)["status"], "failed")
        result.pending_action = {"kwargs": {"path": "D:/workspace/weather.txt"},
                                 "permission": {"requires_confirmation": True}}
        self.assertEqual(score_case(case, result)["status"], "failed")
        result.pending_action["kwargs"]["path"] = "C:/Users/TestUser/Desktop/weather.txt"
        self.assertEqual(score_case(case, result)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
