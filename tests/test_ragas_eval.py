from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import run_ragas_eval as subject


def test_dataset_has_ground_truth_and_safe_corpus() -> None:
    cases = subject.load_cases(subject.DATASET)
    assert len(cases) == 8
    assert all(case["reference"] and case["relevant_sources"] for case in cases)
    assert len(subject.load_cases(subject.DATASET, 2)) == 2


def test_local_paper_cases_are_grounded_in_cited_pdf_pages() -> None:
    pytest.importorskip("pymupdf")
    import pymupdf

    path = subject.ROOT / "eval" / "dataset" / "ragas_multimodal_local.jsonl"
    required = ("blip.pdf", "blip-2.pdf", "recognize_anything.pdf", "mllm_survey.pdf")
    if any(not (subject.ROOT / "多模态" / name).exists() for name in required):
        pytest.skip("Local paper assets are intentionally not committed")
    cases = subject.load_cases(path)
    assert len(cases) == 24
    assert len({case["id"] for case in cases}) == len(cases)
    documents = {}
    for case in cases:
        assert case["gold_pages"] and case["evidence_anchor"]
        assert set(case["gold_pages"]) == set(case["evidence_anchor"])
        assert set(case["relevant_sources"]) == set(case["gold_pages"])
        for name, pages in case["gold_pages"].items():
            document = documents.setdefault(name, pymupdf.open(subject.ROOT / "多模态" / name))
            assert all(1 <= page <= len(document) for page in pages)
            text = re.sub(r"\s+", " ", " ".join(document[page - 1].get_text() for page in pages)).casefold()
            anchor = re.sub(r"\s+", " ", case["evidence_anchor"][name]).casefold()
            assert anchor in text, (case["id"], name, pages)


def test_dataset_rejects_duplicate_and_external_corpus(tmp_path: Path) -> None:
    case = {"id": "x", "query": "q", "reference": "r", "corpus": ["../../secret.txt"]}
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus file"):
        subject.load_cases(path)
    case["corpus"] = ["eval/dataset/fixtures/docs/rag_p2_nebula.md"]
    path.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique IDs"):
        subject.load_cases(path)


def test_report_excludes_failed_and_missing_scores(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    rows = [
        {"case_id": "a", "subset": "x", "status": "scored", "metrics": {
            "faithfulness": 0.8, "answer_relevancy": None, "context_precision": 0.6, "context_recall": 1.0,
        }, "source_recall": 1.0, "total_latency_ms": 1234},
        {"case_id": "b", "subset": "x", "status": "error", "metrics": {}, "error": "judge failed"},
    ]
    subject.write_report(rows, report, dataset=subject.DATASET, judge_model="judge",
                         embedding_model="embed", scored=True)
    text = report.read_text(encoding="utf-8")
    assert "有效评分 1 条" in text
    assert "| faithfulness | 0.8000 | 1 |" in text
    assert "| answer_relevancy | N/A | 0 |" in text
    assert "judge failed" in text
    assert len(report.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_runner_records_sample_without_real_api(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(llm_api_key="fake", embedding_api_key="fake", llm_model="judge",
                             embedding_model="embed", allow_local_fallback=True)
    monkeypatch.setattr(subject, "load_config", lambda: config)
    monkeypatch.setattr(subject, "replace", lambda value, **changes: SimpleNamespace(**{**vars(value), **changes}))
    monkeypatch.setattr(subject, "OpenAICompatibleClient", lambda value: object())
    monkeypatch.setattr(subject, "build_sample", lambda case, client, directory, index_cache: {
        "user_input": case["query"], "retrieved_contexts": ["evidence"],
        "response": "answer", "reference": case["reference"], "source_recall": 1.0,
    })
    monkeypatch.setattr(subject, "score_sample", lambda sample, config: {key: 1.0 for key in subject.METRICS})
    report = tmp_path / "report.md"
    rows = subject.run(subject.load_cases(subject.DATASET, 1), judge=True, report=report, dataset=subject.DATASET)
    assert rows[0]["status"] == "scored"
    assert report.exists()


def test_index_is_reused_for_identical_corpus(tmp_path: Path, monkeypatch) -> None:
    from deskpilot.core.models import Evidence

    class FakeIndex:
        created = 0

        def __init__(self, path):
            self.__class__.created += 1
            self.last_trace = None

        def add_file(self, path):
            pass

        def search_hybrid(self, request):
            return [Evidence("one", "doc", "rag_p2_nebula.md", "Evidence text", 1.0)]

    monkeypatch.setattr(subject, "DocumentIndex", FakeIndex)
    client = SimpleNamespace(chat=lambda *args, **kwargs: "Answer [1]")
    case = subject.load_cases(subject.DATASET, 1)[0]
    cache = {}
    first = subject.build_sample(case, client, tmp_path / "a", cache)
    second = subject.build_sample(case, client, tmp_path / "b", cache)
    assert first["index_cache_hit"] is False
    assert second["index_cache_hit"] is True
    assert FakeIndex.created == 1
