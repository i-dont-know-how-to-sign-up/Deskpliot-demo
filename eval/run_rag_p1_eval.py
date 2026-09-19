from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import OpenAICompatibleClient, local_hash_embedding
from deskpilot.core.config import load_config
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.retrieval import RetrievalFilters, RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "dataset" / "rag_p1_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "eval" / "reports" / "rag_p1_latest.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(case: dict[str, Any], mode: str, strategy: str) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deskpilot_rag_p1_") as raw:
        index = DocumentIndex(Path(raw) / "index.json")
        if mode == "offline":
            index.client.embed = lambda texts: [local_hash_embedding(text) for text in texts]  # type: ignore[method-assign]
            index._configured_embedding_space = lambda: "local-hash-v1:384"  # type: ignore[method-assign]
            index._embedding_space = "local-hash-v1:384"
        for relative in case.get("corpus", []):
            path = (ROOT / relative).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
                raise FileNotFoundError(relative)
            index.add_file(path)

        llm = OpenAICompatibleClient(load_config())
        analyzer = QueryAnalyzer(
            (lambda prompt: llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=700))
            if mode == "api" else None
        )
        analysis = analyzer.analyze(
            str(case["query"]), index_version=f"{len(index.documents)}:{len(index.chunks)}",
            complex_task=bool(case.get("complex")),
        )
        titles = {value.casefold() for value in case.get("filter_titles", [])}
        doc_ids = tuple(doc_id for doc_id, document in index.documents.items() if document.title.casefold() in titles)
        queries = (analysis.standalone_question, *analysis.sub_questions) if analysis.needs_multi_query else (
            analysis.standalone_question,
        )
        request = RetrievalRequest(
            original_query=str(case["query"]), queries=tuple(queries),
            filters=RetrievalFilters(doc_ids=doc_ids), top_k=int(case.get("top_k", 5)),
            use_dense=strategy in {"dense", "hybrid"}, use_sparse=strategy in {"bm25", "hybrid"},
        )
        evidence = index.search_hybrid(request)
        expected_empty = bool(case.get("expected_empty"))
        expected_sources = {Path(value).name.casefold() for value in case.get("relevant_sources", [])}
        contains = [str(value).casefold() for value in case.get("relevant_contains", [])]
        hits: set[str] = set()
        relevance: list[bool] = []
        for item in evidence:
            source = next((name for name in expected_sources if name in item.source_label.casefold()), None)
            content_ok = not contains or any(value in item.text.casefold() for value in contains)
            relevant = bool(source and content_ok and source not in hits)
            relevance.append(relevant)
            if source and content_ok:
                hits.add(source)
        if expected_empty:
            passed = not evidence
            recall = 1.0 if passed else 0.0
            mrr = recall
            ndcg = recall
        else:
            recall = len(hits) / max(1, len(expected_sources))
            mrr = next((1.0 / (rank + 1) for rank, value in enumerate(relevance) if value), 0.0)
            dcg = sum(1.0 / math.log2(rank + 2) for rank, value in enumerate(relevance) if value)
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(expected_sources), len(relevance))))
            ndcg = dcg / ideal if ideal else 0.0
            passed = recall >= 1.0
        trace = index.last_trace.to_dict() if index.last_trace else {}
        return {
            "case_id": case["id"], "subset": case["subset"], "strategy": strategy,
            "status": "passed" if passed else "failed", "recall": round(recall, 6),
            "mrr": round(mrr, 6), "ndcg": round(ndcg, 6),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "analysis_method": analysis.method, "query_count": len(queries),
            "selected": [item.source_label for item in evidence], "trace": trace,
        }


def write_report(results: list[dict[str, Any]], output: Path, mode: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    count = len(results)
    average = lambda key: sum(float(item.get(key, 0)) for item in results) / count if count else 0.0
    lines = [
        "# RAG P1 混合检索评测报告", "", f"- 模式：{mode}", f"- 用例数：{count}",
        f"- 通过数：{sum(item['status'] == 'passed' for item in results)}",
        f"- 平均 Recall：{average('recall'):.4f}", f"- MRR：{average('mrr'):.4f}",
        f"- 平均 nDCG：{average('ndcg'):.4f}", f"- 平均耗时：{average('latency_ms'):.2f} ms", "",
        "| 用例 | 策略 | 状态 | Recall | MRR | nDCG | 查询数 | 耗时(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | {item['strategy']} | {item['status']} | {item['recall']:.4f} | "
            f"{item['mrr']:.4f} | {item['ndcg']:.4f} | {item['query_count']} | {item['latency_ms']:.2f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DeskPilot RAG P1 混合检索与查询扩展评测。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--mode", choices=["offline", "api"], default="offline")
    parser.add_argument("--strategy", choices=["dense", "bm25", "hybrid", "all"], default="hybrid")
    parser.add_argument("--include-local", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    cases = [case for case in cases if not args.case_id or case["id"] in set(args.case_id)]
    if not args.include_local:
        cases = [case for case in cases if not case.get("runtime", {}).get("optional_local")]
    if args.count > 0:
        cases = cases[:args.count]
    strategies = ["dense", "bm25", "hybrid"] if args.strategy == "all" else [args.strategy]
    results = []
    total = len(cases) * len(strategies)
    for number, (case, strategy) in enumerate(
        ((case, strategy) for case in cases for strategy in strategies), start=1,
    ):
        print(f"[{number}/{total}] {case['id']} strategy={strategy}", flush=True)
        try:
            result = run_case(case, args.mode, strategy)
        except Exception as exc:
            result = {"case_id": case["id"], "subset": case["subset"], "strategy": strategy,
                      "status": "error", "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
                      "latency_ms": 0.0, "query_count": 0, "error": repr(exc)}
        results.append(result)
        print(f"  -> {result['status']} recall={result['recall']:.3f} mrr={result['mrr']:.3f}", flush=True)
    write_report(results, args.output, args.mode)
    print(f"报告：{args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
