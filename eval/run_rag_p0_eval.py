from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.rag.vector_index import DocumentIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "dataset" / "rag_p0_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "eval" / "reports" / "rag_p0_latest.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _matched_source(evidence: Any, case: dict[str, Any]) -> str | None:
    source_names = {Path(value).name.casefold() for value in case.get("relevant_sources", [])}
    matched = next((name for name in source_names if name in evidence.source_label.casefold()), None)
    required = [str(value).casefold() for value in case.get("relevant_contains", [])]
    text = evidence.text.casefold()
    content_match = not required or any(value in text for value in required)
    return matched if matched and content_match else None


def _metrics(relevance: list[bool], source_hits: set[str], expected_sources: int) -> dict[str, float]:
    recall = min(1.0, len(source_hits) / max(1, expected_sources))
    reciprocal_rank = next((1.0 / (index + 1) for index, value in enumerate(relevance) if value), 0.0)
    dcg = sum((1.0 / math.log2(index + 2)) for index, value in enumerate(relevance) if value)
    ideal_count = min(max(1, expected_sources), len(relevance))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return {"recall": recall, "mrr": reciprocal_rank, "ndcg": dcg / ideal if ideal else 0.0}


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deskpilot_rag_p0_") as folder:
        index = DocumentIndex(Path(folder) / "index.json")
        # 离线评测必须确定性执行，不能因本机 .env 中存在 Key 而访问网络。
        index.client.embed = lambda texts: [local_hash_embedding(text) for text in texts]  # type: ignore[method-assign]
        index._configured_embedding_space = lambda: "local-hash-v1:384"  # type: ignore[method-assign]
        index._embedding_space = "local-hash-v1:384"
        for relative in case.get("corpus", []):
            path = (ROOT / relative).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
                raise FileNotFoundError(relative)
            index.add_file(path)
        top_k = int(case.get("top_k", 5))
        evidence = index.search(str(case["query"]), top_k=top_k)
        relevance: list[bool] = []
        relevant_source_hits: set[str] = set()
        for item in evidence:
            matched = _matched_source(item, case)
            # 同一来源的重复 chunk 不重复增加 nDCG，防止指标超过 1。
            is_new_relevant = bool(matched and matched not in relevant_source_hits)
            relevance.append(is_new_relevant)
            if matched:
                relevant_source_hits.add(matched)
        expected_names = {Path(value).name.casefold() for value in case.get("relevant_sources", [])}
        metrics = _metrics(relevance, relevant_source_hits, len(expected_names))
        return {
            "case_id": case["id"],
            "subset": case["subset"],
            "status": "passed" if any(relevance) else "failed",
            **{name: round(value, 6) for name, value in metrics.items()},
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "selected": [item.source_label for item in evidence],
            "trace": index.last_trace.to_dict() if index.last_trace else {},
            "index_trace": index.last_index_trace,
        }


def write_report(results: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    report = output.with_suffix(".md")
    count = len(results)
    average = lambda key: sum(float(item.get(key, 0)) for item in results) / count if count else 0.0
    lines = [
        "# RAG P0 检索评测报告",
        "",
        f"- 用例数：{count}",
        f"- 通过数：{sum(item['status'] == 'passed' for item in results)}",
        f"- 平均 Recall：{average('recall'):.4f}",
        f"- MRR：{average('mrr'):.4f}",
        f"- 平均 nDCG：{average('ndcg'):.4f}",
        f"- 平均耗时：{average('latency_ms'):.2f} ms",
        "",
        "| 用例 | 子集 | 状态 | Recall | MRR | nDCG | 耗时(ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | {item['subset']} | {item['status']} | "
            f"{item['recall']:.4f} | {item['mrr']:.4f} | {item['ndcg']:.4f} | {item['latency_ms']:.2f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DeskPilot RAG P0 检索层评测。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=0, help="最多执行多少条；0 表示全部。")
    parser.add_argument("--case-id", action="append", default=[], help="只执行指定 ID，可重复传入。")
    parser.add_argument("--include-local", action="store_true", help="包含多模态目录中的大型真实 PDF。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    selected = [case for case in cases if not args.case_id or case["id"] in set(args.case_id)]
    if not args.include_local:
        selected = [case for case in selected if not case.get("runtime", {}).get("optional_local")]
    if args.count > 0:
        selected = selected[:args.count]
    results: list[dict[str, Any]] = []
    total = len(selected)
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{total}] {case['id']} {case['query']}", flush=True)
        try:
            result = run_case(case)
        except Exception as exc:
            result = {
                "case_id": case["id"], "subset": case["subset"], "status": "error",
                "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency_ms": 0.0,
                "error": repr(exc),
            }
        results.append(result)
        print(f"  -> {result['status']} recall={result['recall']:.3f} mrr={result['mrr']:.3f}", flush=True)
    write_report(results, args.output)
    print(f"报告：{args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
