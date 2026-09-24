from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import cosine_similarity, local_hash_embedding
from deskpilot.rag.retrieval import RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "dataset" / "rag_p2_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "eval" / "reports" / "rag_p2_latest.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank_metrics(relevance: list[bool], relevant_count: int) -> tuple[float, float, float]:
    precision = sum(relevance) / max(1, len(relevance))
    mrr = next((1.0 / (index + 1) for index, value in enumerate(relevance) if value), 0.0)
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(relevance) if value)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(relevant_count, len(relevance))))
    return precision, mrr, dcg / ideal if ideal else 0.0


def _redundancy(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    vectors = [local_hash_embedding(value) for value in texts]
    pairs = [max(0.0, cosine_similarity(vectors[left], vectors[right]))
             for left in range(len(vectors)) for right in range(left + 1, len(vectors))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def run_case(case: dict[str, Any], mode: str, provider: str, expansion: str) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deskpilot_rag_p2_eval_") as raw:
        index = DocumentIndex(Path(raw) / "index.json")
        if mode == "offline":
            index.client.embed = lambda texts: [local_hash_embedding(value) for value in texts]  # type: ignore[method-assign]
            index._configured_embedding_space = lambda: "local-hash-v1:384"  # type: ignore[method-assign]
            index._embedding_space = "local-hash-v1:384"
        for relative in case.get("corpus", []):
            path = (ROOT / relative).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT):
                raise FileNotFoundError(relative)
            index.add_file(path)

        chosen_provider = str(case.get("provider", "lexical")) if provider == "auto" else provider
        chosen_expansion = case.get("expansion", "none") if expansion == "auto" else expansion
        query = str(case["query"])
        request = RetrievalRequest(
            original_query=query,
            queries=tuple(case.get("queries", [query])),
            top_k=int(case.get("top_k", 5)),
            context_expansion=chosen_expansion,
            rerank_provider=chosen_provider,
            max_context_tokens=case.get("max_context_tokens"),
        )
        evidence = index.search_hybrid(request)
        expected_sources = {Path(value).name.casefold() for value in case.get("relevant_sources", [])}
        found_sources = {name for name in expected_sources
                         if any(name in item.source_label.casefold() for item in evidence)}
        # 排序指标按唯一 gold 来源计分；同一文档的多个窗口不能重复贡献 DCG。
        relevance: list[bool] = []
        ranked_source_hits: set[str] = set()
        for item in evidence:
            matched = next(
                (name for name in expected_sources
                 if name in item.source_label.casefold() and name not in ranked_source_hits),
                None,
            )
            relevance.append(matched is not None)
            if matched:
                ranked_source_hits.add(matched)
        precision, mrr, ndcg = _rank_metrics(relevance, len(expected_sources))
        source_recall = len(found_sources) / max(1, len(expected_sources))
        combined = "\n".join(item.text for item in evidence).casefold()
        required = [str(value).casefold() for value in case.get("required_terms", [])]
        forbidden = [str(value).casefold() for value in case.get("forbidden_terms", [])]
        term_recall = sum(value in combined for value in required) / max(1, len(required))
        forbidden_hits = sum(value in combined for value in forbidden)
        trace = index.last_trace.to_dict() if index.last_trace else {}
        rerank = next((value for value in trace.get("events", []) if value["stage"] == "rerank"), {})
        context_tokens = sum(int(item.metadata.get("context_tokens", 0)) for item in evidence)
        passed = source_recall == 1.0 and term_recall == 1.0 and forbidden_hits == 0
        return {
            "case_id": case["id"], "subset": case["subset"], "status": "passed" if passed else "failed",
            "provider": chosen_provider, "effective_provider": rerank.get("detail", {}).get("effective_provider", ""),
            "expansion": chosen_expansion, "source_recall": round(source_recall, 6),
            "term_recall": round(term_recall, 6), "precision_at_k": round(precision, 6),
            "mrr": round(mrr, 6), "ndcg": round(ndcg, 6), "forbidden_hits": forbidden_hits,
            "context_tokens": context_tokens, "redundancy": round(_redundancy([item.text for item in evidence]), 6),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "selected": [item.source_label for item in evidence], "trace": trace,
        }


def write_report(results: list[dict[str, Any]], output: Path, mode: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    count = len(results)
    average = lambda key: sum(float(item.get(key, 0)) for item in results) / count if count else 0.0
    lines = [
        "# RAG P2 精排与上下文扩展评测报告", "", f"- 模式：{mode}", f"- 用例数：{count}",
        f"- 通过数：{sum(item['status'] == 'passed' for item in results)}",
        f"- 平均来源 Recall：{average('source_recall'):.4f}",
        f"- 平均事实项 Recall：{average('term_recall'):.4f}",
        f"- Precision@K：{average('precision_at_k'):.4f}", f"- MRR：{average('mrr'):.4f}",
        f"- nDCG：{average('ndcg'):.4f}", f"- 平均上下文 token：{average('context_tokens'):.2f}",
        f"- 平均冗余度：{average('redundancy'):.4f}", f"- 平均耗时：{average('latency_ms'):.2f} ms", "",
        "| 用例 | 子集 | Provider | 扩展 | 状态 | 来源R | 事实R | P@K | MRR | nDCG | Token | 冗余 | 耗时(ms) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | {item['subset']} | {item['effective_provider'] or item['provider']} | "
            f"{item['expansion']} | {item['status']} | {item['source_recall']:.3f} | "
            f"{item['term_recall']:.3f} | {item['precision_at_k']:.3f} | {item['mrr']:.3f} | "
            f"{item['ndcg']:.3f} | {item['context_tokens']} | {item['redundancy']:.3f} | {item['latency_ms']:.2f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DeskPilot RAG P2 精排、扩展和 MMR 评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--mode", choices=["offline", "api"], default="offline")
    parser.add_argument("--provider", choices=["auto", "disabled", "lexical", "cross_encoder", "colbert", "api", "all"], default="auto")
    parser.add_argument("--expansion", choices=["auto", "none", "sentence_window", "parent", "all"], default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    cases = [case for case in cases if not args.case_id or case["id"] in set(args.case_id)]
    if args.count > 0:
        cases = cases[:args.count]
    providers = ["disabled", "lexical", "colbert"] if args.provider == "all" else [args.provider]
    expansions = ["none", "sentence_window", "parent"] if args.expansion == "all" else [args.expansion]
    jobs = [(case, provider, expansion) for case in cases for provider in providers for expansion in expansions]
    results = []
    for number, (case, provider, expansion) in enumerate(jobs, start=1):
        print(f"[{number}/{len(jobs)}] {case['id']} provider={provider} expansion={expansion}", flush=True)
        try:
            result = run_case(case, args.mode, provider, expansion)
        except Exception as exc:
            result = {"case_id": case["id"], "subset": case["subset"], "status": "error",
                      "provider": provider, "effective_provider": "", "expansion": expansion,
                      "source_recall": 0.0, "term_recall": 0.0, "precision_at_k": 0.0,
                      "mrr": 0.0, "ndcg": 0.0, "forbidden_hits": 0, "context_tokens": 0,
                      "redundancy": 0.0, "latency_ms": 0.0, "selected": [], "error": repr(exc)}
        results.append(result)
        print(f"  -> {result['status']} source_recall={result['source_recall']:.3f} term_recall={result['term_recall']:.3f}", flush=True)
    write_report(results, args.output, args.mode)
    print(f"报告：{args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
