from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from math import ceil
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.rag.retrieval import RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "dataset" / "rag_p2_cases.jsonl"


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, ceil(len(ordered) * ratio) - 1))] if ordered else 0.0


def _disk_bytes(value: str) -> int:
    path = Path(value)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _load_cases(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:count] if count > 0 else rows


def _run_provider(provider: str, cases: list[dict[str, Any]], model: str) -> dict[str, Any]:
    latencies: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    errors: list[str] = []
    old_provider, old_model = os.getenv("RAG_RERANK_PROVIDER"), os.getenv("RAG_RERANK_MODEL")
    os.environ["RAG_RERANK_PROVIDER"] = provider
    if model:
        os.environ["RAG_RERANK_MODEL"] = model
    try:
        with tempfile.TemporaryDirectory(prefix=f"deskpilot_rerank_{provider}_") as raw:
            for number, case in enumerate(cases):
                try:
                    index = DocumentIndex(Path(raw) / f"index_{number}.json")
                    index.client.embed = lambda texts: [local_hash_embedding(text) for text in texts]  # type: ignore[method-assign]
                    index._configured_embedding_space = lambda: "local-hash-v1:384"  # type: ignore[method-assign]
                    index._embedding_space = "local-hash-v1:384"
                    for relative in case.get("corpus", []):
                        index.add_file(ROOT / relative)
                    started = time.perf_counter()
                    results = index.search_hybrid(RetrievalRequest(
                        original_query=str(case["query"]), queries=(str(case["query"]),),
                        top_k=max(1, int(case.get("top_k", 5))),
                        context_expansion=str(case.get("expansion", "none")), rerank_provider=provider,
                    ))
                    latencies.append((time.perf_counter() - started) * 1000)
                    relevant = {Path(item).name.casefold() for item in case.get("relevant_sources", [])}
                    ranks = [rank for rank, item in enumerate(results, 1)
                             if any(name in item.source_label.casefold() for name in relevant)]
                    matched = {name for name in relevant
                               if any(name in item.source_label.casefold() for item in results)}
                    recalls.append(len(matched) / len(relevant) if relevant else 1.0)
                    reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
                except Exception as exc:
                    errors.append(f"{case.get('id', number)}: {type(exc).__name__}: {exc}")
    finally:
        if old_provider is None:
            os.environ.pop("RAG_RERANK_PROVIDER", None)
        else:
            os.environ["RAG_RERANK_PROVIDER"] = old_provider
        if old_model is None:
            os.environ.pop("RAG_RERANK_MODEL", None)
        else:
            os.environ["RAG_RERANK_MODEL"] = old_model
    executed = len(recalls)
    return {
        "provider": provider, "requested": len(cases), "executed": executed, "errors": errors,
        "hit_rate": sum(value > 0 for value in reciprocal_ranks) / executed if executed else 0.0,
        "mean_recall": sum(recalls) / executed if executed else 0.0,
        "mrr": sum(reciprocal_ranks) / executed if executed else 0.0,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_ms": _percentile(latencies, 0.95),
        "model_disk_bytes": _disk_bytes(model) if provider == "cross_encoder" else 0,
        # 三种本地 provider 不消耗生成模型 Token；保留字段便于未来加入 API reranker。
        "total_tokens": 0,
    }


def _render(results: list[dict[str, Any]], dataset: Path) -> str:
    lines = [
        "# RAG Reranker 消融报告", "", f"- 数据集：`{dataset}`", "",
        "| Provider | 执行/请求 | Hit Rate | Recall | MRR | 平均延迟 | P95 | Token | 模型磁盘 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['provider']} | {item['executed']}/{item['requested']} | {item['hit_rate']:.4f} | "
            f"{item['mean_recall']:.4f} | {item['mrr']:.4f} | {item['latency_mean_ms']:.1f} ms | "
            f"{item['latency_p95_ms']:.1f} ms | {item['total_tokens']} | "
            f"{item['model_disk_bytes'] / 1024 / 1024:.1f} MiB |"
        )
    for item in results:
        if item["errors"]:
            lines.extend(["", f"## {item['provider']} 错误", ""] + [f"- {error}" for error in item["errors"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="比较 RRF、lexical 与本地 Cross-Encoder 精排。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--providers", default="rrf,lexical,cross_encoder")
    parser.add_argument("--model", default=os.getenv("RAG_RERANK_MODEL", ""))
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "reports" / "reranker_ablation.md")
    args = parser.parse_args()
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    if "cross_encoder" in providers and not args.model:
        parser.error("包含 cross_encoder 时必须通过 --model 或 RAG_RERANK_MODEL 提供本地模型目录。")
    results = [_run_provider(provider, _load_cases(args.dataset, args.count), args.model) for provider in providers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(results, args.dataset), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
