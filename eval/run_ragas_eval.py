"""Evaluate the isolated DeskPilot document RAG pipeline with RAGAS 0.2.x."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import load_config
from deskpilot.rag.retrieval import RetrievalRequest
from deskpilot.rag.vector_index import DocumentIndex


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "dataset" / "ragas_cases.jsonl"
REPORT = ROOT / "eval" / "reports" / "ragas_latest.md"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_cases(path: Path, count: int = 0) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count < 0:
        raise ValueError("count must be zero or positive")
    if count:
        cases = cases[:count]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Dataset must have unique IDs and at least one case")
    for case in cases:
        if not all(isinstance(case.get(key), str) and case[key].strip() for key in ("id", "query", "reference")):
            raise ValueError(f"Missing id/query/reference: {case.get('id', '?')}")
        if not isinstance(case.get("corpus"), list) or not case["corpus"]:
            raise ValueError(f"Missing corpus: {case['id']}")
        for name in case["corpus"]:
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file():
                raise ValueError(f"Corpus file must exist inside repository: {name}")
    return cases


def build_sample(case: dict[str, Any], client: OpenAICompatibleClient, temporary: Path,
                 index_cache: dict[tuple[str, ...], DocumentIndex] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_key = tuple(sorted(set(case["corpus"])))
    index = index_cache.get(corpus_key) if index_cache is not None else None
    cache_hit = index is not None
    if index is None:
        index = DocumentIndex(temporary / "index.json")
        for name in corpus_key:
            index.add_file((ROOT / name).resolve())
        if index_cache is not None:
            index_cache[corpus_key] = index
    question = case["query"]
    evidence = index.search_hybrid(RetrievalRequest(
        original_query=question,
        queries=(question,),
        top_k=max(1, min(int(case.get("top_k", 4)), 10)),
        context_expansion="parent",
        rerank_provider="lexical",
    ))
    contexts = [item.text for item in evidence if item.text.strip()]
    if not contexts:
        raise RuntimeError("No retrieved contexts; refusing to ask the model without evidence")
    numbered = "\n\n".join(f"[{i}] {item.source_label}\n{item.text}" for i, item in enumerate(evidence, 1))
    answer = client.chat([
        {"role": "system", "content": "仅根据证据回答问题，不能补充证据之外的事实。证据不足时明确说明。重要结论用 [数字] 标注来源。"},
        {"role": "user", "content": f"问题：{question}\n\n证据：\n{numbered}"},
    ], temperature=0, max_tokens=500).strip()
    if not answer:
        raise RuntimeError("Answer model returned an empty response")
    expected = {Path(name).name.casefold() for name in case.get("relevant_sources", [])}
    found = {name for name in expected if any(name in item.source_label.casefold() for item in evidence)}
    return {
        "case_id": case["id"], "subset": case.get("subset", "other"),
        "user_input": question, "retrieved_contexts": contexts,
        "response": answer, "reference": case["reference"],
        "sources": [item.source_label for item in evidence],
        "source_recall": len(found) / len(expected) if expected else None,
        "index_cache_hit": cache_hit,
        "retrieval_trace": index.last_trace.to_dict() if index.last_trace else {},
        "pipeline_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def score_sample(sample: dict[str, Any], config: Any) -> dict[str, float | None]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -r requirements-ragas.txt") from exc

    judge_options = {"extra_body": {"enable_thinking": False}} if (
        config.qwen_enable_thinking is False and
        ("dashscope" in config.llm_base_url.lower() or config.llm_model.lower().startswith("qwen"))
    ) else {}
    judge = ChatOpenAI(model=config.llm_model, api_key=config.llm_api_key,
                       base_url=config.llm_base_url, temperature=0, timeout=90, **judge_options)
    embeddings = OpenAIEmbeddings(model=config.embedding_model, api_key=config.embedding_api_key,
                                  base_url=config.embedding_base_url, check_embedding_ctx_length=False)
    row = {key: sample[key] for key in ("user_input", "retrieved_contexts", "response", "reference")}
    result = evaluate(
        Dataset.from_list([row]),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge, embeddings=embeddings,
        run_config=RunConfig(timeout=90, max_retries=1, max_workers=4),
        raise_exceptions=False, show_progress=False,
    )
    scores = result.scores[0]
    return {name: (float(value) if isinstance((value := scores.get(name)), (int, float))
                   and math.isfinite(value) else None) for name in METRICS}


def write_report(rows: list[dict[str, Any]], path: Path, *, dataset: Path, judge_model: str,
                 embedding_model: str, scored: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    scores = [row for row in rows if row.get("status") == "scored"]
    complete = scored and len(scores) == len(rows) and all(
        all(row.get("metrics", {}).get(name) is not None for name in METRICS) for row in scores
    )
    status = "评分完成" if complete else ("评分部分完成或失败" if scored else "仅准备样本，未执行 RAGAS")
    lines = ["# DeskPilot RAGAS 评测报告", "",
             f"- 时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- 数据集：`{dataset}`；选中 {len(rows)} 条；有效评分 {len(scores)} 条",
             f"- 状态：{status}",
             f"- 答案/评审模型：`{judge_model}`；Embedding：`{embedding_model}`",
             "- 流程：隔离语料建索引 → DeskPilot 混合检索/lexical 精排 → LLM 据证据生成答案 → RAGAS 0.2.x 评分。",
             "- 这不是桌面 Agent 全链路评测；语料包含合成故障文档或本地真实论文，问答和参考答案均为本项目人工编写。",
             "- RAGAS 为 LLM-as-judge 估计值；失败/NaN 不计入均值，也不得视为零分。", "",
             "- 来源 Recall 只看目标文件是否命中；RAGAS 上下文精度还受片段相关性、排序和评审模型影响，两者不应混读。", "",
             "## 汇总", ""]
    if not scores:
        lines.append("没有有效 RAGAS 分数；请检查依赖、API 连通性及逐例错误。")
    else:
        lines.extend(["| 指标 | 均值 | 有效数 |", "|---|---:|---:|"])
        for name in METRICS:
            values = [row["metrics"][name] for row in scores if row["metrics"].get(name) is not None]
            lines.append(f"| {name} | {statistics.mean(values):.4f} | {len(values)} |" if values
                         else f"| {name} | N/A | 0 |")
        recalls = [row["source_recall"] for row in rows if row.get("source_recall") is not None]
        if recalls:
            lines.append(f"\n来源 Recall（确定性辅助指标，非 RAGAS）：{statistics.mean(recalls):.4f}（n={len(recalls)}）。")
        pipeline = [float(row["pipeline_latency_ms"]) for row in scores if row.get("pipeline_latency_ms") is not None]
        elapsed = [float(row["total_latency_ms"]) for row in scores if row.get("total_latency_ms") is not None]
        if pipeline and elapsed:
            lines.append(f"\n平均管线耗时 {statistics.mean(pipeline) / 1000:.2f} s；平均含 RAGAS 评审总耗时 "
                         f"{statistics.mean(elapsed) / 1000:.2f} s。两者不是同一延迟指标。")
    lines.extend(["", "## 逐例结果", "", "| ID | 子集 | 状态 | 忠实度 | 答案相关性 | 上下文精度 | 上下文召回 | 来源召回 | 耗时(s) |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        values = ["N/A" if row.get("metrics", {}).get(name) is None else
                  f"{row['metrics'][name]:.3f}" for name in METRICS]
        source_recall = row.get("source_recall")
        lines.append(f"| {row['case_id']} | {row.get('subset', '')} | {row['status']} | " +
                     " | ".join(values) + f" | {'N/A' if source_recall is None else f'{source_recall:.3f}'} | " +
                     f"{row.get('total_latency_ms', 0) / 1000:.2f} |")
    errors = [row for row in rows if row.get("error")]
    if errors:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- `{row['case_id']}`：{row['error']}" for row in errors)
    lines.extend(["", "逐例答案、上下文及检索 trace 见同名 `.jsonl`；其中可能包含评测语料内容，不要用于私人文档。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cases: list[dict[str, Any]], *, judge: bool, report: Path, dataset: Path) -> list[dict[str, Any]]:
    config = replace(load_config(), allow_local_fallback=False)
    if not config.llm_api_key or not config.embedding_api_key:
        raise RuntimeError("Real LLM and embedding API keys are required; local fallback is not a RAGAS run")
    client = OpenAICompatibleClient(config)
    rows: list[dict[str, Any]] = []
    # Evaluation must not silently depend on an optional Jev service.
    previous_jev = os.environ.get("JEV_MODE")
    os.environ["JEV_MODE"] = "off"
    try:
        with tempfile.TemporaryDirectory(prefix="ragas_", dir=report.parent) as directory:
            index_cache: dict[tuple[str, ...], DocumentIndex] = {}
            for position, case in enumerate(cases, 1):
                started = time.perf_counter()
                row: dict[str, Any] = {"case_id": case["id"], "subset": case.get("subset", "other"),
                                       "status": "error", "metrics": {}}
                try:
                    sample_dir = Path(directory) / case["id"]
                    sample_dir.mkdir()
                    row.update(build_sample(case, client, sample_dir, index_cache))
                    if judge:
                        row["metrics"] = score_sample(row, config)
                        row["status"] = "scored" if any(value is not None for value in row["metrics"].values()) else "error"
                        if row["status"] == "error":
                            row["error"] = "RAGAS returned only NaN or missing metrics"
                    else:
                        row["status"] = "prepared"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                row["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                rows.append(row)
                print(f"[{position}/{len(cases)}] {case['id']}: {row['status']}", flush=True)
                write_report(rows, report, dataset=dataset, judge_model=config.llm_model,
                             embedding_model=config.embedding_model, scored=judge)
    finally:
        if previous_jev is None:
            os.environ.pop("JEV_MODE", None)
        else:
            os.environ["JEV_MODE"] = previous_jev
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", help="Repeat to combine datasets; default is portable 8-case set")
    parser.add_argument("--count", type=int, default=0, help="0 means all cases")
    parser.add_argument("--prepare-only", action="store_true", help="Generate RAG samples but do not call RAGAS")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    datasets = args.dataset or [DATASET]
    cases = [case for dataset in datasets for case in load_cases(dataset)]
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("Combined datasets contain duplicate case IDs")
    if args.count < 0:
        parser.error("count must be zero or positive")
    if args.count:
        cases = cases[:args.count]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = run(cases, judge=not args.prepare_only, report=args.report,
               dataset=", ".join(str(path) for path in datasets))
    print(args.report)
    return 0 if rows and all(row["status"] == ("prepared" if args.prepare_only else "scored") and
                             (args.prepare_only or all(row["metrics"].get(name) is not None for name in METRICS))
                             for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
