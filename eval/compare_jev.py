"""Paired off/gated comparison on the fixed 36-case real-API suite."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import statistics

from deskpilot.core.config import load_config
from deskpilot.decision.jev import JevConfig
from eval.run_eval import aggregate_results, _load_suite


SUITE = "baseline_api_v0.9.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def intervention_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"router_shortcuts": 0, "router_fallbacks": 0,
              "rag_evidence_filtered": 0, "rag_evidence_calls": 0}
    for item in items:
        for step in item.get("step_details", []):
            try:
                detail = json.loads(step.get("detail", ""))
            except (TypeError, ValueError):
                continue
            if step.get("name") == "jev_router":
                counts["router_shortcuts"] += int(detail.get("used_shortcut") is True)
                counts["router_fallbacks"] += int(bool(detail.get("fallback")))
            elif step.get("name") == "retrieval_trace":
                for event in detail.get("events", []):
                    if event.get("stage") != "jev_evidence":
                        continue
                    counts["rag_evidence_calls"] += int(event.get("detail", {}).get("assessed", 0) > 0)
                    counts["rag_evidence_filtered"] += max(
                        0, int(event.get("input_count", 0)) - int(event.get("output_count", 0)))
    return counts


def compare(off: list[dict[str, Any]], gated: list[dict[str, Any]]) -> dict[str, Any]:
    off_by_id = {item["case_id"]: item for item in off}
    gated_by_id = {item["case_id"]: item for item in gated}
    if len(off) != 36 or len(gated) != 36 or off_by_id.keys() != gated_by_id.keys():
        raise ValueError("Both runs must contain the same 36 suite case IDs")
    subsets = ["all", *sorted({item["subset"] for item in off})]
    rows = []
    def observed_f1(items: list[dict[str, Any]]) -> float | None:
        values = [item.get("metrics", {}).get("f1") for item in items]
        available = [float(value) for value in values if value is not None]
        return round(statistics.mean(available), 4) if available else None
    for subset in subsets:
        before = [item for item in off if subset == "all" or item["subset"] == subset]
        after = [gated_by_id[item["case_id"]] for item in before]
        a, b = aggregate_results(before), aggregate_results(after)
        rows.append({"subset": subset, "n": len(before), "off": a, "gated": b,
                     "off_answer_f1": observed_f1(before), "gated_answer_f1": observed_f1(after),
                     "accuracy_delta": round(b["accuracy"] - a["accuracy"], 4),
                     "p95_delta_ms": None if a["latency_p95_ms"] is None or b["latency_p95_ms"] is None else
                     round(b["latency_p95_ms"] - a["latency_p95_ms"], 2),
                     "total_tokens_delta": b["total_tokens"] - a["total_tokens"]})
    regressed = [case_id for case_id in off_by_id if off_by_id[case_id]["status"] == "passed"
                 and gated_by_id[case_id]["status"] != "passed"]
    return {"rows": rows, "regressed_case_ids": regressed,
            "case_ids": list(off_by_id), "jev_model": JevConfig.from_env().model,
            "interventions": intervention_counts(gated)}


def _run(mode: str, destination: Path) -> None:
    environment = os.environ.copy()
    environment["JEV_MODE"] = mode
    environment["ALLOW_LOCAL_FALLBACK"] = "false"
    command = [sys.executable, "-m", "eval.run_eval", "--suite", SUITE, "--mode", "api",
               "--count", "36", "--output", str(destination),
               "--report", str(destination.with_suffix(".md")),
               "--run-name", f"Jev comparison {mode}"]
    print("Running", mode, "36-case API suite", flush=True)
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(f"{mode} run failed with exit code {completed.returncode}; see {destination.with_suffix('.md')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-run", type=Path, help="Existing off-mode JSONL run")
    parser.add_argument("--gated-run", type=Path, help="Existing gated-mode JSONL run")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if bool(args.off_run) != bool(args.gated_run):
        parser.error("--off-run and --gated-run must be supplied together")
    suite = _load_suite(SUITE)
    if len(suite["case_ids"]) != 36:
        parser.error("The fixed baseline API suite no longer contains 36 cases")
    if not args.off_run:
        config = load_config()
        jev = JevConfig.from_env()
        if not config.llm_api_key or not config.embedding_api_key or not jev.api_key:
            parser.error("Set LLM, embedding and OPENROUTER_API_KEY in .env before running paid API comparison")
    directory = args.output_dir or Path("eval/runs") / ("jev_compare_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z"))
    directory.mkdir(parents=True, exist_ok=True)
    off_path = args.off_run or directory / "off.jsonl"
    gated_path = args.gated_run or directory / "gated.jsonl"
    if not args.off_run:
        _run("off", off_path)
        _run("gated", gated_path)
    off, gated = read_jsonl(off_path), read_jsonl(gated_path)
    expected_ids = set(suite["case_ids"])
    if {item["case_id"] for item in off} != expected_ids or {item["case_id"] for item in gated} != expected_ids:
        parser.error("Run case IDs do not match the fixed 36-case suite")
    summary = compare(off, gated)
    jev_events = [event for item in gated for event in item.get("usage_events", [])
                  if isinstance(event, dict) and event.get("kind") == "decision"]
    summary["jev_calls"] = len(jev_events)
    summary["jev_successes"] = sum(event.get("success") is True for event in jev_events)
    summary["jev_errors"] = len(jev_events) - summary["jev_successes"]
    summary["jev_router_calls"] = sum(event.get("stage") == "Jev Router" for event in jev_events)
    summary["jev_rag_calls"] = sum(event.get("stage") == "Jev RAG Evidence" for event in jev_events)
    (directory / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    interventions = summary["interventions"]
    lines = ["# Jev 与原路由对照", "", f"- 套件：`{SUITE}`（36 条，相同用例顺序）",
             f"- Jev：`{summary['jev_model']}`；模式：off vs gated",
             f"- Jev 决策调用：{summary['jev_calls']} 次，成功 {summary['jev_successes']} 次，失败 {summary['jev_errors']} 次。",
             f"- Jev Router 调用 {summary['jev_router_calls']} 次；实际跳过原 Router {interventions['router_shortcuts']} 次；回退 {interventions['router_fallbacks']} 次。",
             f"- Jev RAG 调用 {summary['jev_rag_calls']} 次；实际过滤证据片段 {interventions['rag_evidence_filtered']} 条。",
             "- 差值 = gated - off；负的延迟/Token 差值表示下降。",
             "- 本报告是任务验收指标，非 Jev judge 的相关性/忠实度指标。", "",
             "| 模块 | n | off Accuracy | gated Accuracy | Δ | off 答案F1* | gated 答案F1* | P95 Δ(ms) | Token Δ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['subset']} | {row['n']} | {row['off']['accuracy']:.3f} | "
                     f"{row['gated']['accuracy']:.3f} | {row['accuracy_delta']:+.3f} | "
                     f"{row['off_answer_f1'] if row['off_answer_f1'] is not None else 'n/a'} | "
                     f"{row['gated_answer_f1'] if row['gated_answer_f1'] is not None else 'n/a'} | "
                     f"{row['p95_delta_ms'] if row['p95_delta_ms'] is not None else 'n/a'} | "
                     f"{row['total_tokens_delta']:+d} |")
    lines.extend(["", "- off 通过但 gated 未通过：" + (", ".join(summary["regressed_case_ids"]) or "无"),
                  "- *答案 F1 只统计含参考答案的用例，不代表 36 条总体 F1。",
                  "- 真实 API、网络和缓存波动会影响成对结果；需重复运行和复核退化用例。",
                  "- Jev 的输入/输出 Token 已计入总量，错误调用若服务端未返回 usage 则只记调用事件，不估算为零成本。"])
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(directory / "comparison.md")
    if not args.off_run and (not summary["jev_successes"] or not
                             (interventions["router_shortcuts"] or interventions["rag_evidence_filtered"])):
        print("No Jev intervention observed; do not interpret this as an effective gated-vs-off comparison.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
