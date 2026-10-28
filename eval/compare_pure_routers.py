"""Compare one Jev decision with one Qwen classification, without router chaining."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import load_config
from deskpilot.core.json_utils import parse_json_value
from deskpilot.decision.jev import JevClient, JevConfig


CRITERIA = {
    "direct_answer": "Answer from general knowledge; no external data or action.",
    "single_tool": "Exactly one external read or action is needed.",
    "multi_step": "Two or more dependent actions or an output artifact is needed.",
    "unclear": "The goal or required target is missing or ambiguous.",
}
CASES = [
    ("direct_1", "什么是 RAG？请用三句话回答。", "direct_answer"),
    ("direct_2", "解释一下 Python 的生成器。", "direct_answer"),
    ("tool_1", "搜索最新的上海天气预报。", "single_tool"),
    ("tool_2", "读取当前目录下的 README.md。", "single_tool"),
    ("plan_1", "先检索 AI Agent 论文，再整理成报告写入文件。", "multi_step"),
    ("plan_2", "查找项目文档，写一份摘要并作为附件发送邮件。", "multi_step"),
    ("unclear_1", "帮我处理一下。", "unclear"),
    ("unclear_2", "把它发给那个人。", "unclear"),
]


def classify_jev(client: JevClient, question: str) -> str:
    answer = client.judge(question, {
        "route": {"type": "choice", "instructions": "Choose the complete task class.", "criteria": CRITERIA},
    })["route"]
    choice = answer.get("choice")
    if choice not in CRITERIA:
        raise ValueError("Jev returned an invalid route")
    return str(choice)


def classify_qwen(client: OpenAICompatibleClient, question: str) -> str:
    prompt = "Classify the user request into exactly one category. Return JSON only: {\"route\":\"category\"}.\n"
    prompt += "Categories: " + json.dumps(CRITERIA, ensure_ascii=False) + "\nRequest: " + question
    raw = client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=40)
    parsed = parse_json_value(raw, dict)
    choice = parsed.get("route") if parsed else None
    if choice not in CRITERIA:
        raise ValueError("Qwen returned an invalid route")
    return str(choice)


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reported_calls")}


def run(cases: list[tuple[str, str, str]], repeats: int, jev: JevClient,
        qwen: OpenAICompatibleClient) -> dict[str, Any]:
    clients = {"jev": jev, "qwen": qwen}
    classifiers = {"jev": classify_jev, "qwen": classify_qwen}
    rows: list[dict[str, Any]] = []
    blocked = False
    for repeat in range(repeats):
        for index, (case_id, question, expected) in enumerate(cases):
            order = ("jev", "qwen") if (index + repeat) % 2 == 0 else ("qwen", "jev")
            for name in order:
                client = clients[name]
                before = dict(client.token_usage)
                started = time.perf_counter()
                prediction = ""
                error = ""
                try:
                    prediction = classifiers[name](client, question)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:240]
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                row = {"repeat": repeat + 1, "case_id": case_id, "router": name,
                       "expected": expected, "prediction": prediction, "correct": prediction == expected,
                       "latency_ms": elapsed, "usage": _usage_delta(before, client.token_usage), "error": error}
                rows.append(row)
                print(f"[{repeat + 1}/{repeats} {case_id}] {name}: {elapsed:.0f} ms; "
                      f"route={prediction or 'ERROR'}", flush=True)
                if name == "jev" and "Jev HTTP 403" in error:
                    blocked = True
            if blocked:
                break
        if blocked:
            break
    summary: dict[str, dict[str, float | int | None]] = {}
    for name in ("jev", "qwen"):
        subset = [row for row in rows if row["router"] == name]
        valid = [row for row in subset if not row["error"]]
        latencies = sorted(float(row["latency_ms"]) for row in valid)
        summary[name] = {
            "n": len(subset), "successes": len(valid), "accuracy": round(sum(row["correct"] for row in subset) / len(subset), 4),
            "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "median_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
            "p95_latency_ms": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "reported_input_tokens": sum(row["usage"]["prompt_tokens"] for row in subset),
            "reported_output_tokens": sum(row["usage"]["completion_tokens"] for row in subset),
            "usage_reported_calls": sum(row["usage"]["reported_calls"] for row in subset),
        }
    comparable = len(rows) == 2 * len(cases) * repeats and all(not row["error"] for row in rows)
    return {"timestamp": datetime.now().astimezone().isoformat(), "scope": "four-way routing only; no tool arguments, answer generation, or fallback",
            "comparable": comparable, "blocked_reason": "Jev HTTP 403; check model access/region" if blocked else "",
            "models": {"jev": jev.config.model, "qwen": qwen.config.llm_model},
            "jev_provider": getattr(jev.config, "provider", "unknown"), "summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=len(CASES), help="First N synthetic cases (1-8)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/pure_router/latest.json"))
    args = parser.parse_args()
    if not 1 <= args.count <= len(CASES) or not 1 <= args.repeats <= 5:
        parser.error("count must be 1-8 and repeats must be 1-5")
    config = replace(load_config(), allow_local_fallback=False, qwen_enable_thinking=False)
    jev_config = JevConfig.from_env()
    if not config.llm_api_key:
        parser.error("Configure LLM in .env")
    try:
        jev_config.validate_transport()
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    report = run(CASES[:args.count], args.repeats, JevClient(jev_config), OpenAICompatibleClient(config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return int(not report["comparable"])


if __name__ == "__main__":
    raise SystemExit(main())
