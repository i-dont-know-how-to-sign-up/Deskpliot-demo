"""Compare Jev typed decisions with the existing LLM on labelled RAG/web snippets."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import load_config
from deskpilot.core.json_utils import parse_json_value
from deskpilot.decision.jev import JevClient


DATASET = Path(__file__).parent / "dataset" / "jev_judge_cases.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [item["id"] for item in cases]
    if len(ids) != len(set(ids)) or any(item.get("source_type") not in {"rag", "web"} for item in cases):
        raise ValueError("Duplicate ID or invalid source type")
    for case in cases:
        expected = case.get("expected", {})
        if any(type(expected.get(label)) not in {int, float} or expected[label] not in
               {0, 0.25, 0.5, 0.75, 1} for label in ("relevant", "faithful")):
            raise ValueError(f"Invalid 0-1 score label in {case['id']}")
    return cases


SCORE_LEVELS = [
    "0: Completely unrelated or contradicted; no useful support.",
    "1: Only a small part is relevant or supported.",
    "2: About half of the requested facts or answer claims are covered.",
    "3: Most requested facts or answer claims are covered, with minor gaps.",
    "4: Directly relevant or all answer claims are supported without contradiction.",
]


def _normalize_score(value: Any, *, allow_fractional: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 4 or
            (not allow_fractional and value != int(value))):
        kind = "finite number" if allow_fractional else "integer"
        raise ValueError(f"Judge score must be a {kind} from 0 to 4")
    return float(value) / 4


def judge_jev(client: JevClient, case: dict[str, Any]) -> dict[str, float]:
    questions = {
        "relevant": {"type": "score", "instructions": "How much of the user's information need is covered by the evidence? Score relevance, not truth of the answer.", "criteria": SCORE_LEVELS},
        "faithful": {"type": "score", "instructions": "How many factual claims in the answer are supported by the evidence? Penalize contradictions and unsupported additions.", "criteria": SCORE_LEVELS},
    }
    state = {key: case[key] for key in ("query", "evidence", "answer")}
    with client.stage("Jev Eval Judge"):
        answers = client.judge(state, questions)
    # OpenRouter score is the probability-weighted expected level, so it can be fractional.
    return {key: _normalize_score(answers[key].get("score"), allow_fractional=True)
            for key in questions}


def judge_llm(client: OpenAICompatibleClient, case: dict[str, Any]) -> dict[str, float]:
    prompt = {key: case[key] for key in ("query", "evidence", "answer")}
    raw = client.chat([
        {"role": "system", "content": "Judge only supplied evidence. Return JSON with exactly two integer fields, relevant and faithful, each 0 to 4. Use this common rubric: " + json.dumps(SCORE_LEVELS) + ". Relevant scores coverage of the query; faithful scores support for the answer's factual claims. Do not infer missing facts."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ], temperature=0.0, max_tokens=120)
    parsed = parse_json_value(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge returned invalid JSON scores")
    return {key: _normalize_score(parsed.get(key)) for key in ("relevant", "faithful")}


def score_metrics(rows: list[dict[str, Any]], label: str) -> dict[str, float | int | None]:
    valid = [row for row in rows if row.get("prediction") is not None]
    if not valid:
        return {"n": 0, "mae": None, "rmse": None, "mean_score": None, "within_one_level": None}
    errors = [abs(row["prediction"][label] - row["expected"][label]) for row in valid]
    return {"n": len(valid), "mae": round(statistics.mean(errors), 4),
            "rmse": round((statistics.mean(error * error for error in errors)) ** 0.5, 4),
            "mean_score": round(statistics.mean(row["prediction"][label] for row in valid), 4),
            "within_one_level": round(sum(error <= 0.25 for error in errors) / len(errors), 4)}


def run(cases: list[dict[str, Any]], judge: str, jev: JevClient | None,
        llm: OpenAICompatibleClient | None) -> dict[str, Any]:
    rows = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {judge} {case['id']}", flush=True)
        started = time.perf_counter()
        error = ""
        prediction = None
        try:
            prediction = judge_jev(jev, case) if judge == "jev" else judge_llm(llm, case)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"id": case["id"], "source_type": case["source_type"],
                     "expected": case["expected"], "prediction": prediction, "error": error,
                     "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
    client = jev if judge == "jev" else llm
    metrics = {}
    for source in ("all", "rag", "web"):
        selected = [row for row in rows if source == "all" or row["source_type"] == source]
        metrics[source] = {label: score_metrics(selected, label) for label in ("relevant", "faithful")}
    return {"judge": judge, "model": client.config.model if judge == "jev" else client.config.llm_model,
            "rows": rows, "metrics": metrics,
            "requested": len(rows), "successful": sum(row["prediction"] is not None for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in rows) if rows else None,
            "usage": dict(client.token_usage)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--judge", choices=("jev", "llm", "both"), default="both")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/jev_judge/latest.json"))
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("count must be positive")
    cases = load_cases(args.dataset)[:args.count]
    config = load_config()
    jev = JevClient() if args.judge in {"jev", "both"} else None
    llm = OpenAICompatibleClient(config) if args.judge in {"llm", "both"} else None
    if llm and not config.llm_api_key:
        parser.error("Configure DASHSCOPE_API_KEY/LLM_API_KEY for LLM")
    if jev:
        try:
            jev.config.validate_transport()
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    results = [run(cases, name, jev, llm)
               for name in (("jev", "llm") if args.judge == "both" else (args.judge,))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(result["judge"], "errors", sum(bool(row["error"]) for row in result["rows"]),
              "relevance MAE", result["metrics"]["all"]["relevant"]["mae"],
              "faithfulness MAE", result["metrics"]["all"]["faithful"]["mae"])
    return int(any(row["error"] for result in results for row in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
