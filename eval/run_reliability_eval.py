from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .reliability_scenarios import SCENARIOS, run_scenario
from deskpilot import __version__


def load_cases(content: bytes) -> list[dict]:
    cases = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each JSONL record must be an object")
        identifier = case.get("id", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("Case id must be unique and cannot contain filesystem paths")
        if case.get("scenario") not in SCENARIOS or type(case.get("seed")) is not int:
            raise ValueError(f"Unsupported scenario or non-integer seed: {identifier}")
        seen.add(identifier)
    return cases


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def summarize_contracts(records: list[dict]) -> dict:
    """只汇总真正提供观测字段的样本，不把缺失指标当作零或成功。"""
    checkpoint_cases = [record for record in records if "checkpoint_eligible" in record]
    eligible = sum(record["checkpoint_eligible"] for record in checkpoint_cases)
    reused = sum(record.get("checkpoint_reused", 0) for record in checkpoint_cases)
    replay_cases = [record for record in records if "automatic_replays" in record]
    duplicates = [record for record in records if "duplicate_submissions" in record]
    cancellations = [record["cancel_latency_seconds"] for record in records if "cancel_latency_seconds" in record]
    retry_cases = [record for record in records if record["scenario"] in {"transient_retry", "bounded_retry"}
                   and "attempts" in record]
    return {
        "checkpoint_cases": len(checkpoint_cases), "checkpoint_eligible": eligible,
        "checkpoint_reused": reused, "checkpoint_reuse_rate": reused / eligible if eligible else None,
        "unknown_effect_cases": len(replay_cases),
        "unknown_automatic_replays": sum(record["automatic_replays"] for record in replay_cases),
        "duplicate_observed_cases": len(duplicates),
        "duplicate_submissions": sum(record["duplicate_submissions"] for record in duplicates),
        "cancel_cases": len(cancellations), "cancel_p95_seconds": _percentile95(cancellations),
        "retry_cases": len(retry_cases),
        "retry_attempt_max": max((record["attempts"] for record in retry_cases), default=None),
    }


def _git_metadata() -> dict:
    metadata = {}
    root = Path(__file__).resolve().parents[1]
    # 只信任当前明确运行的项目，不修改用户的全局 safe.directory 配置。
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]
    for key, command in (("commit", [*git, "rev-parse", "HEAD"]),
                         ("dirty", [*git, "status", "--porcelain", "--untracked-files=no"])):
        try:
            process = subprocess.run(command, cwd=root,
                                     capture_output=True, text=True, timeout=5, check=False)
            metadata[key] = (bool(process.stdout.strip()) if key == "dirty" else process.stdout.strip()) if process.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            metadata[key] = None
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline reliability fault-injection evaluation")
    parser.add_argument("--count", type=int, default=None, help="Maximum selected cases; default is the full dataset")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset" / "reliability_cases.jsonl")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).parent / "reports" / "reliability")
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    content = args.dataset.read_bytes()
    try:
        cases = load_cases(content)
    except (ValueError, TypeError) as exc:
        parser.error(f"Invalid dataset: {exc}")
    if args.scenario:
        cases = [case for case in cases if case["scenario"] == args.scenario]
    if args.count is not None:
        cases = cases[:args.count]
    if not cases:
        parser.error("No cases selected")
    records = []
    with tempfile.TemporaryDirectory(prefix="deskpilot_reliability_") as directory:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['id']} {case['scenario']}", flush=True)
            started = time.monotonic()
            try:
                record = run_scenario(case["scenario"], Path(directory) / case["id"], case["seed"])
            except Exception as exc:
                record = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "scenario": case["scenario"],
                          "response_seconds": time.monotonic() - started}
            records.append({"id": case["id"], "seed": case["seed"], **record})
    timings = sorted(float(record.get("response_seconds", 0)) for record in records)
    summary = {"count": len(records), "passed": sum(record["passed"] for record in records),
               "dataset_sha256": hashlib.sha256(content).hexdigest(), "time": datetime.now().astimezone().isoformat(),
               "python": platform.python_version(), "platform": platform.platform(), "real_api": False,
               "project_version": __version__, "git": _git_metadata(),
               "mean_seconds": statistics.mean(timings), "p95_seconds": _percentile95(timings),
               "contracts": summarize_contracts(records)}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "cases": records}
    (args.report_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    scenario_count = len({record["scenario"] for record in records})
    lines = ["# 工程可靠性 P0-P2 故障注入报告", "", f"- 时间：{summary['time']}",
             f"- 项目版本：`{summary['project_version']}`；Git：`{summary['git'].get('commit')}`；dirty：`{summary['git'].get('dirty')}`",
             f"- 数据 SHA-256：`{summary['dataset_sha256']}`", f"- 通过：{summary['passed']}/{summary['count']}",
             f"- 平均/P95 耗时：{summary['mean_seconds']:.3f}s / {summary['p95_seconds']:.3f}s", "",
             "这些指标只覆盖本地确定性故障注入，不是 LLM 答案准确率或生产可用性。未调用真实 API 或真实邮箱。",
             f"本次选中 {scenario_count} 类、{len(records)} 条记录；seed 用于复现，重复记录不代表独立故障类型。", "",
             "| 故障类别 | 通过/总数 | 平均耗时(s) |", "|---|---:|---:|"]
    for scenario in dict.fromkeys(record["scenario"] for record in records):
        selected = [record for record in records if record["scenario"] == scenario]
        lines.append(f"| {scenario} | {sum(record['passed'] for record in selected)}/{len(selected)} | "
                     f"{statistics.mean(float(record.get('response_seconds', 0)) for record in selected):.3f} |")
    contracts = summary["contracts"]
    def display(value):
        return "未观测" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    lines.extend(["", "## 有界恢复与安全指标", "",
                  "| 指标 | 结果 | 观测范围 |", "|---|---:|---|",
                  f"| 检查点复用率 | {display(contracts['checkpoint_reuse_rate'])} | {contracts['checkpoint_cases']} 次恢复；{contracts['checkpoint_reused']}/{contracts['checkpoint_eligible']} 个节点/批次 |",
                  f"| 不确定副作用自动重放数 | {contracts['unknown_automatic_replays']} | {contracts['unknown_effect_cases']} 次 unknown 拒绝重放实验 |",
                  f"| 已观测重复提交数 | {contracts['duplicate_submissions']} | {contracts['duplicate_observed_cases']} 次审批/检查点实验，非 SMTP exactly-once |",
                  f"| 取消响应 P95 / s | {display(contracts['cancel_p95_seconds'])} | {contracts['cancel_cases']} 次，从发出取消请求计时 |",
                  f"| 最大节点 attempt | {display(contracts['retry_attempt_max'])} | {contracts['retry_cases']} 次瞬时/持续网络故障 |",
                  "", "指标仅在对应断言成功并提供观测字段时统计；失败样本详见下表，不将缺失观测当作零或成功。"])
    lines.extend(["", "## 失败详情", ""])
    lines.extend(f"- `{record['id']}`：{record.get('error', '')}" for record in records if not record["passed"])
    (args.report_dir / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Result: {summary['passed']}/{summary['count']}; {args.report_dir / 'latest.md'}")
    return 0 if summary["passed"] == summary["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
