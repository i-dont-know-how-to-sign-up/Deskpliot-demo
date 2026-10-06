from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskpilot import __version__
from deskpilot.core.config import load_config
from eval.run_eval import (
    DATASET,
    _dataset_sha256,
    _git_metadata,
    _load_suite,
    _safe_name,
    aggregate_results,
    aggregate_stage_usage,
    load_cases,
)


SUBSET_NAMES = {
    "DirectQA": "直接问答",
    "RAG-DocQA": "文档问答",
    "Intent-Routing": "意图路由",
    "Tool-Calling": "工具调用",
    "Web-Research": "网页调研",
    "Memory": "会话记忆",
    "Context-Engineering": "上下文工程",
    "Multi-Agent": "多智能体 P0",
    "Multi-Agent-P1": "多智能体 P1",
    "Composite-Workflow": "复合操作",
    "Safety-Permission": "安全权限",
    "Recovery": "故障恢复",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_streaming(
    command: list[str],
    log_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, float]:
    """在终端显示子任务进度，同时把完整输出保存到基线目录。"""
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        environment.update(env_overrides or {})
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait(), time.perf_counter() - started


def _run_test_modules(output_dir: Path) -> tuple[list[dict[str, Any]], int]:
    modules = sorted((ROOT / "tests").glob("test_*.py"))
    results: list[dict[str, Any]] = []
    failed = 0
    for index, path in enumerate(modules, start=1):
        module = f"tests.{path.stem}"
        print(f"\n[tests {index:02d}/{len(modules):02d}] {module}", flush=True)
        log_path = output_dir / "test_logs" / f"{path.stem}.log"
        returncode, duration = _run_streaming(
            [sys.executable, "-m", "pytest", "-q", str(path)],
            log_path,
        )
        summary = _pytest_counts(log_path.read_text(encoding="utf-8", errors="replace"))
        status = "passed" if returncode == 0 else "failed"
        failed += int(returncode != 0)
        results.append({
            "module": module,
            "status": status,
            "returncode": returncode,
            "duration_seconds": round(duration, 3),
            **summary,
        })
    return results, failed


def _pytest_counts(output: str) -> dict[str, int]:
    """从 pytest 汇总行提取真实收集/执行数，避免空模块被误记为通过。"""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for name in counts:
        matches = re.findall(rf"(\d+)\s+{name}\b", output, flags=re.IGNORECASE)
        if matches:
            counts[name] = int(matches[-1])
    return {
        "collected_tests": sum(counts.values()),
        "passed_tests": counts["passed"],
        "failed_tests": counts["failed"],
        "error_tests": counts["errors"],
        "skipped_tests": counts["skipped"],
    }


def _format_ms(value: float | None) -> str:
    return "不适用" if value is None else f"{value:.1f} ms"


def _delta(current: float | int | None, previous: float | int | None, suffix: str = "") -> str:
    if current is None or previous is None:
        return "不适用"
    value = float(current) - float(previous)
    return f"{value:+.4f}{suffix}" if not suffix else f"{value:+.1f}{suffix}"


def _comparison_lines(
    current_results: list[dict[str, Any]], previous_results: list[dict[str, Any]], previous_path: Path,
) -> list[str]:
    lines = [
        "", "## 7. 相对上一基线的差值", "", f"- 对比基线：`{previous_path}`", "",
        "| 范围 | Accuracy Δ | P95 Δ | API 调用 Δ | 输入 Token Δ | 输出 Token Δ | 总 Token Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    scopes = ["__all__"] + sorted({str(item.get("subset", "unknown")) for item in current_results + previous_results})
    for subset in scopes:
        current = current_results if subset == "__all__" else [item for item in current_results if item.get("subset") == subset]
        previous = previous_results if subset == "__all__" else [item for item in previous_results if item.get("subset") == subset]
        now, old = aggregate_results(current), aggregate_results(previous)
        label = "总计" if subset == "__all__" else SUBSET_NAMES.get(subset, subset)
        lines.append(
            f"| {label} | {_delta(now['accuracy'], old['accuracy'])} | "
            f"{_delta(now['latency_p95_ms'], old['latency_p95_ms'], ' ms')} | "
            f"{int(now['reported_calls']) - int(old['reported_calls']):+d} | "
            f"{int(now['prompt_tokens']) - int(old['prompt_tokens']):+d} | "
            f"{int(now['completion_tokens']) - int(old['completion_tokens']):+d} | "
            f"{int(now['total_tokens']) - int(old['total_tokens']):+d} |"
        )
    lines.extend(["", "> 延迟和 Token 负值表示下降；Accuracy 正值表示提升。用例集合变化时应结合执行数和 suite 说明解读。"])
    return lines


def _render_baseline_report(
    *,
    version: str,
    started_at: datetime,
    finished_at: datetime,
    changes: list[str],
    test_results: list[dict[str, Any]],
    offline_results: list[dict[str, Any]],
    api_results: list[dict[str, Any]],
    api_suite: str,
    commands: list[list[str]],
    previous_results: list[dict[str, Any]] | None = None,
    previous_path: Path | None = None,
) -> str:
    git = _git_metadata()
    config = load_config()
    offline = aggregate_results(offline_results)
    api = aggregate_results(api_results)
    combined_results = offline_results + api_results
    combined = aggregate_results(combined_results)
    tests_passed = sum(item["status"] == "passed" for item in test_results)
    tests_collected = sum(int(item.get("collected_tests", 0)) for item in test_results)
    tests_cases_passed = sum(int(item.get("passed_tests", 0)) for item in test_results)
    test_seconds = sum(float(item["duration_seconds"]) for item in test_results)
    dirty = "是" if git["dirty"] is True else "否" if git["dirty"] is False else "未知"
    lines = [
        f"# DeskPilot {version} 版本基线报告",
        "",
        "## 1. 基线信息",
        "",
        f"- 版本：`{version}`",
        f"- 测试开始：{started_at.isoformat(timespec='seconds')}",
        f"- 测试结束：{finished_at.isoformat(timespec='seconds')}",
        f"- 总墙钟时间：{(finished_at - started_at).total_seconds():.1f} 秒",
        f"- Git 提交：`{git['commit']}`",
        f"- 工作区存在未提交修改：{dirty}",
        f"- Python：`{sys.version.split()[0]}`",
        f"- LLM：`{config.llm_model}`",
        f"- Embedding：`{config.embedding_model}`",
        f"- 数据集 SHA-256：`{_dataset_sha256(DATASET)}`",
        f"- 精选 API 套件：`{api_suite}`",
        "- 真实 API 本地 fallback：禁用（API 或 Embedding 失败将记为错误）",
        "- 本版本修改：",
        *[f"  - {item}" for item in changes],
        "",
        "## 2. 总体结论",
        "",
        "| 项目 | 请求/模块数 | 实际执行 | 通过 | 失败 | 错误 | 跳过 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Python 测试模块 | {len(test_results)} | {len(test_results)} | {tests_passed} | {len(test_results) - tests_passed} | 0 | 0 |",
        f"| 离线评测 | {offline['requested']} | {offline['executed']} | {offline['passed']} | {offline['failed']} | {offline['errors']} | {offline['skipped']} |",
        f"| 精选真实 API 评测 | {api['requested']} | {api['executed']} | {api['passed']} | {api['failed']} | {api['errors']} | {api['skipped']} |",
        "",
        f"> 测试模块累计耗时为 {test_seconds:.1f} 秒。离线模式按设计会跳过必须使用 LLM/联网的用例，跳过不等于失败。",
        f"> pytest 共收集 {tests_collected} 条测试，其中通过 {tests_cases_passed} 条；模块通过不再替代真实测试条数。",
        "",
        "## 3. 总体效率与 Token",
        "",
        "| 范围 | 执行数 | 累计响应时间 | 平均响应 | P50 | P95 | API 调用 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (("离线评测", offline), ("真实 API 评测", api), ("评测总计", combined)):
        prompt = str(stats["prompt_tokens"]) if stats["reported_calls"] else "不适用"
        completion = str(stats["completion_tokens"]) if stats["reported_calls"] else "不适用"
        total = str(stats["total_tokens"]) if stats["reported_calls"] else "不适用"
        lines.append(
            f"| {label} | {stats['executed']} | {stats['latency_total_ms']:.1f} ms | "
            f"{_format_ms(stats['latency_mean_ms'])} | {_format_ms(stats['latency_p50_ms'])} | "
            f"{_format_ms(stats['latency_p95_ms'])} | {stats['reported_calls']} | {prompt} | {completion} | {total} |"
        )
    lines.extend([
        "",
        "> Token 只统计模型服务端 `usage` 字段返回的真实数值；服务端未返回时不使用字符数估算。评测总计是本次离线与 API 两批工作负载的总和，不代表单个用户请求延迟。",
        "",
        "## 4. 按功能模块统计",
        "",
        "| 模块 | 执行数 | 通过率 | 累计响应时间 | 平均响应 | P50 | P95 | API 调用 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for subset in sorted({str(item.get("subset", "unknown")) for item in combined_results}):
        stats = aggregate_results([item for item in combined_results if item.get("subset") == subset])
        prompt = str(stats["prompt_tokens"]) if stats["reported_calls"] else "不适用"
        completion = str(stats["completion_tokens"]) if stats["reported_calls"] else "不适用"
        total = str(stats["total_tokens"]) if stats["reported_calls"] else "不适用"
        lines.append(
            f"| {SUBSET_NAMES.get(subset, subset)} | {stats['executed']} | {stats['accuracy']:.4f} | "
            f"{stats['latency_total_ms']:.1f} ms | {_format_ms(stats['latency_mean_ms'])} | "
            f"{_format_ms(stats['latency_p50_ms'])} | {_format_ms(stats['latency_p95_ms'])} | "
            f"{stats['reported_calls']} | {prompt} | {completion} | {total} |"
        )
    stage_stats = aggregate_stage_usage(combined_results)
    lines.extend([
        "",
        "## 5. 模型调用阶段归因",
        "",
        "| 阶段 | 调用数 | Usage 上报 | 错误 | 平均响应 | P50 | P95 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if not stage_stats:
        lines.append("| 无服务端调用记录 | 0 | 0 | 0 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |")
    for stage, stats in sorted(stage_stats.items()):
        lines.append(
            f"| {stage} | {stats['calls']} | {stats['reported_calls']} | {stats['errors']} | "
            f"{_format_ms(stats['latency_mean_ms'])} | {_format_ms(stats['latency_p50_ms'])} | "
            f"{_format_ms(stats['latency_p95_ms'])} | {stats['prompt_tokens']} | "
            f"{stats['completion_tokens']} | {stats['total_tokens']} |"
        )
    lines.extend([
        "",
        "## 6. 功能模块与调用阶段交叉统计",
        "",
        "| 模块 | 阶段 | 调用数 | 平均响应 | P95 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subset in sorted({str(item.get("subset", "unknown")) for item in combined_results}):
        selected = [item for item in combined_results if item.get("subset") == subset]
        for stage, stats in sorted(aggregate_stage_usage(selected).items()):
            lines.append(
                f"| {SUBSET_NAMES.get(subset, subset)} | {stage} | {stats['calls']} | "
                f"{_format_ms(stats['latency_mean_ms'])} | {_format_ms(stats['latency_p95_ms'])} | "
                f"{stats['prompt_tokens']} | {stats['completion_tokens']} | {stats['total_tokens']} |"
            )
    if previous_results is not None and previous_path is not None:
        lines.extend(_comparison_lines(combined_results, previous_results, previous_path))
    lines.extend([
        "", "## 8. 测试模块明细", "",
        "| 模块 | 状态 | 耗时 |", "|---|---|---:|",
    ])
    for item in test_results:
        lines.append(f"| `{item['module']}` | {item['status']} | {item['duration_seconds']:.3f} s |")
    lines.extend(["", "## 9. 失败与错误用例", ""])
    failures = [item for item in combined_results if item.get("status") in {"failed", "error"}]
    if not failures:
        lines.append("本次评测没有失败或运行时错误。")
    else:
        for item in failures:
            lines.append(
                f"- `{item.get('case_id')}` / {SUBSET_NAMES.get(str(item.get('subset')), item.get('subset'))} / "
                f"{item.get('status')} / score={item.get('score', 0)}"
            )
            if item.get("error"):
                lines.append(f"  - 错误：{item['error']}")
    lines.extend(["", "## 10. 复现命令", ""])
    for command in commands:
        lines.extend(["```powershell", subprocess.list2cmdline(command), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 DeskPilot 完整版本基线。")
    parser.add_argument("--project-version", default=__version__)
    parser.add_argument("--api-suite", default="baseline_api_v0.8.0")
    parser.add_argument(
        "--change-summary", action="append", required=True,
        help="当前版本相对上一基线的修改，可重复传入。首个基线可写‘建立首个规范化基线’。",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "eval" / "baselines")
    parser.add_argument(
        "--compare-baseline", type=Path,
        help="上一基线目录；将读取其中 offline_results.jsonl 和 api_results.jsonl 生成差值表。",
    )
    args = parser.parse_args()
    changes = list(args.change_summary)
    suite = _load_suite(args.api_suite)
    dataset_cases = load_cases(DATASET)
    dataset_ids = {str(case.get("id")) for case in dataset_cases}
    missing_suite_ids = [str(item) for item in suite["case_ids"] if str(item) not in dataset_ids]
    if missing_suite_ids:
        parser.error(f"API 套件包含不存在的用例：{', '.join(missing_suite_ids)}")
    config = load_config()
    if importlib.util.find_spec("pytest") is None:
        parser.error("完整基线需要 pytest，请先执行 python -m pip install -r requirements.txt。")
    if not config.llm_api_key or not config.embedding_api_key:
        parser.error("完整基线包含真实 API 用例，请先在 .env 配置 LLM 和 Embedding API Key。")

    started = datetime.now().astimezone()
    timestamp = started.strftime("%Y%m%dT%H%M%S%z")
    version_slug = _safe_name(args.project_version).lstrip("v")
    run_stem = f"deskpilot_baseline_v{version_slug}_{timestamp}"
    output_dir = args.output_root / run_stem
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Baseline output: {output_dir}", flush=True)
    test_results, test_failures = _run_test_modules(output_dir)
    (output_dir / "test_modules.json").write_text(
        json.dumps(test_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    shared = [
        "--project-version", args.project_version,
        "--change-summary", changes[0],
    ]
    for extra in changes[1:]:
        shared.extend(["--change-summary", extra])
    offline_command = [
        sys.executable, "-m", "eval.run_eval", "--count", str(len(dataset_cases)), "--mode", "offline",
        "--run-name", f"{run_stem}-offline",
        "--output", str(output_dir / "offline_results.jsonl"),
        "--report", str(output_dir / "offline_report.md"),
        "--metadata", str(output_dir / "offline_metadata.json"),
        *shared,
    ]
    print(f"\n[baseline] 运行 {len(dataset_cases)} 条离线评测", flush=True)
    offline_code, _ = _run_streaming(offline_command, output_dir / "offline_console.log")

    api_command = [
        sys.executable, "-m", "eval.run_eval", "--suite", args.api_suite, "--mode", "api",
        "--run-name", f"{run_stem}-api",
        "--output", str(output_dir / "api_results.jsonl"),
        "--report", str(output_dir / "api_report.md"),
        "--metadata", str(output_dir / "api_metadata.json"),
        *shared,
    ]
    print("\n[baseline] 运行精选真实 API 评测", flush=True)
    api_code, _ = _run_streaming(
        api_command,
        output_dir / "api_console.log",
        env_overrides={"ALLOW_LOCAL_FALLBACK": "false"},
    )

    finished = datetime.now().astimezone()
    offline_results = _load_jsonl(output_dir / "offline_results.jsonl")
    api_results = _load_jsonl(output_dir / "api_results.jsonl")
    previous_results: list[dict[str, Any]] | None = None
    if args.compare_baseline:
        if not args.compare_baseline.is_dir():
            parser.error(f"对比基线目录不存在：{args.compare_baseline}")
        previous_results = (
            _load_jsonl(args.compare_baseline / "offline_results.jsonl")
            + _load_jsonl(args.compare_baseline / "api_results.jsonl")
        )
        if not previous_results:
            parser.error(f"对比基线缺少可读取的原始结果：{args.compare_baseline}")
    report_path = output_dir / f"{run_stem}.md"
    report_path.write_text(
        _render_baseline_report(
            version=args.project_version,
            started_at=started,
            finished_at=finished,
            changes=changes,
            test_results=test_results,
            offline_results=offline_results,
            api_results=api_results,
            api_suite=args.api_suite,
            commands=[offline_command, api_command],
            previous_results=previous_results,
            previous_path=args.compare_baseline,
        ),
        encoding="utf-8",
    )
    manifest = {
        "project_version": args.project_version,
        "run_stem": run_stem,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "changes": changes,
        "api_suite": args.api_suite,
        "api_case_ids": [str(item) for item in suite["case_ids"]],
        "test_module_count": len(test_results),
        "test_failures": test_failures,
        "offline_returncode": offline_code,
        "api_returncode": api_code,
        "report": report_path.name,
        "compare_baseline": str(args.compare_baseline or ""),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nBaseline report: {report_path}", flush=True)
    return 0 if test_failures == 0 and offline_code == 0 and api_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
