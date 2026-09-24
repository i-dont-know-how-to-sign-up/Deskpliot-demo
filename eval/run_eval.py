from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import OpenAICompatibleClient, local_hash_embedding
from deskpilot.memory.memory_compactor import MemoryCompactor
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.rag.web_research import ResearchResult
from deskpilot.tools.tool_registry import build_default_tool_registry

from .scoring import score_case

DATASET = Path(__file__).parent / "dataset" / "deskpilot_bench.jsonl"
FIXTURES = Path(__file__).parent / "dataset" / "fixtures"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def make_agent(temp_root: Path) -> DocumentQAAgent:
    workspace = temp_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["TOOL_SAFE_ROOTS"] = str(workspace)
    agent = DocumentQAAgent(DocumentIndex(temp_root / "index.json"))
    agent.workspace_root = workspace
    agent.tool_registry = build_default_tool_registry(
        agent.index, agent.web_research_agent, workspace_root=workspace,
    )
    agent.session_store = SessionStore(temp_root / "sessions")
    agent.memory_store = MemoryStore(
        temp_root / "memory.sqlite",
        temp_root / "memory_workspace",
        vector_provider="sqlite",
    )
    agent.memory_compactor = MemoryCompactor(agent.session_store)
    return agent


def prepare_case(case: dict[str, Any], temp_root: Path, agent: DocumentQAAgent) -> None:
    target = temp_root / "workspace"
    for relative in case.get("setup", {}).get("index_files", []):
        source = (ROOT / relative).resolve()
        allowed = ((FIXTURES / "docs").resolve(), (ROOT / "多模态").resolve())
        if not any(source.is_relative_to(root) for root in allowed) or not source.is_file():
            raise ValueError(f"Invalid or missing index fixture: {relative}")
        destination = (target / relative).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise ValueError(f"Index fixture escapes workspace: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        agent.index.add_file(destination)
    fixture = FIXTURES / "workspace"
    if fixture.exists():
        shutil.copytree(fixture, target, dirs_exist_ok=True)
        shutil.copytree(fixture, target / "eval" / "dataset" / "fixtures" / "workspace", dirs_exist_ok=True)
    # 显式命令行目标也放进隔离工作区，不执行仓库中的真实脚本。
    script_fixture = FIXTURES / "check_sample.py"
    if script_fixture.is_file():
        script_target = target / "eval" / "dataset" / "fixtures" / "check_sample.py"
        script_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_fixture, script_target)
    for relative in case.get("setup", {}).get("workspace_files", []):
        source = (FIXTURES / "workspace" / relative).resolve()
        base = (FIXTURES / "workspace").resolve()
        if not source.is_relative_to(base) or not source.is_file():
            raise ValueError(f"Invalid workspace fixture: {relative}")
        destination = (target / relative).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise ValueError(f"Invalid workspace target: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def install_case_mocks(case: dict[str, Any], temp_root: Path, agent: DocumentQAAgent, stack: ExitStack, mode: str) -> None:
    setup = case.get("setup", {})
    if mode == "offline":
        stack.enter_context(patch.object(OpenAICompatibleClient, "chat", return_value=""))
        stack.enter_context(patch.object(
            OpenAICompatibleClient, "embed", side_effect=lambda texts: [local_hash_embedding(text) for text in texts]
        ))
        stack.enter_context(patch.object(agent.web_research_agent, "research", side_effect=lambda topic: ResearchResult(
            topic, "", [], [], [], "", False,
        )))
        for name in ("web.search", "web.read_page"):
            spec = agent.tool_registry.get_tool_spec(name)
            if spec:
                agent.tool_registry.register(replace(spec, handler=lambda **kwargs: []))
        for name in ("email.list_messages", "email.read_thread", "email.search", "email.classify",
                     "email.summarize_thread", "email.create_reply_draft", "email.save_draft", "email.send"):
            spec = agent.tool_registry.get_tool_spec(name)
            if spec:
                agent.tool_registry.register(replace(spec, handler=lambda **kwargs: {
                    "offline": True, "message": "External mail services are disabled in offline evaluation."
                }))
    if "mock_search_results" in setup:
        spec = agent.tool_registry.get_tool_spec("web.search")
        agent.tool_registry.register(replace(spec, handler=lambda query, limit=5: setup["mock_search_results"][:int(limit)]))
    if "mock_research_report" in setup:
        def research(topic: str) -> ResearchResult:
            path = temp_root / "workspace" / "mock_research.md"
            report = str(setup["mock_research_report"])
            path.write_text(report, encoding="utf-8")
            return ResearchResult(topic, report, [], [], [], str(path), False)
        stack.enter_context(patch.object(agent.web_research_agent, "research", side_effect=research))
    if setup.get("mock_email_confirmation"):
        # 模拟权限层的待确认响应；严禁连接真实邮箱或代替用户批准。
        for name in ("email.send", "email.save_draft"):
            spec = agent.tool_registry.get_tool_spec(name)
            agent.tool_registry.register(replace(spec, handler=lambda **kwargs: {
                "permission": {"requires_confirmation": True, "blocked": False, "risk_level": "high"},
                "sent": False, "saved": False,
            }))
    if setup.get("mock_desktop"):
        desktop = temp_root / "Desktop"
        desktop.mkdir(exist_ok=True)
        stack.enter_context(patch.object(agent, "_desktop_directory", return_value=desktop))


def run_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    if case.get("runtime", {}).get("requires_online") and mode != "online":
        return score_case(case, None, "requires_online; use --mode online")
    if case.get("runtime", {}).get("requires_llm") and mode == "offline":
        return score_case(case, None, "requires_llm; use --mode api")
    if case.get("runtime", {}).get("requires_local_files"):
        missing = [item for item in case["setup"]["index_files"] if not (ROOT / item).is_file()]
        if missing:
            return score_case(case, None, f"requires_local_files; missing {missing}")

    with tempfile.TemporaryDirectory(prefix="deskpilot_eval_") as raw:
        temp_root = Path(raw)
        old_cwd = Path.cwd()
        old_safe_roots = os.environ.get("TOOL_SAFE_ROOTS")
        started = time.perf_counter()
        os.chdir(temp_root)
        try:
            agent = make_agent(temp_root)
            with ExitStack() as stack:
                install_case_mocks(case, temp_root, agent, stack, mode)
                prepare_case(case, temp_root, agent)
                os.chdir(temp_root / "workspace")
                messages = case.get("conversation") or [case.get("input", "")]
                session_id = None
                result = None
                for message in messages:
                    result = agent.answer(str(message), session_id=session_id)
                    session_id = result.session_id
            assert result is not None
            pending_arguments = None
            pending = result.pending_action or {}
            if pending.get("action_id"):
                item = agent.pending_actions.inspect(str(pending["action_id"]), result.session_id)
                pending_arguments = dict(item.kwargs) if item is not None else {}
            scored = score_case(
                case,
                result,
                workspace=temp_root / "workspace",
                pending_arguments=pending_arguments,
            )
            scored["latency_ms"] = round((time.perf_counter() - started) * 1000)
            clients = (agent.client, agent.index.client)
            usages = [client.token_usage for client in clients]
            scored["token_usage"] = {
                key: sum(int(usage.get(key, 0)) for usage in usages)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reported_calls")
            }
            return scored
        except Exception as exc:
            return {
                "case_id": case["id"],
                "subset": case["subset"],
                "status": "error",
                "score": 0.0,
                "error": repr(exc),
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        finally:
            os.chdir(old_cwd)
            if old_safe_roots is None:
                os.environ.pop("TOOL_SAFE_ROOTS", None)
            else:
                os.environ["TOOL_SAFE_ROOTS"] = old_safe_roots


def render_report(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    subset_names = {"DirectQA": "直接问答", "RAG-DocQA": "文档问答", "Intent-Routing": "意图路由", "Tool-Calling": "工具调用", "Web-Research": "网页调研", "Memory": "会话记忆", "Context-Engineering": "上下文工程", "Multi-Agent": "多智能体 P0", "Multi-Agent-P1": "多智能体 P1", "Composite-Workflow": "复合操作", "Safety-Permission": "安全权限", "Recovery": "故障恢复"}
    status_names = {"passed": "通过", "failed": "失败", "error": "错误", "skipped": "跳过"}
    executed = [item for item in results if item["status"] != "skipped"]
    passed = [item for item in executed if item["status"] == "passed"]
    average = sum(float(item.get("score", 0)) for item in executed) / len(executed) if executed else 0
    errors = [item for item in executed if item["status"] == "error"]
    latencies = sorted(float(item.get("latency_ms", 0)) for item in executed)
    reference_results = [
        item for item in executed if item.get("metrics", {}).get("exact_match") is not None
    ]
    recovery_results = [item for item in executed if item.get("subset") == "Recovery"]
    token_usages = [item.get("token_usage", {}) for item in executed]
    reported_calls = sum(int(item.get("reported_calls", 0)) for item in token_usages)

    def percentile(values: list[float], percent: float) -> float | None:
        if not values:
            return None
        return values[max(0, min(len(values) - 1, ceil(percent * len(values)) - 1))]

    def average_metric(name: str, selected: list[dict[str, Any]]) -> float | None:
        values = [item.get("metrics", {}).get(name) for item in selected]
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else None

    accuracy = len(passed) / len(executed) if executed else 0.0
    error_rate = len(errors) / len(executed) if executed else 0.0
    recovery_success = (
        sum(item.get("status") == "passed" for item in recovery_results) / len(recovery_results)
        if recovery_results
        else None
    )
    exact_match = average_metric("exact_match", reference_results)
    f1 = average_metric("f1", reference_results)
    communication_efficiency = average_metric("communication_efficiency_proxy", executed)
    lines = [
        "# DeskPilotBench 评测报告",
        "",
        f"- 数据集：{args.dataset}",
        f"- 评测模式：{args.mode}",
        f"- 请求用例数：{len(results)}",
        f"- 实际执行：{len(executed)}",
        f"- 跳过：{len(results) - len(executed)}",
        f"- 通过：{len(passed)}",
        f"- 平均任务完成度：{average:.4f}",
        f"- 准确率：{accuracy:.4f}",
        f"- 错误率：{error_rate:.4f}",
        "",
        "## 指标汇总",
        "",
        "| 维度 | 指标 | 数值 | 覆盖数 |",
        "|---|---|---:|---:|",
        f"| 准确性 | 准确率 | {accuracy:.4f} | {len(executed)} |",
        f"| 准确性 | 精确匹配 | {exact_match:.4f} | {len(reference_results)} |" if exact_match is not None else "| 准确性 | 精确匹配 | 不适用 | 0 |",
        f"| 准确性 | Token F1 | {f1:.4f} | {len(reference_results)} |" if f1 is not None else "| 准确性 | Token F1 | 不适用 | 0 |",
        f"| 效率 | 平均响应时间 | {(sum(latencies) / len(latencies)):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | 平均响应时间 | 不适用 | 0 |",
        f"| 效率 | P50 响应时间 | {percentile(latencies, 0.50):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | P50 响应时间 | 不适用 | 0 |",
        f"| 效率 | P95 响应时间 | {percentile(latencies, 0.95):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | P95 响应时间 | 不适用 | 0 |",
        f"| 效率 | 输入 Token | {sum(int(item.get('prompt_tokens', 0)) for item in token_usages)} | {reported_calls} 次 API 调用 |" if reported_calls else "| 效率 | 输入 Token | 不适用 | 0 次 API 调用 |",
        f"| 效率 | 输出 Token | {sum(int(item.get('completion_tokens', 0)) for item in token_usages)} | {reported_calls} 次 API 调用 |" if reported_calls else "| 效率 | 输出 Token | 不适用 | 0 次 API 调用 |",
        f"| 效率 | 总 Token | {sum(int(item.get('total_tokens', 0)) for item in token_usages)} | {reported_calls} 次 API 调用 |" if reported_calls else "| 效率 | 总 Token | 不适用 | 0 次 API 调用 |",
        f"| 鲁棒性 | 错误率 | {error_rate:.4f} | {len(executed)} |",
        f"| 鲁棒性 | 故障恢复率 | {recovery_success:.4f} | {len(recovery_results)} |" if recovery_success is not None else "| 鲁棒性 | 故障恢复率 | 不适用 | 0 |",
        f"| 协作 | 任务完成度 | {average:.4f} | {len(executed)} |",
        f"| 协作 | 通信效率（代理指标） | {communication_efficiency:.4f} | {len(executed)} |" if communication_efficiency is not None else "| 协作 | 通信效率（代理指标） | 不适用 | 0 |",
        "",
        "> 精确匹配和 Token F1 只统计带有 `expected.reference_answer(s)` 的用例。当前系统是单 Agent，通信效率为重试/失败惩罚代理指标，不代表多智能体通信质量。",
        "",
        "## 分模块统计",
        "",
        "| 模块 | 用例数 | 通过数 | 平均任务完成度 |",
        "|---|---:|---:|---:|",
    ]
    for subset in sorted({item["subset"] for item in results}):
        selected = [item for item in results if item["subset"] == subset]
        run = [item for item in selected if item["status"] != "skipped"]
        ok = [item for item in run if item["status"] == "passed"]
        avg = sum(float(item.get("score", 0)) for item in run) / len(run) if run else 0
        lines.append(f"| {subset_names.get(subset, subset)} | {len(selected)} | {len(ok)} | {avg:.4f} |")
    lines.extend(["", "## 用例明细", ""])
    for item in results:
        lines.append(f"- {item['case_id']} [{status_names.get(item['status'], item['status'])}] 任务完成度={item.get('score', 0)}")
        if item.get("error"):
            lines.append(f"  - error: {item['error']}")
    return "\n".join(lines) + "\n"


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """实时保存结果，避免长时间评测时只能等到最后才看到进展。"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results, args), encoding="utf-8")


def print_progress(
    completed: int,
    total: int,
    case_id: str,
    status: str,
    elapsed_seconds: float,
    *,
    final: bool = False,
) -> None:
    """输出不依赖第三方包的进度条，兼容 Windows PowerShell。"""
    width = 30
    ratio = completed / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "." * (width - filled)
    line = (
        f"[{bar}] {completed:>3}/{total:<3} "
        f"{status:<7} {case_id:<20} elapsed={elapsed_seconds:>7.1f}s"
    )
    # 当前用例开始时覆盖同一行；完成时换行，保留每条用例的最终状态。
    end = "\n" if final else ""
    print("\r" + line[: max(1, 120)], end=end, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the extensible DeskPilotBench dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--subset", action="append")
    parser.add_argument("--case-id", action="append", help="只执行指定用例 ID，可重复传入。")
    parser.add_argument("--mode", choices=["offline", "api", "online"], default="offline")
    parser.add_argument("--output", type=Path, default=Path("eval/runs/latest.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/reports/latest.md"))
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable the live progress bar",
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case.get("id") in selected_ids]
    if args.subset:
        allowed = set(args.subset)
        cases = [case for case in cases if case.get("subset") in allowed]
    cases = cases[: max(0, args.count)]
    results: list[dict[str, Any]] = []
    total = len(cases)
    if args.progress:
        print(f"DeskPilotBench: {total} cases, mode={args.mode}", flush=True)
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{index}"))
        started = time.perf_counter()
        if args.progress:
            print_progress(index - 1, total, case_id, "running", 0.0)
        result = run_case(case, args.mode)
        results.append(result)
        elapsed = (time.perf_counter() - started)
        # 每条用例完成后马上落盘，Ctrl+C 或单条卡住时仍能保留已完成结果。
        write_outputs(results, args)
        if args.progress:
            print_progress(
                index,
                total,
                case_id,
                str(result.get("status", "unknown")),
                elapsed,
                final=True,
            )
    if not cases:
        write_outputs(results, args)
    print(render_report(results, args))
    return 0 if all(item["status"] in {"passed", "skipped"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
