from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import OpenAICompatibleClient, local_hash_embedding
from deskpilot.core.config import load_config
from deskpilot.memory.memory_compactor import MemoryCompactor
from deskpilot.memory.memory_store import MemoryStore
from deskpilot.memory.session_store import SessionStore
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.rag.web_research import ResearchResult, WebSearchClient
from deskpilot.tools.tool_registry import build_default_tool_registry
from deskpilot import __version__

from .scoring import score_case
from .api_health import circuit_breaker_results, is_infrastructure_error, run_api_preflight

DATASET = Path(__file__).parent / "dataset" / "deskpilot_bench.jsonl"
FIXTURES = Path(__file__).parent / "dataset" / "fixtures"
SUITES_DIR = Path(__file__).parent / "suites"


def _safe_name(value: str) -> str:
    """生成可在 Windows/Linux 使用的报告文件名片段。"""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return normalized.strip("-._") or "unknown"


def _now() -> datetime:
    return datetime.now().astimezone()


def _git_metadata() -> dict[str, Any]:
    """只采集可复现信息，不读取远端地址、用户名或凭证。"""
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        ).stdout
        return {"commit": commit or "unknown", "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def _dataset_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _load_suite(name: str) -> dict[str, Any]:
    path = SUITES_DIR / f"{_safe_name(name)}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in SUITES_DIR.glob("*.json"))) or "none"
        raise ValueError(f"Unknown evaluation suite '{name}'. Available: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("case_ids"), list):
        raise ValueError(f"Invalid evaluation suite: {path}")
    return data


def _prepare_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    started = getattr(args, "started_at", _now())
    git = _git_metadata()
    metadata = {
        "project": "DeskPilot",
        "project_version": args.project_version,
        "run_name": args.run_name,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": _now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "dataset": str(args.dataset),
        "dataset_sha256": _dataset_sha256(args.dataset),
        "suite": args.suite or "",
        "changes": list(args.change_summary or ["未提供变更说明"]),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "llm_configured": bool(config.llm_api_key),
        "embedding_configured": bool(config.embedding_api_key),
        "allow_local_fallback": config.allow_local_fallback,
    }
    preflight = getattr(args, "api_preflight", None)
    if isinstance(preflight, dict):
        metadata["api_preflight"] = preflight
    return metadata


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for case in cases:
        runtime = case.setdefault("runtime", {})
        runtime.setdefault("requires_llm", False)
        runtime.setdefault("requires_network", bool(runtime.get("requires_online", False)))
        runtime.setdefault("requires_email", False)
        expected = case.get("expected", {})
        inferred_side_effect = bool(
            expected.get("requires_confirmation")
            or expected.get("pending_tool")
            or (isinstance(expected.get("artifact"), dict) and expected["artifact"].get("exists", True))
        )
        runtime.setdefault("has_side_effect", inferred_side_effect)
    return cases


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
    runtime = case.get("runtime", {})
    if mode == "api" and not (runtime.get("requires_network") or runtime.get("requires_online")):
        def reject_network(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("UNDECLARED_EXTERNAL_NETWORK: fixed API evaluation forbids real web access")

        stack.enter_context(patch.object(WebSearchClient, "search", side_effect=reject_network))
        stack.enter_context(patch.object(WebSearchClient, "fetch_page", side_effect=reject_network))
    if mode == "offline":
        stack.enter_context(patch.object(OpenAICompatibleClient, "chat", return_value=""))
        stack.enter_context(patch.object(
            OpenAICompatibleClient, "embed", side_effect=lambda texts: [local_hash_embedding(text) for text in texts]
        ))
        stack.enter_context(patch.object(agent.web_research_agent, "research", side_effect=lambda topic, max_results=5: ResearchResult(
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
        def research(topic: str, max_results: int = 5) -> ResearchResult:
            path = temp_root / "workspace" / "mock_research.md"
            report = str(setup["mock_research_report"])
            path.write_text(report, encoding="utf-8")
            return ResearchResult(topic, report, [], [], [], str(path), False)
        stack.enter_context(patch.object(agent.web_research_agent, "research", side_effect=research))
        # Planner 可以在同一个 knowledge 节点中合法选择 web.search 或 web.research。
        # 两个入口必须共享同一份固定资料，避免漏 mock 后误访问真实网络。
        report = str(setup["mock_research_report"])
        research_spec = agent.tool_registry.get_tool_spec("web.research")
        if research_spec:
            agent.tool_registry.register(replace(research_spec, handler=lambda topic, max_results=5: report))
        search_spec = agent.tool_registry.get_tool_spec("web.search")
        if search_spec:
            agent.tool_registry.register(replace(search_spec, handler=lambda query, limit=5: [{
                "title": "固定评测资料",
                "url": "https://example.org/mock-research",
                "snippet": report,
            }]))
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
    runtime = case.get("runtime", {})
    requires_network = bool(runtime.get("requires_network") or runtime.get("requires_online"))
    if requires_network and mode != "online":
        return score_case(case, None, "requires_network; use --mode online")
    if runtime.get("requires_llm") and mode == "offline":
        return score_case(case, None, "requires_llm; use --mode api")
    if runtime.get("requires_email") and mode != "online" and not case.get("setup", {}).get("mock_email_confirmation"):
        return score_case(case, None, "requires_email; use --mode online or provide an email mock")
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
            scored["token_usage"] = agent._combined_usage()
            scored["usage_events"] = agent.usage_events()
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


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, ceil(percent * len(ordered)) - 1))]


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按统一口径聚合质量、延迟和服务端报告的真实 Token。"""
    executed = [item for item in results if item.get("status") != "skipped"]
    passed = [item for item in executed if item.get("status") == "passed"]
    errors = [item for item in executed if item.get("status") == "error"]
    latencies = [float(item.get("latency_ms", 0)) for item in executed]
    usages = [item.get("token_usage", {}) for item in executed]
    reported_calls = sum(int(item.get("reported_calls", 0)) for item in usages)
    return {
        "requested": len(results),
        "executed": len(executed),
        "skipped": len(results) - len(executed),
        "passed": len(passed),
        "failed": sum(item.get("status") == "failed" for item in executed),
        "errors": len(errors),
        "accuracy": len(passed) / len(executed) if executed else 0.0,
        "average_score": (
            sum(float(item.get("score", 0)) for item in executed) / len(executed) if executed else 0.0
        ),
        "latency_total_ms": sum(latencies),
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else None,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usages),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usages),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usages),
        "reported_calls": reported_calls,
    }


def aggregate_stage_usage(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按模型调用阶段聚合调用数、真实 Token 和调用自身延迟。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") == "skipped":
            continue
        for event in result.get("usage_events", []):
            if isinstance(event, dict):
                grouped.setdefault(str(event.get("stage") or "Unattributed"), []).append(event)
    summary: dict[str, dict[str, Any]] = {}
    for stage, events in grouped.items():
        latencies = [float(item.get("latency_ms", 0)) for item in events]
        summary[stage] = {
            "calls": len(events),
            "reported_calls": sum(bool(item.get("usage_reported")) for item in events),
            "errors": sum(not bool(item.get("success", True)) for item in events),
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in events),
            "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in events),
            "total_tokens": sum(int(item.get("total_tokens", 0)) for item in events),
            "latency_mean_ms": sum(latencies) / len(latencies) if latencies else None,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }
    return summary


def render_report(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    subset_names = {"DirectQA": "直接问答", "RAG-DocQA": "文档问答", "Intent-Routing": "意图路由", "Tool-Calling": "工具调用", "Web-Research": "网页调研", "Memory": "会话记忆", "Context-Engineering": "上下文工程", "Multi-Agent": "多智能体 P0", "Multi-Agent-P1": "多智能体 P1", "Composite-Workflow": "复合操作", "Safety-Permission": "安全权限", "Recovery": "故障恢复"}
    status_names = {"passed": "通过", "failed": "失败", "error": "错误", "skipped": "跳过"}
    metadata = _prepare_run_metadata(args)
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
    change_lines = [f"  - {item}" for item in metadata["changes"]]
    dirty_text = "是" if metadata["git_dirty"] is True else "否" if metadata["git_dirty"] is False else "未知"
    lines = [
        f"# DeskPilotBench {metadata['project_version']} 评测报告",
        "",
        "## 基线元数据",
        "",
        f"- 运行名称：{metadata['run_name']}",
        f"- DeskPilot 版本：{metadata['project_version']}",
        f"- 测试开始时间：{metadata['started_at']}",
        f"- 报告更新时间：{metadata['finished_at']}",
        f"- Git 提交：{metadata['git_commit']}（工作区有未提交改动：{dirty_text}）",
        f"- Python：{metadata['python_version']}",
        f"- LLM：{metadata['llm_model']}（已配置：{'是' if metadata['llm_configured'] else '否'}）",
        f"- Embedding：{metadata['embedding_model']}（已配置：{'是' if metadata['embedding_configured'] else '否'}）",
        f"- 本地模型降级：{'允许' if metadata['allow_local_fallback'] else '禁用'}",
        f"- 数据集：{args.dataset}",
        f"- 数据集 SHA-256：`{metadata['dataset_sha256']}`",
        f"- 评测套件：{metadata['suite'] or '未指定'}",
        f"- 评测模式：{args.mode}",
        "- 本版本修改：",
        *change_lines,
        "",
        "## 总体结果",
        "",
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
        f"| 效率 | P50 响应时间 | {_percentile(latencies, 0.50):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | P50 响应时间 | 不适用 | 0 |",
        f"| 效率 | P95 响应时间 | {_percentile(latencies, 0.95):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | P95 响应时间 | 不适用 | 0 |",
        f"| 效率 | 累计响应时间 | {sum(latencies):.1f} ms | {len(latencies)} |" if latencies else "| 效率 | 累计响应时间 | 不适用 | 0 |",
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
        "## 分模块质量统计",
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
    difficulties = sorted({str(item.get("difficulty", "unspecified")) for item in results})
    lines.extend([
        "", "## 分难度质量统计", "",
        "| 难度 | 用例数 | 执行数 | 通过数 | 准确率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for difficulty in difficulties:
        selected = [item for item in results if str(item.get("difficulty", "unspecified")) == difficulty]
        stats = aggregate_results(selected)
        lines.append(
            f"| {difficulty} | {len(selected)} | {stats['executed']} | "
            f"{stats['passed']} | {stats['accuracy']:.4f} |"
        )
    lines.extend([
        "",
        "## 分模块效率统计",
        "",
        "| 模块 | 执行数 | 平均响应 | P50 | P95 | API 调用 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    module_stats: dict[str, dict[str, Any]] = {}
    for subset in sorted({item["subset"] for item in results}):
        selected = [item for item in results if item["subset"] == subset]
        stats = aggregate_results(selected)
        module_stats[subset] = stats
        mean = f"{stats['latency_mean_ms']:.1f} ms" if stats["latency_mean_ms"] is not None else "不适用"
        p50 = f"{stats['latency_p50_ms']:.1f} ms" if stats["latency_p50_ms"] is not None else "不适用"
        p95 = f"{stats['latency_p95_ms']:.1f} ms" if stats["latency_p95_ms"] is not None else "不适用"
        token_values = (
            (str(stats["prompt_tokens"]), str(stats["completion_tokens"]), str(stats["total_tokens"]))
            if stats["reported_calls"] else ("不适用", "不适用", "不适用")
        )
        lines.append(
            f"| {subset_names.get(subset, subset)} | {stats['executed']} | {mean} | {p50} | {p95} | "
            f"{stats['reported_calls']} | {token_values[0]} | {token_values[1]} | {token_values[2]} |"
        )
    total_stats = aggregate_results(results)
    total_mean = f"{total_stats['latency_mean_ms']:.1f} ms" if total_stats["latency_mean_ms"] is not None else "不适用"
    total_p50 = f"{total_stats['latency_p50_ms']:.1f} ms" if total_stats["latency_p50_ms"] is not None else "不适用"
    total_p95 = f"{total_stats['latency_p95_ms']:.1f} ms" if total_stats["latency_p95_ms"] is not None else "不适用"
    total_tokens = (
        (str(total_stats["prompt_tokens"]), str(total_stats["completion_tokens"]), str(total_stats["total_tokens"]))
        if total_stats["reported_calls"] else ("不适用", "不适用", "不适用")
    )
    lines.append(
        f"| **总计** | **{total_stats['executed']}** | **{total_mean}** | **{total_p50}** | **{total_p95}** | "
        f"**{total_stats['reported_calls']}** | **{total_tokens[0]}** | **{total_tokens[1]}** | **{total_tokens[2]}** |"
    )
    stage_stats = aggregate_stage_usage(results)
    lines.extend([
        "",
        "## 模型调用阶段归因",
        "",
        "| 阶段 | 调用数 | Usage 上报 | 错误 | 平均响应 | P50 | P95 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if not stage_stats:
        lines.append("| 无服务端调用记录 | 0 | 0 | 0 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |")
    for stage, stats in sorted(stage_stats.items()):
        lines.append(
            f"| {stage} | {stats['calls']} | {stats['reported_calls']} | {stats['errors']} | "
            f"{stats['latency_mean_ms']:.1f} ms | {stats['latency_p50_ms']:.1f} ms | "
            f"{stats['latency_p95_ms']:.1f} ms | {stats['prompt_tokens']} | "
            f"{stats['completion_tokens']} | {stats['total_tokens']} |"
        )
    lines.extend([
        "",
        "## 功能模块与调用阶段交叉统计",
        "",
        "| 模块 | 阶段 | 调用数 | 平均响应 | P95 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subset in sorted({item["subset"] for item in results}):
        selected = [item for item in results if item["subset"] == subset]
        for stage, stats in sorted(aggregate_stage_usage(selected).items()):
            lines.append(
                f"| {subset_names.get(subset, subset)} | {stage} | {stats['calls']} | "
                f"{stats['latency_mean_ms']:.1f} ms | {stats['latency_p95_ms']:.1f} ms | "
                f"{stats['prompt_tokens']} | {stats['completion_tokens']} | {stats['total_tokens']} |"
            )
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
    metadata = _prepare_run_metadata(args)
    metadata["result_summary"] = aggregate_results(results)
    metadata["output"] = str(args.output)
    metadata["report"] = str(args.report)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--subset", action="append")
    parser.add_argument("--case-id", action="append", help="只执行指定用例 ID，可重复传入。")
    parser.add_argument("--suite", help="执行 eval/suites 下的固定用例套件，例如 baseline_api_v0.8.0。")
    parser.add_argument("--mode", choices=["offline", "api", "online"], default="offline")
    parser.add_argument("--project-version", default=__version__, help="写入基线报告的 DeskPilot 版本。")
    parser.add_argument("--run-name", default="DeskPilotBench", help="本次运行名称。")
    parser.add_argument(
        "--change-summary", action="append",
        help="本版本修改说明，可重复传入；会写入报告元数据。",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--skip-api-preflight",
        action="store_true",
        help="skip the LLM and embedding connectivity preflight (not recommended)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable the live progress bar",
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    args.started_at = _now()
    timestamp = args.started_at.strftime("%Y%m%dT%H%M%S%z")
    stem = f"deskpilotbench_v{_safe_name(args.project_version).lstrip('v')}_{args.mode}_{timestamp}"
    args.output = args.output or Path("eval/runs") / f"{stem}.jsonl"
    args.report = args.report or Path("eval/reports") / f"{stem}.md"
    args.metadata = args.metadata or args.output.with_suffix(".metadata.json")
    cases = load_cases(args.dataset)
    selected_case_ids = list(args.case_id or [])
    suite_difficulties: dict[str, str] = {}
    if args.suite:
        suite = _load_suite(args.suite)
        selected_case_ids.extend(str(item) for item in suite["case_ids"])
        suite_difficulties = {
            str(item.get("id")): str(item.get("difficulty", "unspecified"))
            for item in suite.get("cases", []) if isinstance(item, dict) and item.get("id")
        }
        args.run_name = str(suite.get("name") or args.run_name)
    if selected_case_ids:
        by_id = {str(case.get("id")): case for case in cases}
        missing = [case_id for case_id in selected_case_ids if case_id not in by_id]
        if missing:
            parser.error(f"Unknown case IDs: {', '.join(missing)}")
        # 保留 suite 中声明的顺序，便于跨版本对比。
        cases = [by_id[case_id] for case_id in dict.fromkeys(selected_case_ids)]
        for case in cases:
            case["difficulty"] = suite_difficulties.get(
                str(case.get("id")), str(case.get("difficulty", "unspecified")),
            )
    if args.subset:
        allowed = set(args.subset)
        cases = [case for case in cases if case.get("subset") in allowed]
    limit = args.count if args.count is not None else (len(cases) if selected_case_ids else 10)
    cases = cases[: max(0, limit)]
    results: list[dict[str, Any]] = []
    total = len(cases)
    if args.progress:
        print(f"DeskPilotBench: {total} cases, mode={args.mode}", flush=True)
    if args.mode == "api" and not args.skip_api_preflight:
        print("[api-preflight] checking LLM and embedding endpoints...", flush=True)
        args.api_preflight = run_api_preflight()
        if not args.api_preflight.get("ok"):
            results.extend(circuit_breaker_results(
                cases,
                cause=(
                    f"preflight {args.api_preflight.get('stage')} failed: "
                    f"{args.api_preflight.get('error', 'unknown API error')}"
                ),
            ))
            write_outputs(results, args)
            print(
                "[api-preflight] failed: "
                f"{args.api_preflight.get('error')}\n"
                f"[api-preflight] {args.api_preflight.get('guidance')}",
                flush=True,
            )
            return 2
        print("[api-preflight] passed", flush=True)
    consecutive_infrastructure_errors = 0
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{index}"))
        started = time.perf_counter()
        if args.progress:
            print_progress(index - 1, total, case_id, "running", 0.0)
        result = run_case(case, args.mode)
        result["difficulty"] = str(case.get("difficulty", "unspecified"))
        results.append(result)
        if args.mode == "api" and is_infrastructure_error(result):
            consecutive_infrastructure_errors += 1
        else:
            consecutive_infrastructure_errors = 0
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
        if args.mode == "api" and consecutive_infrastructure_errors >= 3:
            cause = str(result.get("error", "repeated API infrastructure failure"))
            remaining = cases[index:]
            results.extend(circuit_breaker_results(remaining, cause=cause))
            write_outputs(results, args)
            print(
                f"[api-circuit-breaker] opened after {consecutive_infrastructure_errors} "
                f"consecutive infrastructure errors; skipped {len(remaining)} remaining cases.",
                flush=True,
            )
            break
    if not cases:
        write_outputs(results, args)
    print(render_report(results, args))
    return 0 if all(item["status"] in {"passed", "skipped"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
