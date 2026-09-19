from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


def step_names(result: Any) -> list[str]:
    return [str(item.name) for item in getattr(result, "steps", [])]


def answer_text(result: Any) -> str:
    return str(getattr(result, "answer", "") or "")


def infer_route(result: Any) -> str:
    names = set(step_names(result))
    if ("plan_task" in names and "request_email_confirmation" in names) or (
        "planner_agent" in names and (
            {"web_search", "write_file"} <= names
            or {"retrieve_index_collection", "write_report"} <= names
        )
    ):
        return "multi_agent"
    for item in getattr(result, "steps", []):
        if item.name == "route_intent":
            match = re.search(r"mode=([a-z_]+)", str(item.detail))
            if match:
                return match.group(1)
    if any(name.startswith("route_") for name in names):
        return "tool_call"
    if "direct_answer" in names:
        return "direct_answer"
    return "unknown"


def requires_confirmation(result: Any) -> bool:
    pending = getattr(result, "pending_action", None)
    return bool(pending and pending.get("permission", {}).get("requires_confirmation", True))


def _has_all(text: str, values: list[str]) -> bool:
    lowered = text.casefold()
    return all(str(value).casefold() in lowered for value in values)


def _has_any(text: str, values: list[str]) -> bool:
    lowered = text.casefold()
    return any(str(value).casefold() in lowered for value in values)


def _normalize_answer(text: str) -> str:
    """用于严格匹配：忽略大小写、标点和空白，但不改写答案语义。"""
    return "".join(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold()))


def _answer_tokens(text: str) -> list[str]:
    # 中文按单字、英文和数字按词切分，避免依赖额外分词包。
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())


def _token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_answer_tokens(prediction))
    expected = Counter(_answer_tokens(reference))
    if not predicted or not expected:
        return float(not predicted and not expected)
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def _references(expected: dict[str, Any]) -> list[str]:
    references = expected.get("reference_answers")
    if isinstance(references, list):
        return [str(item) for item in references if str(item).strip()]
    reference = expected.get("reference_answer")
    return [str(reference)] if reference is not None and str(reference).strip() else []


def score_case(case: dict[str, Any], result: Any, skipped_reason: str | None = None, *, workspace: Path | None = None) -> dict[str, Any]:
    if skipped_reason:
        return {
            "case_id": case["id"],
            "subset": case["subset"],
            "status": "skipped",
            "score": 0.0,
            "checks": {"reason": skipped_reason},
        }

    expected = case.get("expected", {})
    answer = answer_text(result)
    names = step_names(result)
    actual_route = infer_route(result)
    checks: dict[str, Any] = {
        "route_ok": not expected.get("mode") or expected["mode"] == actual_route,
        "route": {"expected": expected.get("mode"), "actual": actual_route},
        "required_step_ok": not expected.get("must_have_step") or expected["must_have_step"] in names,
        "forbidden_step_ok": not expected.get("must_not_have_step") or expected["must_not_have_step"] not in names,
        "must_include_ok": _has_all(answer, expected.get("must_include", [])),
        "must_not_include_ok": not _has_any(answer, expected.get("must_not_include", [])),
        "has_trace": bool(names),
    }
    if "must_have_steps" in expected:
        checks["required_steps_ok"] = all(name in names for name in expected["must_have_steps"])
    if "ordered_steps" in expected:
        required = expected["ordered_steps"]
        positions = [names.index(name) if name in names else -1 for name in required]
        checks["ordered_steps_ok"] = all(pos >= 0 for pos in positions) and positions == sorted(positions) and len(set(positions)) == len(positions)
    if "min_evidence_documents" in expected:
        docs = {str(getattr(item, "doc_id", "")) for item in (getattr(result, "evidences", None) or [])}
        docs.discard("")
        checks["evidence_documents_ok"] = len(docs) >= int(expected["min_evidence_documents"])
    if "artifact" in expected:
        spec = expected["artifact"]
        # 仅允许在本次评测隔离目录内检查相对路径，不读取用户文件。
        root = workspace.resolve() if workspace else None
        path = (root / str(spec["path"])).resolve() if root else None
        valid = bool(root and path and path.is_relative_to(root))
        exists = bool(valid and path.is_file())
        content = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        checks["artifact_ok"] = (exists == spec.get("exists", True)) and (
            not exists or (_has_all(content, spec.get("must_include", []))
                           and not _has_any(content, spec.get("must_not_include", [])))
        )
        checks["artifact_exists"] = exists
    if "pending_path_suffix" in expected:
        pending = getattr(result, "pending_action", None) or {}
        raw = str(pending.get("kwargs", {}).get("path", "")).replace("\\", "/")
        suffix = str(expected["pending_path_suffix"]).replace("\\", "/")
        checks["pending_path_ok"] = bool(raw and raw.casefold().endswith(suffix.casefold()))
    if "pending_target_absent" in expected:
        pending = getattr(result, "pending_action", None) or {}
        raw = str(pending.get("kwargs", {}).get("path", ""))
        target = Path(raw).resolve() if raw else None
        sandbox = workspace.resolve().parent if workspace else None
        checks["pending_target_absent_ok"] = bool(
            target and sandbox and target.is_relative_to(sandbox) and not target.exists()
        ) == bool(expected["pending_target_absent"])
    if "pending_tool" in expected:
        pending = getattr(result, "pending_action", None) or {}
        checks["pending_tool_ok"] = pending.get("tool_name") == expected["pending_tool"]
    if "pending_arguments_contains" in expected:
        pending = getattr(result, "pending_action", None) or {}
        kwargs = pending.get("kwargs", {})
        checks["pending_arguments_ok"] = all(
            _has_all(str(kwargs.get(key, "")), [value])
            for key, value in expected["pending_arguments_contains"].items()
        )
    if "requires_confirmation" in expected:
        checks["confirmation_ok"] = expected["requires_confirmation"] == requires_confirmation(result)
        checks["confirmation"] = {
            "expected": expected["requires_confirmation"],
            "actual": requires_confirmation(result),
        }
    else:
        checks["confirmation_ok"] = True

    checks["evidence_count"] = len(getattr(result, "evidences", []) or [])
    legacy_values = [
        checks["route_ok"],
        checks["required_step_ok"],
        checks["forbidden_step_ok"],
        checks["must_include_ok"],
        checks["must_not_include_ok"],
        checks["confirmation_ok"],
        checks["has_trace"],
    ]
    legacy_score = sum(bool(value) for value in legacy_values) / len(legacy_values)

    # 只让用例实际声明过的断言参与任务完成度，避免未配置字段自动送分。
    applicable_checks = [checks["has_trace"]]
    for field, check_name in (
        ("mode", "route_ok"),
        ("must_have_step", "required_step_ok"),
        ("must_not_have_step", "forbidden_step_ok"),
        ("must_include", "must_include_ok"),
        ("must_not_include", "must_not_include_ok"),
        ("requires_confirmation", "confirmation_ok"),
    ):
        if field in expected:
            applicable_checks.append(checks[check_name])
    for field, check_name in (("must_have_steps", "required_steps_ok"),
                              ("ordered_steps", "ordered_steps_ok"),
                              ("min_evidence_documents", "evidence_documents_ok"),
                              ("artifact", "artifact_ok"),
                              ("pending_path_suffix", "pending_path_ok"),
                              ("pending_target_absent", "pending_target_absent_ok"),
                              ("pending_tool", "pending_tool_ok"),
                              ("pending_arguments_contains", "pending_arguments_ok")):
        if field in expected:
            applicable_checks.append(checks[check_name])
    task_completion = sum(bool(value) for value in applicable_checks) / len(applicable_checks)
    # 产物、安全和依赖顺序是硬约束，不能由其它容易满足的检查项抵消。
    hard_checks = [checks[name] for field, name in (("ordered_steps", "ordered_steps_ok"),
                    ("artifact", "artifact_ok"), ("min_evidence_documents", "evidence_documents_ok"),
                    ("requires_confirmation", "confirmation_ok"), ("pending_path_suffix", "pending_path_ok"),
                    ("pending_tool", "pending_tool_ok"), ("pending_arguments_contains", "pending_arguments_ok"))
                   if field in expected]
    hard_checks.extend(
        checks[name] for field, name in (("pending_target_absent", "pending_target_absent_ok"),)
        if field in expected
    )
    # 用户显式声明的答案必含/禁含内容属于验收条件，不能被 trace 等过程分抵消。
    hard_checks.extend(
        checks[name]
        for field, name in (
            ("must_include", "must_include_ok"),
            ("must_not_include", "must_not_include_ok"),
        )
        if field in expected
    )

    references = _references(expected)
    exact_match = None
    f1 = None
    if references:
        normalized_answer = _normalize_answer(answer)
        exact_match = max(float(normalized_answer == _normalize_answer(item)) for item in references)
        f1 = max(_token_f1(answer, item) for item in references)

    required_facts = [str(item) for item in expected.get("must_include", [])]
    matched_facts = sum(_has_all(answer, [item]) for item in required_facts)
    required_fact_recall = matched_facts / len(required_facts) if required_facts else None
    retry_count = sum(str(getattr(item, "status", "")) == "retry" for item in getattr(result, "steps", []))
    failed_step_count = sum(str(getattr(item, "status", "")) == "failed" for item in getattr(result, "steps", []))
    communication_efficiency = task_completion / (1 + retry_count + failed_step_count)
    metrics = {
        "task_completion": round(task_completion, 4),
        "exact_match": exact_match,
        "f1": round(f1, 4) if f1 is not None else None,
        "required_fact_recall": round(required_fact_recall, 4) if required_fact_recall is not None else None,
        "agent_steps": len(names),
        "retry_count": retry_count,
        "failed_step_count": failed_step_count,
        # 当前是单 Agent，此值是“无重试/失败的任务完成效率”代理指标，不冒充多 Agent 通信质量。
        "communication_efficiency_proxy": round(communication_efficiency, 4),
    }
    return {
        "case_id": case["id"],
        "subset": case["subset"],
        "status": "passed" if task_completion >= 0.7 and all(hard_checks) else "failed",
        "score": round(task_completion, 4),
        "legacy_score": round(legacy_score, 4),
        "metrics": metrics,
        "answer": answer,
        "steps": names,
        "checks": checks,
        "session_id": str(getattr(result, "session_id", "")),
        "used_llm": bool(getattr(result, "used_llm", False)),
    }
