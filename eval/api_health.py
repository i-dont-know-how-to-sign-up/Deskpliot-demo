from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from deskpilot.core.api_clients import OpenAICompatibleClient
from deskpilot.core.config import ModelConfig, load_config


_INFRASTRUCTURE_ERROR_MARKERS = (
    "urlopen error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
    "name or service not known",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "timed out",
    "timeout",
    "winerror 10060",
    "winerror 10061",
    "winerror 10054",
    "network is unreachable",
    "remote end closed connection",
)


def classify_api_error(error: object) -> str:
    """将 API 异常归类，避免把基础设施故障误算成 Agent 能力失败。"""
    message = str(error).casefold()
    if any(marker in message for marker in _INFRASTRUCTURE_ERROR_MARKERS):
        return "connectivity"
    if any(marker in message for marker in ("401", "403", "unauthorized", "invalid api key")):
        return "authentication"
    if "429" in message or "rate limit" in message or "quota" in message:
        return "rate_limit"
    if any(marker in message for marker in ("model not found", "invalid model", "does not exist")):
        return "model"
    return "api"


def api_error_guidance(category: str) -> str:
    guidance = {
        "connectivity": "检查 VPN/代理、防火墙和目标 API 域名的 443 端口，然后重新运行评测。",
        "authentication": "检查 API Key 是否有效、是否属于当前服务商，以及账号是否有模型访问权限。",
        "rate_limit": "检查额度和限流状态，降低调用频率或等待服务端限流窗口恢复。",
        "model": "检查模型名称是否存在，并确认 Base URL 与模型所属服务商一致。",
        "api": "检查服务端响应、Base URL、模型配置和账号权限。",
    }
    return guidance.get(category, guidance["api"])


def run_api_preflight(config: ModelConfig | None = None) -> dict[str, Any]:
    """用极小请求验证 LLM 和 Embedding；预检 usage 不计入正式评测。"""
    effective = replace(config or load_config(), allow_local_fallback=False)
    client = OpenAICompatibleClient(effective)
    checks: list[dict[str, Any]] = []

    try:
        answer = client.chat(
            [{"role": "user", "content": "Reply with OK only."}],
            temperature=0.0,
            max_tokens=8,
        )
        if not answer.strip():
            raise RuntimeError("LLM preflight returned an empty response")
        checks.append({"name": "llm", "ok": True, "model": effective.llm_model})
    except Exception as exc:
        return _failed_preflight("llm", effective.llm_model, exc, checks)

    try:
        vectors = client.embed(["DeskPilot API preflight"])
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding preflight returned an empty vector")
        checks.append({
            "name": "embedding",
            "ok": True,
            "model": effective.embedding_model,
            "dimensions": len(vectors[0]),
        })
    except Exception as exc:
        return _failed_preflight("embedding", effective.embedding_model, exc, checks)

    return {"ok": True, "category": "healthy", "checks": checks}


def is_infrastructure_error(result: dict[str, Any]) -> bool:
    if result.get("status") != "error":
        return False
    return classify_api_error(result.get("error", "")) in {"connectivity", "rate_limit"}


def circuit_breaker_results(
    cases: list[dict[str, Any]],
    *,
    cause: str,
) -> list[dict[str, Any]]:
    reason = f"api_circuit_open: {cause}"
    return [
        {
            "case_id": str(case.get("id", "unknown")),
            "subset": str(case.get("subset", "unknown")),
            "status": "skipped",
            "score": 0.0,
            "reason": reason,
            "difficulty": str(case.get("difficulty", "unspecified")),
        }
        for case in cases
    ]


def _failed_preflight(
    stage: str,
    model: str,
    exc: Exception,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    category = classify_api_error(exc)
    # 截断服务端正文，并且绝不记录请求头或 API Key。
    error = re.sub(r"\s+", " ", str(exc)).strip()[:1000]
    checks.append({"name": stage, "ok": False, "model": model, "error": error})
    return {
        "ok": False,
        "stage": stage,
        "category": category,
        "error": error,
        "guidance": api_error_guidance(category),
        "checks": checks,
    }
