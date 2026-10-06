from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from .models import ContextPacket


def calculate_context_quality(
    gathered: list[ContextPacket], selected: list[ContextPacket], dropped: list[ContextPacket], budget: int
) -> dict[str, object]:
    gathered_tokens = sum(packet.estimated_tokens for packet in gathered)
    selected_tokens = sum(packet.estimated_tokens for packet in selected)
    required = [packet for packet in gathered if packet.required]
    selected_ids = {packet.packet_id for packet in selected}
    retained_required = sum(packet.packet_id in selected_ids for packet in required)
    warnings: list[str] = []
    if selected_tokens > budget:
        warnings.append("context_budget_exceeded")
    if required and retained_required < len(required):
        warnings.append("required_packet_dropped")
    return {
        "estimated_tokens": selected_tokens,
        "gathered_packets": len(gathered),
        "selected_packets": len(selected),
        "dropped_packets": len(dropped),
        "compression_ratio": round(selected_tokens / gathered_tokens, 4) if gathered_tokens else 1.0,
        "average_relevance": round(
            sum(packet.relevance_score for packet in selected) / len(selected), 4
        ) if selected else 0.0,
        "required_retention": round(retained_required / len(required), 4) if required else 1.0,
        "context_utilization": round(selected_tokens / budget, 4) if budget else 0.0,
        "kind_distribution": dict(Counter(packet.kind for packet in selected)),
        "warnings": warnings,
        "packet_attribution": packet_attribution(gathered, selected, dropped),
    }


def packet_attribution(
    gathered: list[ContextPacket], selected: list[ContextPacket], dropped: list[ContextPacket]
) -> list[dict[str, object]]:
    """记录 packet 去留原因；内容本身不进入日志，避免复制隐私正文。"""
    selected_ids = {item.packet_id for item in selected}
    dropped_ids = {item.packet_id for item in dropped}
    result: list[dict[str, object]] = []
    for packet in gathered:
        if packet.packet_id in selected_ids:
            reason = "required" if packet.required else "selected_by_value"
            state = "selected"
        elif packet.packet_id in dropped_ids:
            reason = "below_relevance_or_budget"
            state = "dropped"
        else:
            reason = "role_filtered"
            state = "dropped"
        result.append({
            "packet_id": packet.packet_id,
            "kind": packet.kind,
            "source": packet.source,
            "state": state,
            "reason": reason,
            "estimated_tokens": packet.estimated_tokens,
            "relevance": round(packet.relevance_score, 4),
            "required": packet.required,
        })
    return result


def context_precision_recall(
    selected_packet_ids: Iterable[str], relevant_packet_ids: Iterable[str]
) -> dict[str, float | int]:
    """以 packet ID 为 oracle 计算 Context Precision/Recall。"""
    selected = set(selected_packet_ids)
    relevant = set(relevant_packet_ids)
    true_positive = len(selected & relevant)
    return {
        "selected": len(selected),
        "relevant": len(relevant),
        "true_positive": true_positive,
        "context_precision": round(true_positive / len(selected), 4) if selected else 0.0,
        "context_recall": round(true_positive / len(relevant), 4) if relevant else 1.0,
    }


def summary_fact_consistency(summary: str, source_texts: Iterable[str]) -> dict[str, object]:
    """离线轻量事实一致性检查；用于筛查摘要漂移，不替代事实核验模型。"""
    source = " ".join(source_texts).casefold()
    claims = [item.strip() for item in re.split(r"[。！？!?\n]+", summary) if item.strip()]
    unsupported: list[str] = []
    for claim in claims:
        terms = set(re.findall(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", claim.casefold()))
        if not terms:
            continue
        overlap = sum(term in source for term in terms) / len(terms)
        if overlap < 0.5:
            unsupported.append(claim[:160])
    supported = max(0, len(claims) - len(unsupported))
    return {
        "claims": len(claims),
        "supported_claims": supported,
        "consistency": round(supported / len(claims), 4) if claims else 1.0,
        "unsupported_claims": unsupported,
    }


def lost_in_middle_metrics(packet_ids: list[str], relevant_packet_ids: Iterable[str]) -> dict[str, object]:
    """报告相关 packet 在首/中/尾的位置覆盖，供合成 lost-in-the-middle 用例使用。"""
    relevant = set(relevant_packet_ids)
    positions = [index for index, packet_id in enumerate(packet_ids) if packet_id in relevant]
    size = len(packet_ids)
    middle = [index for index in positions if size and size / 3 <= index < size * 2 / 3]
    return {
        "packet_count": size,
        "relevant_positions": positions,
        "middle_relevant_count": len(middle),
        "relevant_coverage": round(len(positions) / len(relevant), 4) if relevant else 1.0,
        "middle_retained": bool(middle),
    }
