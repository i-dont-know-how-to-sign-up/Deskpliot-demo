from __future__ import annotations

from collections import Counter

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
    }
