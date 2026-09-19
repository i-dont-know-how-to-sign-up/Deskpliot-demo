from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class UsageSnapshot:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reported_calls: int


class UsageCostTracker:
    """持久化真实 API usage，并按任务类型汇总 P50/P95。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def snapshot(self, usage: dict[str, int]) -> UsageSnapshot:
        return UsageSnapshot(*(int(usage.get(key, 0)) for key in (
            "prompt_tokens", "completion_tokens", "total_tokens", "reported_calls"
        )))

    def record_delta(self, before: UsageSnapshot, usage: dict[str, int], task_type: str, model: str) -> dict[str, Any]:
        after = self.snapshot(usage)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type,
            "model": model,
            "prompt_tokens": max(0, after.prompt_tokens - before.prompt_tokens),
            "completion_tokens": max(0, after.completion_tokens - before.completion_tokens),
            "total_tokens": max(0, after.total_tokens - before.total_tokens),
            "reported_calls": max(0, after.reported_calls - before.reported_calls),
        }
        if record["reported_calls"]:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {**record, **self.stats(task_type)}

    def stats(self, task_type: str | None = None) -> dict[str, int]:
        records = self._records(task_type)
        totals = sorted(int(item.get("total_tokens", 0)) for item in records)
        return {
            "samples": len(totals),
            "p50_total_tokens": int(median(totals)) if totals else 0,
            "p95_total_tokens": self._percentile(totals, 0.95),
        }

    def _records(self, task_type: str | None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not task_type or item.get("task_type") == task_type:
                records.append(item)
        return records

    def _percentile(self, values: list[int], ratio: float) -> int:
        if not values:
            return 0
        index = max(0, min(len(values) - 1, int((len(values) - 1) * ratio + 0.5)))
        return values[index]
