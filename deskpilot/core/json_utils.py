from __future__ import annotations

import json
import re
from typing import Any


def parse_json_value(text: str, expected_type: type | tuple[type, ...] | None = None) -> Any | None:
    """从模型输出中提取第一个合法 JSON 值，避免贪心正则跨越多个对象。"""
    source = str(text or "").strip()
    if not source:
        return None

    candidates = [source]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", source, flags=re.IGNORECASE)
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            value = _raw_decode_first(candidate, decoder)
        if value is not None and (expected_type is None or isinstance(value, expected_type)):
            return value
    return None


def _raw_decode_first(text: str, decoder: json.JSONDecoder) -> Any | None:
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None
