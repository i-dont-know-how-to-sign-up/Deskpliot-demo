from __future__ import annotations

import re


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """无 tokenizer 依赖的保守估算；用于预算控制，不用于 API 计费。"""

    if not text:
        return 0
    count = 0
    for token in _TOKEN_PATTERN.findall(text):
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            count += max(1, (len(token) + 3) // 4)
        else:
            count += 1
    return count


def trim_to_token_budget(text: str, max_tokens: int, *, keep_tail: bool = False) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # 二分查找字符边界，避免逐字符反复估算造成长工具输出卡顿。
    marker = "...\n" if keep_tail else "..."
    content_budget = max(1, max_tokens - estimate_tokens(marker))
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[-middle:] if keep_tail else text[:middle]
        if estimate_tokens(candidate) <= content_budget:
            low = middle
        else:
            high = middle - 1
    selected = text[-low:] if keep_tail else text[:low]
    return (marker + selected) if keep_tail else (selected + marker)
