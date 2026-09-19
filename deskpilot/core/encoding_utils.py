from __future__ import annotations

import re
import string
import unicodedata


MOJIBAKE_MARKERS = (
    "\ufffd",
    "\u00c3",
    "\u00c2",
    "\u00e2",
    "\ue000",
    "\ue001",
    "\ue003",
    "\ue004",
    "\ue005",
    "\ue006",
    "\ue00c",
)


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded with the wrong legacy encoding.

    This is deliberately conservative: if the text does not look suspicious,
    it is returned unchanged. It mainly targets strings such as
    "涓汉鍔炲叕" that came from UTF-8 bytes decoded as GBK/GB18030, and
    "ä¸­æ–‡" that came from UTF-8 bytes decoded as Latin-1/CP1252.
    """

    if not text or not _looks_mojibaked(text):
        return text

    candidates = [text]
    for source_encoding in ("gb18030", "latin-1", "cp1252"):
        for error_mode in ("strict", "ignore"):
            try:
                candidates.append(text.encode(source_encoding, errors=error_mode).decode("utf-8", errors="strict"))
            except UnicodeError:
                continue

    return min(candidates, key=_mojibake_score)


def strip_qwen_thinking(text: str) -> str:
    """Remove visible Qwen thinking blocks if the provider returns them."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def text_quality_score(text: str) -> float:
    """Return a rough 0-1 score for whether extracted text is readable."""

    if not text:
        return 0.0
    meaningful = 0
    bad = 0
    for char in text:
        code = ord(char)
        if char.isspace():
            meaningful += 1
        elif char in string.printable:
            meaningful += 1
        elif "\u4e00" <= char <= "\u9fff":
            meaningful += 1
        elif "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af":
            meaningful += 1
        elif "\uff00" <= char <= "\uffef":
            meaningful += 1
        elif "\ue000" <= char <= "\uf8ff":
            bad += 3
        elif "\u0f00" <= char <= "\u0fff":
            bad += 3
        elif code < 32 and char not in "\n\r\t":
            bad += 3
        elif char == "\ufffd":
            bad += 3
        elif _is_common_punctuation(char):
            meaningful += 1
        elif char.isdigit():
            meaningful += 1
        elif char.isalpha():
            bad += 1
        else:
            bad += 1
    total = meaningful + bad
    return meaningful / total if total else 0.0


def is_probably_garbled(text: str, min_length: int = 40) -> bool:
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return False
    private_or_tibetan = sum(1 for char in cleaned if "\ue000" <= char <= "\uf8ff" or "\u0f00" <= char <= "\u0fff")
    exotic_letters = sum(1 for char in cleaned if _is_exotic_letter(char))
    if private_or_tibetan / max(len(cleaned), 1) > 0.03:
        return True
    if exotic_letters / max(len(cleaned), 1) > 0.08:
        return True
    return text_quality_score(cleaned) < 0.62


def _looks_mojibaked(text: str) -> bool:
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return True
    private_use_count = sum(1 for char in text if "\ue000" <= char <= "\uf8ff")
    replacement_count = text.count("\ufffd")
    suspicious_words = sum(text.count(word) for word in ("涓", "鍔", "绋", "妗", "寮", "鎶", "璺", "瀛", "鐢", "绠"))
    return private_use_count + replacement_count + suspicious_words >= 2


def _mojibake_score(text: str) -> int:
    private_use_count = sum(1 for char in text if "\ue000" <= char <= "\uf8ff")
    replacement_count = text.count("\ufffd")
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    suspicious_words = sum(text.count(word) for word in ("涓", "鍔", "绋", "妗", "寮", "鎶", "璺", "瀛", "鐢", "绠"))
    return private_use_count * 4 + replacement_count * 4 + marker_count * 2 + suspicious_words


def _is_common_punctuation(char: str) -> bool:
    return char in "，。！？；：、“”‘’（）《》【】—…·,.!?;:\"'()[]{}<>-/\\|_+=*&^%$#@~`"


def _is_exotic_letter(char: str) -> bool:
    if not char.isalpha():
        return False
    if char in string.ascii_letters:
        return False
    if "\u4e00" <= char <= "\u9fff":
        return False
    if "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af":
        return False
    name = unicodedata.name(char, "")
    return not any(script in name for script in ("CJK", "LATIN", "HIRAGANA", "KATAKANA", "HANGUL"))
