from __future__ import annotations

import math
import random
import smtplib
import ssl
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import TYPE_CHECKING

from .control import TaskControlError

if TYPE_CHECKING:
    from .control import ExecutionControl


class UnknownSideEffect(RuntimeError):
    """请求已派发但结果未知；只能核对，不能自动重放。"""

    state = "unknown"
    side_effect_state = "unknown"
    retryable = False


class ErrorCategory(str, Enum):
    CONTROL = "control"
    UNKNOWN_SIDE_EFFECT = "unknown_side_effect"
    AUTH = "authentication"
    INVALID_INPUT = "invalid_input"
    TRANSIENT = "transient"
    PERMANENT = "permanent"


def classify_error(error: BaseException) -> ErrorCategory:
    if isinstance(error, TaskControlError):
        return ErrorCategory.CONTROL
    if isinstance(error, UnknownSideEffect):
        return ErrorCategory.UNKNOWN_SIDE_EFFECT
    if isinstance(error, smtplib.SMTPAuthenticationError):
        return ErrorCategory.AUTH
    if isinstance(error, ssl.SSLCertVerificationError):
        return ErrorCategory.PERMANENT
    if isinstance(error, urllib.error.HTTPError):
        if error.code in {401, 403}:
            return ErrorCategory.AUTH
        if error.code in {408, 429, 500, 502, 503, 504}:
            return ErrorCategory.TRANSIENT
        return ErrorCategory.INVALID_INPUT if 400 <= error.code < 500 else ErrorCategory.PERMANENT
    if isinstance(error, PermissionError):
        return ErrorCategory.AUTH
    if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
        return ErrorCategory.INVALID_INPUT
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, (ValueError, PermissionError, ssl.SSLCertVerificationError)):
            return classify_error(error.reason)
        return ErrorCategory.TRANSIENT
    if isinstance(error, (TimeoutError, ConnectionError)):
        return ErrorCategory.TRANSIENT
    return ErrorCategory.PERMANENT


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    max_task_retries: int = 8
    max_node_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 10.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.max_task_retries < 0 or self.max_node_retries < 0:
            raise ValueError("Invalid retry limits")
        if any(not math.isfinite(v) or v < 0 for v in (self.base_delay, self.max_delay, self.jitter)):
            raise ValueError("Invalid retry delays")

    def allows(self, error: BaseException, attempt: int, *,
               control: ExecutionControl | None = None, idempotent: bool = True) -> bool:
        if control is not None:
            control.checkpoint()
        if not idempotent or attempt >= self.max_attempts or classify_error(error) != ErrorCategory.TRANSIENT:
            return False
        if control is None:
            return True
        return control.reserve_retry(max_task_retries=self.max_task_retries,
                                     max_node_retries=self.max_node_retries)

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            # 服务器建议超过上限时不提前重放，由调用方放弃本轮自动重试。
            return max(0.0, retry_after)
        delay = self.base_delay * (2 ** min(max(0, attempt - 1), 30))
        return min(self.max_delay, delay + random.uniform(0, self.base_delay * self.jitter))

    def wait(self, attempt: int, *, control: ExecutionControl | None = None,
             retry_after: float | None = None) -> bool:
        delay = self.delay(attempt, retry_after)
        if delay > self.max_delay:
            return False
        if control is None:
            time.sleep(delay)
        else:
            control.wait(delay)
        return True
