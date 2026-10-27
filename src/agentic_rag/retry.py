"""Retry có backoff cho lỗi tạm thời (429 / 5xx). Free tier của Gemini hay gặp 429."""
from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")
_CODES = {408, 429, 500, 502, 503, 504, 529}
_PATTERN = re.compile(r"\b(429|500|502|503|504|529)\b|rate.?limit|resource_exhausted|overloaded|unavailable|timed? ?out", re.I)


def is_retryable(exc: Exception) -> bool:
    for attr in ("status_code", "code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and v in _CODES:
            return True
    return bool(_PATTERN.search(str(exc)))


def retry_call(fn: Callable[[], T], *, attempts: int = 4, base_delay: float = 2.0,
               sleep: Callable[[float], None] = time.sleep) -> T:
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1 or not is_retryable(e):
                raise
            sleep(base_delay * (2 ** i) + random.uniform(0, 0.5))
    raise RuntimeError("unreachable")
