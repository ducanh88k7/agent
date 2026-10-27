"""Rào chắn: ngân sách chạy, làm sạch input, phát hiện nội dung khả nghi trong dữ liệu truy xuất."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from ..llm.base import Usage


class Budget:
    """Giới hạn số lần gọi LLM / tool / token / thời gian cho MỘT lượt hỏi. Chống vòng lặp đốt tiền."""

    def __init__(self, max_llm_calls: int, max_tool_calls: int, max_tokens: int, timeout_s: float):
        self.max_llm_calls, self.max_tool_calls = max_llm_calls, max_tool_calls
        self.max_tokens, self.deadline = max_tokens, time.monotonic() + timeout_s
        self.llm_calls = self.tool_calls = self.tokens = 0
        self.usage = Usage()
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, s) -> "Budget":
        return cls(s.max_llm_calls, s.max_tool_calls, s.max_total_tokens, s.timeout_s)

    def charge_llm(self, usage: Usage) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens += usage.total
            self.usage.add(usage)

    def charge_tool(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def exhausted(self) -> str | None:
        if self.llm_calls >= self.max_llm_calls:
            return "hết số lần gọi LLM"
        if self.tool_calls >= self.max_tool_calls:
            return "hết số lần gọi công cụ"
        if self.tokens >= self.max_tokens:
            return "hết ngân sách token"
        if time.monotonic() > self.deadline:
            return "quá thời gian"
        return None


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_user_input(text: str, max_chars: int = 4000) -> str:
    return _CTRL.sub("", text or "").strip()[:max_chars]


_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)"
    r"|bỏ qua\s+(mọi|tất cả|các)?\s*(hướng dẫn|chỉ dẫn|quy tắc|lệnh)"
    r"|system prompt|you are now|bây giờ bạn là|reveal (your|the) (instructions|prompt)", re.I)


def looks_like_injection(text: str) -> bool:
    """Heuristic rẻ để GẮN CỜ nội dung truy xuất có vẻ chứa câu lệnh. Không phải lá chắn tuyệt đối:
    lớp bảo vệ thật là (1) coi dữ liệu là dữ liệu, (2) quyền tối thiểu cho tool, (3) xác nhận hành động ghi."""
    return bool(_INJECTION.search(text or ""))


def escape_doc_text(text: str) -> str:
    """Không cho nội dung tài liệu tự đóng thẻ <document>."""
    return text.replace("</document>", "</ document>").replace("<document", "< document")
