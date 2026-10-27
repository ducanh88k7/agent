"""Giao diện LLM trung lập nhà cung cấp.

Toàn bộ agent chỉ làm việc với các kiểu ở đây (Message, ToolCall, LLMResponse). Muốn đổi Gemini <-> Claude
hay thêm nhà cung cấp khác chỉ cần viết thêm một adapter, không sửa logic agent.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..retry import retry_call


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema (object)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # chỉ có ở role="assistant"
    tool_call_id: str | None = None  # chỉ có ở role="tool"
    tool_name: str | None = None
    is_error: bool = False
    # Nội dung gốc của provider; gửi lại nguyên vẹn lượt sau (vd Gemini cần giữ thought signature).
    raw: Any = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(calls=1))
    stop_reason: str = "end"
    raw: Any = None

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=list(self.tool_calls), raw=self.raw)


class LLMError(Exception):
    pass


class LLMClient(ABC):
    name = "llm"
    default_model = ""

    def __init__(self, sleep=time.sleep):
        self._sleep = sleep

    @abstractmethod
    def _chat(self, messages: list[Message], system: str, tools: list[ToolSpec] | None, model: str,
              max_tokens: int, temperature: float | None, json_mode: bool) -> LLMResponse: ...

    def chat(self, messages: list[Message], *, system: str = "", tools: list[ToolSpec] | None = None,
             model: str | None = None, max_tokens: int = 2048, temperature: float | None = None,
             json_mode: bool = False) -> LLMResponse:
        model = model or self.default_model
        return retry_call(lambda: self._chat(messages, system, tools, model, max_tokens, temperature, json_mode),
                          sleep=self._sleep)


# ---------------------------------------------------------------- JSON có cấu trúc
def extract_json(text: str) -> Any:
    """Lấy JSON từ output của LLM (chịu được ```json fence và văn bản thừa quanh JSON)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError("Không tìm thấy JSON trong output")
    start = min(starts)
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("JSON không đóng ngoặc đầy đủ")


def complete_json(llm: LLMClient, *, system: str, prompt: str, model: str | None = None,
                  history: list[Message] | None = None, max_tokens: int = 1024) -> tuple[Any, Usage]:
    """Gọi LLM yêu cầu JSON, tự sửa 1 lần nếu parse lỗi. Trả về (dữ liệu, usage tổng)."""
    usage = Usage()
    messages = [*(history or []), Message("user", prompt)]
    resp = llm.chat(messages, system=system, model=model, max_tokens=max_tokens, temperature=0.0, json_mode=True)
    usage.add(resp.usage)
    try:
        return extract_json(resp.text), usage
    except ValueError:
        messages += [resp.to_message(), Message("user", "Output trên không phải JSON hợp lệ. Chỉ trả về đúng một đối tượng JSON, không thêm chữ nào khác.")]
        resp = llm.chat(messages, system=system, model=model, max_tokens=max_tokens, temperature=0.0, json_mode=True)
        usage.add(resp.usage)
        return extract_json(resp.text), usage  # nếu vẫn lỗi thì ValueError, caller tự xử lý fallback
