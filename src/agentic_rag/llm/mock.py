"""LLM giả có kịch bản - dùng cho test và demo offline (không cần API key)."""
from __future__ import annotations

import itertools
import json
from typing import Callable, Union

from .base import LLMClient, LLMResponse, Message, ToolCall, Usage

_ids = itertools.count(1)
Step = Union[LLMResponse, Callable[[list[Message], str, list | None], LLMResponse]]


def reply(text: str, tokens: int = 100) -> LLMResponse:
    return LLMResponse(text=text, usage=Usage(tokens, tokens // 4, 1))


def json_reply(obj) -> LLMResponse:
    return reply(json.dumps(obj, ensure_ascii=False))


def tool_calls(*calls: tuple[str, dict], text: str = "") -> LLMResponse:
    tcs = [ToolCall(f"mock_{next(_ids)}", n, a) for n, a in calls]
    return LLMResponse(text=text, tool_calls=tcs, usage=Usage(150, 30, 1), stop_reason="tool_use")


class MockLLM(LLMClient):
    name = "mock"
    default_model = "mock"

    def __init__(self, script: list[Step] | None = None):
        super().__init__(sleep=lambda s: None)
        self.script = list(script or [])
        self.calls: list[dict] = []

    def _chat(self, messages, system, tools, model, max_tokens, temperature, json_mode):
        self.calls.append({"messages": list(messages), "system": system, "tools": tools, "model": model})
        if not self.script:
            raise AssertionError(f"MockLLM hết kịch bản (lần gọi thứ {len(self.calls)}). System: {system[:80]!r}")
        step = self.script.pop(0)
        return step(messages, system, tools) if callable(step) else step

    def assert_done(self) -> None:
        assert not self.script, f"MockLLM còn {len(self.script)} bước kịch bản chưa dùng"
