"""Adapter cho Claude (anthropic SDK). Chưa chạy thử với API thật trong môi trường dựng project."""
from __future__ import annotations

import os

from .base import LLMClient, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, default_model: str, api_key: str | None = None, **kw):
        super().__init__(**kw)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMError("Thiếu thư viện: pip install anthropic") from e
        self.default_model = default_model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    # Message trung lập -> định dạng Anthropic. Các message role="tool" liên tiếp gộp thành 1 lượt user.
    def _convert(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []

        def push(role: str, blocks: list[dict] | str):
            if out and out[-1]["role"] == role == "user":  # API muốn xen kẽ vai trò: gộp user liên tiếp
                prev = out[-1]["content"]
                prev = [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
                new = [{"type": "text", "text": blocks}] if isinstance(blocks, str) else blocks
                out[-1]["content"] = prev + new
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "user":
                push("user", m.content)
            elif m.role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id,
                               "content": m.content or "(trống)", "is_error": m.is_error}])
            else:
                if m.raw is not None:
                    out.append({"role": "assistant", "content": m.raw})
                    continue
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(trống)"}]})
        return out

    def _chat(self, messages, system, tools: list[ToolSpec] | None, model, max_tokens, temperature, json_mode):
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=self._convert(messages))
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens, 1)
        return LLMResponse(text=text, tool_calls=calls, usage=usage, stop_reason=resp.stop_reason or "end", raw=resp.content)
