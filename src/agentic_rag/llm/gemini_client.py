"""Adapter cho Gemini (SDK `google-genai`). Chưa chạy thử với API thật trong môi trường dựng project."""
from __future__ import annotations

import itertools
import os

from .base import LLMClient, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

_ids = itertools.count(1)


def _gemini_schema(schema: dict) -> dict:
    """JSON schema -> schema Gemini chấp nhận (type viết HOA, bỏ các khóa lạ)."""
    out: dict = {}
    for k, v in schema.items():
        if k == "type":
            out["type"] = str(v).upper()
        elif k == "properties":
            out["properties"] = {n: _gemini_schema(s) for n, s in v.items()}
        elif k == "items":
            out["items"] = _gemini_schema(v)
        elif k in ("description", "enum", "required", "nullable"):
            out[k] = v
    return out


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, default_model: str, api_key: str | None = None, **kw):
        super().__init__(**kw)
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise LLMError("Thiếu thư viện: pip install google-genai") from e
        self.default_model = default_model
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    def _convert(self, messages: list[Message], types) -> list:
        contents: list = []
        for m in messages:
            if m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.content)]))
            elif m.role == "tool":
                payload = {"error": m.content} if m.is_error else {"result": m.content}
                part = types.Part.from_function_response(name=m.tool_name or "tool", response=payload)
                last = contents[-1] if contents else None
                if last is not None and last.role == "user" and last.parts and last.parts[0].function_response:
                    last.parts.append(part)  # gộp nhiều kết quả tool vào cùng một lượt
                else:
                    contents.append(types.Content(role="user", parts=[part]))
            else:
                if m.raw is not None:  # giữ nguyên để không mất thought signature
                    contents.append(m.raw)
                    continue
                parts = [types.Part.from_text(text=m.content)] if m.content else []
                parts += [types.Part(function_call=types.FunctionCall(name=c.name, args=c.arguments)) for c in m.tool_calls]
                contents.append(types.Content(role="model", parts=parts or [types.Part.from_text(text="(trống)")]))
        return contents

    def _chat(self, messages, system, tools: list[ToolSpec] | None, model, max_tokens, temperature, json_mode):
        from google.genai import types

        cfg: dict = {"max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        if tools:
            decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                               parameters=_gemini_schema(t.parameters)) for t in tools]
            cfg["tools"] = [types.Tool(function_declarations=decls)]
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)  # ta tự chạy tool
        elif json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = self.client.models.generate_content(model=model, contents=self._convert(messages, types),
                                                   config=types.GenerateContentConfig(**cfg))
        if not resp.candidates or resp.candidates[0].content is None:
            raise LLMError(f"Gemini không trả nội dung (có thể bị chặn): {getattr(resp, 'prompt_feedback', None)}")
        content = resp.candidates[0].content
        text, calls = [], []
        for p in content.parts or []:
            if getattr(p, "function_call", None):
                fc = p.function_call
                calls.append(ToolCall(fc.id or f"call_{next(_ids)}", fc.name, dict(fc.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text.append(p.text)
        um = resp.usage_metadata
        usage = Usage(getattr(um, "prompt_token_count", 0) or 0,
                      (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0), 1)
        return LLMResponse(text="".join(text), tool_calls=calls, usage=usage, raw=content)
