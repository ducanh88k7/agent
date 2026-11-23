"""Bọc LLMClient: mỗi lần gọi LLM (router, planner, critic, ReAct, tổng hợp, direct...) thành một span `llm_call` trong trace.

Một bản TracedLLM chỉ dùng cho MỘT request (gắn với tracer của request đó), nên không có chuyện lẫn trace giữa các request.
"""
from __future__ import annotations

import time

from ..retry import last_retry_count


class TracedLLM:
    def __init__(self, inner, tracer):
        self._inner, self._tracer = inner, tracer

    def __getattr__(self, name):  # name, default_model, ... đi thẳng xuống client thật
        return getattr(self._inner, name)

    def _record(self, kw: dict, label, t0: float, **extra) -> None:
        model = kw.get("model") or getattr(self._inner, "default_model", "")
        if not label:  # không có nhãn thì lấy đầu system prompt để biết đây là lời gọi nào
            label = (kw.get("system") or "")[:48].replace("\n", " ")
        self._tracer.event("llm_call", label=label, model=model, ms=int((time.monotonic() - t0) * 1000),
                           json=bool(kw.get("json_mode")), **extra)

    def chat(self, messages, *, trace_label=None, **kw):
        t0 = time.monotonic()
        try:
            resp = self._inner.chat(messages, **kw)
        except Exception as e:  # noqa: BLE001
            self._record(kw, trace_label, t0, retries=last_retry_count(), error=f"{type(e).__name__}: {e}"[:200])
            raise
        u = resp.usage
        self._record(kw, trace_label, t0, tokens_in=u.input_tokens, tokens_out=u.output_tokens,
                     tool_calls=[c.name for c in resp.tool_calls], retries=last_retry_count())
        return resp

    def chat_stream(self, messages, **kw):
        t0 = time.monotonic()
        gen = self._inner.chat_stream(messages, **kw)

        def wrapped():
            first_ms, chunks, err = None, 0, None
            try:
                for chunk in gen:
                    if first_ms is None:
                        first_ms = int((time.monotonic() - t0) * 1000)
                    chunks += 1
                    yield chunk
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
                raise
            finally:
                extra = {"stream": True, "chunks": chunks, "ttft_ms": first_ms}
                if err:
                    extra["error"] = err
                self._record(kw, None, t0, **extra)

        return wrapped()