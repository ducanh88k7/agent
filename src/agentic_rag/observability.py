"""Tracer: ghi lại mọi bước (route, LLM, tool, plan, critic...) để debug agent. Không có trace thì gần như không debug được."""
from __future__ import annotations

import json
import threading
import time
import uuid
import os
from pathlib import Path

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None


class Tracer:
    def __init__(self, verbose: bool = False, sink_dir: str | None = None, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.verbose, self.sink_dir = verbose, sink_dir
        self.events: list[dict] = []
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self.langfuse = (
            Langfuse()
            if (
                Langfuse is not None
                and os.getenv("LANGFUSE_PUBLIC_KEY")
                and os.getenv("LANGFUSE_SECRET_KEY")
            )
            else None
        )
        self.trace = self.langfuse.trace(name="agentic_rag", id=self.trace_id) if self.langfuse else None

    def event(self, kind: str, **data) -> None:
         # Khi có tool call hoặc LLM call, push một span lên Langfuse
        if self.trace:
            span = self.trace.span(name=kind, metadata=data)
        span.end()
        ev = {"t": round(time.monotonic() - self._t0, 2), "kind": kind, **data}
        with self._lock:
            self.events.append(ev)
        if self.verbose:
            print(self._fmt(ev), flush=True)

    @staticmethod
    def _fmt(ev: dict) -> str:
        d = {k: v for k, v in ev.items() if k not in ("t", "kind")}
        body = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)[:160] if not isinstance(v, str) else v[:160]}" for k, v in d.items())
        return f"  [{ev['t']:6.2f}s] {ev['kind'].upper():<8} {body}".replace("\n", " ")

    def flush(self) -> str | None:
        if not self.sink_dir:
            return None
        Path(self.sink_dir).mkdir(parents=True, exist_ok=True)
        path = Path(self.sink_dir) / f"{self.trace_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        # Close Langfuse trace and shutdown client if used
        if self.trace:
            try:
                self.trace.end()
            except Exception:
                pass
        if self.langfuse:
            try:
                self.langfuse.shutdown()
            except Exception:
                pass
        return str(path)

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["kind"] == kind)
