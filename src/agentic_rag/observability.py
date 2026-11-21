"""Tracer: ghi lại mọi bước (route, LLM, tool, plan, critic...) để debug agent. Không có trace thì gần như không debug được.

Tương thích với Langfuse v2 (API cũ trace/span) và v3/v4 (OpenTelemetry native).
Cơ chế: thử import theo kiểu v2, nếu không khả dụng thì dùng OTel spans (v3/v4),
nếu cả hai đều không có thì chạy offline (chỉ ghi file).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
import os
from pathlib import Path
from contextlib import contextmanager

# --- Phát hiện phiên bản Langfuse ------------------------------------------
_LANGFUSE_V2 = False   # Langfuse < 3  (API trace/span)
_LANGFUSE_V3 = False   # Langfuse >= 3 (OpenTelemetry native)
Langfuse = None

try:
    import langfuse as _lf_mod
    _lf_ver = tuple(int(x) for x in _lf_mod.__version__.split(".")[:2])
    if _lf_ver >= (3, 0):
        from langfuse import Langfuse as _Lf          # v3/v4
        Langfuse = _Lf
        _LANGFUSE_V3 = True
    else:
        from langfuse import Langfuse as _Lf          # v2
        Langfuse = _Lf
        _LANGFUSE_V2 = True
except Exception:
    pass


# kind của sự kiện -> loại observation hiển thị trong Langfuse
_LF_TYPES = {"llm": "generation", "tool": "tool", "retrieve": "retriever"}


class Tracer:
    def __init__(self, verbose: bool = False, sink_dir: str | None = None, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.verbose, self.sink_dir = verbose, sink_dir
        self.events: list[dict] = []
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._root = None  # span gốc của trace (v3/v4)
        self._last_t = 0.0  # mốc thời gian của sự kiện trước, để tính dt

        has_keys = bool(
            Langfuse is not None
            and os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        )

        if _LANGFUSE_V2 and has_keys:
            # --- v2: API trace/span truyền thống ---
            try:
                self.langfuse = Langfuse()
                self.trace = self.langfuse.trace(name="agentic_rag", id=self.trace_id)
            except Exception:
                self.langfuse = None
                self.trace = None
        elif _LANGFUSE_V3 and has_keys:
            # --- v3/v4: dùng API gốc của Langfuse để mọi bước nằm chung MỘT trace ---
            try:
                self.langfuse = Langfuse()
                lf_trace_id = self.langfuse.create_trace_id(seed=self.trace_id)  # id 32 ký tự hex, suy ra từ trace_id của mình
                self._root = self.langfuse.start_observation(
                    trace_context={"trace_id": lf_trace_id}, name="agentic_rag", as_type="span",
                    metadata={"trace_id": self.trace_id})
                self.trace = True  # sentinel: langfuse đã sẵn sàng
            except Exception:
                self.langfuse = None
                self.trace = None
                self._root = None
        else:
            self.langfuse = None
            self.trace = None

    # -------------------------------------------------------------------------
    def event(self, kind: str, **data) -> None:
        """Ghi một sự kiện; nếu Langfuse đang chạy thì đẩy span lên cloud."""
        now = round(time.monotonic() - self._t0, 2)
        with self._lock:
            dt = round(now - self._last_t, 2)  # thời gian trôi qua kể từ sự kiện trước = thời gian của bước này
            self._last_t = now
        if _LANGFUSE_V2 and self.trace is not None:
            try:
                span = self.trace.span(name=kind, metadata=data)
                span.end()
            except Exception:
                pass
        elif _LANGFUSE_V3 and self._root is not None:
            try:
                meta = json.loads(json.dumps({"t": now, "dt": dt, **data}, ensure_ascii=False, default=str))
                kw: dict = {"name": kind, "as_type": _LF_TYPES.get(kind, "span"), "metadata": meta}
                if kind == "error":
                    kw["level"] = "ERROR"
                if kind == "llm" and "tokens_in" in data:
                    kw["usage_details"] = {"input": int(data.get("tokens_in") or 0), "output": int(data.get("tokens_out") or 0)}
                self._root.start_observation(**kw).end()
            except Exception:
                pass

        ev = {"t": now, "kind": kind, **data}
        with self._lock:
            self.events.append(ev)
        if self.verbose:
            print(self._fmt(ev), flush=True)

    @staticmethod
    def _fmt(ev: dict) -> str:
        d = {k: v for k, v in ev.items() if k not in ("t", "kind")}
        body = " ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)[:160] if not isinstance(v, str) else v[:160]}"
            for k, v in d.items()
        )
        return f"  [{ev['t']:6.2f}s] {ev['kind'].upper():<8} {body}".replace("\n", " ")

    def flush(self) -> str | None:
        """Ghi file trace + đóng kết nối Langfuse."""
        # Ghi JSONL ra disk
        path: str | None = None
        if self.sink_dir:
            Path(self.sink_dir).mkdir(parents=True, exist_ok=True)
            p = Path(self.sink_dir) / f"{self.trace_id}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for ev in self.events:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            path = str(p)

        # Đóng Langfuse
        if _LANGFUSE_V2 and self.trace is not None:
            try:
                self.trace.end()
            except Exception:
                pass
        if _LANGFUSE_V3 and self._root is not None:
            try:
                self._root.end()
            except Exception:
                pass
        if self.langfuse:
            try:
                self.langfuse.flush()
            except Exception:
                pass

        return path

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["kind"] == kind)