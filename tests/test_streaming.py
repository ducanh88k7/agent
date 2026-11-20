"""Unit test cho Streaming SSE và đo đạc TTFT (Time To First Token)."""
import json
import time
from dataclasses import replace

import pytest

from agentic_rag.agent.orchestrator import StreamEvent
from agentic_rag.factory import build_app
from agentic_rag.llm.mock import MockLLM, reply, tool_calls


def test_stream_event_sse_format():
    ev = StreamEvent("token", {"chunk": "Xin chào"})
    sse = ev.to_sse()
    assert sse.startswith("event: token\n")
    assert 'data: {"chunk": "Xin chào"}' in sse
    assert sse.endswith("\n\n")


def test_ask_stream_direct_and_ttft(make_app):
    app, llm = make_app([reply("Chào bạn! Mình có thể giúp gì?")])

    events = list(app.ask_stream("Xin chào"))
    kinds = [e.event for e in events]

    assert "status" in kinds
    assert "token" in kinds
    assert "done" in kinds

    # Kiểm tra ghép các token lại thành câu trả lời
    tokens = [e.data["chunk"] for e in events if e.event == "token"]
    full_text = "".join(tokens)
    assert "Chào bạn" in full_text

    # Kiểm tra đo lường TTFT và độ trễ
    done_ev = next(e for e in events if e.event == "done")
    d = done_ev.data
    assert d["route"] == "direct"
    assert "ttft_ms" in d
    assert "total_time_ms" in d
    assert d["ttft_ms"] >= 0
    assert d["total_time_ms"] >= d["ttft_ms"]
    assert d["answer"] == full_text


def test_ask_stream_empty_question(make_app):
    app, _ = make_app([])
    events = list(app.ask_stream(""))
    kinds = [e.event for e in events]
    assert kinds == ["status", "token", "done"]
    done_ev = next(e for e in events if e.event == "done")
    assert done_ev.data["route"] == "clarify"
    assert "Bạn muốn hỏi điều gì?" in done_ev.data["answer"]


def test_ask_stream_measures_ttft_accurately(make_app):
    # Giả lập model có độ trễ 50ms trước khi trả token đầu tiên
    def delayed_chat(messages, system, tools):
        time.sleep(0.05)
        return reply("Nội dung sau khi suy nghĩ.")

    app, _ = make_app([delayed_chat])

    events = list(app.ask_stream("Hello"))
    done_ev = next(e for e in events if e.event == "done")

    # TTFT phải ghi nhận ít nhất độ trễ ~50ms
    assert done_ev.data["ttft_ms"] >= 40.0
