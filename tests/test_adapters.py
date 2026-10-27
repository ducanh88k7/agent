"""Kiểm tra phần CHUYỂN ĐỔI message/schema của adapter (không gọi mạng, dùng API key giả).
Không thay thế việc thử với API thật - xem README mục 'Đã kiểm tra gì'."""
import pytest

from agentic_rag.llm.base import Message, ToolCall
from agentic_rag.tools import build_registry

HISTORY = [
    Message("user", "Hỏi gì đó"),
    Message("assistant", "", tool_calls=[ToolCall("c1", "calculate", {"expression": "1+1"}), ToolCall("c2", "date_math", {"operation": "weekday", "date": "2027-02-01"})]),
    Message("tool", "= 2", tool_call_id="c1", tool_name="calculate"),
    Message("tool", "Thứ Hai", tool_call_id="c2", tool_name="date_math"),
    Message("user", "[Hệ thống] hãy kết luận"),
]


def test_anthropic_message_conversion():
    pytest.importorskip("anthropic")
    from agentic_rag.llm.anthropic_client import AnthropicClient
    out = AnthropicClient("m", api_key="x")._convert(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]               # xen kẽ vai trò
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    blocks = out[2]["content"]                                                      # 2 tool_result + 1 text gộp trong 1 lượt user
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result", "text"]
    assert [b.get("tool_use_id") for b in blocks[:2]] == ["c1", "c2"]


def test_gemini_message_and_schema_conversion(settings, pipeline):
    pytest.importorskip("google.genai")
    from google.genai import types
    from agentic_rag.llm.gemini_client import GeminiClient
    c = GeminiClient("m", api_key="x")
    contents = c._convert(HISTORY, types)
    assert [x.role for x in contents] == ["user", "model", "user", "user"]
    assert len(contents[2].parts) == 2 and all(p.function_response for p in contents[2].parts)   # kết quả tool gộp một lượt
    assert [p.function_call.name for p in contents[1].parts] == ["calculate", "date_math"]
    # schema của MỌI tool phải được SDK chấp nhận
    from agentic_rag.llm.gemini_client import _gemini_schema
    for spec in build_registry(settings, pipeline).specs():
        types.FunctionDeclaration(name=spec.name, description=spec.description, parameters=_gemini_schema(spec.parameters))
