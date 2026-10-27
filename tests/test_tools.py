import json

from agentic_rag.agent.evidence import EvidenceStore
from agentic_rag.llm.base import ToolCall
from agentic_rag.tools import ToolContext, build_registry, validate_args
from agentic_rag.tools.builtin import safe_eval
from datetime import date
import pytest


def ctx(settings, user=None, confirm=None):
    return ToolContext(EvidenceStore(), settings, date(2026, 12, 20), user, confirm)


def test_validate_args_coerces_and_drops_unknown():
    schema = {"properties": {"days": {"type": "integer"}, "op": {"type": "string", "enum": ["a", "b"]}}, "required": ["op"]}
    assert validate_args(schema, {"days": "5", "op": "a", "student_id": "SV999"}) == ({"days": 5, "op": "a"}, None)
    assert validate_args(schema, {"op": "z"})[1] is not None
    assert "thiếu" in validate_args(schema, {})[1]


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "2**9999", "(1).__class__"])
def test_calculator_rejects_code(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_calculator_and_date_tools(settings, pipeline):
    reg = build_registry(settings, pipeline)
    r = reg.execute(ToolCall("1", "calculate", {"expression": "16 * 520000"}), ctx(settings))
    assert r.ok and "8320000" in r.content
    r = reg.execute(ToolCall("2", "date_math", {"operation": "add_days", "date": "01/02/2027", "days": -14}), ctx(settings))
    assert r.ok and "2027-01-18" in r.content


def test_student_record_bound_to_session_user_not_to_llm_argument(settings, pipeline):
    reg = build_registry(settings, pipeline)
    # LLM cố truyền student_id của người khác: tham số bị loại, dữ liệu vẫn là của người đăng nhập
    r = reg.execute(ToolCall("1", "get_student_record", {"student_id": "SV002"}), ctx(settings, user="SV001"))
    assert json.loads(r.content)["student_id"] == "SV001"
    assert not reg.execute(ToolCall("2", "get_student_record", {}), ctx(settings, user=None)).ok


def test_unknown_tool_lists_available(settings, pipeline):
    r = build_registry(settings, pipeline).execute(ToolCall("1", "rm_rf", {}), ctx(settings))
    assert not r.ok and "search_documents" in r.content


def test_write_tool_requires_confirmation(settings, pipeline, tmp_path):
    from dataclasses import replace
    s = replace(settings, enable_write_tools=True, trace_dir=str(tmp_path))
    reg = build_registry(s, pipeline)
    call = ToolCall("1", "submit_leave_request", {"semester": "HK2 2026-2027", "reason": "ốm"})
    assert not reg.execute(call, ctx(s, "SV001")).ok                                  # không có callback => từ chối
    assert not reg.execute(call, ctx(s, "SV001", confirm=lambda n, a: False)).ok      # người dùng từ chối
    assert reg.execute(call, ctx(s, "SV001", confirm=lambda n, a: True)).ok           # người dùng đồng ý


def test_search_tool_assigns_stable_evidence_ids_and_flags_injection(settings, pipeline):
    reg = build_registry(settings, pipeline)
    c = ctx(settings)
    a = reg.execute(ToolCall("1", "search_documents", {"query": "nộp đơn bảo lưu"}), c)
    b = reg.execute(ToolCall("2", "search_documents", {"query": "điều kiện bảo lưu kết quả học tập"}), c)
    assert '<document id="S1"' in a.content and '<document id="S1"' in b.content   # cùng đoạn => cùng id
    before = len(c.evidence)
    n = reg.execute(ToolCall("3", "search_documents", {"query": "cách nấu phở bò"}), c)
    assert "Không tìm thấy" in n.content and len(c.evidence) == before   # tìm trượt không tạo bằng chứng
