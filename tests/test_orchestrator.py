import re

import pytest

from agentic_rag.agent.memory import Session
from agentic_rag.llm.base import LLMError
from agentic_rag.llm.mock import json_reply, reply, tool_calls

ROUTE = lambda route, q="Q", **kw: json_reply({"route": route, "standalone_question": q, "reason": "t", "confidence": 0.9, "clarifying_question": kw.get("clarify")})  # noqa: E731
PASS = json_reply({"verdict": "pass", "grounded": True, "complete": True, "issues": [], "feedback": ""})


def test_direct_route(make_app):
    app, llm = make_app([reply("Chào bạn! Mình giúp được gì?")])
    a = app.ask("Xin chào")
    assert a.route == "direct" and a.usage["llm_calls"] == 1 and not a.sources
    llm.assert_done()


def test_clarify_route_asks_back_without_more_llm_calls(make_app):
    app, llm = make_app([ROUTE("clarify", clarify="Bạn muốn hỏi về học phí hay ký túc xá?")])
    assert app.ask("Thế còn cái kia?").text == "Bạn muốn hỏi về học phí hay ký túc xá?"
    llm.assert_done()


def test_rag_route_answers_with_citation_and_skips_critic(make_app):
    app, llm = make_app([ROUTE("rag", "Nộp đơn bảo lưu trước bao lâu?"), reply("Ít nhất 2 tuần trước khi bắt đầu học kỳ [S1].")])
    a = app.ask("Nộp đơn bảo lưu trước bao lâu?")
    assert a.route == "rag" and a.sources[0]["id"] == "S1" and "Điều 12" in a.sources[0]["section"]
    assert a.usage["llm_calls"] == 2 and not a.warnings                 # router + answer, không critic
    assert "<document" in llm.calls[1]["messages"][0].content
    llm.assert_done()


def test_rag_route_without_evidence_does_not_call_llm_to_answer(make_app):
    app, llm = make_app([ROUTE("rag", "Cách nấu phở bò ngon")])
    a = app.ask("Cách nấu phở bò ngon?")
    assert "không tìm thấy" in a.text.lower() and a.usage["llm_calls"] == 1
    llm.assert_done()


def test_agent_route_critic_revises_fake_citation(make_app):
    app, llm = make_app([
        ROUTE("agent", "Hạn nộp đơn bảo lưu?"),
        tool_calls(("search_documents", {"query": "thời hạn nộp đơn bảo lưu"})),
        reply("Nộp trước 2 tuần [S9]."),                                  # bịa trích dẫn S9
        json_reply({"verdict": "pass", "grounded": True, "complete": True, "issues": [], "feedback": ""}),  # critic không thấy, nhưng kiểm tra xác định thì thấy
        reply("Nộp trước ít nhất 2 tuần [S1]."),                          # bản sửa
        PASS])
    a = app.ask("Hạn nộp đơn bảo lưu?")
    assert a.text == "Nộp trước ít nhất 2 tuần [S1]." and not a.warnings
    assert any(e["kind"] == "check" and e["citation_issues"] for e in a.events)
    llm.assert_done()


def test_agent_route_gives_up_after_max_reflections_and_strips_fake_citations(make_app):
    bad = reply("Nộp trước 2 tuần [S5].")
    app, llm = make_app([ROUTE("agent"), tool_calls(("search_documents", {"query": "bảo lưu"})), bad, PASS, bad, PASS])
    a = app.ask("hỏi")
    assert "[S5]" not in a.text and a.warnings


def test_critic_revise_verdict_adds_caveat_when_unfixed(make_app):
    rev = json_reply({"verdict": "revise", "grounded": False, "complete": True, "issues": ["số liệu sai"], "feedback": "sửa số"})
    app, llm = make_app([ROUTE("agent"), tool_calls(("search_documents", {"query": "bảo lưu"})),
                         reply("Tối đa 9 học kỳ [S1]."), rev, reply("Tối đa 8 học kỳ [S1]."), rev])
    a = app.ask("hỏi")
    assert "Lưu ý" in a.text and "số liệu sai" in a.warnings


def test_plan_route_with_replan_after_insufficient_step(make_app):
    app, llm = make_app([
        ROUTE("plan", "Học phí và học bổng?"),
        json_reply({"steps": [{"id": "s1", "goal": "tra học phí"}, {"id": "s2", "goal": "tra học bổng theo mức học phí", "depends_on": ["s1"]}]}),   # s2 phụ thuộc s1 => s1 xong mới replan
        tool_calls(("submit_result", {"status": "insufficient", "answer": "không thấy học phí"})),
        json_reply({"steps": [{"id": "r1_1", "goal": "tra học phí ngành CNTT mỗi tín chỉ"}]}),                 # replan thay thế phần còn lại
        tool_calls(("search_documents", {"query": "học phí ngành CNTT mỗi tín chỉ"})),
        lambda m, s, t: tool_calls(("submit_result", {"status": "ok", "answer": "520.000đ/tín chỉ [S1]"})),
        reply("Học phí CNTT là 520.000đ/tín chỉ [S1]."),                                                         # synthesis
        PASS])
    a = app.ask("Học phí và học bổng?")
    assert a.route == "plan" and "520.000" in a.text
    ids = [e["id"] for e in a.events if e["kind"] == "step"]
    assert ids == ["s1", "r1_1"]                                                    # s2 bị thay bởi kế hoạch mới
    llm.assert_done()


def test_llm_failure_is_handled_gracefully(make_app):
    def boom(*_):
        raise LLMError("API sập")
    app, _ = make_app([boom])
    a = app.ask("câu hỏi bất kỳ")
    assert "sự cố" in a.text and a.warnings == ["llm_error"] and "API sập" not in a.text


def test_personal_data_comes_from_session_user(make_app):
    app, llm = make_app([ROUTE("agent"), tool_calls(("get_student_record", {"student_id": "SV001"})), reply("ok")])
    app.ask("hồ sơ của tôi", user_id="SV002")
    tool_msg = [m for m in llm.calls[2]["messages"] if m.role == "tool"][0]
    assert '"SV002"' in tool_msg.content and "SV001" not in tool_msg.content


def test_history_is_passed_to_router_for_follow_up(make_app):
    app, llm = make_app([ROUTE("rag", "Hạn nộp đơn bảo lưu?"), reply("2 tuần [S1]"),
                         ROUTE("rag", "Hạn nộp đơn bảo lưu là bao lâu?"), reply("2 tuần [S1]")])
    s = Session()
    app.ask("Quy chế bảo lưu thế nào?", s)
    app.ask("Thế phải nộp đơn trước bao lâu?", s)
    hist = llm.calls[2]["messages"]
    assert [m.role for m in hist] == ["user", "assistant", "user"] and "bảo lưu" in hist[0].content


def test_trace_is_written_to_disk(make_app, settings):
    import pathlib
    app, _ = make_app([reply("Chào!")])
    a = app.ask("hi")
    assert (pathlib.Path(settings.trace_dir) / f"{a.trace_id}.jsonl").exists()
