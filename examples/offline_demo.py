"""Demo chạy OFFLINE (không cần API key): LLM được thay bằng kịch bản có sẵn, nhưng retrieval/tools/router/planner/critic
là code thật. Mục đích: xem toàn bộ luồng và trace của agent. Chạy:  python examples/offline_demo.py --scenario agent"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.factory import build_app  # noqa: E402
from agentic_rag.llm.mock import MockLLM, json_reply, reply, tool_calls  # noqa: E402
from agentic_rag.retrieval import build_pipeline  # noqa: E402

DOC = re.compile(r'<document id="(S\d+)" source="([^"]*)"')


def find_id(messages, needle: str) -> str:
    """Tìm id [S#] của đoạn tài liệu có nguồn chứa `needle` trong các kết quả tool đã có."""
    for m in reversed(messages):
        if m.role == "tool":
            for sid, src in DOC.findall(m.content):
                if needle.lower() in src.lower():
                    return sid
    return "S1"


def make_app(tmp: str | None = None, **over):
    s = replace(Settings.from_env(), llm_provider="mock", embedding_provider="hash", embedding_model="hash", vector_store="memory",
                index_dir=tmp or tempfile.mkdtemp(), trace_dir=tmp or tempfile.mkdtemp(), docs_dir=str(ROOT / "data/docs"),
                students_path=str(ROOT / "data/students.json"), today="2026-12-20", **over)
    pipe = build_pipeline(s)
    pipe.ingest_dir(s.docs_dir)
    return s, pipe


def scenario_agent(llm: MockLLM):
    q = "Em muốn bảo lưu học kỳ tới (bắt đầu 01/02/2027) thì còn đủ điều kiện không và hạn chót nộp đơn là ngày nào?"
    llm.script += [
        json_reply({"route": "agent", "standalone_question": q, "reason": "cần hồ sơ cá nhân + quy chế + tính ngày", "confidence": 0.9, "clarifying_question": None}),
        # Bước 1: hai việc độc lập => gọi song song
        tool_calls(("search_documents", {"query": "điều kiện và thời hạn nộp đơn xin bảo lưu kết quả học tập", "year": 2025}),
                   ("get_student_record", {"fields": ["discipline", "leave_history", "total_leave_semesters"]})),
        # Bước 2: tính hạn chót bằng công cụ, không tính nhẩm
        tool_calls(("date_math", {"operation": "add_days", "date": "2027-02-01", "days": -14})),
        # Bước 3: trả lời
        lambda m, s, t: reply(f"- **Hạn chót nộp đơn: 18/01/2027** (trước ngày bắt đầu học kỳ ít nhất 14 ngày) [{find_id(m, 'Điều 12')}].\n"
                              f"- **Điều kiện:** bạn không bị kỷ luật, mới bảo lưu 1/4 học kỳ cho phép trong toàn khóa, nên đủ điều kiện về mặt quy chế; "
                              f"vẫn cần lý do chính đáng (ốm đau có xác nhận, hoàn cảnh gia đình, nghĩa vụ quân sự) [{find_id(m, 'Điều 12')}]."),
        json_reply({"verdict": "pass", "grounded": True, "complete": True, "issues": [], "feedback": ""}),
    ]
    return q


def scenario_plan(llm: MockLLM):
    q = "So sánh học phí 16 tín chỉ giữa ngành CNTT và Điện tử viễn thông, và em có đủ điều kiện nhận học bổng không?"
    step = lambda tool, args, needle, text: [  # noqa: E731
        tool_calls((tool, args)),
        lambda m, s, t: tool_calls(("submit_result", {"status": "ok", "answer": text.format(id=find_id(m, needle))}))]
    llm.script += [
        json_reply({"route": "plan", "standalone_question": q, "reason": "nhiều phần độc lập: 2 mức phí, điều kiện học bổng, hồ sơ cá nhân", "confidence": 0.9, "clarifying_question": None}),
        json_reply({"steps": [
            {"id": "s1", "goal": "Tra mức học phí mỗi tín chỉ của ngành Công nghệ thông tin và Điện tử viễn thông", "tool_hint": "search_documents", "depends_on": []},
            {"id": "s2", "goal": "Tra điều kiện và mức học bổng khuyến khích học tập", "tool_hint": "search_documents", "depends_on": []},
            {"id": "s3", "goal": "Lấy điểm, điểm rèn luyện, kỷ luật, số tín chỉ của sinh viên", "tool_hint": "get_student_record", "depends_on": []},
            {"id": "s4", "goal": "Tính học phí 16 tín chỉ cho từng ngành từ mức phí đã tra", "tool_hint": "calculate", "depends_on": ["s1"]}]}),
        *step("search_documents", {"query": "mức học phí mỗi tín chỉ ngành Công nghệ thông tin Điện tử viễn thông"}, "Mức học phí",
              "CNTT 520.000đ/tín chỉ, ĐTVT 500.000đ/tín chỉ [{id}]"),
        *step("search_documents", {"query": "điều kiện xét học bổng khuyến khích học tập"}, "Điều kiện xét học bổng",
              "Cần ≥14 tín chỉ, GPA kỳ ≥3,2, rèn luyện ≥80, không kỷ luật từ khiển trách, không điểm F [{id}]"),
        *step("get_student_record", {}, "zzz", "SV001: CNTT, GPA 3,35, rèn luyện 85, không kỷ luật, 16 tín chỉ, 0 môn F"),
        tool_calls(("calculate", {"expression": "16 * 520000"}), ("calculate", {"expression": "16 * 500000"})),
        tool_calls(("submit_result", {"status": "ok", "answer": "16 tín chỉ: CNTT 8.320.000đ; ĐTVT 8.000.000đ"})),
        # Bước tổng hợp: lấy id trích dẫn từ chính kết quả các bước (s1 = học phí, s2 = học bổng)
        lambda m, s, t: reply("**Học phí 16 tín chỉ:** CNTT 8.320.000đ, ĐTVT 8.000.000đ (chênh 320.000đ) [{0}].\n"
                              "**Học bổng:** bạn đủ điều kiện loại Giỏi (GPA 3,35; rèn luyện 85; 16 tín chỉ; không kỷ luật) [{1}], "
                              "tương đương 100% học phí học kỳ.".format(*re.findall(r"\[(S\d+)\]", m[-1].content)[:2])),
        json_reply({"verdict": "pass", "grounded": True, "complete": True, "issues": [], "feedback": ""}),
    ]
    return q


SCENARIOS = {"agent": scenario_agent, "plan": scenario_plan}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="agent")
    names = list(SCENARIOS) if ap.parse_args().scenario == "all" else [ap.parse_args().scenario]
    for name in names:
        s, pipe = make_app()
        llm = MockLLM()
        q = SCENARIOS[name](llm)
        app = build_app(s, llm=llm, pipeline=pipe)
        print(f"\n{'=' * 78}\nKỊCH BẢN: {name}\nCÂU HỎI : {q}\n{'=' * 78}")
        ans = app.ask(q, user_id="SV001", verbose=True)
        print(f"\nROUTE: {ans.route} | usage: {ans.usage}\n\n{ans.text}\n\nNguồn:")
        for src in ans.sources:
            print(f"  [{src['id']}] {src['section']} (điểm {src['score']})")
        llm.assert_done()
