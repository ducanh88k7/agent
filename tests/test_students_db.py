"""Nguồn hồ sơ sinh viên: JSON và PostgreSQL phải cho CÙNG kết quả.  PG_TEST_DSN=postgresql://postgres@localhost:5433/agentic_rag pytest"""
import json
import os
from dataclasses import replace
from datetime import date

import pytest

from agentic_rag.agent.evidence import EvidenceStore
from agentic_rag.llm.base import ToolCall
from agentic_rag.students import JsonStudentRepo, PgStudentRepo, load_students_to_pg, make_student_repo
from agentic_rag.tools import ToolContext, build_registry

DSN = os.getenv("PG_TEST_DSN")
JSON = "data/students.json"


def test_json_repo_basic():
    repo = JsonStudentRepo(JSON)
    assert repo.get("SV001")["major"] == "Công nghệ thông tin" and repo.get("SV999") is None


def test_unknown_source_is_rejected(settings):
    with pytest.raises(ValueError):
        make_student_repo(replace(settings, student_source="mongo"))


@pytest.fixture()
def pg(settings):
    import psycopg
    if not DSN:
        pytest.skip("đặt PG_TEST_DSN để chạy test PostgreSQL")
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS leave_history, students CASCADE")
    assert load_students_to_pg(DSN, JSON) == 2
    return PgStudentRepo(DSN)


def test_pg_repo_returns_same_record_as_json(pg):
    for sid in ("SV001", "SV002"):
        assert pg.get(sid) == JsonStudentRepo(JSON).get(sid)   # cùng khóa, cùng giá trị (kể cả gpa 3.35 và tổng bảo lưu)
    assert pg.get("SV999") is None


def test_loader_is_idempotent_and_updates(pg):
    import psycopg
    load_students_to_pg(DSN, JSON)                              # chạy lại không nhân đôi lịch sử bảo lưu
    assert pg.get("SV001")["total_leave_semesters"] == 1
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("UPDATE students SET gpa_semester = 1.00 WHERE student_id='SV001'")
    load_students_to_pg(DSN, JSON)                              # nạp lại ghi đè về giá trị trong JSON
    assert pg.get("SV001")["gpa_semester"] == 3.35


def test_sql_injection_in_id_is_harmless(pg):
    assert pg.get("SV001' OR '1'='1") is None


def test_tool_with_pg_source_is_bound_to_session_user(settings, pipeline, pg):
    s = replace(settings, student_source="postgres", database_url=DSN)
    reg = build_registry(s, pipeline)
    ctx = lambda user: ToolContext(EvidenceStore(), s, date(2026, 12, 20), user)  # noqa: E731
    r = reg.execute(ToolCall("1", "get_student_record", {"student_id": "SV002", "fields": ["discipline"]}), ctx("SV001"))
    assert json.loads(r.content) == {"student_id": "SV001", "discipline": "không"}      # tham số lạ bị loại, vẫn là SV001
    assert not reg.execute(ToolCall("2", "get_student_record", {}), ctx("SV404")).ok      # không có hồ sơ
    assert not reg.execute(ToolCall("3", "get_student_record", {}), ctx(None)).ok         # chưa đăng nhập
