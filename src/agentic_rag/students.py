"""Nguồn dữ liệu hồ sơ sinh viên. Tool get_student_record chỉ biết giao diện `get(student_id)`,
nên đổi JSON <-> PostgreSQL <-> API của trường chỉ cần đổi lớp ở đây, không sửa tool hay agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class StudentRepository(Protocol):
    def get(self, student_id: str) -> dict | None: ...


class JsonStudentRepo:
    def __init__(self, path: str):
        self.path = path

    def get(self, student_id: str) -> dict | None:
        return json.loads(Path(self.path).read_text("utf-8")).get(student_id)


class PgStudentRepo:
    """Đọc từ PostgreSQL. Câu lệnh tham số hóa (%s) nên student_id không thể gây SQL injection.
    Mỗi lần gọi mở một kết nối ngắn: đơn giản cho bản học tập; bản thật dùng connection pool (psycopg_pool)."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def get(self, student_id: str) -> dict | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT name, major, gpa_semester, conduct_score, discipline, failed_courses, credits_registered, current_semester "
                "FROM students WHERE student_id = %s", (student_id,)).fetchone()
            if row is None:
                return None
            leaves = conn.execute("SELECT semester, reason FROM leave_history WHERE student_id = %s ORDER BY id", (student_id,)).fetchall()
        name, major, gpa, conduct, disc, failed, credits, sem = row
        return {
            "name": name, "major": major,
            "gpa_semester": float(gpa) if gpa is not None else None,  # numeric -> Decimal; json.dumps không nhận Decimal
            "conduct_score": conduct, "discipline": disc, "failed_courses": failed,
            "credits_registered": credits, "current_semester": sem,
            "leave_history": [{"semester": s, "reason": r} for s, r in leaves],
            "total_leave_semesters": len(leaves),
        }


def make_student_repo(settings) -> StudentRepository:
    if settings.student_source == "postgres":
        return PgStudentRepo(settings.students_database_url or settings.database_url)
    if settings.student_source == "json":
        return JsonStudentRepo(settings.students_path)
    raise ValueError(f"STUDENT_SOURCE không hợp lệ: {settings.student_source} (json | postgres)")


def load_students_to_pg(dsn: str, json_path: str, schema_path: str = "db/schema.sql") -> int:
    """Áp dụng schema rồi nạp students.json vào PostgreSQL. Chạy lại nhiều lần an toàn (upsert)."""
    import psycopg

    data = json.loads(Path(json_path).read_text("utf-8"))
    with psycopg.connect(dsn) as conn:  # một transaction: lỗi giữa chừng thì không ghi dở
        conn.execute(Path(schema_path).read_text("utf-8"))
        for sid, r in data.items():
            conn.execute(
                "INSERT INTO students (student_id, name, major, gpa_semester, conduct_score, discipline, failed_courses, credits_registered, current_semester) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (student_id) DO UPDATE SET name=EXCLUDED.name, major=EXCLUDED.major, "
                "gpa_semester=EXCLUDED.gpa_semester, conduct_score=EXCLUDED.conduct_score, discipline=EXCLUDED.discipline, "
                "failed_courses=EXCLUDED.failed_courses, credits_registered=EXCLUDED.credits_registered, current_semester=EXCLUDED.current_semester",
                (sid, r["name"], r["major"], r.get("gpa_semester"), r.get("conduct_score"), r.get("discipline", "không"),
                 r.get("failed_courses", 0), r.get("credits_registered", 0), r["current_semester"]))
            conn.execute("DELETE FROM leave_history WHERE student_id = %s", (sid,))
            for lv in r.get("leave_history", []):
                conn.execute("INSERT INTO leave_history (student_id, semester, reason) VALUES (%s,%s,%s)", (sid, lv["semester"], lv.get("reason")))
    return len(data)
