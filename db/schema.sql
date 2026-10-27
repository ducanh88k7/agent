-- Dữ liệu sinh viên (nguồn sự thật cho tool get_student_record).
-- Chạy được nhiều lần: python -m agentic_rag.cli load-students   (tự áp dụng file này)

CREATE TABLE IF NOT EXISTS students (
    student_id          text PRIMARY KEY,
    name                text    NOT NULL,
    major               text    NOT NULL,
    gpa_semester        numeric(3,2) CHECK (gpa_semester BETWEEN 0 AND 4),
    conduct_score       integer CHECK (conduct_score BETWEEN 0 AND 100),
    discipline          text    NOT NULL DEFAULT 'không',   -- không | khiển trách | cảnh cáo | ...
    failed_courses      integer NOT NULL DEFAULT 0 CHECK (failed_courses >= 0),
    credits_registered  integer NOT NULL DEFAULT 0 CHECK (credits_registered >= 0),
    current_semester    text    NOT NULL
);

-- Tách lịch sử bảo lưu thành bảng riêng; tổng số học kỳ bảo lưu được ĐẾM, không lưu trùng một con số dễ lệch.
CREATE TABLE IF NOT EXISTS leave_history (
    id          bigserial PRIMARY KEY,
    student_id  text NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    semester    text NOT NULL,
    reason      text,
    UNIQUE (student_id, semester)
);
