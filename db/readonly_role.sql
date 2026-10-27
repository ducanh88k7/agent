-- TÙY CHỌN nhưng nên làm: tài khoản CHỈ ĐỌC cho agent.
-- Nếu agent bị lừa (prompt injection) hay code có lỗi, nó cũng không thể sửa/xóa hồ sơ, không đọc được bảng khác.
-- Chạy bằng tài khoản quản trị, nhớ ĐỔI mật khẩu:
--   docker compose exec -T db psql -U postgres -d agentic_rag < db/readonly_role.sql
CREATE ROLE agent_reader LOGIN PASSWORD 'doi-mat-khau-nay';
GRANT CONNECT ON DATABASE agentic_rag TO agent_reader;
GRANT USAGE ON SCHEMA public TO agent_reader;
GRANT SELECT ON students, leave_history TO agent_reader;
-- Sau đó đặt trong .env:  STUDENTS_DATABASE_URL=postgresql://agent_reader:<mật khẩu>@localhost:5433/agentic_rag
