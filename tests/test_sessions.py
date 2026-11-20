"""Unit test cho module Session và SessionStore (Memory & Postgres)."""
import os
from dataclasses import replace

import pytest

from agentic_rag.agent.memory import PgSessionStore, Session, SessionStore, make_session_store
from agentic_rag.config import Settings

PG_DSN = os.getenv("PG_TEST_DSN")


def test_session_turns_and_history():
    s = Session(id="test1", user_id="U1", max_turns=2)
    s.add_turn("Chào bạn", "Chào bạn, mình có thể giúp gì?")
    s.add_turn("Học phí là bao nhiêu?", "Học phí là 500k/tín chỉ.")

    msgs = s.history()
    assert len(msgs) == 4
    assert msgs[0].role == "user" and msgs[0].content == "Chào bạn"
    assert msgs[3].role == "assistant" and "500k" in msgs[3].content

    # Vượt quá max_turns=2: chỉ lấy 2 lượt (4 messages) gần nhất trong history()
    s.add_turn("Còn học bổng?", "Học bổng loại A, B, C.")
    msgs_after = s.history()
    assert len(msgs_after) == 4
    assert msgs_after[0].content == "Học phí là bao nhiêu?"
    assert msgs_after[2].content == "Còn học bổng?"


def test_session_on_update_callback():
    called = []
    s = Session(id="test_cb", on_update=lambda sess: called.append(sess.id))
    s.add_turn("q1", "a1")
    assert called == ["test_cb"]


def test_memory_session_store():
    store = SessionStore(max_turns=3)
    s1 = store.get(None, user_id="U1")
    assert s1.id and s1.user_id == "U1"

    # Lấy lại session cũ bằng session_id
    s1_retrieved = store.get(s1.id)
    assert s1_retrieved is s1

    # Tạo session mới riêng biệt
    s2 = store.get(None, user_id="U2")
    assert s2.id != s1.id


def test_make_session_store_factory():
    s_mem = replace(Settings.from_env(), session_store="memory")
    store_mem = make_session_store(s_mem)
    assert isinstance(store_mem, SessionStore)

    s_bad = replace(Settings.from_env(), session_store="redis")
    with pytest.raises(ValueError, match="SESSION_STORE không hợp lệ"):
        make_session_store(s_bad)


@pytest.fixture()
def pg_store():
    if not PG_DSN:
        pytest.skip("Đặt biến PG_TEST_DSN để test PostgreSQL session store")
    import psycopg

    with psycopg.connect(PG_DSN, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS chat_sessions CASCADE")

    return PgSessionStore(PG_DSN, max_turns=4)


def test_pg_session_store_lifecycle(pg_store):
    # 1. Tạo session mới
    sess = pg_store.get(None, user_id="SV001")
    sid = sess.id
    assert sid

    # 2. Thêm turn -> tự động trigger save vào DB
    sess.add_turn("Hạn nộp học phí khi nào?", "Hạn là ngày 15/12.")

    # 3. Mở store mới kết nối lại DB và lấy session qua id
    new_store = PgSessionStore(PG_DSN, max_turns=4)
    loaded = new_store.get(sid)
    assert loaded.id == sid
    assert loaded.user_id == "SV001"
    assert len(loaded.turns) == 1
    assert loaded.turns[0] == ("Hạn nộp học phí khi nào?", "Hạn là ngày 15/12.")

    # 4. Thêm turn thứ 2 và kiểm tra cập nhật
    loaded.add_turn("Thế còn bảo lưu?", "Hạn bảo lưu là tuần 4.")
    reloaded = new_store.get(sid)
    assert len(reloaded.turns) == 2
    assert reloaded.turns[1][0] == "Thế còn bảo lưu?"
