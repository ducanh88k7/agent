"""Bộ nhớ hội thoại ngắn hạn (cửa sổ N lượt gần nhất). Chỉ giữ văn bản hỏi/đáp, KHÔNG giữ vết tool để tiết kiệm token.
Hỗ trợ cả lưu trữ trong bộ nhớ RAM (MemorySessionStore) và lưu trữ bền vững trong PostgreSQL (PgSessionStore)."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

from ..llm.base import Message


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    user_id: str | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    max_turns: int = 6
    on_update: Callable[[Session], None] | None = field(default=None, repr=False)

    def history(self) -> list[Message]:
        out: list[Message] = []
        for q, a in self.turns[-self.max_turns:]:
            out += [Message("user", q), Message("assistant", a)]
        return out

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-self.max_turns * 2:]
        if self.on_update:
            self.on_update(self)


class SessionStore:
    """Trong bộ nhớ tiến trình (đủ cho học tập và test nhanh)."""

    def __init__(self, max_turns: int = 6):
        self._s: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_turns = max_turns

    def get(self, session_id: str | None, user_id: str | None = None) -> Session:
        with self._lock:
            if session_id and session_id in self._s:
                return self._s[session_id]
            s = Session(id=session_id or uuid.uuid4().hex[:10], user_id=user_id, max_turns=self.max_turns)
            self._s[s.id] = s
            return s


class PgSessionStore:
    """Lưu trữ session trong PostgreSQL. Hỗ trợ multi-worker API và không mất phiên khi restart server."""

    def __init__(self, dsn: str, max_turns: int = 6, auto_init: bool = True):
        self.dsn = dsn
        self.max_turns = max_turns
        if auto_init:
            self._ensure_table()

    def _ensure_table(self) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id  text PRIMARY KEY,
                    user_id     text,
                    turns       jsonb NOT NULL DEFAULT '[]'::jsonb,
                    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def save(self, session: Session) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            conn.execute("""
                INSERT INTO chat_sessions (session_id, user_id, turns, updated_at)
                VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) DO UPDATE SET
                    turns = EXCLUDED.turns,
                    updated_at = CURRENT_TIMESTAMP;
            """, (session.id, session.user_id, json.dumps(session.turns, ensure_ascii=False)))

    def get(self, session_id: str | None, user_id: str | None = None) -> Session:
        import psycopg

        if session_id:
            with psycopg.connect(self.dsn) as conn:
                row = conn.execute(
                    "SELECT user_id, turns FROM chat_sessions WHERE session_id = %s",
                    (session_id,)
                ).fetchone()
                if row:
                    stored_user_id, raw_turns = row
                    turns = [tuple(t) for t in raw_turns] if isinstance(raw_turns, list) else []
                    return Session(
                        id=session_id,
                        user_id=stored_user_id,
                        turns=turns,
                        max_turns=self.max_turns,
                        on_update=self.save,
                    )

        # Tạo mới session nếu chưa có
        sid = session_id or uuid.uuid4().hex[:10]
        s = Session(id=sid, user_id=user_id, max_turns=self.max_turns, on_update=self.save)
        self.save(s)
        return s


def make_session_store(settings) -> SessionStore | PgSessionStore:
    backend = getattr(settings, "session_store", "memory")
    if backend == "postgres":
        return PgSessionStore(settings.database_url, max_turns=settings.history_turns)
    if backend == "memory":
        return SessionStore(max_turns=settings.history_turns)
    raise ValueError(f"SESSION_STORE không hợp lệ: {backend!r} (memory | postgres)")
