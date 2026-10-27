"""Bộ nhớ hội thoại ngắn hạn (cửa sổ N lượt gần nhất). Chỉ giữ văn bản hỏi/đáp, KHÔNG giữ vết tool để tiết kiệm token."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from ..llm.base import Message


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    user_id: str | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    max_turns: int = 6

    def history(self) -> list[Message]:
        out: list[Message] = []
        for q, a in self.turns[-self.max_turns:]:
            out += [Message("user", q), Message("assistant", a)]
        return out

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-self.max_turns * 2:]


class SessionStore:
    """Trong bộ nhớ tiến trình (đủ cho học tập). Bản thật: Redis/DB."""

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
