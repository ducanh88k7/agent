"""API HTTP (FastAPI). Chạy: python -m agentic_rag.cli serve   hoặc   uvicorn agentic_rag.app:app

Lưu ý bảo mật: ở đây user_id lấy từ header X-User-Id để demo. Bản thật PHẢI lấy từ xác thực (JWT/session),
vì tool get_student_record tin tưởng user_id này. Tool ghi (submit_leave_request) bị tắt khi chạy qua API vì chưa có luồng xác nhận.
"""
from __future__ import annotations

import os
from dataclasses import replace

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent.memory import SessionStore
from .config import Settings
from .factory import build_app

settings = replace(Settings.from_env(), enable_write_tools=False)
app = FastAPI(title="Agentic RAG", version="0.1.0")
_engine = None
_sessions = SessionStore(settings.history_turns)


def engine():
    global _engine
    if _engine is None:
        _engine = build_app(settings)
    return _engine


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}})


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "chunks": e.pipeline.store.count(), "llm": settings.llm_provider}


@app.post("/chat")
def chat(req: ChatRequest, x_user_id: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "API key không hợp lệ")
    session = _sessions.get(req.session_id, x_user_id)
    if session.user_id != x_user_id:  # chống dùng session_id của người khác
        raise HTTPException(403, "Session không thuộc về người dùng này")
    ans = engine().ask(req.message, session, user_id=x_user_id)
    return {**ans.to_dict(), "session_id": session.id}
