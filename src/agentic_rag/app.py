"""API HTTP (FastAPI). Chạy: python -m agentic_rag.cli serve   hoặc   uvicorn agentic_rag.app:app"""

from __future__ import annotations

import os
import time
from dataclasses import replace

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pathlib

from pydantic import BaseModel, Field

from .agent.memory import SessionStore, make_session_store
from .config import Settings
from .factory import build_app

# -------------------------------------------------
# Configuration & FastAPI app
# -------------------------------------------------
settings = replace(Settings.from_env(), enable_write_tools=False)
app = FastAPI(title="Agentic RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # cho mọi origin (development); hạn chế khi production
    allow_methods=["*"],        # GET, POST, OPTIONS, v.v.
    allow_headers=["*"],        # Accept, Content-Type, X-User-Id, X-Api-Key …
)

# Serve simple UI for SSE demo (frontend/index.html)
frontend_path = pathlib.Path(__file__).parents[2] / "frontend"

_engine = None
_sessions = make_session_store(settings)

def engine():
    global _engine
    if _engine is None:
        _engine = build_app(settings)
    return _engine

# -------------------------------------------------
# Request models
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)

# -------------------------------------------------
# Error handling
# -------------------------------------------------
@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}},
    )

# -------------------------------------------------
# Health endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    e = engine()
    return {
        "status": "ok",
        "chunks": e.pipeline.store.count(),
        "llm": settings.llm_provider,
        "session_store": settings.session_store,
    }

# -------------------------------------------------
# Synchronous chat endpoint
# -------------------------------------------------
@app.post("/chat")
def chat(
    req: ChatRequest,
    x_user_id: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "API key không hợp lệ")
    session = _sessions.get(req.session_id, x_user_id)
    if session.user_id != x_user_id:
        raise HTTPException(403, "Session không thuộc về người dùng này")
    ans = engine().ask(req.message, session, user_id=x_user_id)
    return {**ans.to_dict(), "session_id": session.id}

# -------------------------------------------------
# Streaming chat endpoint (SSE) – includes TTFT measurement
# -------------------------------------------------
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    x_user_id: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "API key không hợp lệ")
    session = _sessions.get(req.session_id, x_user_id)
    if session.user_id != x_user_id:
        raise HTTPException(403, "Session không thuộc về người dùng này")

    request_start = time.time()

    def event_generator():
        first = True
        for event in engine().ask_stream(req.message, session, user_id=x_user_id):
            if first:
                ttft_ms = int((time.time() - request_start) * 1000)
                # Thêm TTFT vào metadata của event đầu tiên
                if hasattr(event, "metadata") and isinstance(event.metadata, dict):
                    event.metadata["ttft_ms"] = ttft_ms
                else:
                    try:
                        event = event.copy(update={"metadata": {"ttft_ms": ttft_ms}})
                    except Exception:
                        pass
                first = False
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
