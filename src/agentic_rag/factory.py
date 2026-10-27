"""Lắp ráp toàn bộ hệ thống từ Settings. Cho phép tiêm llm/pipeline giả khi test."""
from __future__ import annotations

from .agent.memory import SessionStore
from .agent.orchestrator import AgenticRAG
from .config import Settings
from .llm import make_llm
from .retrieval import build_pipeline
from .tools import build_registry


def build_app(settings: Settings | None = None, *, llm=None, pipeline=None) -> AgenticRAG:
    settings = settings or Settings.from_env()
    pipeline = pipeline or build_pipeline(settings)
    llm = llm or make_llm(settings)
    return AgenticRAG(settings, llm, pipeline, build_registry(settings, pipeline))


__all__ = ["build_app", "SessionStore", "Settings"]
