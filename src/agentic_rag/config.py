"""Cấu hình tập trung. Mọi tham số chỉnh được bằng biến môi trường (.env) mà không sửa code."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv là tùy chọn
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# (model chính, model nhanh/rẻ dùng cho router, critic). Kiểm tra tên model mới nhất trong tài liệu nhà cung cấp.
_PROVIDER_MODELS = {
    "gemini": ("gemini-2.5-flash", "gemini-2.5-flash-lite"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
    "mock": ("mock", "mock"),
}
_EMBED_MODELS = {"gemini": "gemini-embedding-001", "sentence-transformers": "BAAI/bge-m3", "hash": "hash"}


def _get(name: str, default):
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _int(name: str, default: int) -> int:
    return int(_get(name, default))


def _float(name: str, default: float) -> float:
    return float(_get(name, default))


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    llm_provider: str = "gemini"  # gemini | anthropic | mock
    main_model: str = ""
    fast_model: str = ""
    max_output_tokens: int = 2048
    # --- Embedding ---
    embedding_provider: str = "hash"  # hash (offline, chỉ để thử) | sentence-transformers | gemini
    embedding_model: str = "hash"
    embedding_dim: int = 768  # pgvector HNSW chỉ index được vector <= 2000 chiều
    # --- Lưu trữ ---
    vector_store: str = "memory"  # memory | pgvector
    database_url: str = "postgresql://postgres:postgres@localhost:5432/agentic_rag"
    index_dir: str = ".index"
    docs_dir: str = "data/docs"
    students_path: str = "data/students.json"
    student_source: str = "json"  # json | postgres
    students_database_url: str | None = None  # vd tài khoản chỉ-đọc; để trống = dùng DATABASE_URL
    trace_dir: str = "runs"
    # --- Retrieval ---
    reranker: str = "lexical"  # none | lexical | cross-encoder
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float | None = None  # None = dùng ngưỡng mặc định của reranker
    k_dense: int = 30
    k_sparse: int = 30
    k_candidates: int = 20
    k_final: int = 4
    child_max_chars: int = 400
    max_parent_chars: int = 2500
    # --- Agent ---
    max_llm_calls: int = 30
    max_tool_calls: int = 16
    max_total_tokens: int = 80_000
    timeout_s: float = 120.0
    react_max_steps: int = 8
    step_max_steps: int = 5
    max_plan_steps: int = 6
    max_replans: int = 1
    parallel_steps: bool = False  # True: chạy song song các bước độc lập (coi chừng rate limit free tier)
    max_tool_chars: int = 6000
    critic_mode: str = "auto"  # auto (chỉ route agent/plan) | always | off
    max_reflections: int = 1
    history_turns: int = 6
    enable_write_tools: bool = False
    today: str | None = None  # ép ngày hiện tại (YYYY-MM-DD) cho demo/test
    domain: str = "tra cứu quy chế đào tạo, học phí, học bổng, ký túc xá của Trường Đại học Mẫu"

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _get("LLM_PROVIDER", "gemini")
        main, fast = _PROVIDER_MODELS.get(provider, _PROVIDER_MODELS["gemini"])
        emb = _get("EMBEDDING_PROVIDER", "hash")
        rr = _get("RERANKER", "lexical")
        thr = os.getenv("RERANK_THRESHOLD")
        return cls(
            llm_provider=provider,
            main_model=_get("MAIN_MODEL", main),
            fast_model=_get("FAST_MODEL", fast),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 2048),
            embedding_provider=emb,
            embedding_model=_get("EMBEDDING_MODEL", _EMBED_MODELS.get(emb, "hash")),
            embedding_dim=_int("EMBEDDING_DIM", 768),
            vector_store=_get("VECTOR_STORE", "memory"),
            database_url=_get("DATABASE_URL", cls.database_url),
            index_dir=_get("INDEX_DIR", ".index"),
            docs_dir=_get("DOCS_DIR", "data/docs"),
            students_path=_get("STUDENTS_PATH", "data/students.json"),
            student_source=_get("STUDENT_SOURCE", "json"),
            students_database_url=os.getenv("STUDENTS_DATABASE_URL") or None,
            trace_dir=_get("TRACE_DIR", "runs"),
            reranker=rr,
            reranker_model=_get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            rerank_threshold=float(thr) if thr not in (None, "") else None,
            k_dense=_int("K_DENSE", 30),
            k_sparse=_int("K_SPARSE", 30),
            k_candidates=_int("K_CANDIDATES", 20),
            k_final=_int("K_FINAL", 4),
            child_max_chars=_int("CHILD_MAX_CHARS", 400),
            max_llm_calls=_int("MAX_LLM_CALLS", 30),
            max_tool_calls=_int("MAX_TOOL_CALLS", 16),
            max_total_tokens=_int("MAX_TOTAL_TOKENS", 80_000),
            timeout_s=_float("TIMEOUT_S", 120.0),
            react_max_steps=_int("REACT_MAX_STEPS", 8),
            step_max_steps=_int("STEP_MAX_STEPS", 5),
            max_plan_steps=_int("MAX_PLAN_STEPS", 6),
            max_replans=_int("MAX_REPLANS", 1),
            parallel_steps=_bool("PARALLEL_STEPS", False),
            critic_mode=_get("CRITIC_MODE", "auto"),
            max_reflections=_int("MAX_REFLECTIONS", 1),
            history_turns=_int("HISTORY_TURNS", 6),
            enable_write_tools=_bool("ENABLE_WRITE_TOOLS", False),
            today=os.getenv("TODAY") or None,
            domain=_get("DOMAIN", cls.domain),
        )
