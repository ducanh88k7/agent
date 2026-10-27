import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.factory import build_app  # noqa: E402
from agentic_rag.llm.mock import MockLLM  # noqa: E402
from agentic_rag.retrieval import build_pipeline  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    return replace(Settings.from_env(), llm_provider="mock", embedding_provider="hash", embedding_model="hash", vector_store="memory",
                   reranker="lexical", rerank_threshold=None, index_dir=str(tmp_path / "idx"), trace_dir=str(tmp_path / "runs"),
                   docs_dir=str(ROOT / "data/docs"), students_path=str(ROOT / "data/students.json"), today="2026-12-20")


@pytest.fixture()
def pipeline(settings):
    p = build_pipeline(settings)
    p.ingest_dir(settings.docs_dir)
    return p


@pytest.fixture()
def make_app(settings, pipeline):
    def _make(script, **over):
        llm = MockLLM(script)
        s = replace(settings, **over) if over else settings
        return build_app(s, llm=llm, pipeline=pipeline), llm
    return _make
