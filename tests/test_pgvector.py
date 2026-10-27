"""Chạy khi có Postgres + pgvector:  PG_TEST_DSN=postgresql://postgres@localhost:5433/agentic_rag pytest tests/test_pgvector.py"""
import os
from dataclasses import replace

import pytest

from agentic_rag.evaluation import eval_retrieval
from agentic_rag.retrieval import build_pipeline

DSN = os.getenv("PG_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="đặt PG_TEST_DSN để chạy test pgvector")


@pytest.fixture()
def pg_pipeline(settings):
    import psycopg
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS rag_chunks, rag_parents, rag_meta CASCADE")
    p = build_pipeline(replace(settings, vector_store="pgvector", database_url=DSN))
    p.ingest_dir(settings.docs_dir)
    return p


def test_pg_retrieval_matches_golden_set(pg_pipeline):
    r = eval_retrieval(pg_pipeline, "data/eval/retrieval.jsonl")
    assert r["recall_at_k"] >= 0.9 and r["no_answer_accuracy"] == 1.0, r


def test_pg_filters_idempotent_ingest_and_fingerprint(pg_pipeline, settings):
    n = pg_pipeline.store.count()
    pg_pipeline.ingest_dir(settings.docs_dir)
    assert pg_pipeline.store.count() == n
    r = pg_pipeline.retrieve("giờ giấc", {"doc_type": "nội quy"})
    assert r.passages and all(p.metadata["doc_type"] == "nội quy" for p in r.passages)
    with pytest.raises(Exception):  # số chiều khác => cột vector(768) không khớp / fingerprint lệch
        build_pipeline(replace(settings, vector_store="pgvector", database_url=DSN, embedding_dim=512))
