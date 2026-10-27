"""PostgreSQL + pgvector. Dense: cosine (<=>) với HNSW. Sparse: full-text search ('simple', đã bỏ dấu).

Lưu ý: HNSW của pgvector chỉ index được vector <= 2000 chiều, nên chọn EMBEDDING_DIM <= 2000 (vd 768).
"""
from __future__ import annotations

import json

import numpy as np

from ..text import tokenize
from .chunking import Chunk, Parent
from .stores import VectorStore


def _fts(text: str) -> str:
    return " ".join(tokenize(text, stop=True))


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, dim: int = 768):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        boot = psycopg.connect(dsn, autocommit=True)
        boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
        boot.close()
        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)
        self._init_schema()

    def _init_schema(self):
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS rag_meta (key text PRIMARY KEY, value jsonb)")
        c.execute("CREATE TABLE IF NOT EXISTS rag_parents (id text PRIMARY KEY, doc_id text, title text, text text, metadata jsonb)")
        c.execute(f"""CREATE TABLE IF NOT EXISTS rag_chunks (
            id text PRIMARY KEY, doc_id text, parent_id text REFERENCES rag_parents(id) ON DELETE CASCADE,
            body text, text text, metadata jsonb, embedding vector({self.dim}),
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', fts_text)) STORED, fts_text text)""")
        c.execute("CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin (tsv)")
        c.execute("CREATE INDEX IF NOT EXISTS rag_chunks_meta ON rag_chunks USING gin (metadata)")
        if self.dim <= 2000:
            c.execute("CREATE INDEX IF NOT EXISTS rag_chunks_hnsw ON rag_chunks USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def _where(filters: dict | None) -> tuple[str, list]:
        conds, params = [], []
        for k, want in (filters or {}).items():
            if want is None:
                continue
            if isinstance(want, (list, tuple, set)):
                conds.append("lower(metadata->>%s) = ANY(%s)")
                params += [k, [str(w).lower() for w in want]]
            else:
                conds.append("lower(metadata->>%s) = %s")
                params += [k, str(want).lower()]
        return (" AND ".join(conds) if conds else "TRUE"), params

    def add(self, parents, chunks, vectors):
        with self.conn.transaction():
            for p in parents:
                self.conn.execute(
                    "INSERT INTO rag_parents VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, title=EXCLUDED.title, metadata=EXCLUDED.metadata",
                    (p.id, p.doc_id, p.title, p.text, json.dumps(p.metadata, ensure_ascii=False)))
            for c, v in zip(chunks, vectors):
                self.conn.execute(
                    "INSERT INTO rag_chunks (id, doc_id, parent_id, body, text, metadata, embedding, fts_text) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (c.id, c.doc_id, c.parent_id, c.body, c.text, json.dumps(c.metadata, ensure_ascii=False), np.asarray(v, np.float32), _fts(c.text)))

    def delete_doc(self, doc_id):
        self.conn.execute("DELETE FROM rag_chunks WHERE doc_id=%s", (doc_id,))
        self.conn.execute("DELETE FROM rag_parents WHERE doc_id=%s", (doc_id,))

    def dense_search(self, qvec, k, filters=None):
        where, params = self._where(filters)
        q = np.asarray(qvec, np.float32)
        rows = self.conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) FROM rag_chunks WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
            [q, *params, q, k]).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def sparse_search(self, query, k, filters=None):
        terms = sorted({t for t in _fts(query).split() if t.isalnum()})
        if not terms:
            return []
        where, params = self._where(filters)
        tsq = " | ".join(terms)  # OR giữa các từ; xếp hạng bằng ts_rank_cd
        rows = self.conn.execute(
            f"SELECT id, ts_rank_cd(tsv, to_tsquery('simple', %s)) AS r FROM rag_chunks "
            f"WHERE tsv @@ to_tsquery('simple', %s) AND {where} ORDER BY r DESC LIMIT %s",
            [tsq, tsq, *params, k]).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get_chunks(self, ids):
        if not ids:
            return {}
        rows = self.conn.execute("SELECT id, doc_id, parent_id, body, text, metadata FROM rag_chunks WHERE id = ANY(%s)", (list(ids),)).fetchall()
        return {r[0]: Chunk(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows}

    def get_parents(self, ids):
        if not ids:
            return {}
        rows = self.conn.execute("SELECT id, doc_id, title, text, metadata FROM rag_parents WHERE id = ANY(%s)", (list(ids),)).fetchall()
        return {r[0]: Parent(r[0], r[1], r[2], r[3], r[4]) for r in rows}

    def all_chunk_texts(self):
        return [r[0] for r in self.conn.execute("SELECT text FROM rag_chunks ORDER BY id").fetchall()]

    def count(self):
        return self.conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]

    def get_meta(self):
        r = self.conn.execute("SELECT value FROM rag_meta WHERE key='index'").fetchone()
        return r[0] if r else {}

    def set_meta(self, meta):
        self.conn.execute("INSERT INTO rag_meta VALUES ('index', %s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (json.dumps(meta),))
