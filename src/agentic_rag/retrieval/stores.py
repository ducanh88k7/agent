"""Kho lưu chunk + vector. Hai cài đặt: MemoryStore (numpy, lưu ra đĩa) và PgVectorStore (pg_store.py)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk, Parent


class IndexMismatchError(RuntimeError):
    pass


def match_filters(meta: dict, filters: dict | None) -> bool:
    """Lọc metadata: giá trị đơn => so bằng (không phân biệt hoa/thường); list => thuộc tập."""
    for k, want in (filters or {}).items():
        if want is None:
            continue
        have = meta.get(k)
        if isinstance(want, (list, tuple, set)):
            if str(have).lower() not in {str(w).lower() for w in want}:
                return False
        elif str(have).lower() != str(want).lower():
            return False
    return True


class VectorStore(ABC):
    @abstractmethod
    def add(self, parents: list[Parent], chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    @abstractmethod
    def delete_doc(self, doc_id: str) -> None: ...
    @abstractmethod
    def dense_search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[str, float]]: ...
    @abstractmethod
    def sparse_search(self, query: str, k: int, filters: dict | None = None) -> list[tuple[str, float]]: ...
    @abstractmethod
    def get_chunks(self, ids: list[str]) -> dict[str, Chunk]: ...
    @abstractmethod
    def get_parents(self, ids: list[str]) -> dict[str, Parent]: ...
    @abstractmethod
    def all_chunk_texts(self) -> list[str]: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def get_meta(self) -> dict: ...
    @abstractmethod
    def set_meta(self, meta: dict) -> None: ...

    def save(self) -> None:  # chỉ MemoryStore cần
        pass


class MemoryStore(VectorStore):
    def __init__(self, path: str | None = None, dim: int = 768):
        self.path = Path(path) if path else None
        self.dim = dim
        self.parents: dict[str, Parent] = {}
        self.chunks: dict[str, Chunk] = {}
        self.order: list[str] = []
        self.matrix = np.zeros((0, dim), np.float32)
        self.meta: dict = {}
        self._bm25: BM25Index | None = None

    # ---- ghi
    def add(self, parents, chunks, vectors):
        for p in parents:
            self.parents[p.id] = p
        for c in chunks:
            self.chunks[c.id] = c
            self.order.append(c.id)
        if len(chunks):
            self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])
        self._bm25 = None

    def delete_doc(self, doc_id):
        keep = [i for i, cid in enumerate(self.order) if self.chunks[cid].doc_id != doc_id]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, self.dim), np.float32)
        self.order = [self.order[i] for i in keep]
        self.chunks = {cid: self.chunks[cid] for cid in self.order}
        self.parents = {pid: p for pid, p in self.parents.items() if p.doc_id != doc_id}
        self._bm25 = None

    # ---- đọc
    def _allowed(self, filters):
        return None if not filters else {cid for cid in self.order if match_filters(self.chunks[cid].metadata, filters)}

    def dense_search(self, qvec, k, filters=None):
        if not self.order:
            return []
        scores = self.matrix @ qvec  # vector đã chuẩn hóa => tích vô hướng = cosine
        allowed = self._allowed(filters)
        idx = np.argsort(-scores)
        out = []
        for i in idx:
            cid = self.order[i]
            if allowed is None or cid in allowed:
                out.append((cid, float(scores[i])))
                if len(out) >= k:
                    break
        return out

    def sparse_search(self, query, k, filters=None):
        if self._bm25 is None:
            self._bm25 = BM25Index()
            self._bm25.build([(cid, self.chunks[cid].text) for cid in self.order])
        return self._bm25.search(query, k, self._allowed(filters))

    def get_chunks(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}

    def get_parents(self, ids):
        return {i: self.parents[i] for i in ids if i in self.parents}

    def all_chunk_texts(self):
        return [self.chunks[c].text for c in self.order]

    def count(self):
        return len(self.order)

    def get_meta(self):
        return dict(self.meta)

    def set_meta(self, meta):
        self.meta = dict(meta)

    # ---- đĩa
    def save(self):
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "chunks.json").write_text(json.dumps([asdict(self.chunks[c]) for c in self.order], ensure_ascii=False), "utf-8")
        (self.path / "parents.json").write_text(json.dumps([asdict(p) for p in self.parents.values()], ensure_ascii=False), "utf-8")
        (self.path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), "utf-8")
        np.save(self.path / "vectors.npy", self.matrix)

    def load(self) -> bool:
        if not self.path or not (self.path / "chunks.json").exists():
            return False
        chunks = [Chunk(**d) for d in json.loads((self.path / "chunks.json").read_text("utf-8"))]
        self.parents = {d["id"]: Parent(**d) for d in json.loads((self.path / "parents.json").read_text("utf-8"))}
        self.chunks = {c.id: c for c in chunks}
        self.order = [c.id for c in chunks]
        self.matrix = np.load(self.path / "vectors.npy")
        self.meta = json.loads((self.path / "meta.json").read_text("utf-8"))
        self._bm25 = None
        return True
