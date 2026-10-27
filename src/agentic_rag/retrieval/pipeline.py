"""Pipeline RAG nâng cao: ingest (parse -> chunk -> embed -> lưu) và retrieve (hybrid -> RRF -> rerank -> parent)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .embeddings import Embedder
from .reranker import Reranker
from .stores import IndexMismatchError, VectorStore


@dataclass
class Passage:
    id: str  # parent_id: đơn vị trả về cho LLM
    source: str
    section: str
    text: str
    score: float
    matched_chunks: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        parts = self.section.split(" > ")
        return f"{self.source} > {' > '.join(parts[1:] if len(parts) > 1 else parts)}"  # bỏ tên tài liệu cho gọn


@dataclass
class RetrievalResult:
    query: str
    passages: list[Passage]
    best_score: float = 0.0
    debug: dict = field(default_factory=dict)


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: điểm = Σ 1/(k + hạng). Chỉ dùng thứ hạng nên không cần quy đổi thang điểm."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class RAGPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker: Reranker, settings):
        self.store, self.embedder, self.reranker, self.s = store, embedder, reranker, settings
        self._check_fingerprint()
        if store.count():
            reranker.fit(store.all_chunk_texts())

    def _check_fingerprint(self) -> None:
        meta = self.store.get_meta()
        if self.store.count() and meta and meta != self.embedder.fingerprint:
            raise IndexMismatchError(
                f"Index được tạo bằng {meta} nhưng cấu hình hiện tại là {self.embedder.fingerprint}. "
                "Hãy chạy lại `ingest` (hoặc đặt lại EMBEDDING_PROVIDER/EMBEDDING_DIM cho khớp).")

    # ---------------------------------------------------------------- ingest
    def ingest_text(self, doc_id: str, text: str, source: str, metadata: dict | None = None) -> int:
        parents, chunks = chunk_document(text, doc_id, source, metadata, child_max_chars=self.s.child_max_chars)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.delete_doc(doc_id)  # idempotent: ingest lại cùng file không bị trùng
        self.store.add(parents, chunks, vectors)
        return len(chunks)

    def ingest_dir(self, path: str | Path, patterns=("*.md", "*.txt")) -> dict:
        stats = {}
        for pat in patterns:
            for f in sorted(Path(path).rglob(pat)):
                stats[f.name] = self.ingest_text(f.stem, f.read_text("utf-8"), f.name)
        self.store.set_meta(self.embedder.fingerprint)
        self.store.save()
        self.reranker.fit(self.store.all_chunk_texts())
        return stats

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, query: str, filters: dict | None = None, *, k_final: int | None = None,
                 threshold: float | None = None) -> RetrievalResult:
        s = self.s
        thr = threshold if threshold is not None else (s.rerank_threshold if s.rerank_threshold is not None
                                                       else self.reranker.default_threshold)
        qvec = self.embedder.embed_query(query)
        dense = self.store.dense_search(qvec, s.k_dense, filters)      # tìm theo NGHĨA
        sparse = self.store.sparse_search(query, s.k_sparse, filters)  # tìm theo TỪ KHÓA
        fused = rrf_fuse([[c for c, _ in dense], [c for c, _ in sparse]])[: s.k_candidates]
        chunks = self.store.get_chunks([c for c, _ in fused])
        cand = [(cid, chunks[cid]) for cid, _ in fused if cid in chunks]

        scores = self.reranker.score(query, [c.text for _, c in cand])
        ranked = sorted(zip(cand, scores), key=lambda x: -x[1])
        best = ranked[0][1] if ranked else 0.0

        # gộp child -> parent, giữ điểm của child tốt nhất
        by_parent: dict[str, tuple[float, list[str]]] = {}
        for (cid, ch), sc in ranked:
            if sc < thr:
                continue
            sc0, ids = by_parent.get(ch.parent_id, (sc, []))
            by_parent[ch.parent_id] = (max(sc0, sc), ids + [cid])
        top = sorted(by_parent.items(), key=lambda x: -x[1][0])[: (k_final or s.k_final)]
        parents = self.store.get_parents([p for p, _ in top])
        passages = []
        for pid, (sc, cids) in top:
            p = parents.get(pid)
            if not p:
                continue
            text = p.text
            if len(text) > s.max_parent_chars:  # parent quá dài: chỉ giữ các child đã khớp
                text = "\n…\n".join(chunks[c].body for c in cids if c in chunks)[: s.max_parent_chars]
            passages.append(Passage(pid, p.metadata.get("source", p.doc_id), p.title, text, round(float(sc), 4), cids, p.metadata))
        debug = {
            "dense": [(c, round(v, 3)) for c, v in dense[:5]],
            "sparse": [(c, round(v, 3)) for c, v in sparse[:5]],
            "fused": [(c, round(v, 4)) for c, v in fused[:5]],
            "reranked": [(cid, round(sc, 3)) for (cid, _), sc in ranked[:5]],
            "threshold": thr,
        }
        return RetrievalResult(query, passages, round(float(best), 4), debug)
