"""Embedder: biến văn bản thành vector. Dùng CÙNG một embedder cho tài liệu và câu hỏi.

- HashingEmbedder: chạy offline, không cần model; chỉ để thử/test. Chất lượng ngữ nghĩa thấp.
- SentenceTransformerEmbedder: model local (vd BAAI/bge-m3, tốt cho tiếng Việt).
- GeminiEmbedder: gemini-embedding-001 qua API (free tier dùng được).
Đổi embedder/số chiều => phải ingest lại toàn bộ (index lưu dấu vân tay để phát hiện lệch).
"""
from __future__ import annotations

import math
import os
import zlib
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from ..retry import retry_call
from ..text import tokenize


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.maximum(n, 1e-12)).astype(np.float32)


class Embedder(ABC):
    name = "embedder"
    dim = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    @property
    def fingerprint(self) -> dict:
        return {"embedder": self.name, "dim": self.dim}


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = "hash"

    def _one(self, text: str) -> np.ndarray:
        toks = tokenize(text, stop=True)
        feats: Counter = Counter()
        for w in toks:
            feats["w:" + w] += 1.0
            p = f"#{w}#"
            for i in range(len(p) - 2):
                feats["c:" + p[i:i + 3]] += 0.3
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] += 1.0
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in feats.items():
            h = zlib.crc32(f.encode("utf-8"))
            v[h % self.dim] += (1 if (h >> 20) & 1 else -1) * math.sqrt(c)
        return v

    def embed_documents(self, texts):
        return _normalize(np.stack([self._one(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return _normalize(self._one(text))


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())
        # multilingual-e5 cần "query: " / "passage: "; bge-m3 thì không
        self.qp, self.dp = query_prefix, doc_prefix

    def embed_documents(self, texts):
        return _normalize(np.asarray(self.model.encode([self.dp + t for t in texts], batch_size=32)))

    def embed_query(self, text):
        return _normalize(np.asarray(self.model.encode(self.qp + text)))


class GeminiEmbedder(Embedder):
    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, api_key: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model, self.dim, self.name = model, dim, f"gemini:{model}"

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        out: list = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            resp = retry_call(lambda b=batch: self.client.models.embed_content(
                model=self.model, contents=b,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)))
            out.extend(e.values for e in resp.embeddings)
        # Vector đã cắt chiều (<3072) không tự chuẩn hóa => phải normalize
        return _normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def make_embedder(settings) -> Embedder:
    p = settings.embedding_provider
    if p == "hash":
        return HashingEmbedder(settings.embedding_dim)
    if p == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if p == "gemini":
        return GeminiEmbedder(settings.embedding_model, settings.embedding_dim)
    raise ValueError(f"EMBEDDING_PROVIDER không hợp lệ: {p}")
