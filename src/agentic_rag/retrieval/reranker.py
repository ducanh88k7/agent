"""Rerank: chấm lại các ứng viên bằng model đọc CẢ câu hỏi lẫn đoạn văn (cross-encoder).

- LexicalReranker: offline, không cần model (độ phủ từ khóa có trọng số IDF + cụm 2 từ). Để thử/test.
- CrossEncoderReranker: bge-reranker-v2-m3 (tốt cho tiếng Việt). Khuyến nghị cho bản thật.
Điểm đưa về [0,1]; chunk dưới ngưỡng bị loại => hệ thống biết nói "không tìm thấy".
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from ..text import bigrams, tokenize


class Reranker(ABC):
    default_threshold = 0.0

    def fit(self, texts: list[str]) -> None:  # học thống kê corpus (nếu cần)
        pass

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class NoReranker(Reranker):
    def score(self, query, texts):
        return [1.0] * len(texts)


class LexicalReranker(Reranker):
    default_threshold = 0.3

    def __init__(self):
        self.df: Counter = Counter()
        self.n = 0

    def fit(self, texts):
        self.df, self.n = Counter(), len(texts)
        for t in texts:
            self.df.update(set(tokenize(t, stop=True)))

    def _idf(self, tok: str) -> float:
        return math.log(1 + (self.n - self.df.get(tok, 0) + 0.5) / (self.df.get(tok, 0) + 0.5)) if self.n else 1.0

    def score(self, query, texts):
        q = tokenize(query, stop=True)
        if not q:
            return [0.0] * len(texts)
        qset, qbi = set(q), set(bigrams(q))
        total = sum(self._idf(t) for t in qset) or 1.0
        out = []
        for text in texts:
            d = tokenize(text, stop=True)
            dset, dbi = set(d), set(bigrams(d))
            cov = sum(self._idf(t) for t in qset if t in dset) / total
            bi = len(qbi & dbi) / len(qbi) if qbi else 0.0
            out.append(min(1.0, 0.75 * cov + 0.25 * bi))
        return out


class CrossEncoderReranker(Reranker):
    default_threshold = 0.3

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # pip install sentence-transformers

        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query, texts):
        if not texts:
            return []
        s = np.asarray(self.model.predict([(query, t) for t in texts]), dtype=np.float64)
        if s.min() < 0 or s.max() > 1:  # logit thô => sigmoid
            s = 1 / (1 + np.exp(-s))
        return s.tolist()


def make_reranker(settings) -> Reranker:
    if settings.reranker == "none":
        return NoReranker()
    if settings.reranker == "lexical":
        return LexicalReranker()
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    raise ValueError(f"RERANKER không hợp lệ: {settings.reranker}")
