"""BM25 thuần Python (tìm theo từ khóa). Tiếng Việt: unigram + bigram âm tiết, bỏ dấu để chịu lỗi gõ."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..text import bigrams, tokenize


def _terms(text: str) -> list[str]:
    toks = tokenize(text, stop=True)
    return toks + bigrams(toks)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dl: list[int] = []
        self.avgdl = 0.0

    def build(self, docs: list[tuple[str, str]]) -> None:
        self.ids = [i for i, _ in docs]
        self.inv = defaultdict(list)
        self.dl = []
        for idx, (_, text) in enumerate(docs):
            terms = _terms(text)
            self.dl.append(len(terms))
            for t, c in Counter(terms).items():
                self.inv[t].append((idx, c))
        self.avgdl = sum(self.dl) / len(self.dl) if self.dl else 0.0

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        for t in set(_terms(query)):
            postings = self.inv.get(t)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, tf in postings:
                if allowed is not None and self.ids[idx] not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / (self.avgdl or 1))
                scores[idx] += idf * tf * (self.k1 + 1) / norm
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.ids[i], s) for i, s in top]
