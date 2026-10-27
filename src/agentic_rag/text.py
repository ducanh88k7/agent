"""Xử lý văn bản tiếng Việt dùng chung cho BM25, embedder offline và reranker."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+", re.UNICODE)

# Stopword để ở dạng CÓ dấu và lọc TRƯỚC khi bỏ dấu, tránh nhầm "bao" (bao lâu) với "bảo" (bảo lưu).
_STOPWORDS = {
    "và", "là", "của", "các", "có", "cho", "được", "trong", "với", "một", "những", "này", "khi", "thì",
    "bao", "nhiêu", "gì", "nào", "như", "thế", "em", "tôi", "mình", "ạ", "nhé", "hay", "hoặc", "để", "từ",
    "ở", "tại", "theo", "về", "bị", "đã", "sẽ", "đang", "không", "cần", "ơi", "vậy", "sao", "thì", "cũng",
    "the", "a", "of", "is", "and", "to",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fold_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str, *, fold: bool = True, stop: bool = False) -> list[str]:
    words = _WORD.findall(nfc(text).lower())
    if stop:
        words = [w for w in words if w not in _STOPWORDS]
    if fold:
        words = [fold_accents(w) for w in words]
    return words


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
