"""EvidenceStore: sổ ghi các đoạn tài liệu agent đã thấy trong MỘT lượt hỏi. Mỗi đoạn có id [S1], [S2]...

Câu trả lời chỉ được trích dẫn các id này; code kiểm tra id có tồn tại thật (chống bịa trích dẫn).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from ..retrieval.pipeline import Passage

_CITE = re.compile(r"\[(S\d+)\]")


@dataclass
class EvidenceItem:
    id: str
    passage: Passage


class EvidenceStore:
    def __init__(self):
        self._items: dict[str, EvidenceItem] = {}
        self._by_parent: dict[str, str] = {}
        self._lock = threading.Lock()

    def add(self, passage: Passage) -> str:
        with self._lock:
            if passage.id in self._by_parent:  # cùng một đoạn => cùng một id
                return self._by_parent[passage.id]
            eid = f"S{len(self._items) + 1}"
            self._items[eid] = EvidenceItem(eid, passage)
            self._by_parent[passage.id] = eid
            return eid

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def get(self, eid: str) -> EvidenceItem | None:
        return self._items.get(eid)

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def cited_ids(text: str) -> list[str]:
        seen: list[str] = []
        for m in _CITE.findall(text or ""):
            if m not in seen:
                seen.append(m)
        return seen

    def check_citations(self, answer: str) -> list[str]:
        """Kiểm tra xác định (không tốn LLM). Trả về danh sách vấn đề."""
        issues = []
        cited = self.cited_ids(answer)
        unknown = [c for c in cited if c not in self._items]
        if unknown:
            issues.append(f"Trích dẫn không tồn tại: {', '.join('[' + u + ']' for u in unknown)}")
        if self._items and not cited:
            issues.append("Câu trả lời dùng tài liệu nhưng chưa có trích dẫn [S#] nào")
        return issues

    def strip_unknown(self, answer: str) -> str:
        return _CITE.sub(lambda m: m.group(0) if m.group(1) in self._items else "", answer)

    def sources_for(self, answer: str) -> list[dict]:
        ids = [i for i in self.cited_ids(answer) if i in self._items]
        return [{"id": i, "source": self._items[i].passage.source, "section": self._items[i].passage.label,
                 "score": self._items[i].passage.score} for i in ids]

    def render(self, max_chars_each: int = 1500) -> str:
        return "\n".join(
            f'<document id="{e.id}" source="{e.passage.label}">\n{e.passage.text[:max_chars_each]}\n</document>'
            for e in self._items.values())
