"""Chia tài liệu thành Parent (đơn vị ĐỌC, vd một Điều) và Child (đơn vị TÌM, vài khoản nhỏ).

Tìm bằng child cho vector sắc nét, nhưng trả parent cho LLM để đủ ngữ cảnh (parent-child / small-to-big).
Mỗi child được gắn breadcrumb "[Tài liệu > Chương > Điều]" trước khi embed (contextual header).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_ITEM = re.compile(r"^\s*(\d+[.)]|[-*•])\s+")
_SENT = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Parent:
    id: str
    doc_id: str
    title: str  # breadcrumb
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    parent_id: str
    body: str  # nội dung gốc của child
    text: str  # "[breadcrumb]\nbody" - thứ được embed / đưa vào BM25
    metadata: dict = field(default_factory=dict)


def _coerce(v: str):
    v = v.strip()
    return int(v) if re.fullmatch(r"-?\d+", v) else v


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = _coerce(v)
    return meta, text[m.end():]


def split_sections(body: str) -> list[tuple[list[str], str]]:
    """Trả về [(đường dẫn heading, nội dung)] cho mỗi mục có nội dung."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], str]] = []
    lines: list[str] = []
    path: list[str] = []
    for line in body.splitlines():
        m = _HEAD.match(line)
        if m:
            text = "\n".join(lines).strip()
            if text:
                sections.append((path, text))
            lines = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            path = [t for _, t in stack]
        else:
            lines.append(line)
    text = "\n".join(lines).strip()
    if text:
        sections.append((path, text))
    return sections


def _units(text: str) -> list[str]:
    units: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if buf:
                units.append(" ".join(buf))
                buf = []
        elif _ITEM.match(line) and buf:
            units.append(" ".join(buf))
            buf = [line.strip()]
        else:
            buf.append(line.strip())
    if buf:
        units.append(" ".join(buf))
    return units


def _explode(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]
    out, cur = [], ""
    for s in _SENT.split(unit):
        while len(s) > max_chars:  # câu quá dài: cắt cứng
            out.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _pack(units: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for u in (p for unit in units for p in _explode(unit, max_chars)):
        if cur and size + len(u) + 1 > max_chars:
            out.append("\n".join(cur))
            last = cur[-1]
            cur = [last] if len(last) <= overlap_chars else []  # overlap: lặp lại đơn vị cuối nếu ngắn
            size = sum(len(x) + 1 for x in cur)
        cur.append(u)
        size += len(u) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def chunk_document(text: str, doc_id: str, source: str, base_meta: dict | None = None, *,
                   child_max_chars: int = 400, overlap_chars: int = 80) -> tuple[list[Parent], list[Chunk]]:
    front, body = parse_front_matter(text)
    meta = {**(base_meta or {}), **front, "source": source}
    parents: list[Parent] = []
    chunks: list[Chunk] = []
    sections = split_sections(body) or [([source], body.strip())]
    for pi, (path, content) in enumerate(sections):
        title = " > ".join(path) if path else source
        pid = f"{doc_id}:p{pi}"
        leaf = path[-1] if path else source
        pmeta = {**meta, "section": title}
        parents.append(Parent(pid, doc_id, title, f"{leaf}\n{content}", pmeta))
        for ci, piece in enumerate(_pack(_units(content), child_max_chars, overlap_chars)):
            chunks.append(Chunk(f"{pid}:c{ci}", doc_id, pid, piece, f"[{title}]\n{piece}", dict(pmeta)))
    return parents, chunks
