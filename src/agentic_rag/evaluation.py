"""Đánh giá: đo RETRIEVAL riêng (offline, không cần LLM) và END-TO-END (cần LLM: route, tool đã dùng, nội dung, chi phí).
Nguyên tắc: mỗi lần đổi một biến (chunk size, reranker, prompt...) thì chạy lại để so sánh."""
from __future__ import annotations

import json
from pathlib import Path

from .text import fold_accents


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def eval_retrieval(pipeline, path: str | Path, k: int | None = None) -> dict:
    rows, hits, rr, na_ok, na_n = [], 0, 0.0, 0, 0
    answerable = 0
    for item in load_jsonl(path):
        res = pipeline.retrieve(item["q"], k_final=k)
        labels = [p.label for p in res.passages]
        if item["expect"]:
            answerable += 1
            rank = next((i for i, lab in enumerate(labels, 1) if any(e.lower() in lab.lower() for e in item["expect"])), None)
            hits += rank is not None
            rr += 1 / rank if rank else 0.0
            rows.append({"q": item["q"], "ok": rank is not None, "rank": rank, "top": labels[:1]})
        else:  # câu ngoài phạm vi: đúng khi KHÔNG trả đoạn nào
            na_n += 1
            na_ok += not labels
            rows.append({"q": item["q"], "ok": not labels, "rank": None, "top": labels[:1]})
    return {"recall_at_k": hits / answerable if answerable else 0.0, "mrr": rr / answerable if answerable else 0.0,
            "no_answer_accuracy": na_ok / na_n if na_n else None, "n": len(rows), "rows": rows}


def eval_e2e(app, path: str | Path, verbose: bool = False) -> dict:
    rows = []
    for item in load_jsonl(path):
        ans = app.ask(item["q"], user_id=item.get("user"), verbose=verbose)
        tools = [e["name"] for e in ans.events if e["kind"] == "tool"]
        low = fold_accents(ans.text.lower())
        checks = {
            "route": ans.route in item.get("expect_route", [ans.route]),
            "tools": all(t in tools for t in item.get("expect_tools", [])),
            "content": all(fold_accents(s.lower()) in low for s in item.get("must_contain", [])),
            "no_warnings": not ans.warnings,
        }
        rows.append({"q": item["q"], "route": ans.route, "tools": tools, "checks": checks, "ok": all(checks.values()),
                     "llm_calls": ans.usage.get("llm_calls"), "tokens": ans.usage.get("input_tokens", 0) + ans.usage.get("output_tokens", 0)})
    n = len(rows) or 1
    return {"pass_rate": sum(r["ok"] for r in rows) / n, "avg_llm_calls": sum(r["llm_calls"] or 0 for r in rows) / n,
            "avg_tokens": sum(r["tokens"] for r in rows) / n, "n": len(rows), "rows": rows}
