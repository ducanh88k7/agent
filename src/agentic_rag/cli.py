"""CLI: python -m agentic_rag.cli {ingest|ask|chat|eval-retrieval|eval-e2e|serve|trace}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.memory import Session
from .config import Settings


def _print_answer(ans, show_usage: bool = True) -> None:
    print(f"\n{ans.text}\n")
    if ans.sources:
        print("Nguồn:")
        for s in ans.sources:
            print(f"  [{s['id']}] {s['section']}")
    if ans.warnings:
        print(f"⚠ cảnh báo: {', '.join(ans.warnings)}")
    if show_usage:
        u = ans.usage
        print(f"(route={ans.route} | {u.get('llm_calls', 0)} lần gọi LLM, {u.get('tool_calls', 0)} lần gọi tool, "
              f"{u.get('input_tokens', 0) + u.get('output_tokens', 0)} token | trace={ans.trace_id})")


def _confirm(tool: str, args: dict) -> bool:
    print(f"\n⚠ Agent muốn thực hiện hành động GHI: {tool}({json.dumps(args, ensure_ascii=False)})")
    return input("  Cho phép? [y/N] ").strip().lower() in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentic-rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="Nạp tài liệu vào index"); p.add_argument("path", nargs="?")
    for name in ("ask", "chat"):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--user", default=None, help="Mã sinh viên (phiên đăng nhập giả lập), vd SV001")
        p.add_argument("-v", "--verbose", action="store_true", help="In trace từng bước")
    p = sub.add_parser("eval-retrieval"); p.add_argument("--file", default="data/eval/retrieval.jsonl")
    p = sub.add_parser("eval-e2e"); p.add_argument("--file", default="data/eval/e2e.jsonl"); p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("trace", help="Xem lại một trace đã lưu"); p.add_argument("trace_id")
    a = ap.parse_args(argv)
    settings = Settings.from_env()

    if a.cmd == "trace":
        path = Path(settings.trace_dir) / f"{a.trace_id}.jsonl"
        for line in path.read_text("utf-8").splitlines():
            print(line)
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("agentic_rag.app:app", host="127.0.0.1", port=a.port)
        return 0

    from .factory import build_app
    if a.cmd == "ingest":
        from .retrieval import build_pipeline
        pipe = build_pipeline(settings)
        stats = pipe.ingest_dir(a.path or settings.docs_dir)
        print(f"Đã nạp {sum(stats.values())} chunk từ {len(stats)} file: {stats}")
        return 0
    if a.cmd == "eval-retrieval":
        from .evaluation import eval_retrieval
        from .retrieval import build_pipeline
        r = eval_retrieval(build_pipeline(settings), a.file)
        for row in r["rows"]:
            print(("OK  " if row["ok"] else "FAIL"), f"rank={row['rank']}", row["q"], "->", row["top"])
        print(f"\nRecall@k={r['recall_at_k']:.2f}  MRR={r['mrr']:.2f}  no-answer-acc={r['no_answer_accuracy']}  (n={r['n']})")
        return 0

    app = build_app(settings)
    if app.pipeline.store.count() == 0:
        print("Index trống. Chạy trước:  python -m agentic_rag.cli ingest", file=sys.stderr)
        return 1
    if a.cmd == "eval-e2e":
        from .evaluation import eval_e2e
        r = eval_e2e(app, a.file, verbose=a.verbose)
        for row in r["rows"]:
            print(("OK  " if row["ok"] else "FAIL"), row["route"], row["tools"], row["checks"], row["q"])
        print(f"\npass={r['pass_rate']:.2f}  avg LLM calls={r['avg_llm_calls']:.1f}  avg tokens={r['avg_tokens']:.0f}")
        return 0

    confirm = _confirm if settings.enable_write_tools else None
    if a.cmd == "ask":
        _print_answer(app.ask(a.question, user_id=a.user, confirm=confirm, verbose=a.verbose))
        return 0
    session = Session(user_id=a.user, max_turns=settings.history_turns)
    print("Chat (gõ 'exit' để thoát). User:", a.user or "(chưa đăng nhập)")
    while True:
        try:
            q = input("\nBạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q.lower() in ("exit", "quit"):
            return 0
        if q:
            _print_answer(app.ask(q, session, confirm=confirm, verbose=a.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
