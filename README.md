# Agentic RAG - dự án học tập (không dùng framework)

Một hệ thống hỏi đáp tài liệu có **agent**: tự chọn đường đi (router), tự gọi công cụ (ReAct), chia nhỏ câu hỏi khó
(plan-and-execute), tự kiểm tra câu trả lời (critic). Toàn bộ vòng lặp agent viết tay, ~2.700 dòng Python (đã gồm comment), để bạn đọc
và hiểu từng bước. Khi đã hiểu, có thể chuyển sang LangGraph/LlamaIndex hoặc nâng lên multi-agent (xem `docs/UPGRADE_PATH.md`).

Dữ liệu mẫu là quy chế/học phí/ký túc xá của một "Trường Đại học Mẫu" (tài liệu bịa, chỉ để demo).

```
Câu hỏi ─► Guardrail ─► ROUTER ─┬─ direct ──► trả lời thẳng (chào hỏi)
                                ├─ clarify ─► hỏi lại người dùng
                                ├─ rag ─────► Advanced RAG cố định (hybrid → RRF → rerank → parent) → trả lời
                                ├─ agent ───► ReAct: [search_documents | get_student_record | calculate | date_math]
                                └─ plan ────► Planner → PlanExecutor (mỗi bước = 1 ReAct nhỏ, có replan) → Synthesizer
                                                              │
                  kiểm tra trích dẫn (miễn phí) ─► CRITIC (route agent/plan) ─► sửa tối đa 1 lần ─► Câu trả lời + nguồn
```

## 1. Chạy thử trong 2 phút (không cần API key)

```bash
pip install -r requirements.txt
export PYTHONPATH=src                      # Windows PowerShell: $env:PYTHONPATH="src"

python examples/offline_demo.py --scenario agent   # xem trace một lượt agent: gọi tool song song, tính ngày, critic
python examples/offline_demo.py --scenario plan    # xem plan-and-execute: 4 bước có phụ thuộc, tổng hợp
pytest -q                                          # 48 test (+ 2 test pgvector nếu đặt PG_TEST_DSN)
```

Demo offline dùng **LLM giả có kịch bản**; retrieval, tool, router, planner, critic, guardrail là code thật.
Mục đích là để bạn nhìn thấy *luồng*, không phải đánh giá chất lượng LLM.

## 2. Chạy với LLM thật (Gemini free tier hoặc Claude)

```bash
cp .env.example .env        # điền GEMINI_API_KEY (hoặc ANTHROPIC_API_KEY + LLM_PROVIDER=anthropic)
pip install google-genai    # hoặc: pip install anthropic

python -m agentic_rag.cli ingest                       # nạp data/docs vào index
python -m agentic_rag.cli ask -v --user SV001 "Em muốn bảo lưu học kỳ bắt đầu 01/02/2027, hạn chót nộp đơn là ngày nào?"
python -m agentic_rag.cli chat --user SV001            # hội thoại nhiều lượt, có nhớ ngữ cảnh
python -m agentic_rag.cli serve                        # API: POST /chat, GET /health
```

`-v` in trace từng bước (route → LLM → tool → plan → critic). Mỗi lượt cũng được lưu ở `runs/<trace_id>.jsonl`.
Câu thử: *"Học phí 16 tín chỉ ngành CNTT và ĐTVT khác nhau bao nhiêu, em có được học bổng không?"* (→ route `plan`),
*"Nộp đơn bảo lưu trước bao lâu?"* (→ `rag`), *"Em (SV002) có được bảo lưu không?"* (→ `agent`, SV002 đang bị kỷ luật cảnh cáo).

**Chất lượng retrieval:** mặc định `EMBEDDING_PROVIDER=hash` + `RERANKER=lexical` chạy offline nhưng ngữ nghĩa yếu.
Cho dữ liệu thật hãy dùng `EMBEDDING_PROVIDER=gemini` (hoặc `sentence-transformers` với BGE-M3) và `RERANKER=cross-encoder`
(`bge-reranker-v2-m3`), rồi ingest lại. Đổi embedder/số chiều mà quên ingest lại sẽ báo lỗi rõ ràng (`IndexMismatchError`).

PostgreSQL: `docker compose up -d`, đặt `VECTOR_STORE=pgvector`, `pip install "psycopg[binary]" pgvector`, ingest lại.
`EMBEDDING_DIM` phải ≤ 2000 thì HNSW mới index được (mặc định 768).

## 3. Nên đọc code theo thứ tự nào

| # | File | Học được gì |
|---|---|---|
| 1 | `retrieval/chunking.py` | Parent-child chunking, breadcrumb, metadata từ front matter |
| 2 | `retrieval/pipeline.py` | Hybrid (dense + BM25) → RRF → rerank → ngưỡng → gộp về parent |
| 3 | `tools/base.py`, `tools/builtin.py` | Tool = hàm + schema; kiểm tra tham số; xác nhận hành động ghi; RAG là *một tool* |
| 4 | `agent/react.py` | **Vòng lặp agent** (đọc kỹ `ReActAgent.run`): gọi tool song song, phát hiện lặp, ngân sách, ép kết thúc |
| 5 | `agent/router.py` | Chọn đường rẻ nhất; luật trước, LLM nhỏ sau; fallback an toàn |
| 6 | `agent/planner.py` | Kế hoạch có phụ thuộc, chạy theo lớp, replan khi bước thất bại, tổng hợp |
| 7 | `agent/critic.py`, `agent/evidence.py` | Kiểm tra trích dẫn bằng code + LLM critic + sửa |
| 8 | `agent/orchestrator.py` | Ghép tất cả lại trong một lượt hỏi |
| 9 | `agent/prompts.py` | Toàn bộ prompt ở một chỗ |

## 4. Những gì mình đã thêm ngoài yêu cầu (và vì sao)

- **Ngân sách chạy** (`guardrails.Budget`): giới hạn số lần gọi LLM/tool/token/thời gian. Agent không có nó có thể lặp đốt tiền.
- **Phát hiện vòng lặp**: gọi lại cùng tool + cùng tham số lần thứ 3 bị chặn; sau đó ép agent kết luận.
- **Trích dẫn kiểm chứng được**: mỗi đoạn tài liệu có id `[S1]`; code kiểm tra id có thật, xóa id bịa (`EvidenceStore`).
- **Không tìm thấy thì không bịa**: reranker có ngưỡng; không đoạn nào đạt ngưỡng → trả "không tìm thấy" *không gọi LLM sinh*.
- **Ràng buộc quyền truy cập ở code**: `get_student_record` không nhận mã sinh viên từ LLM, chỉ dùng `ctx.user_id` của phiên đăng nhập; tham số lạ bị loại.
- **Tool ghi cần xác nhận** (`submit_leave_request`, bật bằng `ENABLE_WRITE_TOOLS=true`, chỉ CLI): human-in-the-loop.
- **Chống prompt injection ở mức cơ bản**: nội dung tài liệu/tool bọc trong thẻ và coi là *dữ liệu*; gắn cờ mẫu câu lệnh khả nghi; không cho tài liệu tự đóng thẻ.
- **Tracing** (`observability.py`): mọi bước ghi ra JSONL. Không có trace thì gần như không debug được agent.
- **Đánh giá** (`evaluation.py`, `data/eval/`): đo retrieval riêng (Recall@k, MRR, tỷ lệ từ chối đúng câu ngoài phạm vi) và end-to-end (route, tool đã dùng, nội dung, số lần gọi LLM, token).
- **Retry/backoff** khi 429/5xx (free tier hay gặp); **adapter nhà cung cấp** để đổi Gemini ↔ Claude chỉ bằng biến môi trường.
- **Dấu vân tay index**: phát hiện lệch embedder/số chiều (lỗi dimension mismatch kinh điển).
- **Router/critic dùng model rẻ** (`FAST_MODEL`), planner/agent dùng model chính (`MAIN_MODEL`).

## 5. Cấu hình hay chỉnh (`.env`)

| Biến | Ý nghĩa |
|---|---|
| `CRITIC_MODE` | `auto` (chỉ route agent/plan), `always`, `off` |
| `PARALLEL_STEPS` | chạy song song các bước plan độc lập (mặc định tắt vì rate limit free tier) |
| `MAX_LLM_CALLS`, `MAX_TOOL_CALLS`, `MAX_TOTAL_TOKENS`, `TIMEOUT_S` | ngân sách một lượt hỏi |
| `K_FINAL`, `RERANK_THRESHOLD`, `CHILD_MAX_CHARS` | tinh chỉnh retrieval (chỉnh bằng golden set, mỗi lần một biến) |
| `TODAY` | ép ngày hiện tại để demo/test hạn chót |

## 6. Thêm một công cụ mới (ví dụ)

```python
# tools/builtin.py
def make_gpa_tool() -> Tool:
    def run(args, ctx): return f"GPA tích lũy dự kiến: {args['current'] + 0.1:.2f}"
    return Tool("estimate_gpa", "Ước tính GPA ... (nói rõ KHI NÀO dùng và KHI NÀO KHÔNG dùng)",
                {"type": "object", "properties": {"current": {"type": "number"}}, "required": ["current"]},
                run, tags=("calc",))
# rồi đăng ký trong build_registry(): reg.register(make_gpa_tool())
```
Mô tả tool chính là "prompt" của tool: agent chọn tool dựa vào nó. Thêm một test trong `tests/test_tools.py`.

## 7. Đã kiểm tra gì, chưa kiểm tra gì (đọc kỹ)

**Đã chạy và đạt:** 48 test (retrieval, chunking, tool, vòng lặp ReAct, router, planner/replan, critic/sửa trích dẫn,
xử lý lỗi LLM, API contract, chuyển đổi message/schema của adapter Gemini và Claude); 2 test `PgVectorStore` trên
PostgreSQL 16 + pgvector 0.6.0 thật; hai kịch bản demo offline chạy trọn vẹn.

**CHƯA kiểm tra:**
- **Gọi API thật của Gemini/Claude** (môi trường dựng project không có key/mạng tới API). Adapter đã được kiểm tra phần chuyển đổi
  dữ liệu bằng SDK thật nhưng chưa nhận phản hồi thật. Tên model mặc định có thể đã cũ: sửa `MAIN_MODEL`/`FAST_MODEL`.
- **Prompt với LLM thật**: router/planner/critic được viết cẩn thận và có fallback khi JSON hỏng, nhưng chắc chắn cần chỉnh sau
  khi chạy thật. Hãy chạy `eval-e2e` và đọc trace.
- `GeminiEmbedder`, `SentenceTransformerEmbedder`, `CrossEncoderReranker` (cần key hoặc tải model).
- Chỉ số Recall/MRR = 1.00 trên `data/eval/retrieval.jsonl` **không chứng minh chất lượng thực tế**: bộ 12 câu này do mình viết cùng với
  corpus nhỏ. Hãy tự tạo golden set 30-100 câu từ tài liệu của bạn.

**Giới hạn đã biết (bản học tập):** chỉ ingest `.md/.txt` (PDF: dùng Docling/Unstructured rồi xuất markdown); session lưu trong RAM;
`user_id` ở API lấy từ header (bản thật phải từ JWT/session); chưa streaming; chưa có tóm tắt lịch sử dài; phát hiện injection chỉ là heuristic.

## 8. Lộ trình gợi ý

1. Chạy `eval-retrieval`, thay corpus mẫu bằng tài liệu của bạn, viết golden set riêng.
2. Chạy với Gemini, đọc trace, chỉnh `prompts.py` theo lỗi thấy được.
3. Chuyển sang `EMBEDDING_PROVIDER=gemini|sentence-transformers` + `RERANKER=cross-encoder`, so điểm với baseline.
4. Thêm LLM-as-judge vào `eval-e2e`, theo dõi chi phí/độ trễ mỗi route.
5. Nâng cấp multi-agent hoặc chuyển sang framework: xem `docs/UPGRADE_PATH.md`.
