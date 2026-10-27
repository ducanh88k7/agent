# Kiến trúc chi tiết

## 1. Vòng đời một lượt hỏi (`AgenticRAG.ask`)

1. **Chuẩn bị**: tạo `Tracer`, `Budget`, `EvidenceStore`, `ToolContext` (mang `user_id` từ phiên đăng nhập, ngày hôm nay, callback xác nhận). Làm sạch input.
2. **Router** (`agent/router.py`): luật regex cho chào hỏi (0 lần gọi LLM); còn lại một lần gọi model rẻ trả JSON
   `{route, standalone_question, confidence, clarifying_question}`. Câu hỏi được viết lại cho độc lập ngữ cảnh ("Thế phải nộp đơn trước bao lâu?" → "Hạn nộp đơn bảo lưu là bao lâu?").
   Output hỏng → fallback sang `agent` (đủ mạnh, an toàn). API sập (`LLMError`) → báo lỗi, không che bằng fallback.
3. **Thực thi theo route** (bảng dưới).
4. **Kiểm tra cuối** (`_finalize`): (a) kiểm tra trích dẫn bằng code, miễn phí; (b) critic LLM nếu route là agent/plan và có bằng chứng;
   (c) nếu có vấn đề: sửa tối đa `MAX_REFLECTIONS` lần, hết lượt thì xóa trích dẫn bịa và thêm lưu ý.
5. **Ghi nhớ + trace**: lưu lượt hỏi/đáp vào session, ghi trace JSONL, trả `Answer(text, route, sources, warnings, usage, trace_id)`.

| Route | Khi nào | Số lần gọi LLM (điển hình) | Ghi chú |
|---|---|---|---|
| `direct` | chào hỏi, chuyện chung | 0-1 | không truy xuất; không bịa quy định (prompt yêu cầu hỏi lại) |
| `clarify` | thiếu thông tin cốt lõi | 1 (router) | trả câu hỏi làm rõ, không tốn thêm |
| `rag` | tra cứu một ý | 2 (router + trả lời) | pipeline cố định; không đoạn nào đạt ngưỡng → từ chối không gọi LLM |
| `agent` | cần dữ liệu cá nhân/tính toán + tài liệu | 3-8 + critic | ReAct, tool song song |
| `plan` | nhiều phần độc lập, so sánh, tổng hợp | 8-20 | planner + mỗi bước một ReAct nhỏ + synth + critic |

Nguyên tắc: **route rẻ nhất đủ trả lời**. Agent/plan mạnh hơn nhưng đắt, chậm, khó đoán hơn.

## 2. Vòng lặp ReAct (`agent/react.py`)

```
messages = [lịch sử, câu hỏi]
lặp tối đa N bước:
    nếu Budget cạn → ép kết luận
    resp = LLM(messages, tools)
    nếu không có tool_calls → đó là câu trả lời, xong
    nếu có submit_result (chỉ trong bước của plan) → nộp {status, answer}, xong
    chạy các tool_calls (song song nếu đều là tool chỉ-đọc), đưa kết quả vào messages
    nếu một thao tác lặp quá 2 lần → chặn; lặp nhiều → ép kết luận
```

Điểm cần nhớ: **LLM chỉ đề nghị gọi tool; code của ta thực thi.** Vì thế `ToolRegistry.execute` là điểm kiểm soát duy nhất:
kiểm tra schema, loại tham số lạ, xin xác nhận với tool ghi, bắt exception và trả lỗi về cho agent để nó tự sửa.

`submit_result` là mẫu "kết thúc có cấu trúc": thay vì đoán trạng thái từ văn bản tự do, agent của mỗi bước gọi tool này với
`status ∈ {ok, insufficient, failed}`. Executor dựa vào status để quyết định replan.

## 3. Plan-and-execute (`agent/planner.py`)

1. `Planner.make_plan`: LLM trả danh sách bước `{id, goal, tool_hint, depends_on}`. `parse_steps` kiểm tra: id duy nhất, phụ thuộc hợp lệ và chỉ trỏ về bước đứng trước (không vòng), cắt theo `MAX_PLAN_STEPS`. Hỏng → kế hoạch một bước.
2. `PlanExecutor.run`: lặp theo **lớp** (các bước đã sẵn sàng). Mỗi bước chạy một `ReActAgent` nhỏ nhận: câu hỏi gốc + mục tiêu bước + kết quả các bước phụ thuộc.
   Có thể chạy song song các bước cùng lớp (`PARALLEL_STEPS=true`).
3. Sau mỗi lớp, nếu có bước không `ok` và còn lượt `MAX_REPLANS`: `Planner.replan` nhận các bước đã xong + các bước còn lại và trả phần còn cần làm (có thể viết lại, bỏ, hoặc rỗng).
4. `synthesize`: một lần gọi LLM ghép kết quả thành câu trả lời, nói rõ phần nào không trả lời được.

Vì mỗi bước dùng chung `EvidenceStore` của lượt hỏi, trích dẫn `[S#]` nhất quán xuyên suốt các bước.

## 4. Retrieval (`retrieval/`)

Offline: parse → chunk theo cấu trúc (parent = một Điều/mục, child = nhóm khoản ~400 ký tự, gắn breadcrumb) → embed child → lưu.
Online: embed câu hỏi → **dense** top-30 + **sparse (BM25)** top-30 (cùng bộ lọc metadata) → **RRF** → top-20 ứng viên →
**rerank** → bỏ dưới ngưỡng → gộp child về parent (giữ điểm tốt nhất) → top-4 parent trả cho LLM.

Lỗi hay gặp và nơi xử lý: chunk mất ngữ cảnh (breadcrumb + parent), sót từ khóa (BM25), thứ hạng thô (rerank),
câu ngoài phạm vi (ngưỡng), lệch embedder (dấu vân tay index), bản cũ lẫn bản mới (metadata `year`).

## 4. Lớp bảo vệ

| Rủi ro | Cách xử lý | File |
|---|---|---|
| Vòng lặp đốt tiền | `Budget` (LLM/tool/token/thời gian), chặn lặp, `max_steps` | `guardrails.py`, `react.py` |
| Bịa trích dẫn | id `[S#]` phát từ `EvidenceStore`; kiểm tra + xóa id lạ | `evidence.py`, `orchestrator.py` |
| Bịa khi thiếu dữ liệu | ngưỡng rerank; prompt "không tìm thấy thì nói rõ"; critic kiểm tra bám tài liệu | `pipeline.py`, `prompts.py`, `critic.py` |
| Truy cập dữ liệu người khác | `user_id` từ phiên, tool không có tham số chọn người; tham số lạ bị loại | `builtin.py`, `base.py` |
| Hành động ghi ngoài ý muốn | `requires_confirmation` + callback; API tắt tool ghi | `base.py`, `app.py` |
| Prompt injection trong tài liệu/tool | coi là dữ liệu, bọc thẻ, escape thẻ đóng, gắn cờ mẫu khả nghi | `guardrails.py`, `builtin.py` |
| Tool trả quá nhiều | cắt `MAX_TOOL_CHARS` | `react.py` |
| API LLM lỗi tạm thời | retry/backoff cho 429/5xx; lỗi hạ tầng báo gọn, không lộ chi tiết | `retry.py`, `orchestrator.py` |
| Router/critic/planner trả JSON hỏng | sửa 1 lần, rồi fallback (agent / pass / kế hoạch 1 bước) | `llm/base.py`, từng module |

Prompt injection là bài toán chưa có lời giải trọn vẹn: heuristic chỉ gắn cờ. Lớp phòng thủ thật là quyền tối thiểu cho tool,
xác nhận hành động ghi, và không cho agent vừa đọc nội dung không tin cậy vừa có quyền hành động nhạy cảm mà không qua người.

## 5. Trace

Mỗi sự kiện: `{t, kind, ...}` với `kind ∈ route | llm | tool | plan | step | retrieve | agent_done | critic | check | budget | guard | error`.
Xem bằng `-v`, hoặc `python -m agentic_rag.cli trace <trace_id>`. Khi trả lời sai, đi theo thứ tự:
đoạn đúng có trong index không → có trong top ứng viên không (`retrieve.fused`) → qua rerank không (`retrieve.reranked`) →
có vào prompt không (`tool` result) → LLM dùng đúng không (câu trả lời + `check`).

## 6. Quyết định thiết kế

- **Không dùng framework ở bản đầu**: để thấy rõ vòng lặp. Ranh giới module (llm / retrieval / tools / agent) được cắt sao cho thay từng phần bằng framework được.
- **LLM trung lập nhà cung cấp** (`llm/base.py`): agent chỉ biết `Message/ToolCall/LLMResponse`. Adapter giữ `raw` của lượt assistant để gửi lại nguyên vẹn (cần cho thought signature của Gemini).
- **RAG là một tool**, không phải một hệ riêng: agent gọi `search_documents` như mọi tool; mọi tối ưu retrieval vẫn nguyên giá trị.
- **Parent-child + breadcrumb** thay vì chunk phẳng: tìm chính xác, đọc đủ ngữ cảnh.
- **Fail-safe có chủ đích**: router lỗi → agent; critic lỗi → giữ câu trả lời (fail-open, có trace); planner lỗi → kế hoạch một bước; hết ngân sách → trả lời tốt nhất kèm nêu rõ phần thiếu.
