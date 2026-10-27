# Lộ trình nâng cấp

## A. Từ Single-Agent lên Multi-Agent (khi nào và làm thế nào)

**Đừng nâng cấp chỉ vì nghe hay.** Multi-agent đắt hơn, chậm hơn, khó debug hơn. Chỉ nâng khi `eval-e2e` + trace cho thấy một agent đơn thất bại vì:
(1) quá nhiều tool khiến nó chọn sai; (2) prompt hệ thống quá dài/mâu thuẫn vì phải phục vụ nhiều miền; (3) cần model/quyền khác nhau cho từng miền.

Codebase đã chuẩn bị sẵn nền:

| Đã có | Dùng để làm gì |
|---|---|
| `Tool.tags` + `ToolRegistry.subset(names=, tags=)` | chia kho tool theo miền: `docs`, `data`, `calc`, `write` |
| `AgentSpec(name, system_prompt, tools, model, max_steps)` | mô tả một agent chuyên biệt |
| `ReActAgent.run(...)` + `make_submit_tool()` | chạy một agent con và nhận `{status, answer}` có cấu trúc |
| `Budget`, `EvidenceStore`, `Tracer` truyền qua `ToolContext.extras` | ngân sách, trích dẫn, trace dùng chung cho mọi agent con |
| `Planner/PlanExecutor` | đã là "supervisor" dạng kế hoạch: mỗi bước có thể giao cho agent chuyên biệt |

### Bước 1: định nghĩa các agent chuyên biệt

```python
# agent/specialists.py  (bản phác, chưa chạy thử)
def build_specialists(registry, settings, today, domain) -> dict[str, AgentSpec]:
    return {
        "docs_agent": AgentSpec("docs_agent", prompts.step_system(today, domain), registry.subset(tags=["docs"]), settings.main_model, 5),
        "data_agent": AgentSpec("data_agent", DATA_PROMPT,  registry.subset(tags=["data", "calc"]), settings.fast_model, 4),
    }
```

### Bước 2: tool `delegate` cho supervisor (mẫu handoff)

```python
def make_delegate_tool(llm, settings, specialists) -> Tool:
    def run(args, ctx):
        spec = specialists[args["agent"]]
        sub = ReActAgent(llm, spec, ctx.extras["tracer"], settings)
        out = sub.run([Message("user", args["task"])], ctx, ctx.extras["budget"], finish_tool=make_submit_tool())
        return f"[{out.status}] {out.text}"           # kết quả quay về supervisor
    return Tool("delegate", "Giao một việc cho agent chuyên biệt (docs_agent | data_agent). Viết task đầy đủ, tự chứa.",
                {"type": "object", "properties": {"agent": {"type": "string", "enum": list(specialists)},
                                                  "task": {"type": "string"}}, "required": ["agent", "task"]}, run)
```

### Bước 3: supervisor = ReActAgent chỉ có tool `delegate`

Thay `_agent()` trong `orchestrator.py` bằng một `ReActAgent` mà `tools=ToolRegistry([delegate])`. Các route `rag`, `direct`, `clarify`, `plan` giữ nguyên.
Sau đó **chạy lại cùng golden set** và so với single-agent: pass rate, số lần gọi LLM, token, độ trễ. Nếu không tốt hơn rõ rệt thì đừng giữ.

### Lưu ý khi chuyển sang multi-agent
- Agent con chỉ nhận `task` + ngữ cảnh tối thiểu (đừng đổ cả lịch sử): tiết kiệm token, giảm nhiễu.
- Trích dẫn `[S#]` vẫn nhất quán vì chung `EvidenceStore`.
- Mỗi agent con cần mô tả rõ năng lực để supervisor giao đúng; sai ở đây là lỗi phổ biến nhất.
- Giữ giới hạn độ sâu (agent con không được gọi `delegate`) để tránh đệ quy.

## B. Chuyển sang framework (LangGraph / LlamaIndex...)

| Thành phần tự viết | Tương đương LangGraph | Ghi chú |
|---|---|---|
| `Router` + `_route` | node phân loại + `add_conditional_edges` | state: `question`, `route`, `messages`, `evidence` |
| `ReActAgent` | `create_react_agent` hoặc node LLM + `ToolNode` | giữ nguyên `Tool.func` làm tool |
| `Planner` / `PlanExecutor` / `synthesize` | các node plan → execute (lặp) → synthesize, cạnh có điều kiện replan | pattern "plan-and-execute" có sẵn mẫu |
| `Critic` + `_finalize` | node review → cạnh `revise`/`end` (giới hạn vòng) | |
| `Budget` | `recursion_limit` + bộ đếm trong state | tự thêm giới hạn token/thời gian |
| `Tracer` | LangSmith hoặc Langfuse | không phải tự viết nữa |
| `Session` | checkpointer (SQLite/Postgres) | |
| `retrieval/` | giữ nguyên, bọc thành tool/retriever | framework không cải thiện chất lượng retrieval cho bạn |

Thứ tự khuyên dùng: (1) giữ `retrieval/` và `tools/` nguyên vẹn; (2) dựng lại luồng router → agent → critic bằng LangGraph, **chạy `eval-e2e` trên cả hai bản** để chắc hành vi không tụt;
(3) chuyển trace sang Langfuse/LangSmith; (4) mới bỏ code tự viết tương ứng.

## C. Hướng mở rộng khác
- Parse PDF/bảng (Docling), nhiều loại tài liệu, ingest tăng dần theo hash nội dung.
- Contextual Retrieval (LLM viết ngữ cảnh cho từng chunk lúc index), query rewriting nhiều biến thể, MMR.
- LLM-as-judge cho `eval-e2e`, bộ ca khó (retrieve lần đầu trượt, câu mơ hồ, tài liệu chứa lệnh độc hại).
- Streaming (SSE), semantic cache, tóm tắt lịch sử dài, bộ nhớ dài hạn.
- Xác thực thật (JWT) cho API; tách quyền đọc/ghi; hàng đợi cho tool ghi.
