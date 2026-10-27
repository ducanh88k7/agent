"""Router: chọn đường đi rẻ nhất đủ trả lời câu hỏi (direct / rag / agent / plan / clarify) và viết lại câu hỏi cho độc lập ngữ cảnh."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm.base import LLMClient, LLMError, Message, Usage, complete_json
from ..observability import Tracer
from . import prompts

ROUTES = ("direct", "rag", "agent", "plan", "clarify")
_SMALLTALK = re.compile(r"^\s*(xin chào|chào( bạn)?|hello|hi|hey|cảm ơn( bạn)?|thanks?( you)?|ok(ay)?|tạm biệt|bye)\W*$", re.I)


@dataclass
class RouteDecision:
    route: str
    standalone_question: str
    reason: str = ""
    confidence: float = 1.0
    clarifying_question: str | None = None
    usage: Usage | None = None


class Router:
    def __init__(self, llm: LLMClient, settings, tool_overview: str, tracer: Tracer):
        self.llm, self.s, self.tools, self.tracer = llm, settings, tool_overview, tracer

    def route(self, question: str, history: list[Message]) -> RouteDecision:
        # 1) Luật rẻ trước: chào hỏi/cảm ơn không cần tốn một lần gọi LLM
        if _SMALLTALK.match(question):
            d = RouteDecision("direct", question, "heuristic: small talk", 1.0)
            self.tracer.event("route", route=d.route, reason=d.reason, source="heuristic")
            return d
        # 2) LLM nhỏ/rẻ phân loại
        try:
            data, usage = complete_json(self.llm, system=prompts.router_system(self.tools, self.s.domain),
                                        prompt=f"Latest user message: {question}", history=history[-6:],
                                        model=self.s.fast_model, max_tokens=400)
            route = str(data.get("route", "")).strip().lower()
            if route not in ROUTES:
                raise ValueError(f"route không hợp lệ: {route!r}")
            d = RouteDecision(route, str(data.get("standalone_question") or question).strip(), str(data.get("reason", "")),
                              float(data.get("confidence", 0.5) or 0.5), data.get("clarifying_question"), usage)
            if d.route == "clarify" and not d.clarifying_question:
                d.route = "agent"  # không có câu hỏi làm rõ thì để agent tự xử lý
        except LLMError:
            raise  # API sập là sự cố hạ tầng: để orchestrator báo lỗi, không che bằng fallback
        except Exception as e:  # noqa: BLE001 - output router hỏng (JSON/route sai): rơi về agent (an toàn, đủ mạnh)
            self.tracer.event("route", route="agent", reason=f"router lỗi ({type(e).__name__}), dùng agent mặc định", source="fallback")
            return RouteDecision("agent", question, "fallback", 0.0)
        self.tracer.event("route", route=d.route, reason=d.reason, confidence=d.confidence, standalone=d.standalone_question, source="llm")
        return d
