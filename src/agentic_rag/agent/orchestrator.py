"""Điều phối một lượt hỏi: guardrail -> router -> (direct | rag | agent | plan | clarify) -> kiểm tra trích dẫn + critic -> trả lời."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ..llm.base import LLMClient, LLMError, Message
from ..observability import Tracer
from ..retrieval.pipeline import RAGPipeline
from ..tools.base import ToolContext, ToolRegistry
from . import prompts
from .critic import Critic
from .evidence import EvidenceStore
from .guardrails import Budget, sanitize_user_input
from .memory import Session
from .planner import PlanExecutor, Planner, synthesize
from .react import AgentSpec, ReActAgent
from .router import Router

log = logging.getLogger(__name__)
CAVEAT = "\n\n_Lưu ý: một số ý trong câu trả lời có thể chưa được tài liệu xác nhận đầy đủ. Vui lòng kiểm tra lại với phòng Đào tạo._"


@dataclass
class Answer:
    text: str
    route: str
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace_id: str = ""
    usage: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"answer": self.text, "route": self.route, "sources": self.sources, "warnings": self.warnings,
                "trace_id": self.trace_id, "usage": self.usage}


class AgenticRAG:
    def __init__(self, settings, llm: LLMClient, pipeline: RAGPipeline, registry: ToolRegistry):
        self.s, self.llm, self.pipeline, self.registry = settings, llm, pipeline, registry

    # ------------------------------------------------------------------ API chính
    def ask(self, question: str, session: Session | None = None, *, user_id: str | None = None,
            confirm=None, verbose: bool = False) -> Answer:
        s = self.s
        session = session or Session(user_id=user_id, max_turns=s.history_turns)
        user_id = user_id or session.user_id
        tracer = Tracer(verbose=verbose, sink_dir=s.trace_dir)
        budget = Budget.from_settings(s)
        evidence = EvidenceStore()
        today = date.fromisoformat(s.today) if s.today else date.today()
        ctx = ToolContext(evidence=evidence, settings=s, today=today, user_id=user_id, confirm=confirm)
        ctx.extras["budget"] = budget  # để tool 'delegate' (multi-agent sau này) dùng chung ngân sách
        ctx.extras["tracer"] = tracer
        question = sanitize_user_input(question)
        warnings: list[str] = []
        route = "error"
        try:
            if not question:
                return Answer("Bạn muốn hỏi điều gì?", "clarify", trace_id=tracer.trace_id)
            history = session.history()
            decision = Router(self.llm, s, self.registry.describe(), tracer).route(question, history)
            if decision.usage:
                budget.charge_llm(decision.usage)
            route, q = decision.route, decision.standalone_question
            if route == "clarify":
                text = decision.clarifying_question or "Bạn có thể nói rõ hơn được không?"
            elif route == "direct":
                text = self._direct(question, history, ctx, budget)
            elif route == "rag":
                text = self._rag(q, ctx, tracer, budget)
            elif route == "plan":
                text = self._plan(q, ctx, tracer, budget)
            else:
                route = "agent"
                text = self._agent(q, history, ctx, tracer, budget)
            text, w = self._finalize(q, text, route, ctx, tracer, budget)
            warnings += w
        except LLMError as e:
            log.exception("LLM error")
            tracer.event("error", type="LLMError", message=str(e))
            text, warnings = "Xin lỗi, hệ thống AI đang gặp sự cố. Bạn vui lòng thử lại sau.", ["llm_error"]
        except Exception as e:  # noqa: BLE001 - không để lộ chi tiết nội bộ cho người dùng
            log.exception("Unhandled error")
            tracer.event("error", type=type(e).__name__, message=str(e))
            text, warnings = "Xin lỗi, đã có lỗi khi xử lý câu hỏi. Bạn vui lòng thử lại.", ["internal_error"]
        else:
            session.add_turn(question, text)
        tracer.flush()
        u = budget.usage
        return Answer(text, route, evidence.sources_for(text), warnings, tracer.trace_id,
                      {"llm_calls": budget.llm_calls, "tool_calls": budget.tool_calls, "input_tokens": u.input_tokens,
                       "output_tokens": u.output_tokens}, tracer.events)

    # ------------------------------------------------------------------ các route
    def _direct(self, question, history, ctx, budget) -> str:
        resp = self.llm.chat([*history, Message("user", question)], system=prompts.direct_system(ctx.today.isoformat(), self.s.domain),
                             model=self.s.fast_model, max_tokens=self.s.max_output_tokens)
        budget.charge_llm(resp.usage)
        return resp.text.strip()

    def _rag(self, q, ctx, tracer, budget) -> str:
        """Pipeline cố định (Advanced RAG): retrieve -> trả lời bám tài liệu. Rẻ, nhanh, dễ đoán."""
        res = self.pipeline.retrieve(q)
        tracer.event("retrieve", query=q, found=len(res.passages), best=res.best_score, **{k: v for k, v in res.debug.items() if k in ("fused", "reranked")})
        if not res.passages:  # không đoạn nào đủ liên quan => không gọi LLM sinh, tránh bịa
            return "Mình không tìm thấy thông tin này trong tài liệu hiện có. Bạn có thể hỏi lại cụ thể hơn hoặc liên hệ phòng Đào tạo để được hỗ trợ."
        blocks = "\n".join(f'<document id="{ctx.evidence.add(p)}" source="{p.label}">\n{p.text}\n</document>' for p in res.passages)
        resp = self.llm.chat([Message("user", f"<question>{q}</question>\n{blocks}")], system=prompts.rag_answer_system(ctx.today.isoformat(), self.s.domain),
                             model=self.s.main_model, max_tokens=self.s.max_output_tokens)
        budget.charge_llm(resp.usage)
        return resp.text.strip()

    def _agent(self, q, history, ctx, tracer, budget) -> str:
        spec = AgentSpec("agent", prompts.agent_system(ctx.today.isoformat(), self.s.domain), self.registry, self.s.main_model, self.s.react_max_steps)
        run = ReActAgent(self.llm, spec, tracer, self.s).run([*history, Message("user", q)], ctx, budget)
        tracer.event("agent_done", status=run.status, stop=run.stop, steps=run.steps, tool_calls=run.tool_calls)
        return run.text

    def _plan(self, q, ctx, tracer, budget) -> str:
        planner = Planner(self.llm, self.s, self.registry.describe(), tracer)
        steps = planner.make_plan(q, budget)
        results = PlanExecutor(self.llm, self.s, self.registry, planner, tracer).run(q, steps, ctx, budget)
        return synthesize(self.llm, self.s, q, results, ctx, budget)

    # ------------------------------------------------------------------ kiểm tra cuối + reflection
    def _finalize(self, q, answer, route, ctx, tracer, budget) -> tuple[str, list[str]]:
        ev, warnings = ctx.evidence, []
        use_critic = len(ev) > 0 and (self.s.critic_mode == "always" or (self.s.critic_mode == "auto" and route in ("agent", "plan")))
        critic = Critic(self.llm, self.s, tracer)
        for attempt in range(self.s.max_reflections + 1):
            issues = ev.check_citations(answer)  # 1) kiểm tra xác định, miễn phí
            verdict = critic.review(q, answer, ev, budget) if use_critic and not budget.exhausted() else None  # 2) LLM critic
            problems = issues + (verdict.issues if verdict and verdict.verdict == "revise" else [])
            needs_fix = bool(issues) or bool(verdict and verdict.verdict == "revise")
            tracer.event("check", attempt=attempt, citation_issues=issues, critic=verdict.verdict if verdict else "skipped")
            if not needs_fix:
                return answer, warnings
            if attempt == self.s.max_reflections or budget.exhausted():
                answer = ev.strip_unknown(answer)  # không để lọt trích dẫn bịa
                warnings += problems or ["chưa qua kiểm tra"]
                return (answer + CAVEAT if verdict and verdict.verdict == "revise" else answer), warnings
            feedback = "; ".join(problems) + (f". {verdict.feedback}" if verdict and verdict.feedback else "")
            answer = critic.revise(q, answer, feedback, ev, budget)
        return answer, warnings
