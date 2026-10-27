"""Plan-and-execute: Planner chia câu hỏi phức tạp thành các bước; PlanExecutor chạy từng bước bằng một ReAct agent nhỏ;
gặp bước thất bại thì replan; cuối cùng Synthesizer ghép kết quả thành câu trả lời."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message, Usage, complete_json
from ..observability import Tracer
from ..tools.base import ToolContext, ToolRegistry
from . import prompts
from .evidence import EvidenceStore
from .guardrails import Budget
from .react import AgentSpec, ReActAgent, make_submit_tool


@dataclass
class PlanStep:
    id: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    tool_hint: str | None = None


@dataclass
class StepResult:
    id: str
    goal: str
    status: str  # ok | insufficient | failed | skipped
    answer: str


def parse_steps(data: dict, max_steps: int, known: set[str] = frozenset()) -> list[PlanStep]:
    """Kiểm tra kế hoạch do LLM sinh: id duy nhất, phụ thuộc hợp lệ, không vòng, không quá max_steps."""
    raw = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("'steps' phải là danh sách")
    steps: list[PlanStep] = []
    ids: set[str] = set()
    for i, d in enumerate(raw[:max_steps], start=1):
        sid = str(d.get("id") or f"s{i}")
        if sid in ids or sid in known:
            raise ValueError(f"id bước trùng: {sid}")
        goal = str(d.get("goal", "")).strip()
        if not goal:
            raise ValueError(f"bước {sid} thiếu goal")
        hint = d.get("tool_hint")
        steps.append(PlanStep(sid, goal, [str(x) for x in (d.get("depends_on") or [])], None if hint in (None, "null", "") else str(hint)))
        ids.add(sid)
    for s in steps:  # phụ thuộc chỉ được trỏ tới bước đứng TRƯỚC (đảm bảo không có vòng)
        for dep in s.depends_on:
            if dep not in known and dep not in ids:
                raise ValueError(f"bước {s.id} phụ thuộc id không tồn tại: {dep}")
        later = {x.id for x in steps[steps.index(s):]}
        if later & set(s.depends_on):
            raise ValueError(f"bước {s.id} phụ thuộc vào bước đứng sau nó (vòng/ngược thứ tự)")
    return steps


class Planner:
    def __init__(self, llm: LLMClient, settings, tool_overview: str, tracer: Tracer):
        self.llm, self.s, self.tools, self.tracer = llm, settings, tool_overview, tracer

    def make_plan(self, question: str, budget: Budget) -> list[PlanStep]:
        try:
            data, usage = complete_json(self.llm, system=prompts.planner_system(self.tools, self.s.max_plan_steps),
                                        prompt=f"Question: {question}", model=self.s.main_model, max_tokens=900)
            budget.charge_llm(usage)
            steps = parse_steps(data, self.s.max_plan_steps)
            if not steps:
                raise ValueError("kế hoạch rỗng")
        except Exception as e:  # noqa: BLE001 - kế hoạch hỏng => một bước duy nhất là chính câu hỏi
            self.tracer.event("plan", note=f"planner lỗi ({type(e).__name__}: {e}), dùng kế hoạch 1 bước")
            steps = [PlanStep("s1", question)]
        self.tracer.event("plan", steps=[{"id": s.id, "goal": s.goal, "depends_on": s.depends_on, "tool": s.tool_hint} for s in steps])
        return steps

    def replan(self, question: str, results: dict[str, StepResult], pending: list[PlanStep], budget: Budget) -> list[PlanStep] | None:
        done = "\n".join(f"[{r.id}] ({r.status}) {r.goal}\n  -> {r.answer[:400]}" for r in results.values())
        todo = "\n".join(f"[{s.id}] {s.goal} (depends_on={s.depends_on})" for s in pending) or "(không còn)"
        try:
            data, usage = complete_json(self.llm, system=prompts.replan_system(self.tools, self.s.max_plan_steps),
                                        prompt=f"Question: {question}\n\nFinished steps:\n{done}\n\nPending steps:\n{todo}",
                                        model=self.s.main_model, max_tokens=900)
            budget.charge_llm(usage)
            steps = parse_steps(data, self.s.max_plan_steps, known=set(results))
        except Exception as e:  # noqa: BLE001
            self.tracer.event("plan", note=f"replan lỗi ({type(e).__name__}: {e}), giữ nguyên kế hoạch")
            return None
        self.tracer.event("plan", replan=[{"id": s.id, "goal": s.goal, "depends_on": s.depends_on} for s in steps])
        return steps


class PlanExecutor:
    def __init__(self, llm: LLMClient, settings, tools: ToolRegistry, planner: Planner, tracer: Tracer):
        self.llm, self.s, self.tools, self.planner, self.tracer = llm, settings, tools, planner, tracer

    def _run_step(self, question: str, step: PlanStep, results: dict[str, StepResult], ctx: ToolContext, budget: Budget) -> StepResult:
        deps = "\n".join(f"[{d}] {results[d].goal}\n  -> ({results[d].status}) {results[d].answer}" for d in step.depends_on if d in results)
        hint = f"\nSuggested tool: {step.tool_hint}" if step.tool_hint else ""
        user = (f"<overall_question>{question}</overall_question>\n<your_step id=\"{step.id}\">{step.goal}</your_step>{hint}\n"
                f"<results_of_previous_steps>\n{deps or '(none)'}\n</results_of_previous_steps>")
        agent = ReActAgent(self.llm, AgentSpec(f"step:{step.id}", prompts.step_system(ctx.today.isoformat(), self.s.domain),
                                               self.tools, self.s.main_model, self.s.step_max_steps), self.tracer, self.s)
        run = agent.run([Message("user", user)], ctx, budget, finish_tool=make_submit_tool())
        status = "insufficient" if run.status == "partial" else run.status
        self.tracer.event("step", id=step.id, status=status, tool_calls=run.tool_calls, answer=run.text[:200])
        return StepResult(step.id, step.goal, status, run.text)

    def run(self, question: str, plan: list[PlanStep], ctx: ToolContext, budget: Budget) -> list[StepResult]:
        results: dict[str, StepResult] = {}
        pending = list(plan)
        replans = 0
        while pending:
            if budget.exhausted():
                for s in pending:
                    results[s.id] = StepResult(s.id, s.goal, "skipped", f"Bỏ qua vì {budget.exhausted()}.")
                break
            ready = [s for s in pending if all(d in results for d in s.depends_on)]
            if not ready:  # phụ thuộc không thể thỏa
                for s in pending:
                    results[s.id] = StepResult(s.id, s.goal, "failed", "Không thực hiện được do phụ thuộc không thỏa.")
                break
            if self.s.parallel_steps and len(ready) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(ready))) as ex:
                    outs = list(ex.map(lambda st: self._run_step(question, st, results, ctx, budget), ready))
            else:
                outs = [self._run_step(question, st, results, ctx, budget) for st in ready]
            for st, r in zip(ready, outs):
                results[st.id] = r
            pending = [s for s in pending if s.id not in results]
            bad = [r for r in outs if r.status != "ok"]
            if bad and replans < self.s.max_replans and not budget.exhausted():
                new = self.planner.replan(question, results, pending, budget)
                replans += 1
                if new is not None:
                    pending = new
        return list(results.values())


def synthesize(llm: LLMClient, settings, question: str, results: list[StepResult], ctx: ToolContext, budget: Budget) -> str:
    body = "\n".join(f"[{r.id}] ({r.status}) {r.goal}\n{r.answer}" for r in results)
    resp = llm.chat([Message("user", f"<question>{question}</question>\n<step_results>\n{body}\n</step_results>")],
                    system=prompts.synth_system(ctx.today.isoformat()), model=settings.main_model, max_tokens=settings.max_output_tokens)
    budget.charge_llm(resp.usage)
    return resp.text.strip()
