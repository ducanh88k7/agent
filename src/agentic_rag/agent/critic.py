"""Reflection: một lượt kiểm tra câu trả lời nháp (bám tài liệu? đủ ý? trích dẫn đúng?) rồi sửa nếu cần.
Chạy sau kiểm tra xác định (EvidenceStore.check_citations) vốn không tốn LLM."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message, complete_json
from ..observability import Tracer
from . import prompts
from .evidence import EvidenceStore
from .guardrails import Budget


@dataclass
class Verdict:
    verdict: str = "pass"  # pass | revise
    grounded: bool = True
    complete: bool = True
    issues: list[str] = field(default_factory=list)
    feedback: str = ""


class Critic:
    def __init__(self, llm: LLMClient, settings, tracer: Tracer):
        self.llm, self.s, self.tracer = llm, settings, tracer

    def review(self, question: str, answer: str, evidence: EvidenceStore, budget: Budget) -> Verdict:
        prompt = f"<question>{question}</question>\n<evidence>\n{evidence.render()}\n</evidence>\n<draft_answer>\n{answer}\n</draft_answer>"
        try:
            data, usage = complete_json(self.llm, system=prompts.CRITIC_SYSTEM, prompt=prompt, model=self.s.fast_model, max_tokens=600)
            budget.charge_llm(usage)
            v = Verdict(str(data.get("verdict", "pass")).lower(), bool(data.get("grounded", True)), bool(data.get("complete", True)),
                        [str(i) for i in data.get("issues", [])], str(data.get("feedback", "")))
            if v.verdict not in ("pass", "revise"):
                v.verdict = "pass"
        except Exception as e:  # noqa: BLE001 - critic hỏng thì "fail-open": giữ câu trả lời, ghi trace
            self.tracer.event("critic", note=f"critic lỗi ({type(e).__name__}), bỏ qua")
            return Verdict()
        self.tracer.event("critic", verdict=v.verdict, grounded=v.grounded, complete=v.complete, issues=v.issues)
        return v

    def revise(self, question: str, answer: str, feedback: str, evidence: EvidenceStore, budget: Budget) -> str:
        prompt = (f"<question>{question}</question>\n<evidence>\n{evidence.render()}\n</evidence>\n"
                  f"<draft_answer>\n{answer}\n</draft_answer>\n<reviewer_feedback>\n{feedback}\n</reviewer_feedback>")
        resp = self.llm.chat([Message("user", prompt)], system=prompts.REVISE_SYSTEM, model=self.s.main_model, max_tokens=self.s.max_output_tokens)
        budget.charge_llm(resp.usage)
        self.tracer.event("critic", action="revised")
        return resp.text.strip()
