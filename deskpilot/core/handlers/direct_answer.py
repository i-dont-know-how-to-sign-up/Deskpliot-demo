from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import StepStatus
from ..models import AgentStep


@dataclass
class DirectAnswerOutcome:
    answer: str
    used_llm: bool
    steps: list[AgentStep] = field(default_factory=list)


class DirectAnswerHandler:
    """封装 DirectQA 的生成、降级和枚举一致性复核策略。"""

    def __init__(
        self,
        *,
        answer: Callable[[str, Any], str],
        fallback: Callable[[str, Any], str],
        needs_review: Callable[[str], bool],
        review: Callable[[str, str, int], str],
        api_error: Callable[[], str],
    ) -> None:
        self.answer = answer
        self.fallback = fallback
        self.needs_review = needs_review
        self.review = review
        self.api_error = api_error

    def handle(self, question: str, context: Any, direct_response: str = "") -> DirectAnswerOutcome:
        result = direct_response or self.answer(question, context)
        used_llm = bool(result)
        steps: list[AgentStep] = []
        if not result:
            error = self.api_error()
            if error:
                steps.append(AgentStep("llm_api_call", StepStatus.FAILED, error))
            result = self.fallback(question, context)
        elif self.needs_review(result):
            reviewed = self.review(question, result, int(getattr(context, "output_budget", 0) or 0))
            if reviewed:
                result = reviewed
                steps.append(AgentStep(
                    "review_enumeration", StepStatus.SUCCESS, "已复核枚举数量和统计口径。",
                ))
        steps.append(AgentStep(
            "direct_answer",
            StepStatus.SUCCESS,
            "已由意图路由器判断为通用问题并直接回答。"
            if used_llm else "未调用 LLM API，已使用本地 fallback 生成回答。",
        ))
        return DirectAnswerOutcome(result, used_llm, steps)
