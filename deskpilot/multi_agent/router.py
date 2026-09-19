from __future__ import annotations

from dataclasses import dataclass

from .planner import PlannerAgent
from .schemas import TaskPlan


@dataclass
class RouteDecision:
    route: str
    score: int
    reason: str
    plan: TaskPlan


class MultiAgentRouter:
    """根据任务复杂度选择单智能体或多智能体路径。"""

    def __init__(self, planner: PlannerAgent | None = None):
        self.planner = planner or PlannerAgent()

    def route(self, question: str) -> RouteDecision:
        plan = self.planner.build_plan(question)
        if plan.complexity_score >= 7 or plan.route == "multi_agent":
            route = "multi_agent"
            reason = "存在多领域依赖或多个执行节点"
        else:
            route = "single_agent"
            reason = "单领域、低复杂度任务，避免额外智能体开销"
        if any(step.requires_human for step in plan.steps):
            reason += "；计划包含外部副作用，必须人工确认"
        return RouteDecision(route, plan.complexity_score, reason, plan)

