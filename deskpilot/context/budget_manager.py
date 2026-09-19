from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleBudget:
    role: str
    input_tokens: int
    output_tokens: int
    complexity: int


class AdaptiveBudgetManager:
    """根据 Planner 复杂度分配角色预算，避免所有请求固定使用最大窗口。"""

    BASE_INPUT = {"planner": 2600, "router": 2200, "executor": 2600, "answer": 3600, "shared": 4200}
    BASE_OUTPUT = {"planner": 900, "router": 500, "executor": 900, "answer": 1200, "shared": 900}

    def __init__(self, minimum_input: int = 900, maximum_input: int = 9000) -> None:
        self.minimum_input = minimum_input
        self.maximum_input = maximum_input

    def allocate(self, role: str, complexity: int = 3) -> RoleBudget:
        normalized_role = role if role in self.BASE_INPUT else "shared"
        bounded_complexity = max(0, min(int(complexity), 10))
        factor = 0.70 + bounded_complexity * 0.09
        input_tokens = int(self.BASE_INPUT[normalized_role] * factor)
        output_tokens = int(self.BASE_OUTPUT[normalized_role] * (0.75 + bounded_complexity * 0.07))
        return RoleBudget(
            role=normalized_role,
            input_tokens=max(self.minimum_input, min(input_tokens, self.maximum_input)),
            output_tokens=max(256, min(output_tokens, 4096)),
            complexity=bounded_complexity,
        )
