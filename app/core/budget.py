"""
Budget enforcement + circuit breaker, both scoped per run (not globally --
each agent task gets its own fresh budget). In-memory per-process dict is
enough for a single-instance deployment; swap RunBudget's storage for
Redis (matching Project 2's pattern) if you need this to survive process
restarts or coordinate across multiple worker processes running agents
concurrently.
"""
import os
from dataclasses import dataclass, field

MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "10"))
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "20000"))
MAX_COST_USD = float(os.environ.get("AGENT_MAX_COST_USD", "0.50"))
MAX_CONSECUTIVE_TOOL_FAILURES = int(os.environ.get("AGENT_MAX_CONSECUTIVE_FAILURES", "3"))


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class CircuitBreakerTripped(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class RunBudget:
    run_id: str
    steps_used: int = 0
    tokens_used: int = 0
    cost_used_usd: float = 0.0
    consecutive_failures: int = 0
    max_steps: int = MAX_STEPS
    max_tokens: int = MAX_TOKENS
    max_cost_usd: float = MAX_COST_USD
    max_consecutive_failures: int = MAX_CONSECUTIVE_TOOL_FAILURES

    def record_step(self, tokens: int, cost_usd: float) -> None:
        self.steps_used += 1
        self.tokens_used += tokens
        self.cost_used_usd += cost_usd
        self._check_budget()

    def record_tool_success(self) -> None:
        self.consecutive_failures = 0

    def record_tool_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise CircuitBreakerTripped(
                f"{self.consecutive_failures} consecutive tool failures "
                f"(threshold: {self.max_consecutive_failures})"
            )

    def _check_budget(self) -> None:
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded(f"Step limit reached ({self.steps_used}/{self.max_steps})")
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceeded(f"Token budget exceeded ({self.tokens_used}/{self.max_tokens})")
        if self.cost_used_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Cost budget exceeded (${self.cost_used_usd:.4f}/${self.max_cost_usd:.4f})"
            )

    def as_dict(self) -> dict:
        return {
            "steps_used": self.steps_used, "max_steps": self.max_steps,
            "tokens_used": self.tokens_used, "max_tokens": self.max_tokens,
            "cost_used_usd": round(self.cost_used_usd, 6), "max_cost_usd": self.max_cost_usd,
            "consecutive_failures": self.consecutive_failures,
        }
