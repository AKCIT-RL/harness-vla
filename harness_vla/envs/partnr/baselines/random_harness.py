"""Trivial harness used for smoke tests: explores rooms, then declares Done.

Classification: beta-only (smoke baseline; no LLM).
"""

import random
from typing import Optional

from partnr_harness.contracts import (
    AgentHarness,
    Decision,
    DoneDecision,
    StepContext,
    ToolCall,
)


class RandomHarness(AgentHarness):
    """Explores random rooms for a fixed number of decisions.

    Exercises the full bridge pipeline (decide -> skill -> result) without an
    LLM. Not meant to solve tasks.
    """

    def __init__(
        self, max_decisions_before_done: int = 6, seed: Optional[int] = 0
    ) -> None:
        self.max_decisions_before_done = max_decisions_before_done
        self._rng = random.Random(seed)

    def decide(self, context: StepContext) -> Decision:
        if context.decision_index >= self.max_decisions_before_done:
            return DoneDecision(rationale="decision budget reached")

        tool_names = {tool.name for tool in context.tools}
        if context.rooms:
            room = self._rng.choice(context.rooms)
            if "Explore" in tool_names:
                return ToolCall("Explore", room, rationale="random exploration")
            if "Navigate" in tool_names:
                return ToolCall("Navigate", room, rationale="random navigation")
        return DoneDecision(rationale="no applicable tool/room")
