"""Example harness that reacts to non-privileged failure signals.

Classification: beta-only (smoke baseline; no LLM).

Demonstrates the recovery hooks enabled by ``StepContext.signals``:
- ``skill_failure_loop``: stop retrying the failing target (blacklist it);
- ``duplicate_subtask``: skip work the partner already finished.
"""

from typing import Optional, Set

from partnr_harness.contracts import (
    DUPLICATE_SUBTASK,
    SKILL_FAILURE_LOOP,
    Decision,
    DoneDecision,
    StepContext,
    ToolCall,
)

from harness_vla.envs.partnr.baselines.random_harness import RandomHarness


class RecoveryAwareHarness(RandomHarness):
    def __init__(
        self, max_decisions_before_done: int = 6, seed: Optional[int] = 0
    ) -> None:
        super().__init__(max_decisions_before_done, seed)
        self._blacklist: Set[str] = set()

    def decide(self, context: StepContext) -> Decision:
        signal_kinds = {signal.kind for signal in context.signals}

        for signal in context.signals:
            if signal.kind in (SKILL_FAILURE_LOOP, DUPLICATE_SUBTASK):
                self._blacklist.add(signal.data.get("args", ""))

        context.rooms = [room for room in context.rooms if room not in self._blacklist]
        decision = super().decide(context)

        if isinstance(decision, ToolCall) and signal_kinds:
            decision.rationale += f" (recovered from: {sorted(signal_kinds)})"
        return decision
