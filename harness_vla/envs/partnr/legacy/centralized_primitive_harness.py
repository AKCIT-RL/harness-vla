"""beta-only: historical PARTNR prompt harness, kept for equivalence control; not the paper harness.

Centralized primitive harness: ONE LLM brain driving ALL agents (beta-only).

Same atomic-primitive vocabulary and JSON protocol as ``PrimitiveHarness``,
but the model controls every agent at once. Each decision is a JSON object
keyed by agent label (``agent_0``, ``agent_1``, ...) whose values are the
per-agent action objects ``{"action": ..., "args": ...}``. The special key
``"all"`` with ``{"action": "Done"}`` ends the episode for everyone.

One chat call per decision boundary; partner observation via the simulator's
action history (updated by the evaluation runner after all planners act in a
tick, so both agents see the same history). Agents not mentioned in the
response keep executing their current skill (or idle with Wait if they have
none).
"""

import json
import re
from typing import Dict, Optional

from partnr_harness.contracts import (
    Decision,
    DoneDecision,
    StepContext,
    ToolCall,
)
from harness_vla.envs.partnr.legacy.llm_prefix_harness import LLMPrefixHarness
from harness_vla.envs.partnr.legacy.primitive_harness import (
    _first_balanced_object,
    extract_json_object,
)

_MULTI_SCHEMA = (
    '{"agent_0":{"action":"<tool|Done>","args":"<comma-separated args>"},'
    '"agent_1":{"action":"<tool>","args":"..."}}'
)
_AGENT_KEY = re.compile(r"^agent_\d+$")

_DECOMPOSE_SCHEMA = (
    '{"subgoals":[{"id":"sg1","description":"...","assigned_to":"agent_0"},'
    '{"id":"sg2","description":"...","assigned_to":"agent_1"}]}'
)


class CentralizedPrimitiveHarness(LLMPrefixHarness):
    """Single LLM instance emitting per-agent atomic primitive actions.

    Optionally maintains persistent subgoal *claims* per agent (R1): the task
    is decomposed once into subgoals assigned to agents; the planner can then
    terminate deterministically when every claim is marked done, instead of
    relying on the LLM to reliably emit Done (which it often retracts).
    """

    def __init__(self, *args, use_claims: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_claims = use_claims
        # label -> {"id", "description", "done"}
        self._claims: Dict[str, Dict[str, object]] = {}
        self.decompose_meta: Optional[Dict[str, object]] = None

    def _content_done(self, content: str) -> bool:
        return _first_balanced_object(content) is not None

    # --------------------------------------------------------------- claims

    def all_claims_done(self) -> bool:
        """True when claims are enabled, exist, and all are marked done."""
        return bool(self.use_claims and self._claims) and all(
            c["done"] for c in self._claims.values()
        )

    def claims(self) -> Dict[str, Dict[str, object]]:
        return dict(self._claims)

    def on_episode_start(self, context) -> None:
        super().on_episode_start(context)
        self._claims = {}
        self.decompose_meta = None
        if not self.use_claims:
            return
        agents = (context.extras or {}).get("agents") or {}
        labels = sorted(agents.keys())
        self._claims = self._decompose(context.instruction, labels)

    def _decompose(self, instruction: str, labels) -> Dict[str, Dict[str, object]]:
        """LLM call (retried once if unparseable) splitting the task into
        per-agent subgoal claims."""
        prompt = (
            f"Task: {instruction}\n\n"
            f"Agents: {', '.join(labels)}.\n"
            "Break this task into subgoals and assign each to an agent. "
            "Return exactly one JSON object with schema "
            f"{_DECOMPOSE_SCHEMA} and nothing else."
        )
        messages = [
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": prompt},
        ]
        parsed: Dict[str, object] = {}
        raw = ""
        meta: Dict[str, object] = {}
        attempts = 0
        while attempts < 2:
            attempts += 1
            raw = self._post(messages)
            meta = dict(self.last_meta)
            if meta.get("llm_failure"):
                continue
            candidate = extract_json_object(raw)
            if isinstance(candidate, dict) and candidate:
                parsed = candidate
                break
        self.decompose_meta = {
            "attempts": attempts,
            "parsed": bool(parsed),
            "raw": raw[:2000],
            "meta": meta,
        }
        claims: Dict[str, Dict[str, object]] = {}
        subgoals = parsed.get("subgoals", [])
        if not isinstance(subgoals, list):
            subgoals = []
        for sg in subgoals:
            if not isinstance(sg, dict):
                continue
            assigned = sg.get("assigned_to")
            if assigned not in labels:
                continue
            entry = {
                "id": sg.get("id"),
                "description": sg.get("description", ""),
                "done": False,
            }
            if assigned in claims:
                claims[assigned]["description"] += "; " + sg.get("description", "")
            else:
                claims[assigned] = entry
        # every agent must hold a claim; agents without one get a standby claim
        for label in labels:
            claims.setdefault(
                label,
                {"id": f"{label}_standby", "description": "assist / standby",
                 "done": False},
            )
        return claims

    # ------------------------------------------------------------- prompting

    def _build_user_message(self, context: StepContext) -> str:
        world = context.world_description
        if len(world) > self.world_char_budget:
            world = world[: self.world_char_budget] + "\n[...truncated...]"

        extras = context.extras or {}
        tools_by_agent = extras.get("tools_by_agent") or {
            f"agent_{context.agent_uid}": [
                {"name": t.name, "description": t.description} for t in context.tools
            ]
        }
        tools_section = "\n\n".join(
            "Tools for %s:\n%s"
            % (
                label,
                "\n".join(
                    "- %s: %s" % (t["name"], str(t.get("description", "")).strip())
                    for t in tools
                ),
            )
            for label, tools in sorted(tools_by_agent.items())
        )
        history_by_agent = extras.get("recent_actions_by_agent") or {}
        history_section = "\n\n".join(
            "Recent actions of %s:\n%s" % (label, "\n".join(actions) or "(none)")
            for label, actions in sorted(history_by_agent.items())
        )
        pending = extras.get("pending_results") or {}
        pending_section = "\n".join(
            "- %s: %s[%s] -> %s"
            % (
                label,
                result.get("tool"),
                result.get("args"),
                result.get("response"),
            )
            for label, result in sorted(pending.items())
            if result
        )
        busy = extras.get("busy_agents") or []
        in_flight = extras.get("in_flight_actions") or {}

        parts = [
            f"Task: {context.instruction}",
            "You control ALL agents at once. Issue one atomic action per agent "
            "per turn. An agent may only use its own tools. To end the whole "
            'episode, answer {"all":{"action":"Done","args":""}}. Agents you '
            "omit keep executing their current action.",
            tools_section,
            f"House state:\n{world}",
        ]
        if history_section:
            parts.append(history_section)
        if self.use_claims and self._claims:
            claims_section = "\n".join(
                f"- {label}: [{'DONE' if c['done'] else 'TODO'}] {c['description']}"
                for label, c in sorted(self._claims.items())
            )
            parts.append(
                "Assigned subgoals (claims). Return Done for an agent only when "
                "ITS claim is complete:\n" + claims_section
            )
        if pending_section:
            parts.append("Latest action results:\n" + pending_section)
        if busy:
            lines = ["Agents still executing (do NOT issue actions for them):"]
            for label in busy:
                action = in_flight.get(label)
                lines.append(f"- {label}: running {action}; result pending"
                             if action else f"- {label}")
            parts.append("\n".join(lines))
        if context.signals:
            parts.append(
                "Warnings:\n"
                + "\n".join(f"- {signal.message}" for signal in context.signals)
            )
        parts.append(
            "Return exactly one JSON object and nothing else. Use schema "
            f"{_MULTI_SCHEMA}. Put the raw bracket-argument contents in args, "
            "without brackets, using exact entity names from the house state."
        )
        return "\n\n".join(parts)

    # --------------------------------------------------------------- parsing

    def decide(self, context: StepContext) -> Dict[str, Optional[Decision]]:
        messages = [
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": self._build_user_message(context)},
        ]
        raw = self._post(messages)
        meta = dict(self.last_meta)
        rationale = raw.strip()[:200]
        deciding = (context.extras or {}).get("deciding_agents") or [
            f"agent_{context.agent_uid}"
        ]
        if meta.get("llm_failure"):
            return {label: self._llm_failure(raw, meta) for label in deciding}

        parsed = extract_json_object(raw)
        if parsed is None or not isinstance(parsed, dict):
            return {
                label: ToolCall(
                    "InvalidResponse",
                    rationale,
                    rationale=rationale,
                    raw_output=raw,
                    error_kind="parse_error",
                    meta=meta,
                )
                for label in deciding
            }

        return self._to_decisions(parsed, context, rationale, raw, meta)

    def _to_decisions(
        self,
        parsed: dict,
        context: StepContext,
        rationale: str,
        raw: str,
        meta: dict,
    ) -> Dict[str, Optional[Decision]]:
        tools_by_agent = (context.extras or {}).get("tools_by_agent") or {}
        decisions: Dict[str, Optional[Decision]] = {}
        for label, payload in parsed.items():
            if label == "all":
                if isinstance(payload, dict) and payload.get("action") == "Done":
                    decisions["all"] = DoneDecision(
                        rationale=rationale, raw_output=raw, meta=meta
                    )
                continue
            if not _AGENT_KEY.match(label):
                continue
            decision = self._per_agent_decision(
                payload, label, tools_by_agent, rationale, raw, meta
            )
            decisions[label] = decision
        if not decisions:
            deciding = (context.extras or {}).get("deciding_agents") or []
            return {
                label: ToolCall(
                    "InvalidResponse",
                    "no per-agent actions found in LLM output",
                    rationale=rationale,
                    raw_output=raw,
                    error_kind="compile_error",
                    meta=meta,
                )
                for label in deciding
            }
        return decisions

    def _per_agent_decision(
        self,
        payload,
        label: str,
        tools_by_agent: dict,
        rationale: str,
        raw: str,
        meta: dict,
    ) -> Decision:
        def invalid(message: str, kind: str) -> ToolCall:
            return ToolCall(
                "InvalidResponse",
                message,
                rationale=rationale,
                raw_output=raw,
                error_kind=kind,
                meta=meta,
            )

        if not isinstance(payload, dict):
            return invalid(f"invalid payload for {label}", "compile_error")
        action = payload.get("action")
        args = payload.get("args")
        if isinstance(args, list) and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in args
        ):
            args = ",".join("None" if item is None else str(item) for item in args)
        if isinstance(action, str) and "[" in action and action.endswith("]"):
            bracketed_tool, embedded_args = action.split("[", 1)
            action = bracketed_tool
            if embedded_args[:-1] or not args:
                args = embedded_args[:-1]
        if isinstance(args, str) and args.startswith("[") and args.endswith("]"):
            args = args[1:-1]
        if not isinstance(action, str) or (args is not None and not isinstance(args, str)):
            return invalid(f"malformed action/args for {label}", "compile_error")
        args = args or ""
        if action == "Done":
            # Per-agent Done latches that agent's claim as complete (R1). The
            # planner converts it to a Wait; the episode ends when ALL claims
            # are done (planner-side), making termination deterministic.
            if self.use_claims and label in self._claims:
                self._claims[label]["done"] = True
            return DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
        if action == "Place":
            parts = [part.strip() for part in args.split(",")]
            args = ",".join(parts[:3] + ["None", "None"])
        available = {t["name"] for t in tools_by_agent.get(label, [])}
        if available and action not in available:
            return invalid(
                f"Action {action!r} is not one of the tools of {label}.",
                "compile_error",
            )
        return ToolCall(action, args, rationale=rationale, raw_output=raw, meta=meta)
