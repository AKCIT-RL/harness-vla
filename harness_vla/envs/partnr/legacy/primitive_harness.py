"""beta-only: historical PARTNR prompt harness, kept for equivalence control; not the paper harness.

Primitive-action harness: one atomic JSON decision per turn (single agent).

Beta-only vocabulary agreed on 2026-08-25: Navigate, Pick, Place (3 args, no
next_to), Open, Close, PowerOn/PowerOff/Clean/Fill/PourInPlace, Wait (short)
and Done. Macros (Rearrange, Explore) are excluded; full observability.
"""

import json
import re
from typing import Optional

from partnr_harness.contracts import (
    Decision,
    DoneDecision,
    StepContext,
    ToolCall,
)
from harness_vla.envs.partnr.legacy.llm_prefix_harness import LLMPrefixHarness

_ACTION_SCHEMA = '{"action":"<tool|Done>","args":"<comma-separated args>"}'

_BRACKET_CALL = re.compile(r"^\s*([A-Za-z_]+)\s*\[([^\]]*)\]")
_UNICODE_BRACKETS = str.maketrans({"【": "{", "】": "}", "［": "[", "］": "]"})


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model output."""
    if not text:
        return None
    cleaned = text.strip().translate(_UNICODE_BRACKETS)
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    candidates = [cleaned]
    brace = _first_balanced_object(cleaned)
    if brace is not None:
        candidates.append(brace)

    for cand in candidates:
        for attempt in (cand, _strip_trailing_commas(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


class PrimitiveHarness(LLMPrefixHarness):
    """Single-agent harness restricted to atomic primitives, JSON protocol."""

    def _content_done(self, content: str) -> bool:
        return _first_balanced_object(content) is not None

    def _build_user_message(self, context: StepContext) -> str:
        message, _, _ = super()._build_user_message(context).rpartition("\n\n")
        tool_contracts = "\n".join(
            "- %s: %s" % (tool.name, tool.description.strip())
            for tool in context.tools
        )
        own_actions = (
            "\n\nOwn recent actions:\n" + "\n".join(context.own_recent_actions)
            if context.own_recent_actions
            else ""
        )
        return message + "\n\n" + (
            "Full tool argument contracts:\n%s\n\n"
            "Return exactly one JSON object and nothing else. Use schema %s. "
            "Choose one atomic action per turn from the available tools, or "
            "Done when the whole task is complete. Put the raw contents "
            "of the bracket arguments in args, without brackets, using "
            "exact entity names from the house state."
            "%s"
            % (tool_contracts, _ACTION_SCHEMA, own_actions)
        )

    def decide(self, context: StepContext) -> Decision:
        messages = [
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": self._build_user_message(context)},
        ]
        raw = self._post(messages)
        meta = dict(self.last_meta)
        if meta.get("llm_failure"):
            return self._llm_failure(raw, meta)
        parsed = extract_json_object(raw)
        rationale = raw.strip()[:200]
        if parsed is None:
            # Fallback: models sometimes mimic the history format Tool[args].
            bracket = _BRACKET_CALL.match(raw.strip())
            if bracket and any(
                tool.name == bracket.group(1) for tool in context.tools
            ):
                parsed = {"action": bracket.group(1), "args": bracket.group(2)}
            elif bracket and bracket.group(1) == "Done":
                return DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
            else:
                return ToolCall(
                    "InvalidResponse",
                    rationale,
                    rationale=rationale,
                    raw_output=raw,
                    error_kind="parse_error",
                    meta=meta,
                )

        action = parsed.get("action")
        args = parsed.get("args")
        if isinstance(args, list) and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in args
        ):
            args = ",".join("None" if item is None else str(item) for item in args)
        if isinstance(action, str) and "[" in action and action.endswith("]"):
            bracketed_tool, embedded_args = action.split("[", 1)
            embedded_args = embedded_args[:-1]
            if any(tool.name == bracketed_tool for tool in context.tools):
                action = bracketed_tool
                # Embedded args are the semantic target; prefer them when set.
                if embedded_args or not args:
                    args = embedded_args
        if isinstance(action, str) and (args is None or args == ""):
            if action == "Done":
                args = ""
        if isinstance(args, str) and args.startswith("[") and args.endswith("]"):
            args = args[1:-1]
        if not isinstance(action, str) or not isinstance(args, str):
            return ToolCall(
                "InvalidResponse",
                rationale,
                rationale=rationale,
                raw_output=raw,
                error_kind="compile_error",
                meta=meta,
            )
        if action == "Done":
            return DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
        if action == "Place":
            # No next_to in this vocabulary: constraint/reference forced to None.
            parts = [part.strip() for part in args.split(",")]
            args = ",".join(parts[:3] + ["None", "None"])

        available_tools = {tool.name for tool in context.tools}
        if action not in available_tools:
            error = "Action %r is not one of the available tools." % action
            return ToolCall(
                "InvalidResponse",
                error,
                rationale=rationale,
                raw_output=raw,
                error_kind="compile_error",
                meta=meta,
            )
        return ToolCall(action, args, rationale=rationale, raw_output=raw, meta=meta)
