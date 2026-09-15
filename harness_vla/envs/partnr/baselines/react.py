"""ReAct baseline as an ``AgentHarness`` (beta-only reimplementation).

Mirrors the PARTNR zero-shot ReAct planner (``habitat_llm.planner.
zero_shot_react_planner`` + ``conf/instruct/zero_shot_prompt.yaml``, kept
upstream untouched as the reference) on top of the environment contract:

* prompt = role/system text, task, world description, tool list with
  descriptions, ``Thought: ... / <Action[args]> / Assigned!`` output format;
* the FULL accumulated transcript of this episode (every previous action and
  its ``Agent_<uid>_Observation``) is appended on every turn — no history
  window. This is the ReAct property compared against the harness;
* one LLM call per decision; unparsable output is retried once with a format
  reminder, then the agent declares Done (``rationale="parse failure"``).

Differences from upstream, on purpose: the world description is refreshed
every turn (upstream sends it once and appends ``Objects:`` updates), and the
transport is the same OpenAI-compatible/Ollama client used by the other
harnesses so sampling/seeding/thinking flags stay comparable. No
``habitat_llm`` import.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from partnr_harness.contracts import (
    ActionResult,
    AgentHarness,
    Decision,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    StepContext,
    ToolCall,
    ToolSpec,
)

from harness_vla.envs.partnr.legacy.llm_prefix_harness import LLMPrefixHarness

_LOG = logging.getLogger(__name__)

STOPWORD = "Assigned!"
END_EXPRESSION = "Done[]"

_ACTION_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*?)\]", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an agent that solves multi-agent planning problems. The task "
    "assigned to you will be situated in a house and will generally involve "
    "navigating to objects, picking and placing them on different receptacles "
    "to achieve rearrangement. You strictly follow any format specifications "
    "and pay attention to the previous actions taken in order to avoid "
    "repeating mistakes. Rooms do not need to be explored more than once."
)
PARTNER_PROMPT = (
    "There will be another agent trying to solve the same task that you are at "
    "the same time. You may find that that agent has picked up relevant objects "
    "or is in the process of completing parts of the task. If that is the case "
    "you may want to move on to a different part of the task."
)
EXPLORE_PROMPT = (
    "Rooms do not need to be explored more than once.\n"
    "This means if you have explored the living room and have not found the "
    "object, then you should explore the kitchen, if a relevant object is still "
    "not found, you should explore the hallway etc..."
)
REPEAT_PROMPT = (
    "Many calls to the same action in a row are a sign that something has gone "
    "wrong and you should try a different action."
)
FORMAT_PROMPT = (
    "What is the next action to make progress towards completing the task?\n"
    "Return your response in the following format\n\n"
    "Thought: <reasoning for why you are taking the next action>\n"
    "<next action call>\n"
    f"{STOPWORD}\n\n"
    "Here is an example:\n"
    "Thought: Since there are no objects found I should explore a room I have "
    "not explored yet\n"
    "Explore[<room name>]\n"
    f"{STOPWORD}"
)
_FORMAT_REMINDER = (
    "Your previous reply did not follow the format. Reply with exactly:\n"
    "Thought: <reasoning>\n<ToolName[arguments]>\n"
    f"{STOPWORD}\n"
    f"Use one of the Possible Actions listed above, or {END_EXPRESSION} when "
    "the task is finished."
)


def parse_react_response(text: str) -> Optional[Dict[str, str]]:
    """Return ``{"thought", "name", "args", "action"}`` or ``None``.

    Mirrors ``zero_shot_action_parser``: the action is the last line before
    ``Assigned!``; ``Done[]`` anywhere terminates. Falls back to the last
    bracket call in the text when the last line is not a call.
    """
    if not text or not text.strip():
        return None
    body = text.split(STOPWORD, 1)[0].strip()
    thought = ""
    match = re.search(r"Thought:\s*(.*)", body)
    if match:
        thought = match.group(1).split("\n", 1)[0].strip()
    if END_EXPRESSION in body:
        return {"thought": thought, "name": "Done", "args": "", "action": END_EXPRESSION}
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    call = _ACTION_CALL.fullmatch(lines[-1]) if lines else None
    if call is None:
        calls = list(_ACTION_CALL.finditer(body))
        call = calls[-1] if calls else None
    if call is None:
        return None
    name, args = call.group(1), call.group(2).strip()
    if name == "Thought":
        return None
    return {"thought": thought, "name": name, "args": args, "action": f"{name}[{args}]"}


def tool_descriptions(tools: List[ToolSpec]) -> str:
    lines = []
    for tool in tools:
        description = " ".join(str(tool.description or "").split())
        lines.append(f"- {tool.name}: {description}" if description else f"- {tool.name}")
    lines.append("- Done: Used to indicate that the agent has finished the task. Example (Done[])")
    return "\n".join(lines)


class ReActHarness(AgentHarness):
    """PARTNR zero-shot ReAct as an external harness (single agent per instance)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11435/v1",
        model: str = "gemma3:12b",
        temperature: float = 0.0,
        max_tokens: int = 512,
        num_ctx: Optional[int] = None,
        timeout_s: float = 240.0,
        enable_thinking: bool = False,
        disable_thinking: bool = False,
        max_retries: int = 3,
        repeat_penalty: float = 1.0,
        repeat_last_n: int = 64,
        thinking_budget: int = 0,
        world_char_budget: int = 12000,
        max_transcript_chars: int = 0,
        system_prompt: str = SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            _LOG.warning("ReActHarness ignoring unknown kwargs: %s", sorted(kwargs))
        self.world_char_budget = int(world_char_budget)
        self.max_transcript_chars = int(max_transcript_chars)
        self.system_prompt_text = system_prompt
        self._llm = LLMPrefixHarness(
            prefix=system_prompt,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            world_char_budget=world_char_budget,
            enable_thinking=enable_thinking,
            disable_thinking=disable_thinking,
            num_ctx=num_ctx,
            max_retries=max_retries,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            thinking_budget=thinking_budget,
        )
        self.transcript: List[Dict[str, Any]] = []
        self.instruction: str = ""
        self.agent_uid: int = 0
        self.has_partner: bool = False
        self._tools: List[ToolSpec] = []
        self.last_prompt: str = ""
        self.last_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------ lifecycle

    def on_episode_start(self, context: EpisodeContext) -> None:
        self._llm.on_episode_start(context)
        self.transcript = []
        self.instruction = context.instruction
        self.agent_uid = context.agent_uid
        self.has_partner = bool(context.partner_uids)
        self._tools = list(context.tools)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        pass

    def on_action_result(self, result: ActionResult) -> None:
        pending = next(
            (entry for entry in reversed(self.transcript) if entry["observation"] is None),
            None,
        )
        if pending is None:
            return
        pending["observation"] = " ".join(str(result.response or "").split())
        pending["success_hint"] = result.success_hint

    # --------------------------------------------------------------- prompt

    def system_prompt(self) -> str:
        parts = [self.system_prompt_text]
        if self.has_partner:
            parts.append(PARTNER_PROMPT)
        parts.extend([EXPLORE_PROMPT, REPEAT_PROMPT])
        return "\n\n".join(parts)

    def render_transcript(self) -> str:
        entries = [self._render_entry(entry) for entry in self.transcript]
        omitted = 0
        if self.max_transcript_chars > 0:
            while entries and sum(len(e) + 1 for e in entries) > self.max_transcript_chars:
                entries.pop(0)
                omitted += 1
        if omitted:
            entries.insert(0, f"[... {omitted} earlier turn(s) omitted ...]")
        return "\n".join(entries)

    def _render_entry(self, entry: Dict[str, Any]) -> str:
        observation = entry["observation"]
        if observation is None:
            observation = "(no observation yet)"
        thought = entry.get("thought") or ""
        return (
            f"Thought: {thought}\n{entry['action']}\n{STOPWORD}\n"
            f"Agent_{self.agent_uid}_Observation:{observation}"
        )

    def build_user_prompt(self, context: StepContext) -> str:
        world = context.world_description or ""
        if not world:
            state = (context.extras or {}).get("state") or {}
            world = str(state.get("text") or "")
        if len(world) > self.world_char_budget:
            world = world[: self.world_char_budget] + "\n[...truncated...]"
        tools = self._tools or list(context.tools)
        parts = [
            f"Task: {context.instruction or self.instruction}",
            world,
            f"Possible Actions:\n{tool_descriptions(tools)}",
            FORMAT_PROMPT,
        ]
        transcript = self.render_transcript()
        if transcript:
            parts.append(f"Previous actions and observations:\n{transcript}")
        return "\n\n".join(part for part in parts if part)

    # --------------------------------------------------------------- decide

    def decide(self, context: StepContext) -> Decision:
        if not self._tools:
            self._tools = list(context.tools)
        available = {tool.name for tool in context.tools} or {tool.name for tool in self._tools}
        user = self.build_user_prompt(context)
        self.last_prompt = user
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user},
        ]

        raw = ""
        parsed: Optional[Dict[str, str]] = None
        meta: Dict[str, Any] = {}
        attempts = 0
        for attempt in range(2):
            attempts = attempt + 1
            raw = self._llm._post(messages)
            meta = dict(self._llm.last_meta)
            if meta.get("llm_failure"):
                self.last_meta = meta
                return self._llm._llm_failure(raw, meta)
            parsed = parse_react_response(raw)
            if parsed is not None and (parsed["name"] == "Done" or parsed["name"] in available):
                break
            reminder = _FORMAT_REMINDER
            if parsed is not None:
                reminder = (
                    f"Action {parsed['name']!r} is not one of the Possible Actions "
                    f"({', '.join(sorted(available))}). " + _FORMAT_REMINDER
                )
                parsed = None
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": reminder},
            ]
        meta.update({
            "parse_attempts": attempts,
            "transcript_len": len(self.transcript),
            "prompt_chars": len(messages[0]["content"]) + len(user),
        })
        self.last_meta = meta

        if parsed is None:
            rationale = "parse failure"
            if raw.strip() and parse_react_response(raw) is not None:
                rationale = "unknown tool"
            return DoneDecision(rationale=rationale, raw_output=raw, meta=meta)

        thought = parsed["thought"]
        if parsed["name"] == "Done":
            self._record(parsed, decision_index=context.decision_index)
            return DoneDecision(rationale=thought, raw_output=raw, meta=meta)
        self._record(parsed, decision_index=context.decision_index)
        return ToolCall(parsed["name"], parsed["args"], rationale=thought, raw_output=raw, meta=meta)

    def _record(self, parsed: Dict[str, str], decision_index: int) -> None:
        self.transcript.append({
            "decision_index": decision_index,
            "thought": parsed["thought"],
            "name": parsed["name"],
            "args": parsed["args"],
            "action": parsed["action"],
            "observation": None,
            "success_hint": None,
        })
