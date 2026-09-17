"""Prompt construction for the PARTNR Harness VLA adapter.

Mirrors ``harness_vla.planner.prompts`` (role preamble, fixed primitive
library, Global Memory, structured state, history, single-JSON output) with
the PARTNR primitive vocabulary. Paper-confirmed: the full episode transcript
is the agent's short-term memory (``max_history=0`` renders every step).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from harness_vla.envs.partnr.primitives import format_action

PARTNR_ROLE_SUFFIX = (
    "A benchmark driver is already running and waiting for your commands. Each "
    "turn you read the house state and the transcript of what has happened so "
    "far, then choose ONE primitive to execute. The simulator executes it and "
    "reports the outcome; you iterate until the whole task is complete."
)

PRIMITIVE_REFERENCE = """\
Fixed primitive library (you may ONLY use these; you cannot invent new ones).
Bracket syntax: Name[arg1, arg2, ...] with exact entity names from the house state.
  Navigate[target]                 -> walk to a furniture, object or room.
  Pick[object]                     -> grasp an object you are next to (one at a time).
  Place[object, relation, furniture, spatial_constraint, reference]
                                   -> put the held object on/within a furniture you are next to;
                                      relation is on or within; spatial_constraint is next_to or
                                      None; reference is an object for next_to or None.
  Open[furniture] / Close[furniture]
  PowerOn[object] / PowerOff[object] / Clean[object] / Fill[object] / Pour[object]
  Wait[]                           -> idle for a short time (e.g. while a partner finishes).
  Done[]                           -> declare the whole task finished."""

GENERIC_GUIDANCE = """\
Operating rules (task-independent):
  - Navigate to the furniture that holds an object before Pick; Navigate to the destination furniture before Place.
  - You can hold one object at a time; Place it before picking another.
  - Do not repeat an action that already succeeded; read the transcript before acting.
  - If an action failed, change something (target, position, order) instead of re-issuing it unchanged.
  - If your partner holds an object, do not try to pick it; if your partner is handling a sub-task, pick another.
  - Emit Done[] only when every part of the instruction is satisfied for the whole team."""

OUTPUT_CONTRACT = """\
Respond with a SINGLE JSON object and nothing else, in this exact shape:
{
  "thought": "<one or two sentences: current state, why this primitive now>",
  "action": "Name[arg1, arg2]"
}
Do not output multiple actions. Do not wrap the JSON in markdown fences.
Output only the JSON object."""


def build_partnr_system_prompt(prefix: str, tools_text: str, global_memory_text: str) -> str:
    """Static part: role, primitive library, env tool contracts, memory, output."""
    parts = [" ".join(part for part in (prefix.strip(), PARTNR_ROLE_SUFFIX) if part)]
    parts.append(PRIMITIVE_REFERENCE)
    if tools_text.strip():
        parts.append("Tool contracts reported by the environment:\n" + tools_text.strip())
    parts.append(GENERIC_GUIDANCE)
    if global_memory_text.strip():
        parts.append(global_memory_text.strip())
    parts.append(OUTPUT_CONTRACT)
    return "\n\n".join(parts)


def tools_text_from_specs(tools: Sequence[Any]) -> str:
    lines = []
    for tool in tools:
        name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, Mapping) else None)
        if not name:
            continue
        description = getattr(tool, "description", None)
        if description is None and isinstance(tool, Mapping):
            description = tool.get("description", "")
        description = " ".join(str(description or "").split())
        lines.append(f"  - {name}: {description}" if description else f"  - {name}")
    return "\n".join(lines)


# ------------------------------------------------------------------- state

def _states_text(states: Any) -> str:
    if isinstance(states, Mapping):
        return ", ".join(f"{key}={value}" for key, value in sorted(states.items()))
    return str(states) if states else ""


def render_state(
    state: Mapping[str, Any],
    agent_uid: Optional[int] = None,
    agent_names: Optional[Mapping[str, str]] = None,
) -> str:
    """Compact text of the structured state (rooms, furniture, objects, agents).

    ``agent_uid`` renders that agent as "you" and the others as "partner";
    ``agent_names`` (uid -> display name) overrides this for a controller that
    addresses several agents by label.
    """
    names = {str(key): str(value) for key, value in (agent_names or {}).items()}

    def who(uid: Any) -> str:
        key = str(uid)
        if key in names:
            return names[key]
        if agent_uid is not None and key == str(agent_uid):
            return "you"
        return "partner"

    lines: List[str] = []
    rooms = list(state.get("rooms") or [])
    furniture: Dict[str, Any] = state.get("furniture") or {}
    objects: Dict[str, Any] = state.get("objects") or {}
    agents: Dict[str, Any] = state.get("agents") or {}

    lines.append("Rooms: " + (", ".join(rooms) if rooms else "(none listed)"))
    lines.append("Furniture by room:")
    by_room: Dict[str, List[str]] = {}
    for name, entry in furniture.items():
        entry = entry or {}
        label = name
        states = _states_text(entry.get("states"))
        if states:
            label += f" ({states})"
        by_room.setdefault(str(entry.get("room") or "unknown_room"), []).append(label)
    for room in rooms + sorted(set(by_room) - set(rooms)):
        if room in by_room:
            lines.append(f"  {room}: " + ", ".join(by_room[room]))
    if not by_room:
        lines.append("  (none listed)")

    lines.append("Objects:")
    if not objects:
        lines.append("  (none listed)")
    for name, entry in objects.items():
        entry = entry or {}
        if entry.get("held_by") is not None:
            holder = entry["held_by"]
            holder_name = names.get(str(holder))
            if holder_name is None:
                holder_name = "you" if agent_uid is not None and str(holder) == str(agent_uid) else f"agent {holder}"
            where = f"held by {holder_name}"
        elif entry.get("furniture"):
            relation = entry.get("relation") or "on"
            where = f"{relation} {entry['furniture']}"
            if entry.get("room"):
                where += f" ({entry['room']})"
        else:
            where = f"in {entry.get('room') or 'unknown location'}"
        states = _states_text(entry.get("states"))
        lines.append(f"  {name}: {where}" + (f"; {states}" if states else ""))

    lines.append("Agents:")
    if not agents:
        lines.append("  (none listed)")
    for key, entry in agents.items():
        entry = entry or {}
        holding = entry.get("holding") or "nothing"
        lines.append(f"  agent {key} ({who(key)}): in {entry.get('room') or 'unknown room'}, holding {holding}")
    return "\n".join(lines)


def _truncate(text: str, budget: int) -> str:
    if budget and len(text) > budget:
        return text[:budget] + "\n[...truncated to budget...]"
    return text


# -------------------------------------------------------------------- turn

def render_history_line(index: int, record: Mapping[str, Any]) -> str:
    action = record.get("action") or format_action(record.get("name", "?"), record.get("args", ""))
    status = record.get("status") or "pending"
    message = " ".join(str(record.get("message") or "").split())
    line = f"  {index}. {action} -> {status}"
    if message:
        line += f": {message}"
    return line


def build_partnr_turn_prompt(
    step_ctx_like: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    resolved_task_memory: Optional[Sequence[str]] = None,
    max_history: int = 0,
    state_char_budget: int = 12000,
) -> str:
    """Per-turn user message.

    ``step_ctx_like`` is a dict view of ``StepContext`` (``instruction``,
    ``agent_uid``, ``decision_index``, ``max_decisions``, ``world_description``,
    ``partner_recent_actions``, ``signals``, ``extras.state``). ``history`` is
    the full-episode list of ``{"decision_index", "action", "status",
    "message"}``; ``max_history=0`` renders all of it (paper-confirmed).
    """
    ctx = step_ctx_like
    agent_uid = ctx.get("agent_uid")
    lines: List[str] = []
    instruction = str(ctx.get("instruction", "")).rstrip(".")
    lines.append(f"Task instruction: {instruction}")
    decision_index = ctx.get("decision_index")
    max_decisions = ctx.get("max_decisions")
    if decision_index is not None and max_decisions:
        lines.append(f"Decision {int(decision_index) + 1} of {max_decisions} (budget).")
    lines.append("")

    state = (ctx.get("extras") or {}).get("state")
    if isinstance(state, Mapping) and (state.get("objects") or state.get("furniture")):
        lines.append("House state (authoritative, current):")
        lines.append(_truncate(render_state(state, agent_uid), state_char_budget))
    else:
        lines.append("House state (authoritative, current):")
        lines.append(_truncate(str(ctx.get("world_description") or "(unavailable)"), state_char_budget))
    lines.append("")

    if resolved_task_memory:
        lines.append("Task Specific Memory structural prior (not an execution script):")
        lines.append(
            "  All entity names below were re-grounded from the CURRENT house state. "
            "Use this only as a structural prior; the current state and transcript "
            "take precedence."
        )
        for index, command in enumerate(resolved_task_memory, 1):
            lines.append(f"  {index}. {command}")
        lines.append("")

    partner = list(ctx.get("partner_recent_actions") or [])
    if partner:
        lines.append("Partner recent actions (most recent last):")
        lines.extend(f"  - {item}" for item in partner)
        lines.append("")

    signals = ctx.get("signals") or []
    messages = []
    for signal in signals:
        message = getattr(signal, "message", None)
        if message is None and isinstance(signal, Mapping):
            message = signal.get("message")
        if message:
            messages.append(str(message))
    if messages:
        lines.append("Warnings derived from the transcript:")
        lines.extend(f"  - {message}" for message in messages)
        lines.append("")

    shown = list(history) if not max_history or max_history <= 0 else list(history)[-max_history:]
    if shown:
        omitted = len(history) - len(shown)
        header = "Transcript of your actions this episode (most recent last)"
        if omitted:
            header += f" [{omitted} earlier steps omitted]"
        lines.append(header + ":")
        for offset, record in enumerate(shown):
            index = record.get("decision_index")
            index = int(index) + 1 if isinstance(index, int) else omitted + offset + 1
            lines.append(render_history_line(index, record))
        lines.append("")
    else:
        lines.append("No actions executed yet this episode.")
        lines.append("")

    lines.append(
        "Choose the single best next primitive to make progress toward the task. "
        "Respond with the JSON object described above."
    )
    return "\n".join(lines)


def step_context_to_dict(context: Any) -> Dict[str, Any]:
    """Dict view of an attrs ``StepContext`` (or pass a dict through)."""
    if isinstance(context, Mapping):
        return dict(context)
    keys = (
        "instruction", "agent_uid", "decision_index", "max_decisions",
        "world_description", "rooms", "partner_recent_actions", "own_recent_actions",
        "signals", "extras",
    )
    return {key: getattr(context, key, None) for key in keys}


# ------------------------------------------------------------------- team
# Prompts for the Harness VLA controllers (``agent.HarnessVlaCentralized`` and
# ``decentralized.HarnessVlaDecentralized``): one JSON object per turn keyed
# by agent label. The same builders serve both; the decentralized agent just
# controls a single label and reads its partner's published transcript.

TEAM_GUIDANCE = """\
Operating rules (task-independent):
  - Navigate to the furniture that holds an object before Pick; Navigate to the destination furniture before Place.
  - Each agent holds one object at a time; Place it before picking another.
  - Do not repeat an action that already succeeded; read the transcript before acting.
  - If an action failed, change something (target, position, order) instead of re-issuing it unchanged.
  - Never send two agents after the same object; split the work so both agents are useful.
  - An agent still executing an action cannot receive a new one this turn."""

TEAM_DONE_RULE = (
    '  - Emit Done[] for an agent only when its part is complete; emit "all": "Done[]" '
    "only when every part of the instruction is satisfied for the whole team."
)
SOLO_DONE_RULE = (
    "  - Emit Done[] only when your own part is complete and nothing is left that "
    "you can still do to finish the whole task."
)


def team_output_contract(labels: Sequence[str], allow_all: bool) -> str:
    """Output shape: ``thought`` plus one key per controlled agent label."""
    lines = ["Respond with a SINGLE JSON object and nothing else, in this exact shape:", "{"]
    lines.append('  "thought": "<one or two sentences: current state, who does what now and why>",')
    entries = [f'  "{label}": "Name[arg1, arg2]" or null' for label in labels]
    if allow_all:
        entries.append('  "all": null or "Done[]"')
    lines.append(",\n".join(entries))
    lines.append("}")
    lines.append(
        "Include a key only for the agents listed as awaiting a decision this turn; "
        "use null to keep an agent idle. Each value is exactly one primitive call "
        "Name[args] with exact entity names from the house state, or Done[]."
    )
    if allow_all:
        lines.append('Set "all": "Done[]" when the WHOLE task is complete for the team.')
    lines.append("Do not wrap the JSON in markdown fences. Output only the JSON object.")
    return "\n".join(lines)


def tools_text_by_agent(tools_by_agent: Mapping[str, Sequence[Any]]) -> str:
    """Tool contracts grouped per agent label."""
    blocks = []
    for label in sorted(tools_by_agent):
        text = tools_text_from_specs(tools_by_agent[label])
        blocks.append(f"Tools of {label}:\n{text}" if text else f"Tools of {label}: (none reported)")
    return "\n".join(blocks)


def build_team_system_prompt(
    role: str,
    tools_by_agent: Mapping[str, Sequence[Any]],
    global_memory_text: str,
    labels: Sequence[str],
    allow_all: bool,
) -> str:
    """Static part for a controller: role, primitives, per-agent tools, memory, output."""
    parts = [" ".join(part for part in (role.strip(), PARTNR_ROLE_SUFFIX) if part)]
    parts.append(PRIMITIVE_REFERENCE)
    tools_text = tools_text_by_agent(tools_by_agent)
    if tools_text.strip():
        parts.append("Tool contracts reported by the environment:\n" + tools_text)
    parts.append(TEAM_GUIDANCE + "\n" + (TEAM_DONE_RULE if allow_all else SOLO_DONE_RULE))
    if global_memory_text.strip():
        parts.append(global_memory_text.strip())
    parts.append(team_output_contract(labels, allow_all))
    return "\n\n".join(parts)


def render_team_transcript_line(index: int, record: Mapping[str, Any]) -> str:
    """``  3. agent_0: Pick[apple_0] -> success: ...`` (label-prefixed)."""
    line = render_history_line(index, record)
    label = record.get("agent")
    if label:
        head, _, tail = line.partition(". ")
        return f"{head}. {label}: {tail}"
    return line


def render_team_transcript(
    history: Sequence[Mapping[str, Any]], max_history: int = 0
) -> List[str]:
    shown = list(history) if not max_history or max_history <= 0 else list(history)[-max_history:]
    omitted = len(history) - len(shown)
    lines: List[str] = []
    for offset, record in enumerate(shown):
        index = record.get("decision_index")
        index = int(index) + 1 if isinstance(index, int) else omitted + offset + 1
        lines.append(render_team_transcript_line(index, record))
    return lines


def build_team_turn_prompt(
    view: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    resolved_task_memory: Optional[Sequence[str]] = None,
    partner_task_memory: Optional[Sequence[str]] = None,
    partner_transcript: Optional[Sequence[Mapping[str, Any]]] = None,
    claims: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_history: int = 0,
    state_char_budget: int = 12000,
) -> str:
    """Per-turn user message for a controller.

    ``view`` keys: ``instruction``, ``decision_index``, ``max_decisions``,
    ``state`` (structured) or ``world_description``, ``agent_names`` (uid ->
    display name), ``deciding`` (labels awaiting a decision), ``busy`` (label
    -> in-flight action), ``signals``. ``history`` records carry ``agent``.
    """
    lines: List[str] = []
    instruction = str(view.get("instruction", "")).rstrip(".")
    lines.append(f"Task instruction: {instruction}")
    decision_index = view.get("decision_index")
    max_decisions = view.get("max_decisions")
    if decision_index is not None and max_decisions:
        lines.append(f"Decision {int(decision_index) + 1} of {max_decisions} (budget).")
    lines.append("")

    state = view.get("state")
    lines.append("House state (authoritative, current):")
    if isinstance(state, Mapping) and (state.get("objects") or state.get("furniture")):
        lines.append(_truncate(render_state(state, None, view.get("agent_names")), state_char_budget))
    else:
        lines.append(_truncate(str(view.get("world_description") or "(unavailable)"), state_char_budget))
    lines.append("")

    if claims:
        lines.append("Assigned subgoals (claims). Emit Done[] for an agent only when ITS claim is complete:")
        for label, claim in sorted(claims.items()):
            flag = "DONE" if claim.get("done") else "TODO"
            lines.append(f"  - {label}: [{flag}] {claim.get('description', '')}")
        lines.append("")

    if resolved_task_memory:
        lines.append("Task Specific Memory structural prior (not an execution script):")
        lines.append(
            "  All entity names below were re-grounded from the CURRENT house state. "
            "Use this only as a structural prior; the current state and transcript "
            "take precedence."
        )
        for index, command in enumerate(resolved_task_memory, 1):
            lines.append(f"  {index}. {command}")
        lines.append("")

    if partner_task_memory:
        lines.append("Partner's remembered plan for this task (re-grounded; the partner may follow it):")
        for index, command in enumerate(partner_task_memory, 1):
            lines.append(f"  {index}. {command}")
        lines.append("")

    deciding = list(view.get("deciding") or [])
    busy: Mapping[str, Any] = view.get("busy") or {}
    if deciding:
        lines.append("Agents awaiting your decision this turn: " + ", ".join(deciding))
    if busy:
        lines.append("Agents still executing (do NOT command them this turn):")
        for label, action in sorted(busy.items()):
            lines.append(f"  - {label}: running {action}; result pending" if action else f"  - {label}")
    if deciding or busy:
        lines.append("")

    messages = []
    for signal in view.get("signals") or []:
        message = getattr(signal, "message", None)
        if message is None and isinstance(signal, Mapping):
            message = signal.get("message")
        if message:
            messages.append(str(message))
    if messages:
        lines.append("Warnings derived from the transcript:")
        lines.extend(f"  - {message}" for message in messages)
        lines.append("")

    if partner_transcript is not None:
        lines.append("Partner's transcript this episode, as published by the partner (most recent last):")
        partner_lines = render_team_transcript(partner_transcript, max_history)
        lines.extend(partner_lines or ["  (nothing published yet)"])
        lines.append("")

    own_header = (
        "Transcript of your actions this episode (most recent last)"
        if partner_transcript is not None
        else "Team transcript this episode (most recent last)"
    )
    transcript_lines = render_team_transcript(history, max_history)
    if transcript_lines:
        omitted = len(history) - len(transcript_lines)
        if omitted:
            own_header += f" [{omitted} earlier steps omitted]"
        lines.append(own_header + ":")
        lines.extend(transcript_lines)
    else:
        lines.append("No actions executed yet this episode.")
    lines.append("")

    lines.append(
        "Choose the single best next primitive for each agent awaiting a decision. "
        "Respond with the JSON object described above."
    )
    return "\n".join(lines)
