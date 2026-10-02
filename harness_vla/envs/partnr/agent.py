"""``HarnessVlaCentralized``: the Harness VLA controller for PARTNR.

One LLM brain, one JSON decision per turn keyed by agent label, the FULL
team transcript as short-term memory (``max_history=0``), Task Specific
Memory re-grounded on the current state, Global Memory, and the bootstrap ->
frozen deployment split (``phase``). Plugged behind the env's
``CentralizedHarnessPlanner`` it commands every agent; with a single
controlled label (see ``decentralized.HarnessVlaDecentralized``) it is the
per-agent brain behind ``HarnessBridgePlanner``.

Persistent subgoal *claims* (optional, ``use_claims``): the task is
decomposed once into per-agent claims; a per-agent ``Done[]`` latches the
claim and the env terminates deterministically once every claim is done.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from partnr_harness.contracts import (
    ActionResult,
    AgentHarness,
    Decision,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    StepContext,
    ToolCall,
    classify_action_result,
)

from harness_vla.envs.partnr.llm_client import LLMClient
from harness_vla.envs.partnr.memory import (
    TASK_MEMORY_SCHEMA_VERSION,
    bootstrap_label,
    load_partnr_task_memory,
    normalize_task_key,
    partnr_global_memory,
    resolve_partnr_task_memory,
    write_partnr_task_memory,
)
from harness_vla.envs.partnr.primitives import (
    active_primitive_names,
    extract_json_object,
    use_vocabulary,
    format_action,
    parse_action,
    symbolize_partnr_action,
)
from harness_vla.envs.partnr.prompts import (
    build_team_system_prompt,
    build_team_turn_prompt,
    step_context_to_dict,
)
from harness_vla.memory.global_memory import GlobalMemory
from harness_vla.planner.phase_policy import Phase

_LOG = logging.getLogger(__name__)

DEFAULT_ROLE = (
    "You are the central controller for a household task. You command TWO "
    "agents: agent_0 is a Spot robot and agent_1 is a human. You act through "
    "atomic primitives, one action per agent per turn."
)
DEFAULT_AGENT_DESCRIPTIONS = {"agent_0": "Spot robot", "agent_1": "human"}

_AGENT_LABEL = re.compile(r"^agent_\d+$")
_LABELED_COMMAND = re.compile(r"^(?P<label>agent_\d+)\s*:\s*(?P<command>.+)$")
_MESSAGE_CHARS = 300
_NULL_VALUES = {"", "null", "none", "idle", "wait"}

_CORRECTION = (
    "Your previous reply could not be parsed. Reply with ONLY one JSON object "
    'of the form {"thought": "...", "<agent_label>": "Name[args]"} containing a '
    "key for each agent awaiting a decision (value: one primitive call or "
    "Done[] or null). No prose, no markdown fences."
)
_DECOMPOSE_SCHEMA = (
    '{"subgoals":[{"id":"sg1","description":"...","assigned_to":"agent_0"},'
    '{"id":"sg2","description":"...","assigned_to":"agent_1"}]}'
)


def agent_label(uid: Any) -> str:
    return f"agent_{uid}"


def label_uid(label: str) -> Optional[int]:
    try:
        return int(str(label).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def load_global_memory(path: Optional[str]) -> GlobalMemory:
    """Ledger (schema 3 ``entries``) or rendered ``{success_rules, failure_models}``."""
    if not path or not Path(path).is_file():
        return GlobalMemory(success_rules=[], failure_models=[])
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _LOG.warning("global memory at %s unreadable; using empty memory", path)
        return GlobalMemory(success_rules=[], failure_models=[])
    if isinstance(data, dict) and "entries" in data:
        return partnr_global_memory(path)
    return GlobalMemory(
        success_rules=list((data or {}).get("success_rules", [])),
        failure_models=list((data or {}).get("failure_models", [])),
    )


# --------------------------------------------------------- labeled memory

def labeled_skeleton(history: Sequence[Mapping[str, Any]]) -> List[str]:
    """``agent_0: Pick[<obj:apple>]`` for each successful primitive, in order."""
    commands: List[str] = []
    for step in history:
        if not step.get("ok") or step.get("error_kind"):
            continue
        name = step.get("name")
        if name not in active_primitive_names() or name in ("Wait", "Done"):
            continue
        vocab = step.get("vocab") or {}
        symbolic = symbolize_partnr_action(
            step.get("action") or format_action(name, step.get("args", "")),
            objects=vocab.get("objects", ()),
            furniture=vocab.get("furniture", ()),
            rooms=vocab.get("rooms", ()),
        )
        label = step.get("agent")
        commands.append(f"{label}: {symbolic}" if label else symbolic)
    return commands


def split_labeled_command(command: str) -> Tuple[Optional[str], str]:
    match = _LABELED_COMMAND.match(command.strip())
    if match is None:
        return None, command.strip()
    return match.group("label"), match.group("command").strip()


def resolve_labeled_task_memory(entry: Optional[Mapping], state: Mapping) -> List[str]:
    """Re-ground a labeled skeleton on ``state`` (labels pass through)."""
    if not entry:
        return []
    resolved: List[str] = []
    for command in entry.get("commands", []):
        label, bare = split_labeled_command(str(command))
        bound = resolve_partnr_task_memory({"commands": [bare]}, state)
        if not bound:
            continue
        resolved.append(f"{label}: {bound[0]}" if label else bound[0])
    return resolved


def promote_labeled_transcript(
    task_memory_dir: str,
    instruction: str,
    history: Sequence[Mapping[str, Any]],
    source: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Merge one successful labeled transcript into the task's memory file
    (shortest skeleton wins, ``support`` incremented)."""
    commands = labeled_skeleton(history)
    if not commands:
        return None
    for command in commands:
        _label, bare = split_labeled_command(command)
        parsed = parse_action(bare)
        if parsed is None:
            return None
    task_key = normalize_task_key(instruction)
    existing = load_partnr_task_memory(task_memory_dir, instruction)
    source = dict(source or {})
    source.setdefault("num_decisions", len(history))
    if existing is None:
        entry = {
            "schema_version": TASK_MEMORY_SCHEMA_VERSION,
            "classification": "paper-compatible",
            "environment": "partnr",
            "task_key": task_key,
            "instruction": instruction,
            "commands": commands,
            "support": 1,
            "sources": [source],
        }
    else:
        entry = dict(existing)
        if len(commands) < len(existing.get("commands", [])):
            entry["commands"] = commands
        entry["support"] = int(existing.get("support", 0)) + 1
        entry["sources"] = list(existing.get("sources", [])) + [source]
    return write_partnr_task_memory({task_key: entry}, task_memory_dir)[0]


# ------------------------------------------------------------------ agent

class HarnessVlaCentralized(AgentHarness):
    """Harness VLA controller: one JSON decision per turn, keyed by agent label."""

    def __init__(
        self,
        role: str = DEFAULT_ROLE,
        base_url: str = "http://localhost:11435/v1",
        model: str = "gemma4:12b",
        temperature: float = 0.0,
        repeat_penalty: float = 1.0,
        repeat_last_n: int = 64,
        thinking_budget: int = 0,
        max_tokens: int = 512,
        timeout_s: float = 240.0,
        world_char_budget: int = 12000,
        enable_thinking: bool = False,
        disable_thinking: bool = False,
        num_ctx: Optional[int] = None,
        max_retries: int = 3,
        max_history: int = 0,
        zero_shot: bool = True,
        task_memory_dir: Optional[str] = None,
        global_memory_path: Optional[str] = None,
        phase: str = Phase.DEPLOYMENT.value,
        use_claims: bool = True,
        agent_descriptions: Optional[Mapping[str, str]] = None,
        profile: Any = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            _LOG.warning("%s ignoring unknown kwargs: %s", type(self).__name__, sorted(kwargs))
        # ``profile`` (duck-typed, all attributes optional) adapts the prompts to
        # another environment: ``role``, ``role_suffix``, ``primitive_reference``,
        # ``guidance``, ``agent_descriptions``, ``vocabulary`` (primitives.Vocabulary).
        self.profile = profile
        if profile is not None:
            role = getattr(profile, "role", None) or role
            agent_descriptions = getattr(profile, "agent_descriptions", None) or agent_descriptions
            vocabulary = getattr(profile, "vocabulary", None)
            if vocabulary is not None:
                use_vocabulary(vocabulary)
        self.role = role
        self.world_char_budget = int(world_char_budget)
        self.max_history = int(max_history)
        self.max_retries = int(max_retries)
        self.zero_shot = bool(zero_shot)
        self.task_memory_dir = task_memory_dir
        self.global_memory_path = global_memory_path
        self.phase = Phase(phase)
        self.use_claims = bool(use_claims)
        self.agent_descriptions = dict(agent_descriptions or DEFAULT_AGENT_DESCRIPTIONS)
        self._llm = LLMClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            enable_thinking=enable_thinking,
            disable_thinking=disable_thinking,
            num_ctx=num_ctx,
            max_retries=max_retries,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            thinking_budget=thinking_budget,
        )
        self.history: List[Dict[str, Any]] = []
        self.system_prompt: str = ""
        self.global_memory = GlobalMemory(success_rules=[], failure_models=[])
        self.task_memory: Optional[Dict[str, Any]] = None
        self.last_prompt: str = ""
        self.last_meta: Dict[str, Any] = {}
        self.decompose_meta: Optional[Dict[str, Any]] = None
        self._claims: Dict[str, Dict[str, Any]] = {}
        self._episode: Optional[EpisodeContext] = None
        self._labels: List[str] = []
        self._uid_by_label: Dict[str, int] = {}
        self._tools_by_label: Dict[str, List[Dict[str, Any]]] = {}
        self.controlled: List[str] = []

    # ------------------------------------------------------------ overrides
    # Hooks specialised by the decentralized agent.

    def _own_task_memory_dir(self) -> Optional[str]:
        return self.task_memory_dir

    def _own_global_memory_path(self) -> Optional[str]:
        return self.global_memory_path

    def _controlled_labels(self, context: EpisodeContext) -> List[str]:
        return list(self._labels)

    def _partner_task_memory(self, state: Optional[Mapping[str, Any]]) -> Optional[List[str]]:
        return None

    def _partner_transcript(self) -> Optional[List[Dict[str, Any]]]:
        return None

    def _publish(self, record: Mapping[str, Any]) -> None:
        """Called after a record is created or completed (no-op here)."""

    @property
    def allow_all(self) -> bool:
        return len(self.controlled) > 1

    # ------------------------------------------------------------ lifecycle

    def on_episode_start(self, context: EpisodeContext) -> None:
        self._episode = context
        self._llm.start_episode(context.agent_uid)
        self.history = []
        self._claims = {}
        self.decompose_meta = None
        agents = (context.extras or {}).get("agents") or {
            agent_label(context.agent_uid): {
                "uid": context.agent_uid,
                "tools": [
                    {"name": tool.name, "description": getattr(tool, "description", "")}
                    for tool in context.tools
                ],
            }
        }
        self._labels = sorted(agents)
        self._uid_by_label = {label: int(agents[label].get("uid", label_uid(label) or 0)) for label in self._labels}
        self._tools_by_label = {label: list(agents[label].get("tools") or []) for label in self._labels}
        self.controlled = self._controlled_labels(context)

        if self.zero_shot:
            self.global_memory = GlobalMemory(success_rules=[], failure_models=[])
            self.task_memory = None
        else:
            self.global_memory = load_global_memory(self._own_global_memory_path())
            try:
                self.task_memory = load_partnr_task_memory(self._own_task_memory_dir(), context.instruction)
            except (OSError, ValueError, json.JSONDecodeError):
                _LOG.warning("task memory for %r unreadable; ignoring", context.instruction)
                self.task_memory = None
        memory_text = ""
        if self.global_memory.success_rules or self.global_memory.failure_models:
            memory_text = self.global_memory.render()
        self.system_prompt = build_team_system_prompt(
            self.role,
            {label: self._tools_by_label.get(label, []) for label in self.controlled},
            memory_text,
            self.controlled,
            self.allow_all,
            role_suffix=getattr(self.profile, "role_suffix", None),
            primitive_reference=getattr(self.profile, "primitive_reference", None),
            guidance=getattr(self.profile, "guidance", None),
        )
        if self.use_claims and self.allow_all:
            self._claims = self._decompose(context.instruction, self.controlled)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        task_memory_dir = self._own_task_memory_dir()
        if self.phase != Phase.BOOTSTRAP or not task_memory_dir or self._episode is None:
            return
        # Privileged channel: the env only fills ``bootstrap_label`` when its
        # ``expose_bootstrap_label`` flag is on; deployment never reads it.
        label = bootstrap_label(summary)
        if label is None or not label.get("task_success"):
            return
        try:
            promote_labeled_transcript(
                task_memory_dir,
                self._episode.instruction,
                [record for record in self.history if record.get("agent") in self.controlled],
                source={"episode_id": summary.episode_id, "label_source": label.get("source")},
            )
        except (OSError, ValueError):
            _LOG.warning("task memory promotion failed", exc_info=True)

    # --------------------------------------------------------------- claims

    def all_claims_done(self) -> bool:
        return bool(self.use_claims and self._claims) and all(
            claim["done"] for claim in self._claims.values()
        )

    def claims(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._claims)

    def _decompose(self, instruction: str, labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """One LLM call (retried once) splitting the task into per-agent claims."""
        roles = ", ".join(
            f"{label} ({self.agent_descriptions[label]})" if label in self.agent_descriptions else label
            for label in labels
        )
        prompt = (
            f"Task: {instruction}\n\n"
            f"Agents: {roles}.\n"
            "Break this task into subgoals and assign each to an agent so both "
            "agents are useful. Return exactly one JSON object with schema "
            f"{_DECOMPOSE_SCHEMA} and nothing else."
        )
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": prompt},
        ]
        parsed: Dict[str, Any] = {}
        raw = ""
        meta: Dict[str, Any] = {}
        attempts = 0
        while attempts < 2:
            attempts += 1
            raw = self._llm.post(messages)
            meta = dict(self._llm.last_meta)
            if meta.get("llm_failure"):
                continue
            candidate = extract_json_object(raw)
            if isinstance(candidate, dict) and candidate:
                parsed = candidate
                break
        self.decompose_meta = {"attempts": attempts, "parsed": bool(parsed), "raw": raw[:2000], "meta": meta}
        claims: Dict[str, Dict[str, Any]] = {}
        subgoals = parsed.get("subgoals", [])
        if not isinstance(subgoals, list):
            subgoals = []
        for subgoal in subgoals:
            if not isinstance(subgoal, dict):
                continue
            assigned = subgoal.get("assigned_to")
            if assigned not in labels:
                continue
            description = str(subgoal.get("description", ""))
            if assigned in claims:
                claims[assigned]["description"] += "; " + description
            else:
                claims[assigned] = {"id": subgoal.get("id"), "description": description, "done": False}
        for label in labels:
            claims.setdefault(label, {"id": f"{label}_standby", "description": "assist / standby", "done": False})
        return claims

    # ------------------------------------------------------------------ view

    def _view(self, context: StepContext) -> Dict[str, Any]:
        ctx = step_context_to_dict(context)
        extras = dict(ctx.get("extras") or {})
        own = agent_label(ctx.get("agent_uid"))
        deciding = [label for label in (extras.get("deciding_agents") or [own]) if label in self.controlled]
        in_flight = dict(extras.get("in_flight_actions") or {})
        busy = {label: in_flight.get(label) for label in (extras.get("busy_agents") or in_flight)}
        return {
            "instruction": ctx.get("instruction"),
            "decision_index": ctx.get("decision_index"),
            "max_decisions": ctx.get("max_decisions"),
            "state": extras.get("state"),
            "rendered_state": extras.get("rendered_state"),
            "state_title": extras.get("state_title"),
            "world_description": ctx.get("world_description"),
            "rooms": ctx.get("rooms"),
            "agent_names": self._agent_names(),
            "deciding": deciding,
            "busy": busy,
            "signals": ctx.get("signals") or [],
        }

    def _agent_names(self) -> Dict[str, str]:
        names = {}
        for label, uid in self._uid_by_label.items():
            description = self.agent_descriptions.get(label)
            names[str(uid)] = f"{label}, {description}" if description else label
        return names

    @staticmethod
    def _vocab(state: Optional[Mapping[str, Any]], rooms: Optional[Sequence[str]]) -> Dict[str, set]:
        state = state or {}
        return {
            "objects": set(state.get("objects") or {}),
            "furniture": set(state.get("furniture") or {}),
            "rooms": set(state.get("rooms") or []) | set(rooms or []),
        }

    def _tool_names(self, label: str) -> set:
        return {str(tool.get("name")) for tool in self._tools_by_label.get(label, []) if tool.get("name")}

    def _wait_tool(self, label: str) -> str:
        return next((name for name in self._tool_names(label) if name.lower().startswith("wait")), "Wait")

    # --------------------------------------------------------------- record

    def _record(
        self,
        label: str,
        decision_index: Any,
        name: str,
        args: str,
        vocab: Dict[str, set],
        error_kind: Optional[str] = None,
        status: Optional[str] = None,
        message: str = "",
    ) -> Dict[str, Any]:
        record = {
            "decision_index": decision_index,
            "agent": label,
            "name": name,
            "args": args,
            "action": format_action(name, args),
            "status": status,
            "message": message,
            "ok": name == "Done",
            "error_kind": error_kind,
            "vocab": vocab,
        }
        self.history.append(record)
        self._publish(record)
        return record

    # --------------------------------------------------------------- decide

    def decide(self, context: StepContext) -> Dict[str, Optional[Decision]]:
        return self._decide_team(self._view(context))

    def _decide_team(self, view: Mapping[str, Any]) -> Dict[str, Optional[Decision]]:
        deciding: List[str] = list(view.get("deciding") or [])
        state = view.get("state")
        vocab = self._vocab(state if isinstance(state, Mapping) else None, view.get("rooms"))
        resolved: Optional[List[str]] = None
        if not self.zero_shot and self.task_memory and isinstance(state, Mapping):
            resolved = resolve_labeled_task_memory(self.task_memory, state) or None
        partner_memory = self._partner_task_memory(state if isinstance(state, Mapping) else None)
        partner_transcript = self._partner_transcript()
        user = build_team_turn_prompt(
            view,
            self.history,
            resolved_task_memory=resolved,
            partner_task_memory=partner_memory or None,
            partner_transcript=partner_transcript,
            claims=self._claims if self.use_claims and self._claims else None,
            max_history=self.max_history,
            state_char_budget=self.world_char_budget,
        )
        self.last_prompt = user
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

        attempts = 0
        raw = ""
        meta: Dict[str, Any] = {}
        parsed: Optional[Dict[str, Any]] = None
        while attempts <= self.max_retries:
            attempts += 1
            raw = self._llm.post(messages)
            meta = dict(self._llm.last_meta)
            if meta.get("llm_failure"):
                self.last_meta = meta
                message = f"LLM unavailable: {meta.get('llm_error', 'unknown')}"
                return {
                    label: ToolCall("InvalidResponse", message, rationale=message[:200],
                                    raw_output=raw, error_kind="llm_failure", meta=meta)
                    for label in deciding
                }
            parsed = self._parse_team(raw, deciding)
            if parsed is not None:
                break
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _CORRECTION},
            ]
        meta.update({
            "parse_attempts": attempts,
            "history_len": len(self.history),
            "prompt_chars": len(self.system_prompt) + len(user),
            "task_memory_used": bool(resolved),
            "partner_task_memory_used": bool(partner_memory),
            "partner_transcript_len": len(partner_transcript) if partner_transcript is not None else None,
            "phase": self.phase.value,
            "deciding": list(deciding),
        })
        self.last_meta = meta
        rationale = self._thought(raw)
        decision_index = view.get("decision_index")

        decisions: Dict[str, Optional[Decision]] = {}
        if parsed is None:
            # Parse failure after retries: idle the deciding agents so the sim advances.
            for label in deciding:
                wait = self._wait_tool(label)
                self._record(label, decision_index, wait, "", vocab, error_kind="parse_error")
                decisions[label] = ToolCall(wait, "", rationale=rationale, raw_output=raw,
                                            error_kind="parse_error", meta=meta)
            return decisions

        for label in deciding:
            value = parsed.get(label)
            if value is None:
                decisions[label] = None
                continue
            name, args_list = value
            if name == "Done":
                if self.use_claims and label in self._claims:
                    self._claims[label]["done"] = True
                self._record(label, decision_index, "Done", "", vocab, status="done",
                             message="declared its part complete")
                decisions[label] = DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
                continue
            if name == "Place":
                args_list = (args_list + ["None"] * 5)[:5]
                args_list = ["None" if not item or item.lower() == "none" else item for item in args_list]
            args = ",".join(args_list)
            tools = self._tool_names(label)
            if tools and name not in tools:
                self._record(label, decision_index, name, args, vocab, error_kind="compile_error")
                decisions[label] = ToolCall(
                    "InvalidResponse", f"Action {name!r} is not one of the tools of {label}.",
                    rationale=rationale, raw_output=raw, error_kind="compile_error", meta=meta,
                )
                continue
            self._record(label, decision_index, name, args, vocab)
            decisions[label] = ToolCall(name, args, rationale=rationale, raw_output=raw, meta=meta)

        if parsed.get("all") is not None and self.allow_all:
            decisions["all"] = DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
        return decisions

    def _parse_team(self, raw: str, deciding: Sequence[str]) -> Optional[Dict[str, Any]]:
        """``{label: (name, args) | None, "all": True | None}`` or ``None`` if unusable."""
        obj = extract_json_object(raw or "")
        if not isinstance(obj, dict):
            return None
        result: Dict[str, Any] = {}
        found = False
        for label in deciding:
            if label not in obj:
                continue
            found = True
            value = self._parse_value(obj[label])
            if value is False:
                return None
            result[label] = value
        if not found and len(deciding) == 1 and "action" in obj:
            # Single-agent shape {"thought", "action": "Name[args]"}.
            value = self._parse_value({"action": obj.get("action"), "args": obj.get("args")})
            if value is False:
                return None
            result[deciding[0]] = value
            found = True
        all_value = obj.get("all")
        if all_value is not None:
            parsed_all = self._parse_value(all_value)
            if parsed_all is not None and parsed_all is not False and parsed_all[0] == "Done":
                result["all"] = True
                found = True
        if not found:
            return None
        return result

    @staticmethod
    def _parse_value(value: Any) -> Any:
        """``None`` (idle), ``(name, args)`` or ``False`` when malformed."""
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip().lower() in _NULL_VALUES:
                return None
            parsed = parse_action(value)
            return parsed if parsed is not None else False
        if isinstance(value, Mapping):
            action = value.get("action", value.get("tool"))
            if action is None:
                return None
            parsed = parse_action(json.dumps({"action": action, "args": value.get("args")}))
            return parsed if parsed is not None else False
        return False

    @staticmethod
    def _thought(raw: str) -> str:
        parsed = extract_json_object(raw or "")
        if isinstance(parsed, dict) and isinstance(parsed.get("thought"), str):
            return parsed["thought"].strip()[:200]
        return (raw or "").strip()[:200]

    # --------------------------------------------------------------- result

    def on_action_result(self, result: ActionResult) -> None:
        label = agent_label(result.agent_uid) if getattr(result, "agent_uid", None) is not None else None
        pending = [record for record in self.history if record["status"] is None]
        if label is not None:
            pending = [record for record in pending if record.get("agent") == label]
        elif len(pending) > 1:
            same = [r for r in pending if r["name"] == result.tool and r["args"] == (result.args or "")]
            pending = same or pending
        if not pending:
            return
        record = pending[-1]
        record["status"] = "success" if result.success_hint else "failure"
        record["ok"] = bool(result.success_hint)
        record["message"] = " ".join(str(result.response or "").split())[:_MESSAGE_CHARS]
        record["failure_kind"] = result.failure_kind or classify_action_result(result.response or "")
        self._publish(record)
