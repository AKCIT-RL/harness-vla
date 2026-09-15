"""``PartnrHarnessAdapter``: the Harness VLA agent behind PARTNR's ``AgentHarness``.

Paper-confirmed pieces: one primitive per turn with closed-loop feedback, the
FULL episode transcript as short-term memory (``max_history=0``), Task
Specific Memory re-grounded on the current state, Global Memory, and the
bootstrap -> frozen deployment split (``phase``). The LLM transport is the
legacy ``PrimitiveHarness`` one so ``num_ctx``/thinking flags/seeding stay
identical to the beta-only control condition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    classify_action_result,
)

from harness_vla.envs.partnr.legacy.primitive_harness import PrimitiveHarness
from harness_vla.envs.partnr.memory import (
    bootstrap_label,
    load_partnr_task_memory,
    partnr_global_memory,
    promote_transcript_to_task_memory,
    resolve_partnr_task_memory,
)
from harness_vla.envs.partnr.primitives import (
    extract_json_object,
    format_action,
    parse_action,
)
from harness_vla.envs.partnr.prompts import (
    build_partnr_system_prompt,
    build_partnr_turn_prompt,
    step_context_to_dict,
    tools_text_from_specs,
)
from harness_vla.memory.global_memory import GlobalMemory
from harness_vla.planner.phase_policy import Phase

_LOG = logging.getLogger(__name__)

_CORRECTION = (
    "Your previous reply could not be parsed. Reply with ONLY one JSON object "
    '{"thought": "...", "action": "Name[args]"} using a primitive from the '
    "library (or Done[]). No prose, no markdown fences."
)
_MESSAGE_CHARS = 300


def _load_global_memory(path: Optional[str]) -> GlobalMemory:
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


class PartnrHarnessAdapter(AgentHarness):
    """Harness VLA agent for PARTNR (single agent per instance)."""

    def __init__(
        self,
        prefix: str = "You are a robot performing a household task. You act through atomic primitives, one per turn.",
        base_url: str = "http://localhost:11435/v1",
        model: str = "gemma3:12b",
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
        **kwargs: Any,
    ) -> None:
        if kwargs:
            _LOG.warning("PartnrHarnessAdapter ignoring unknown kwargs: %s", sorted(kwargs))
        self.prefix = prefix
        self.world_char_budget = int(world_char_budget)
        self.max_history = int(max_history)
        self.max_retries = int(max_retries)
        self.zero_shot = bool(zero_shot)
        self.task_memory_dir = task_memory_dir
        self.global_memory_path = global_memory_path
        self.phase = Phase(phase)
        # Transport identical to the legacy control condition (fair comparison).
        self._llm = PrimitiveHarness(
            prefix=prefix,
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
        self.history: List[Dict[str, Any]] = []
        self.system_prompt: str = ""
        self.global_memory: GlobalMemory = GlobalMemory(success_rules=[], failure_models=[])
        self.task_memory: Optional[Dict[str, Any]] = None
        self.last_prompt: str = ""
        self.last_meta: Dict[str, Any] = {}
        self._tools: List[str] = []
        self._episode: Optional[EpisodeContext] = None

    # ------------------------------------------------------------ lifecycle

    def on_episode_start(self, context: EpisodeContext) -> None:
        self._episode = context
        self._llm.on_episode_start(context)
        self.history = []
        self._tools = [tool.name for tool in context.tools]
        if self.zero_shot:
            self.global_memory = GlobalMemory(success_rules=[], failure_models=[])
            self.task_memory = None
        else:
            self.global_memory = _load_global_memory(self.global_memory_path)
            try:
                self.task_memory = load_partnr_task_memory(self.task_memory_dir, context.instruction)
            except (OSError, ValueError, json.JSONDecodeError):
                _LOG.warning("task memory for %r unreadable; ignoring", context.instruction)
                self.task_memory = None
        memory_text = ""
        if self.global_memory.success_rules or self.global_memory.failure_models:
            memory_text = self.global_memory.render()
        self.system_prompt = build_partnr_system_prompt(
            self.prefix, tools_text_from_specs(context.tools), memory_text
        )

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        if self.phase != Phase.BOOTSTRAP or not self.task_memory_dir or self._episode is None:
            return
        # Privileged channel: the env only fills ``bootstrap_label`` when its
        # ``expose_bootstrap_label`` flag is on; deployment never reads it.
        label = bootstrap_label(summary)
        if label is None or not label.get("task_success"):
            return
        try:
            promote_transcript_to_task_memory(
                self.task_memory_dir,
                self._episode.instruction,
                self.history,
                source={"episode_id": summary.episode_id, "label_source": label.get("source")},
            )
        except (OSError, ValueError):
            _LOG.warning("task memory promotion failed", exc_info=True)

    # --------------------------------------------------------------- decide

    def _vocab(self, state: Optional[Dict[str, Any]], rooms: Optional[List[str]]) -> Dict[str, set]:
        state = state or {}
        return {
            "objects": set(state.get("objects") or {}),
            "furniture": set(state.get("furniture") or {}),
            "rooms": set(state.get("rooms") or []) | set(rooms or []),
        }

    def _record(self, context: StepContext, name: str, args: str, vocab: Dict[str, set],
                error_kind: Optional[str] = None, status: Optional[str] = None) -> None:
        self.history.append({
            "decision_index": context.decision_index,
            "name": name,
            "args": args,
            "action": format_action(name, args),
            "status": status,
            "message": "",
            "ok": name == "Done",
            "error_kind": error_kind,
            "vocab": vocab,
        })

    def decide(self, context: StepContext) -> Decision:
        if self._episode is None:
            self._tools = [tool.name for tool in context.tools]
            self.system_prompt = build_partnr_system_prompt(
                self.prefix, tools_text_from_specs(context.tools), ""
            )
        ctx = step_context_to_dict(context)
        state = (ctx.get("extras") or {}).get("state")
        vocab = self._vocab(state, ctx.get("rooms"))
        resolved = None
        if not self.zero_shot and self.task_memory and isinstance(state, dict):
            resolved = resolve_partnr_task_memory(self.task_memory, state) or None
        user = build_partnr_turn_prompt(
            ctx, self.history, resolved, self.max_history, self.world_char_budget
        )
        self.last_prompt = user
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        tools = set(self._tools) or {tool.name for tool in context.tools}

        attempts = 0
        raw = ""
        meta: Dict[str, Any] = {}
        parsed = None
        while attempts <= self.max_retries:
            attempts += 1
            raw = self._llm._post(messages)
            meta = dict(self._llm.last_meta)
            if meta.get("llm_failure"):
                self.last_meta = meta
                return self._llm._llm_failure(raw, meta)
            parsed = parse_action(raw)
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
            "phase": self.phase.value,
        })
        self.last_meta = meta
        rationale = self._thought(raw)

        if parsed is None:
            # Parse failure after retries: idle so the simulator advances.
            wait = next((name for name in tools if name.lower().startswith("wait")), "Wait")
            self._record(context, wait, "", vocab, error_kind="parse_error")
            return ToolCall(wait, "", rationale=rationale, raw_output=raw,
                            error_kind="parse_error", meta=meta)

        name, args_list = parsed
        if name == "Done":
            self._record(context, "Done", "", vocab, status="done")
            return DoneDecision(rationale=rationale, raw_output=raw, meta=meta)
        if name == "Place":
            args_list = (args_list + ["None"] * 5)[:5]
            args_list = ["None" if not item or item.lower() == "none" else item for item in args_list]
        args = ",".join(args_list)
        if name not in tools:
            self._record(context, name, args, vocab, error_kind="compile_error")
            return ToolCall(
                "InvalidResponse",
                f"Action {name!r} is not one of the available tools.",
                rationale=rationale, raw_output=raw, error_kind="compile_error", meta=meta,
            )
        self._record(context, name, args, vocab)
        return ToolCall(name, args, rationale=rationale, raw_output=raw, meta=meta)

    @staticmethod
    def _thought(raw: str) -> str:
        parsed = extract_json_object(raw or "")
        if isinstance(parsed, dict) and isinstance(parsed.get("thought"), str):
            return parsed["thought"].strip()[:200]
        return (raw or "").strip()[:200]

    # --------------------------------------------------------------- result

    def on_action_result(self, result: ActionResult) -> None:
        pending = next(
            (record for record in reversed(self.history) if record["status"] is None), None
        )
        if pending is None:
            return
        pending["status"] = "success" if result.success_hint else "failure"
        pending["ok"] = bool(result.success_hint)
        pending["message"] = " ".join(str(result.response or "").split())[:_MESSAGE_CHARS]
        pending["failure_kind"] = result.failure_kind or classify_action_result(result.response or "")
