"""PARTNR Task Memory and Global Memory built from environment JSONL traces.

Trace schema (written by ``partnr_harness.logging_sink.JsonlDecisionSink`` via
the bridge planner): one JSON record per line with ``type`` in
``episode_start`` (``context``: EpisodeContext), ``decision`` (``context``:
StepContext incl. ``extras.state``; ``decision``: ToolCall/DoneDecision),
``result`` (``result``: ActionResult with ``success_hint``), ``llm_failure`` and
``episode_end`` (``summary``: EpisodeSummary). The official
``task_state_success`` is not part of the harness JSONL by default; the env
exposes it only as ``summary.extras.bootstrap_label`` (``{"task_success",
"source": "official_stats"}``) when ``plan_config.expose_bootstrap_label`` is
on (bootstrap phase). Old logs with ``task_state_success`` on the
``episode_end`` record or in ``extras`` are still understood, as are explicit
per-episode overrides supplied by the caller.

Classification:

* Task Memory (``build_partnr_task_memory`` / ``resolve_partnr_task_memory``):
  **paper-confirmed** mechanism (symbolic successful command skeleton per task,
  only from successful episodes, re-grounded at deployment); the PARTNR
  symbolization itself is **paper-compatible**.
* Global Memory (``partnr_global_memory``, ``extract_global_memory_candidates``,
  ``promote_candidates``): the memory and the bootstrap->deployment freeze are
  **paper-confirmed**; the LLM-based extraction procedure is
  **paper-compatible** (the paper describes Global Memory being distilled from
  experience but does not specify the extraction procedure).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness_vla.envs.partnr.primitives import (
    PARTNR_PRIMITIVE_NAMES,
    extract_json_object,
    format_action,
    instance_category,
    parse_action,
    parse_symbol,
    symbolize_partnr_action,
)
from harness_vla.memory.global_memory import (
    GlobalMemory,
    GlobalMemoryEvidence,
    GlobalMemoryLedger,
    validate_trace_hash_provenance,
)
from harness_vla.trace_io import write_json_atomic

_LOG = logging.getLogger(__name__)

TASK_MEMORY_SCHEMA_VERSION = 1
# Fields of a StepContext/decision record that would leak privileged or
# instance-specific information into memory.
FORBIDDEN_TASK_MEMORY_KEYS = frozenset({
    "world_description", "state", "text", "raw_output", "meta", "extras",
})

# PARTNR has no manual seed: every rule must come from bootstrap evidence.
PARTNR_SEED_SUCCESS_RULES: List[str] = []
PARTNR_SEED_FAILURE_MODELS: List[str] = []

ChatFn = Callable[[str, str], str]


# ------------------------------------------------------------------ traces

def normalize_task_key(instruction: str) -> str:
    """Instruction text -> stable task key (lowercase, collapsed spaces, no final period)."""
    text = " ".join(str(instruction).strip().lower().split())
    return text.rstrip(".").strip()


def task_key_filename(task_key: str) -> str:
    """Filesystem-safe file name for one task key (slug + short hash)."""
    slug = re.sub(r"[^a-z0-9]+", "_", task_key).strip("_")[:60] or "task"
    digest = hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}.json"


def load_episode_trace(path: str) -> Dict[str, Any]:
    """Read one env JSONL file into an episode dict (incomplete last line dropped)."""
    payload = Path(path).read_bytes()
    lines = payload.decode("utf-8").split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]  # incomplete trailing record
    records = [json.loads(line) for line in lines if line.strip()]
    return {
        "path": str(path),
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "records": records,
    }


def _as_episode(trace: Any) -> Dict[str, Any]:
    if isinstance(trace, dict) and "records" in trace:
        return trace
    if isinstance(trace, list):
        return {"path": "", "trace_sha256": "", "records": trace}
    raise TypeError("trace must be a list of records or a dict with 'records'")


def episode_instruction(records: Sequence[Mapping]) -> str:
    for record in records:
        context = record.get("context") or {}
        if record.get("type") in ("episode_start", "decision") and context.get("instruction"):
            return str(context["instruction"])
    return ""


def episode_id_of(records: Sequence[Mapping]) -> str:
    for record in records:
        if record.get("type") == "episode_start":
            return str((record.get("context") or {}).get("episode_id", ""))
    for record in records:
        if record.get("type") == "episode_end":
            return str((record.get("summary") or {}).get("episode_id", ""))
    return ""


def bootstrap_label(summary: Any) -> Optional[Dict[str, Any]]:
    """``{"task_success": bool, "source": str}`` from an EpisodeSummary (or its
    dict form), or ``None`` when the env did not expose the label."""
    extras = getattr(summary, "extras", None)
    if extras is None and isinstance(summary, Mapping):
        extras = summary.get("extras")
    label = (extras or {}).get("bootstrap_label")
    if not isinstance(label, Mapping) or "task_success" not in label:
        return None
    return {"task_success": bool(label["task_success"]), "source": str(label.get("source", ""))}


def episode_success(episode: Mapping) -> Optional[bool]:
    """Official success of an episode, or ``None`` when unknown.

    Reads ``summary.extras.bootstrap_label`` (current env channel); the
    ``task_state_success`` forms are kept for old logs and caller overrides.
    """
    if episode.get("task_state_success") is not None:
        return float(episode["task_state_success"]) == 1.0
    for record in episode.get("records", []):
        if record.get("type") != "episode_end":
            continue
        if record.get("task_state_success") is not None:
            return float(record["task_state_success"]) == 1.0
        label = bootstrap_label(record.get("summary") or {})
        if label is not None:
            return label["task_success"]
        extras = (record.get("summary") or {}).get("extras") or {}
        if extras.get("task_state_success") is not None:
            return float(extras["task_state_success"]) == 1.0
    return None


def _vocabulary(state: Mapping) -> Tuple[set, set, set]:
    return (
        set((state or {}).get("objects") or {}),
        set((state or {}).get("furniture") or {}),
        set((state or {}).get("rooms") or []),
    )


def episode_transcript(records: Sequence[Mapping]) -> List[Dict[str, Any]]:
    """Pair each decision with its result: ``[{action, name, args, status, ok, vocab}]``."""
    transcript: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    for record in records:
        kind = record.get("type")
        if kind == "decision":
            decision = record.get("decision") or {}
            context = record.get("context") or {}
            state = (context.get("extras") or {}).get("state") or {}
            objects, furniture, rooms = _vocabulary(state)
            rooms |= set(context.get("rooms") or [])
            name = decision.get("name")
            if name is None:  # DoneDecision
                name, args = "Done", ""
            else:
                args = decision.get("args") or ""
            pending = {
                "decision_index": context.get("decision_index"),
                "name": name,
                "args": args,
                "action": format_action(name, args),
                "error_kind": decision.get("error_kind"),
                "status": None,
                "ok": name == "Done",
                "vocab": {"objects": objects, "furniture": furniture, "rooms": rooms},
            }
            transcript.append(pending)
        elif kind == "result" and pending is not None:
            result = record.get("result") or {}
            pending["status"] = "success" if result.get("success_hint") else "failure"
            pending["ok"] = bool(result.get("success_hint"))
            pending["failure_kind"] = result.get("failure_kind")
            pending = None
    return transcript


def symbolic_steps(records: Sequence[Mapping]) -> List[Dict[str, Any]]:
    """Normalized, instance-free steps for ``memory.global_memory_extractor``."""
    steps: List[Dict[str, Any]] = []
    for index, step in enumerate(episode_transcript(records)):
        vocab = step.get("vocab") or {}
        parsed = parse_action(step["action"])
        instances = {
            arg for arg in (parsed[1] if parsed else [])
            if instance_category(arg) is not None
        }
        decision_index = step.get("decision_index")
        error = step.get("failure_kind") or step.get("error_kind")
        ok = step.get("ok")
        if step.get("error_kind"):
            ok = False
        elif step.get("status") is None and step["name"] != "Done":
            ok = None
        steps.append({
            "turn": int(decision_index) + 1 if isinstance(decision_index, int) else index + 1,
            "action": symbolize_partnr_action(
                step["action"],
                objects=vocab.get("objects", ()),
                furniture=vocab.get("furniture", ()),
                rooms=vocab.get("rooms", ()),
            ),
            "primitive": str(step["name"]),
            "ok": ok,
            "error": None if ok else error,
            "instances": instances,
        })
    return steps


# ------------------------------------------------------------- task memory

def skeleton_from_transcript(transcript: Sequence[Mapping]) -> List[str]:
    """Symbolic skeleton: successful primitive calls only, no instance ids."""
    commands: List[str] = []
    for step in transcript:
        if not step.get("ok") or step.get("error_kind"):
            continue
        if step["name"] not in PARTNR_PRIMITIVE_NAMES or step["name"] == "Wait":
            continue
        vocab = step.get("vocab") or {}
        commands.append(
            symbolize_partnr_action(
                step["action"],
                objects=vocab.get("objects", ()),
                furniture=vocab.get("furniture", ()),
                rooms=vocab.get("rooms", ()),
            )
        )
    return commands


def _contains_instance_id(commands: Iterable[str]) -> bool:
    for command in commands:
        parsed = parse_action(command)
        if parsed is None:
            continue
        for arg in parsed[1]:
            if parse_symbol(arg) is None and instance_category(arg) is not None:
                return True
    return False


def build_partnr_task_memory(
    traces: Sequence[Any],
    *,
    success_overrides: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Task key -> symbolic Task Memory entry, from successful episodes only.

    ``traces`` is a list of episodes (each a list of JSONL records or a dict
    from ``load_episode_trace``). ``success_overrides`` maps episode id or
    trace path to the official ``task_state_success`` when the JSONL lacks it.
    When several successful episodes share a task key the shortest skeleton
    is kept and ``support`` counts the contributing episodes.
    """
    success_overrides = dict(success_overrides or {})
    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        episode = dict(_as_episode(trace))
        records = episode["records"]
        episode_id = episode_id_of(records)
        for key in (episode_id, episode.get("path", "")):
            if key and key in success_overrides:
                episode["task_state_success"] = float(bool(success_overrides[key]))
        if episode_success(episode) is not True:
            continue
        instruction = episode_instruction(records)
        if not instruction:
            continue
        transcript = episode_transcript(records)
        commands = skeleton_from_transcript(transcript)
        if not commands or _contains_instance_id(commands):
            continue
        candidates[normalize_task_key(instruction)].append({
            "instruction": instruction,
            "commands": commands,
            "source": {
                "episode_id": episode_id,
                "trace_path": episode.get("path", ""),
                "trace_sha256": episode.get("trace_sha256", ""),
                "num_decisions": len(transcript),
            },
        })

    memory: Dict[str, Dict[str, Any]] = {}
    for task_key, entries in candidates.items():
        entries.sort(key=lambda item: (len(item["commands"]), item["source"]["episode_id"]))
        best = entries[0]
        memory[task_key] = {
            "schema_version": TASK_MEMORY_SCHEMA_VERSION,
            "classification": "paper-compatible",
            "environment": "partnr",
            "task_key": task_key,
            "instruction": best["instruction"],
            "commands": list(best["commands"]),
            "support": len(entries),
            "sources": [item["source"] for item in entries],
        }
    return memory


def validate_partnr_task_memory(entry: Mapping) -> None:
    if entry.get("schema_version") != TASK_MEMORY_SCHEMA_VERSION:
        raise ValueError("unsupported PARTNR task memory schema")
    if entry.get("classification") != "paper-compatible":
        raise ValueError("invalid memory classification")
    if set(entry) & FORBIDDEN_TASK_MEMORY_KEYS:
        raise ValueError("task memory contains forbidden keys")
    commands = entry.get("commands")
    if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
        raise ValueError("task memory commands must be a list of strings")
    if _contains_instance_id(commands):
        raise ValueError("task memory contains instance ids")


def write_partnr_task_memory(memory: Mapping[str, Mapping], task_memory_dir: str) -> List[str]:
    """Write one JSON file per task key; returns the written paths."""
    out_dir = Path(task_memory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task_key, entry in sorted(memory.items()):
        validate_partnr_task_memory(entry)
        path = out_dir / task_key_filename(task_key)
        write_json_atomic(path, dict(entry))
        written.append(str(path))
    return written


def load_partnr_task_memory(task_memory_dir: Optional[str], instruction: str) -> Optional[Dict]:
    """Load the validated entry for ``instruction`` or ``None``."""
    if not task_memory_dir:
        return None
    path = Path(task_memory_dir) / task_key_filename(normalize_task_key(instruction))
    if not path.is_file():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    validate_partnr_task_memory(entry)
    return entry


def promote_transcript_to_task_memory(
    task_memory_dir: str,
    instruction: str,
    transcript: Sequence[Mapping],
    source: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Merge one successful in-memory transcript into the task's memory file.

    Keeps the shortest skeleton seen so far and increments ``support``.
    Returns the written path, or ``None`` when the transcript yields no
    symbolic skeleton (nothing is written).
    """
    commands = skeleton_from_transcript(transcript)
    if not commands or _contains_instance_id(commands):
        return None
    task_key = normalize_task_key(instruction)
    existing = load_partnr_task_memory(task_memory_dir, instruction)
    source = dict(source or {})
    source.setdefault("num_decisions", len(transcript))
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
    written = write_partnr_task_memory({task_key: entry}, task_memory_dir)
    return written[0]


def resolve_partnr_task_memory(memory_entry: Optional[Mapping], state: Mapping) -> List[str]:
    """Bind ``<kind:category>`` symbols to instances present in ``state``.

    First match by category prefix (sorted for determinism); a command with
    any unresolvable symbol is skipped. Literal arguments pass through.
    """
    if not memory_entry:
        return []
    objects, furniture, rooms = _vocabulary(state)
    pools = {
        "obj": sorted(objects),
        "furn": sorted(furniture),
        "room": sorted(rooms),
        "ent": sorted(objects | furniture | rooms),
    }
    resolved: List[str] = []
    for command in memory_entry.get("commands", []):
        parsed = parse_action(command)
        if parsed is None:
            continue
        name, args = parsed
        bound: List[str] = []
        for arg in args:
            symbol = parse_symbol(arg)
            if symbol is None:
                bound.append(arg)
                continue
            kind, category = symbol
            match = next(
                (item for item in pools.get(kind, []) if instance_category(item) == category),
                None,
            )
            if match is None:
                bound = []
                break
            bound.append(match)
        else:
            resolved.append(format_action(name, bound))
    return resolved


# ----------------------------------------------------------- global memory

def partnr_global_memory(ledger_path: Optional[str] = None) -> GlobalMemory:
    """Empty-seeded Global Memory plus promoted rules from ``ledger_path`` (if any)."""
    base = GlobalMemory(
        success_rules=list(PARTNR_SEED_SUCCESS_RULES),
        failure_models=list(PARTNR_SEED_FAILURE_MODELS),
    )
    if not ledger_path or not Path(ledger_path).exists():
        return base
    return GlobalMemory.from_ledger(
        ledger_path,
        include_seed=True,
        base_memory=base,
        primitive_names=PARTNR_PRIMITIVE_NAMES,
        provenance_validator=validate_trace_hash_provenance,
    )


def partnr_global_memory_ledger(ledger_path: str, *, read_only: bool = False) -> GlobalMemoryLedger:
    return GlobalMemoryLedger.load(
        ledger_path,
        read_only=read_only,
        primitive_names=PARTNR_PRIMITIVE_NAMES,
        provenance_validator=validate_trace_hash_provenance,
    )


_EXTRACTION_SYSTEM = (
    "You distill task-independent operating knowledge for a household robot "
    "that acts through the primitives "
    + ", ".join(PARTNR_PRIMITIVE_NAMES)
    + ". Rules must generalize across houses and tasks: NO instance ids "
    "(cup_0, table_13, kitchen_1), NO room or object names specific to this "
    "episode, NO restatement of the task. Reply with ONE JSON object only: "
    '{"success_rules": [str, ...], "failure_models": [str, ...]} with at most '
    "3 rules in total, each naming the primitive it applies to."
)

_INSTANCE_ID = re.compile(r"\b[a-z][a-z_]*_\d+\b")


def _extraction_user(transcript: Sequence[Mapping], success: bool) -> str:
    lines = [
        f"Outcome: task {'succeeded' if success else 'failed'}.",
        "Action transcript (symbolized, in order):",
    ]
    for step in transcript:
        vocab = step.get("vocab") or {}
        action = symbolize_partnr_action(
            step["action"],
            objects=vocab.get("objects", ()),
            furniture=vocab.get("furniture", ()),
            rooms=vocab.get("rooms", ()),
        )
        lines.append(f"  {action} -> {step.get('status') or 'n/a'}"
                     + (f" ({step['failure_kind']})" if step.get("failure_kind") else ""))
    lines.append("")
    lines.append("Write generic rules about HOW to use the primitives (what to do before "
                 "what, how to react to failures), not about this task.")
    return "\n".join(lines)


def _rule_texts(payload: Any, key: str) -> List[Tuple[str, Optional[str]]]:
    items = (payload or {}).get(key) if isinstance(payload, dict) else None
    out: List[Tuple[str, Optional[str]]] = []
    for item in items or []:
        if isinstance(item, str):
            out.append((item.strip(), None))
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            out.append((item["text"].strip(), item.get("primitive")))
    return [(text, primitive) for text, primitive in out if text]


def _attribute_primitive(text: str, hint: Optional[str], transcript: Sequence[Mapping]) -> Optional[str]:
    if hint in PARTNR_PRIMITIVE_NAMES:
        return hint
    for name in PARTNR_PRIMITIVE_NAMES:
        if re.search(rf"\b{name}\b", text):
            return name
    counts = Counter(step["name"] for step in transcript if step["name"] in PARTNR_PRIMITIVE_NAMES)
    return counts.most_common(1)[0][0] if counts else None


def extract_global_memory_candidates(
    traces: Sequence[Any],
    chat_fn: ChatFn,
    *,
    success_overrides: Optional[Mapping[str, bool]] = None,
    max_rules: int = 3,
) -> List[GlobalMemoryEvidence]:
    """Ask the LLM for <=3 generic rules per completed bootstrap trace.

    Paper-compatible: the paper describes Global Memory extracted from
    experience but not the extraction procedure. Rules quoting instance ids
    are dropped; each candidate is trace-backed (sha256 + turns) and enters
    the ledger as *pending* until ``promote_candidates`` decides.
    """
    success_overrides = dict(success_overrides or {})
    candidates: List[GlobalMemoryEvidence] = []
    for trace in traces:
        episode = dict(_as_episode(trace))
        records = episode["records"]
        if not episode.get("path") or not episode.get("trace_sha256"):
            _LOG.warning("skipping trace without path/sha256 (no provenance)")
            continue
        episode_id = episode_id_of(records)
        for key in (episode_id, episode.get("path", "")):
            if key and key in success_overrides:
                episode["task_state_success"] = float(bool(success_overrides[key]))
        success = episode_success(episode)
        if success is None:
            continue
        transcript = episode_transcript(records)
        if not transcript:
            continue
        raw = chat_fn(_EXTRACTION_SYSTEM, _extraction_user(transcript, success))
        payload = extract_json_object(raw or "")
        rules = [("success_rule", t, p) for t, p in _rule_texts(payload, "success_rules")]
        rules += [("failure_model", t, p) for t, p in _rule_texts(payload, "failure_models")]
        turns = tuple(
            int(step["decision_index"]) + 1
            for step in transcript
            if isinstance(step.get("decision_index"), int)
        ) or (1,)
        for kind, text, hint in rules[:max_rules]:
            if _INSTANCE_ID.search(text):
                continue
            primitive = _attribute_primitive(text, hint, transcript)
            if primitive is None:
                continue
            # Ledger policy: success rules need a verified success, failure
            # models a verified failure (see global_memory._decide_candidate).
            postcondition_met = kind == "success_rule"
            if postcondition_met and not success:
                continue
            candidates.append(GlobalMemoryEvidence(
                kind=kind,
                text=text,
                trace_sha256=episode["trace_sha256"],
                turns=turns,
                trace_path=episode["path"],
                primitive=primitive,
                postcondition_met=postcondition_met,
                structured_evidence={
                    "primitive_postcondition_met": postcondition_met,
                    "task_state_success": bool(success),
                    "episode_id": episode_id,
                    "extraction": "llm",
                },
            ))
    return candidates


def _normalize_rule(text: str) -> str:
    return " ".join(text.split()).lower().strip()


def promote_candidates(ledger: GlobalMemoryLedger, min_support: int = 2) -> List[str]:
    """Promote pending rules supported by >= ``min_support`` distinct traces."""
    groups: Dict[Tuple[str, str], List] = defaultdict(list)
    for decision in ledger.decisions:
        if decision.status != "pending":
            continue
        groups[(decision.kind, _normalize_rule(decision.candidate.text))].append(decision.candidate)
    promoted: List[str] = []
    for (_kind, _normalized), members in sorted(groups.items()):
        support = len({member.trace_sha256 for member in members})
        if support < min_support:
            continue
        interpretation = members[0].text.strip()
        for member in members:
            ledger.promote(member.identity, semantic_interpretation=interpretation)
        promoted.append(interpretation)
    return promoted
