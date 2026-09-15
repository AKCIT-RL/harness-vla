"""Automatic Global Memory extractor for the bootstrap phase.

Classification: **paper-compatible**. The paper says Global Memory is distilled
from bootstrap experience into general, reusable guidance but does not specify
the extractor. This module makes that step automatic and auditable:

1. ``summarize_bootstrap_traces``: episode records -> compact symbolic
   :class:`EpisodeDigest` (success flag, failed actions with error classes,
   failure -> alternative "recovery" patterns, repeated loops).
2. ``propose_global_memory_candidates``: digests -> :class:`GlobalMemoryCandidate`
   either rule-based (deterministic, always available) or LLM-based (``llm``
   callable, same ``messages -> str`` shape the planners use).
3. ``promote_candidates``: phase-gated automatic promotion into a
   :class:`GlobalMemoryLedger` with per-episode trace provenance.

Candidates are phrased over the primitive vocabulary and error classes only;
any rule quoting a concrete instance name seen in the digests is rejected.

The generic defaults understand LIBERO-style trace records (``turn``,
``primitive``, ``primitive_postcondition_met``, ``termination_reason``).
Environment adapters supply ``transcript_fn`` to normalize their own trace
schema (see ``envs/partnr/memory.symbolic_steps``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from harness_vla.memory.global_memory import (
    GlobalMemoryEvidence,
    GlobalMemoryLedger,
)
from harness_vla.memory.task_memory import FORBIDDEN_OUTPUT_KEYS
from harness_vla.planner.phase_policy import Phase, PhasePolicy
from harness_vla.primitives.primitives import PRIMITIVE_NAMES

__all__ = [
    "EpisodeDigest",
    "GlobalMemoryCandidate",
    "PromotionReport",
    "LlmFn",
    "TranscriptFn",
    "promote_candidates",
    "propose_global_memory_candidates",
    "summarize_bootstrap_traces",
]

# ``llm(messages) -> str`` with OpenAI-style ``[{"role", "content"}]`` messages.
LlmFn = Callable[[List[Dict[str, str]]], str]
# ``transcript_fn(records) -> [step]`` where each step is a mapping with
# ``turn`` (int), ``action`` (symbolic str), ``primitive`` (str),
# ``ok`` (bool | None), ``error`` (str | None) and ``instances`` (iterable of
# concrete instance names that appeared in the raw action).
TranscriptFn = Callable[[Sequence[Mapping]], Sequence[Mapping[str, Any]]]

_INSTANCE_ID = re.compile(r"\b[A-Za-z][A-Za-z_\-]*_\d+\b")
_UNKNOWN_ERROR = "unknown_failure"


# ---------------------------------------------------------------- digests

@dataclass(frozen=True)
class RecoveryPattern:
    """A failed action followed by a different alternative action."""

    failed_primitive: str
    error_class: str
    recovery_primitive: str
    recovery_ok: bool
    turns: Tuple[int, ...]


@dataclass(frozen=True)
class LoopPattern:
    """The same symbolic action failing ``count`` consecutive times."""

    primitive: str
    error_class: str
    count: int
    turns: Tuple[int, ...]


@dataclass(frozen=True)
class EpisodeDigest:
    episode_id: str
    task_success: Optional[bool]
    steps: Tuple[Dict[str, Any], ...]
    failures: Tuple[Tuple[str, str, int], ...]  # (primitive, error_class, turn)
    recoveries: Tuple[RecoveryPattern, ...]
    loops: Tuple[LoopPattern, ...]
    instances: frozenset = frozenset()
    trace_path: str = ""
    trace_sha256: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["instances"] = sorted(self.instances)
        return data


def _error_class(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or _UNKNOWN_ERROR


def _default_transcript(records: Sequence[Mapping]) -> List[Dict[str, Any]]:
    """LIBERO-style records -> normalized steps (primitive name as action)."""
    steps: List[Dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        primitive = record.get("primitive")
        if not primitive:
            continue
        invocation = record.get("invocation") or {}
        labels = record.get("object_labels") or {}
        instances = {
            str(invocation[key])
            for key in ("target", "object", "destination")
            if isinstance(invocation.get(key), str)
        }
        args = [
            f"<{labels[invocation[key]]}>" if invocation.get(key) in labels else "<?>"
            for key in ("target", "object", "destination")
            if key in invocation
        ]
        met = record.get("primitive_postcondition_met")
        steps.append({
            "turn": int(record.get("turn", index)),
            "action": f"{primitive}[{', '.join(args)}]" if args else f"{primitive}[]",
            "primitive": str(primitive),
            "ok": met if met in {True, False} else None,
            "error": None if met is True else record.get("termination_reason"),
            "instances": instances,
        })
    return steps


def _default_success(episode: Mapping) -> Optional[bool]:
    if episode.get("task_success") is not None:
        return float(episode["task_success"]) == 1.0
    for record in reversed(episode.get("records") or []):
        if record.get("task_success") is not None:
            return float(record["task_success"]) == 1.0
    return None


def _as_episode(item: Any) -> Dict[str, Any]:
    if isinstance(item, Mapping) and "records" in item:
        return dict(item)
    if isinstance(item, (list, tuple)):
        return {"records": list(item)}
    raise TypeError("episode must be a list of records or a mapping with 'records'")


def summarize_bootstrap_traces(
    records: Sequence[Any],
    *,
    transcript_fn: Optional[TranscriptFn] = None,
    success_fn: Optional[Callable[[Mapping], Optional[bool]]] = None,
) -> List[EpisodeDigest]:
    """Bootstrap episodes -> one compact symbolic digest per episode.

    ``records`` is a list of episodes; each episode is either a list of trace
    records or a mapping with ``records`` and optional ``episode_id``,
    ``path``/``trace_path``, ``trace_sha256`` and ``task_success``.
    """
    transcript_fn = _default_transcript if transcript_fn is None else transcript_fn
    success_fn = _default_success if success_fn is None else success_fn
    digests: List[EpisodeDigest] = []
    for index, item in enumerate(records):
        episode = _as_episode(item)
        trace_path = str(episode.get("trace_path") or episode.get("path") or "")
        episode_id = str(
            episode.get("episode_id")
            or (Path(trace_path).stem if trace_path else "")
            or f"episode_{index}"
        )
        steps = [dict(step) for step in transcript_fn(episode["records"])]
        instances: set = set()
        for step in steps:
            instances |= {str(name) for name in step.pop("instances", ()) or ()}
        failures = tuple(
            (step["primitive"], _error_class(step.get("error")), int(step["turn"]))
            for step in steps
            if step.get("ok") is False
        )
        digests.append(EpisodeDigest(
            episode_id=episode_id,
            task_success=success_fn(episode),
            steps=tuple(steps),
            failures=failures,
            recoveries=tuple(_recoveries(steps)),
            loops=tuple(_loops(steps)),
            instances=frozenset(instances),
            trace_path=trace_path,
            trace_sha256=str(episode.get("trace_sha256") or ""),
        ))
    return digests


def _recoveries(steps: Sequence[Mapping]) -> List[RecoveryPattern]:
    out: List[RecoveryPattern] = []
    for index, step in enumerate(steps):
        if step.get("ok") is not False:
            continue
        # Skip identical retries; the first different action is the alternative.
        for later in steps[index + 1:]:
            if later["action"] == step["action"]:
                continue
            out.append(RecoveryPattern(
                failed_primitive=step["primitive"],
                error_class=_error_class(step.get("error")),
                recovery_primitive=later["primitive"],
                recovery_ok=later.get("ok") is True,
                turns=(int(step["turn"]), int(later["turn"])),
            ))
            break
    return out


def _loops(steps: Sequence[Mapping]) -> List[LoopPattern]:
    out: List[LoopPattern] = []
    index = 0
    while index < len(steps):
        step = steps[index]
        end = index + 1
        while (
            end < len(steps)
            and steps[end]["action"] == step["action"]
            and steps[end].get("ok") is False
        ):
            end += 1
        if step.get("ok") is False and end - index >= 2:
            out.append(LoopPattern(
                primitive=step["primitive"],
                error_class=_error_class(step.get("error")),
                count=end - index,
                turns=tuple(int(item["turn"]) for item in steps[index:end]),
            ))
        index = end
    return out


# ------------------------------------------------------------- candidates

@dataclass(frozen=True)
class GlobalMemoryCandidate:
    kind: str
    text: str
    primitive: str
    source: str  # "rule" | "llm"
    evidence_episode_ids: Tuple[str, ...]
    support: int
    success_rate: Optional[float] = None
    evidence_turns: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    pattern: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["evidence_episode_ids"] = list(self.evidence_episode_ids)
        data["evidence_turns"] = {key: list(value) for key, value in self.evidence_turns.items()}
        return data


def _leaks_instances(text: str, digests: Sequence[EpisodeDigest]) -> bool:
    lowered = text.lower()
    for digest in digests:
        if any(name.lower() in lowered for name in digest.instances):
            return True
    return bool(_INSTANCE_ID.search(text))


def _rule_candidates(digests: Sequence[EpisodeDigest], min_support: int) -> List[GlobalMemoryCandidate]:
    recovery_groups: Dict[Tuple[str, str, str], Dict[str, List[RecoveryPattern]]] = defaultdict(
        lambda: defaultdict(list)
    )
    loop_groups: Dict[Tuple[str, str], Dict[str, List[LoopPattern]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for digest in digests:
        for recovery in digest.recoveries:
            key = (recovery.failed_primitive, recovery.error_class, recovery.recovery_primitive)
            recovery_groups[key][digest.episode_id].append(recovery)
        for loop in digest.loops:
            loop_groups[(loop.primitive, loop.error_class)][digest.episode_id].append(loop)

    candidates: List[GlobalMemoryCandidate] = []
    for (failed, error, alternative), per_episode in sorted(recovery_groups.items()):
        if len(per_episode) < min_support:
            continue
        observations = [item for items in per_episode.values() for item in items]
        successes = sum(1 for item in observations if item.recovery_ok)
        candidates.append(GlobalMemoryCandidate(
            kind="failure_model",
            text=(
                f"If {failed} fails with {error}, do not repeat the same call; "
                f"perform {alternative} first and only then retry {failed}."
            ),
            primitive=failed,
            source="rule",
            evidence_episode_ids=tuple(sorted(per_episode)),
            support=len(per_episode),
            success_rate=successes / len(observations),
            evidence_turns={
                episode_id: tuple(sorted({turn for item in items for turn in item.turns}))
                for episode_id, items in per_episode.items()
            },
            pattern={
                "type": "recovery",
                "failed_primitive": failed,
                "error_class": error,
                "recovery_primitive": alternative,
                "observations": len(observations),
                "recovery_successes": successes,
            },
        ))
    for (primitive, error), per_episode in sorted(loop_groups.items()):
        if len(per_episode) < min_support:
            continue
        candidates.append(GlobalMemoryCandidate(
            kind="failure_model",
            text=(
                f"Repeating {primitive} after {error} keeps failing; change the "
                f"approach instead of retrying the identical call."
            ),
            primitive=primitive,
            source="rule",
            evidence_episode_ids=tuple(sorted(per_episode)),
            support=len(per_episode),
            success_rate=None,
            evidence_turns={
                episode_id: tuple(sorted({turn for item in items for turn in item.turns}))
                for episode_id, items in per_episode.items()
            },
            pattern={"type": "loop", "primitive": primitive, "error_class": error},
        ))
    return candidates


def _render_digest(digest: EpisodeDigest) -> str:
    outcome = {True: "succeeded", False: "failed"}.get(digest.task_success, "unknown")
    lines = [f"Episode {digest.episode_id}: task {outcome}."]
    for step in digest.steps:
        status = {True: "ok", False: "FAILED"}.get(step.get("ok"), "n/a")
        error = f" ({_error_class(step.get('error'))})" if step.get("ok") is False else ""
        lines.append(f"  t{step['turn']}: {step['action']} -> {status}{error}")
    for recovery in digest.recoveries:
        lines.append(
            f"  recovery: {recovery.failed_primitive}/{recovery.error_class} -> "
            f"{recovery.recovery_primitive} ({'ok' if recovery.recovery_ok else 'failed'})"
        )
    for loop in digest.loops:
        lines.append(f"  loop: {loop.primitive}/{loop.error_class} x{loop.count}")
    return "\n".join(lines)


def build_extraction_messages(
    digests: Sequence[EpisodeDigest], primitive_names: Sequence[str]
) -> List[Dict[str, str]]:
    system = (
        "You distill task-independent operating knowledge for a robot that acts "
        "through the primitives " + ", ".join(primitive_names) + ". Write general "
        "guidelines about HOW to use the primitives (what to do before what, how "
        "to react to failures). Rules must not mention concrete instance names, "
        "room or object names, or restate any task. Each rule must name the "
        "primitive it applies to. Reply with ONE JSON list only: "
        '[{"rule": str, "kind": "success_rule"|"failure_model", '
        '"evidence_episode_ids": [str, ...]}]. Use only episode ids listed below.'
    )
    user = "\n\n".join(_render_digest(digest) for digest in digests)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_json_list(raw: str) -> List[Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    for candidate in (text, text[text.find("["): text.rfind("]") + 1]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("rules", parsed.get("guidelines"))
        if isinstance(parsed, list):
            return parsed
    return []


def _llm_candidates(
    digests: Sequence[EpisodeDigest],
    llm: LlmFn,
    primitive_names: Sequence[str],
    min_support: int,
) -> List[GlobalMemoryCandidate]:
    by_id = {digest.episode_id: digest for digest in digests}
    raw = llm(build_extraction_messages(digests, primitive_names))
    candidates: List[GlobalMemoryCandidate] = []
    for item in _parse_json_list(raw):
        if not isinstance(item, dict) or set(item) & FORBIDDEN_OUTPUT_KEYS:
            continue
        text = item.get("rule")
        ids = item.get("evidence_episode_ids")
        if not isinstance(text, str) or not text.strip() or not isinstance(ids, list):
            continue
        ids = [str(value) for value in ids]
        # Hallucinated provenance taints the whole item.
        if not ids or any(value not in by_id for value in ids):
            continue
        ids = tuple(sorted(set(ids)))
        if len(ids) < min_support:
            continue
        primitive = next(
            (name for name in primitive_names if re.search(rf"\b{re.escape(name)}\b", text)),
            None,
        )
        if primitive is None:
            continue
        kind = "success_rule" if item.get("kind") == "success_rule" else "failure_model"
        if kind == "success_rule" and not all(by_id[value].task_success for value in ids):
            kind = "failure_model"
        candidates.append(GlobalMemoryCandidate(
            kind=kind,
            text=" ".join(text.split()),
            primitive=primitive,
            source="llm",
            evidence_episode_ids=ids,
            support=len(ids),
            success_rate=None,
            evidence_turns={
                value: tuple(int(step["turn"]) for step in by_id[value].steps) or (1,)
                for value in ids
            },
            pattern={"type": "llm"},
        ))
    return candidates


def propose_global_memory_candidates(
    digests: Sequence[EpisodeDigest],
    llm: Optional[LlmFn] = None,
    *,
    min_support: int = 2,
    primitive_names: Optional[Sequence[str]] = None,
) -> List[GlobalMemoryCandidate]:
    """Digests -> candidate guidance with episode provenance.

    Rule-based (``llm is None``): recurring failure -> alternative patterns and
    failure loops seen in >= ``min_support`` episodes. LLM-based: one call over
    all digests, JSON list parsed tolerantly; items with unknown evidence ids
    are dropped. Both modes reject rules that leak concrete instance names.
    """
    names = tuple(PRIMITIVE_NAMES if primitive_names is None else primitive_names)
    if llm is None:
        candidates = _rule_candidates(digests, min_support)
    else:
        candidates = _llm_candidates(digests, llm, names, min_support)
    return [
        candidate for candidate in candidates
        if candidate.primitive in names and not _leaks_instances(candidate.text, digests)
    ]


# -------------------------------------------------------------- promotion

@dataclass
class PromotionReport:
    promoted: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    evidence_added: int = 0

    def to_dict(self) -> dict:
        return {
            "promoted": list(self.promoted),
            "rejected": [list(item) for item in self.rejected],
            "evidence_added": self.evidence_added,
        }


def _memory_write_allowed(policy: Union[PhasePolicy, Phase, str]) -> bool:
    if isinstance(policy, PhasePolicy):
        return policy.memory_write_allowed
    return Phase(policy) is Phase.BOOTSTRAP


def promote_candidates(
    ledger: GlobalMemoryLedger,
    candidates: Sequence[GlobalMemoryCandidate],
    policy: Union[PhasePolicy, Phase, str],
    *,
    digests: Sequence[EpisodeDigest],
    min_support: int = 2,
    min_success_rate: float = 0.5,
) -> PromotionReport:
    """Automatically promote qualifying candidates into ``ledger``.

    Criterion: ``support >= min_support`` and, when the candidate carries an
    observed recovery ``success_rate`` (rule-based recoveries), ``success_rate
    >= min_success_rate``. Each promoted rule is written once per evidence
    episode as trace-backed :class:`GlobalMemoryEvidence` (trace path, sha256,
    turns) and promoted with the rule text as semantic interpretation. Raises
    ``PermissionError`` when ``policy`` forbids memory writes (deployment).
    """
    if not _memory_write_allowed(policy):
        raise PermissionError("memory writes are forbidden during deployment")
    by_id = {digest.episode_id: digest for digest in digests}
    report = PromotionReport()
    for candidate in candidates:
        if candidate.support < min_support:
            report.rejected.append((candidate.text, f"support {candidate.support} < {min_support}"))
            continue
        if candidate.success_rate is not None and candidate.success_rate < min_success_rate:
            report.rejected.append((
                candidate.text,
                f"recovery success rate {candidate.success_rate:.2f} < {min_success_rate:.2f}",
            ))
            continue
        reason = f"auto:{candidate.source} support={candidate.support}"
        if candidate.success_rate is not None:
            reason += f" success_rate={candidate.success_rate:.2f}"
        evidence: List[GlobalMemoryEvidence] = []
        for episode_id in candidate.evidence_episode_ids:
            digest = by_id.get(episode_id)
            if digest is None or not digest.trace_path or not digest.trace_sha256:
                continue
            postcondition_met = candidate.kind == "success_rule"
            evidence.append(GlobalMemoryEvidence(
                kind=candidate.kind,
                text=candidate.text,
                trace_sha256=digest.trace_sha256,
                turns=tuple(candidate.evidence_turns.get(episode_id) or (1,)),
                trace_path=digest.trace_path,
                primitive=candidate.primitive,
                postcondition_met=postcondition_met,
                structured_evidence={
                    "primitive_postcondition_met": postcondition_met,
                    "task_state_success": digest.task_success,
                    "episode_id": episode_id,
                    "extraction": candidate.source,
                    "promotion_reason": reason,
                    "pattern": dict(candidate.pattern),
                },
            ))
        if not evidence:
            report.rejected.append((candidate.text, "no trace provenance for evidence episodes"))
            continue
        for item in evidence:
            if ledger.add(item):
                report.evidence_added += 1
            ledger.promote(item.identity, semantic_interpretation=candidate.text)
        report.promoted.append(candidate.text)
    return report
