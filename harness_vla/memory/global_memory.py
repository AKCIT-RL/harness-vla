"""Global Memory for the Harness VLA beta.

Global Memory stores *task-independent* operating knowledge for the fixed
primitive library: reusable success rules and failure models (e.g. empty grasp,
false visual success). Rules can be seeded manually or promoted explicitly from
trace-backed candidates. Task Specific Memory and the bootstrap/deployment
lifecycle are implemented separately in ``libero_memory_lifecycle.py``.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness_vla.primitives.primitives import PRIMITIVE_NAMES
from harness_vla.trace_io import (
    load_complete_jsonl_bytes,
    write_json_atomic,
)


# Manual seed rules, adapted from the Harness VLA paper (Appendix A). These are
# task-independent and describe how to use the fixed primitive library well.
SEED_SUCCESS_RULES: List[str] = [
    "Use the vla_act primitive for contact-rich phases such as irregular grasping, "
    "constrained placement, or fixture interaction. After a stable grasp, prefer "
    "analytic primitives (move_to, release) for long transport and precise placement.",
    "Stage the end-effector into a favorable pre-contact pose with move_to / "
    "rotate_wrist before invoking vla_act, then invoke vla_act only for the local "
    "contact-rich operation.",
    "vla_act is retryable: if a contact attempt fails, re-stage the robot with "
    "analytic primitives and invoke vla_act again rather than abandoning the task.",
    "Reserve analytic primitives for non-contact structure: grounding the target, "
    "free-space transport, posture adjustment, and release.",
]

SEED_FAILURE_MODELS: List[str] = [
    "Empty grasp: if the gripper closes but the object does not move with the "
    "end-effector, treat the attempt as an empty grasp. Re-localize the object and "
    "re-stage before retrying vla_act.",
    "False success: do not terminate from visual proximity alone. Rely on the "
    "benchmark success signal and the latest execution feedback before concluding "
    "the task is done.",
    "Unstable staging: if the pre-contact pose does not expose the target in a "
    "reachable configuration, adjust position/orientation with analytic primitives "
    "before invoking vla_act.",
    "Detach during transport: commanding the gripper open (gripper=\"open\" or "
    "release) while holding an object detaches it immediately and the object is "
    "left behind. After a verified grasp, transport with move_to using "
    "gripper=\"close\" (or omit gripper) and open the gripper only at the "
    "destination via release or vla_act place.",
    "Repeated grasp at the same pose: if a grasp attempt fails (empty_grasp or "
    "grasp_unverified), do NOT retry the identical approach. Re-stage first: "
    "change the wrist orientation with rotate_wrist and/or approach the target "
    "from a different offset with move_to, then retry vla_act grasp. Repeating "
    "the same pose reproduces the same failed contact geometry.",
    "Path planning failure during transport: if move_to fails with 'Could not "
    "create path' while holding an object, the direct straight-line path is "
    "blocked (usually by the table surface or the destination object). Do NOT "
    "repeat the same move_to. First lift: move_to the current X,Y with a much "
    "higher Z (10-15 voxels above the destination height), then transport "
    "laterally at that safe height, and only then descend above the "
    "destination before releasing.",
]


@dataclass
class GlobalMemory:
    """Cross-task memory rendered from a seed and promoted trace evidence."""

    success_rules: List[str] = field(default_factory=lambda: list(SEED_SUCCESS_RULES))
    failure_models: List[str] = field(default_factory=lambda: list(SEED_FAILURE_MODELS))
    provenance: Dict[str, List[dict]] = field(default_factory=dict, repr=False)

    @classmethod
    def seeded(cls) -> "GlobalMemory":
        """Return a GlobalMemory populated with the fixed manual seed."""
        return cls()

    @classmethod
    def load(cls, path: str) -> "GlobalMemory":
        """Load from a JSON file, falling back to the seed if absent/invalid."""
        if not path or not os.path.exists(path):
            return cls.seeded()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                success_rules=list(data.get("success_rules", SEED_SUCCESS_RULES)),
                failure_models=list(data.get("failure_models", SEED_FAILURE_MODELS)),
            )
        except (OSError, json.JSONDecodeError, TypeError):
            return cls.seeded()

    @classmethod
    def from_ledger(
        cls,
        ledger_path: str,
        *,
        include_seed: bool = True,
        base_memory: Optional["GlobalMemory"] = None,
    ) -> "GlobalMemory":
        """Render only promoted ledger entries, optionally after a seed/base."""
        base = base_memory or cls.seeded()
        memory = cls(
            success_rules=list(base.success_rules) if include_seed else [],
            failure_models=list(base.failure_models) if include_seed else [],
        )
        ledger = GlobalMemoryLedger.load(ledger_path, read_only=True)
        content = {
            "success_rule": memory.success_rules,
            "failure_model": memory.failure_models,
        }
        canonical = {
            _normalize_content(text): text
            for values in content.values()
            for text in values
        }
        for decision in ledger.decisions:
            if decision.status != "promoted":
                continue
            candidate = decision.candidate
            rendered_candidate = decision.semantic_interpretation
            normalized = _normalize_content(rendered_candidate)
            if normalized not in canonical:
                canonical[normalized] = rendered_candidate.strip()
                content[candidate.kind].append(canonical[normalized])
            rendered_text = canonical[normalized]
            memory.provenance.setdefault(rendered_text, []).append(
                decision.to_dict()
            )
        return memory

    def provenance_for(self, text: str) -> List[dict]:
        """Return trace provenance for rendered content without exposing mutation."""
        normalized = _normalize_content(text)
        for rendered_text, provenance in self.provenance.items():
            if _normalize_content(rendered_text) == normalized:
                return [dict(item) for item in provenance]
        return []

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "success_rules": self.success_rules,
                    "failure_models": self.failure_models,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def render(self) -> str:
        """Render the memory as prompt text."""
        lines = ["Global Memory (task-independent operating knowledge):", "", "Success rules:"]
        for i, rule in enumerate(self.success_rules, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")
        lines.append("Failure models:")
        for i, fm in enumerate(self.failure_models, 1):
            lines.append(f"  {i}. {fm}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GlobalMemoryEvidence:
    """A trace-backed candidate, not an automatically promoted memory entry."""

    kind: str
    text: str
    trace_sha256: str
    turns: tuple[int, ...]
    trace_path: str = ""
    primitive: str = ""
    postcondition_met: Optional[bool] = None
    structured_evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"success_rule", "failure_model"}:
            raise ValueError("evidence kind must be success_rule or failure_model")
        if not self.text.strip():
            raise ValueError("evidence text must be non-empty")
        if len(self.trace_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.trace_sha256
        ):
            raise ValueError("trace_sha256 must be a SHA-256 hex digest")
        if not self.turns or any(
            isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0
            for turn in self.turns
        ):
            raise ValueError("evidence must reference positive trace turns")
        if self.postcondition_met is not None and not isinstance(
            self.postcondition_met, bool
        ):
            raise ValueError("postcondition_met must be true, false, or null")
        if not isinstance(self.structured_evidence, dict):
            raise ValueError("structured_evidence must be a mapping")

    @property
    def identity(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "text": " ".join(self.text.split()),
                "trace_sha256": self.trace_sha256,
                "turns": list(self.turns),
                "trace_path": self.trace_path,
                "primitive": self.primitive,
                "postcondition_met": self.postcondition_met,
                "structured_evidence": self.structured_evidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.identity,
            "kind": self.kind,
            "text": self.text,
            "trace_sha256": self.trace_sha256,
            "turns": list(self.turns),
            "trace_path": self.trace_path,
            "primitive": self.primitive,
            "postcondition_met": self.postcondition_met,
            "structured_evidence": self.structured_evidence,
        }


def _normalize_content(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class GlobalMemoryDecision:
    """An explicit, deterministic disposition of one memory candidate."""

    candidate: GlobalMemoryEvidence
    status: str
    reason: str
    semantic_interpretation: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pending", "promoted", "rejected"}:
            raise ValueError("decision status must be pending, promoted, or rejected")
        if not self.reason.strip():
            raise ValueError("decision reason must be non-empty")
        if self.status == "promoted" and not self.semantic_interpretation.strip():
            raise ValueError(
                "promoted decision requires a non-empty semantic interpretation"
            )
        if self.status != "promoted" and self.semantic_interpretation:
            raise ValueError(
                "only promoted decisions may contain a semantic interpretation"
            )

    @property
    def kind(self) -> str:
        return self.candidate.kind

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "semantic_interpretation": self.semantic_interpretation,
        }


def _candidate_from_dict(item: dict) -> GlobalMemoryEvidence:
    return GlobalMemoryEvidence(
        kind=item["kind"],
        text=item["text"],
        trace_sha256=item["trace_sha256"],
        turns=tuple(item["turns"]),
        trace_path=item.get("trace_path", ""),
        primitive=item.get("primitive", ""),
        postcondition_met=item.get("postcondition_met"),
        structured_evidence=dict(item.get("structured_evidence", {})),
    )


def _decide_candidate(
    candidate: GlobalMemoryEvidence, semantic_interpretation: str = ""
) -> GlobalMemoryDecision:
    if not candidate.trace_path or not candidate.primitive or not candidate.structured_evidence:
        return GlobalMemoryDecision(candidate, "rejected", "incomplete provenance")
    if candidate.primitive not in PRIMITIVE_NAMES:
        return GlobalMemoryDecision(candidate, "rejected", "unknown primitive")
    if candidate.postcondition_met is None:
        return GlobalMemoryDecision(candidate, "pending", "ambiguous postcondition")
    if candidate.structured_evidence.get(
        "primitive_postcondition_met"
    ) is not candidate.postcondition_met:
        return GlobalMemoryDecision(
            candidate, "rejected", "structured outcome mismatch"
        )
    expected_kind = (
        "success_rule" if candidate.postcondition_met else "failure_model"
    )
    if candidate.kind != expected_kind:
        return GlobalMemoryDecision(candidate, "rejected", "outcome/category mismatch")
    if semantic_interpretation.strip():
        return GlobalMemoryDecision(
            candidate,
            "promoted",
            "explicit semantically validated interpretation",
            semantic_interpretation.strip(),
        )
    return GlobalMemoryDecision(
        candidate,
        "pending",
        "awaiting semantically validated interpretation",
    )


def _trace_records_for_candidate(candidate: GlobalMemoryEvidence) -> List[dict]:
    if not candidate.trace_path:
        raise ValueError("Global Memory evidence has no trace path")
    trace_path = Path(candidate.trace_path)
    if not trace_path.is_file():
        raise ValueError("Global Memory evidence trace does not exist")
    payload = trace_path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ValueError("Global Memory evidence trace is incomplete")
    if hashlib.sha256(payload).hexdigest() != candidate.trace_sha256:
        raise ValueError("Global Memory evidence trace hash mismatch")
    return load_complete_jsonl_bytes(payload)


def _validate_candidate_provenance(candidate: GlobalMemoryEvidence) -> None:
    records = _trace_records_for_candidate(candidate)
    for turn in candidate.turns:
        matches = [record for record in records if record.get("turn") == turn]
        if len(matches) != 1:
            raise ValueError("Global Memory evidence turn must exist exactly once")
        record = matches[0]
        if record.get("primitive") != candidate.primitive:
            raise ValueError("Global Memory evidence primitive mismatch")
        if record.get("primitive_postcondition_met") is not candidate.postcondition_met:
            raise ValueError("Global Memory evidence postcondition mismatch")
        expected = {
            "primitive_postcondition_met": record.get("primitive_postcondition_met"),
            "termination_reason": record.get("termination_reason"),
            "task_success": record.get("task_success"),
            "episode_status": record.get("episode_status"),
        }
        if candidate.structured_evidence != expected:
            raise ValueError("Global Memory structured evidence mismatch")


@dataclass
class GlobalMemoryLedger:
    """Persist trace-backed candidates and explicit promotion decisions."""

    evidence: List[GlobalMemoryEvidence] = field(default_factory=list)
    decisions: List[GlobalMemoryDecision] = field(default_factory=list)
    read_only: bool = False

    @classmethod
    def load(cls, path: str, *, read_only: bool = False) -> "GlobalMemoryLedger":
        ledger_path = Path(path)
        if not ledger_path.exists():
            return cls(read_only=read_only)
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        schema_version = data.get("schema_version", 1)
        if type(schema_version) is not int or schema_version not in {1, 2, 3}:
            raise ValueError("unsupported Global Memory ledger schema version")
        if schema_version in {2, 3}:
            decisions = []
            for item in data.get("entries", []):
                candidate = _candidate_from_dict(item["candidate"])
                _validate_candidate_provenance(candidate)
                interpretation = item.get("semantic_interpretation", "")
                if not isinstance(interpretation, str):
                    raise ValueError("semantic interpretation must be a string")
                decision = _decide_candidate(candidate, interpretation)
                if (
                    item.get("status"),
                    item.get("reason"),
                    interpretation,
                ) != (
                    decision.status,
                    decision.reason,
                    decision.semantic_interpretation,
                ):
                    raise ValueError(
                        "persisted Global Memory decision violates promotion policy"
                    )
                decisions.append(decision)
            evidence = [item.candidate for item in decisions]
        else:
            evidence = [_candidate_from_dict(item) for item in data.get("evidence", [])]
            decisions = [_decide_candidate(item) for item in evidence]
            for candidate in evidence:
                if candidate.trace_path:
                    _validate_candidate_provenance(candidate)
        return cls(evidence=evidence, decisions=decisions, read_only=read_only)

    def add(self, candidate: GlobalMemoryEvidence) -> bool:
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        if any(existing.identity == candidate.identity for existing in self.evidence):
            return False
        self.evidence.append(candidate)
        self.evidence.sort(key=lambda item: item.identity)
        self.decisions.append(_decide_candidate(candidate))
        self.decisions.sort(key=lambda item: item.candidate.identity)
        return True

    def decision_for(self, candidate: GlobalMemoryEvidence) -> GlobalMemoryDecision:
        for decision in self.decisions:
            if decision.candidate.identity == candidate.identity:
                return decision
        raise KeyError(candidate.identity)

    def promote(
        self, candidate_id: str, *, semantic_interpretation: str
    ) -> GlobalMemoryDecision:
        """Explicitly promote one complete candidate using a semantic rule."""
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        if not isinstance(semantic_interpretation, str) or not semantic_interpretation.strip():
            raise ValueError("semantic interpretation must be non-empty")
        for index, decision in enumerate(self.decisions):
            if decision.candidate.identity != candidate_id:
                continue
            promoted = _decide_candidate(decision.candidate, semantic_interpretation)
            if promoted.status != "promoted":
                raise ValueError(
                    "candidate provenance does not permit explicit promotion"
                )
            self.decisions[index] = promoted
            return promoted
        raise KeyError(candidate_id)

    def audit(self) -> dict:
        counts = {
            "candidates": len(self.decisions),
            "promoted": 0,
            "rejected": 0,
            "pending": 0,
        }
        for decision in self.decisions:
            counts[decision.status] += 1
        return {
            "counts": counts,
            "candidate_ids": [item.candidate.identity for item in self.decisions],
        }

    def process_trace(self, trace_path: str, *, run_status: str, ledger_path: str) -> dict:
        """Classify a completed trace and atomically persist the canonical ledger."""
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        trace_payload = Path(trace_path).read_bytes()
        candidates = evidence_from_completed_trace_bytes(
            trace_payload, trace_path=str(trace_path), run_status=run_status
        )
        added = sum(1 for candidate in candidates if self.add(candidate))
        self.save(ledger_path)
        audit = self.audit()
        audit.update({
            "trace_path": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
            "ledger_sha256": hashlib.sha256(Path(ledger_path).read_bytes()).hexdigest(),
            "added": added,
        })
        return audit

    def save(self, path: str) -> None:
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        write_json_atomic(path, {
            "schema_version": 3,
            "entries": [item.to_dict() for item in self.decisions],
        })


def evidence_from_completed_trace(
    path: str, *, run_status: str
) -> List[GlobalMemoryEvidence]:
    """Extract structured candidates; promotion remains an explicit later decision."""
    trace_path = Path(path)
    return evidence_from_completed_trace_bytes(
        trace_path.read_bytes(), trace_path=str(trace_path), run_status=run_status
    )


def evidence_from_completed_trace_bytes(
    payload: bytes, *, trace_path: str, run_status: str
) -> List[GlobalMemoryEvidence]:
    """Extract candidates and provenance from one complete trace payload."""
    if run_status != "completed":
        raise ValueError("only a completed run can produce Global Memory evidence")
    if not payload.endswith(b"\n"):
        raise ValueError("incomplete trace cannot produce Global Memory evidence")
    records = load_complete_jsonl_bytes(payload)
    if not records:
        raise ValueError("empty trace cannot produce Global Memory evidence")
    trace_sha256 = hashlib.sha256(payload).hexdigest()
    candidates = []
    for index, record in enumerate(records, 1):
        reason = record.get("termination_reason")
        primitive = record.get("primitive", "primitive")
        if record.get("primitive_postcondition_met") in {True, False}:
            if record.get("episode_status") != "completed":
                raise ValueError(
                    "Global Memory evidence requires completed episode records"
                )
            if "task_success" not in record:
                raise ValueError(
                    "Global Memory evidence requires task_success"
                )
        if record.get("primitive_postcondition_met") is False:
            candidates.append(GlobalMemoryEvidence(
                kind="failure_model",
                text=f"{primitive} did not satisfy its postcondition in the recorded physical state.",
                trace_sha256=trace_sha256,
                turns=(int(record.get("turn", index)),),
                trace_path=trace_path,
                primitive=primitive,
                postcondition_met=False,
                structured_evidence={
                    "primitive_postcondition_met": False,
                    "termination_reason": reason,
                    "task_success": record.get("task_success"),
                    "episode_status": record.get("episode_status"),
                },
            ))
        elif record.get("primitive_postcondition_met") is True:
            candidates.append(GlobalMemoryEvidence(
                kind="success_rule",
                text=f"{primitive} satisfied its postcondition in the recorded physical state.",
                trace_sha256=trace_sha256,
                turns=(int(record.get("turn", index)),),
                trace_path=trace_path,
                primitive=primitive,
                postcondition_met=True,
                structured_evidence={
                    "primitive_postcondition_met": True,
                    "termination_reason": reason,
                    "task_success": record.get("task_success"),
                    "episode_status": record.get("episode_status"),
                },
            ))
    return candidates
