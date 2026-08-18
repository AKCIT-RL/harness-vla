"""Standalone bootstrap and deployment lifecycle for LIBERO memories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from harness_vla.memory.global_memory import (
    GlobalMemory,
    GlobalMemoryLedger,
)
from harness_vla.planner.phase_policy import (
    Phase,
    PhaseManifest,
    PhaseOperation,
    validate_phase_manifest,
)
from harness_vla.memory.task_memory import (
    TaskMemoryDecision,
    load_task_memory,
    promote_task_memory,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _trace_turn_count(trace_path: Path) -> int:
    payload = trace_path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ValueError("incomplete trace cannot enter a memory lifecycle")
    try:
        records = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid trace JSONL") from error
    return len([record for record in records if record.get("status") != "initialization_reset"])


@dataclass(frozen=True)
class BootstrapMemoryResult:
    """Serializable audit plus the Task Memory promotion decision."""

    task_memory: TaskMemoryDecision
    audit: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.audit)


def bootstrap_memories(
    manifest: PhaseManifest,
    *,
    seed: int,
    trace_path: str,
    episode_result_path: str,
    task_memory_dir: str,
    global_ledger_path: str,
    run_status: str = "completed",
) -> BootstrapMemoryResult:
    """Validate bootstrap provenance, then process both memory classes."""
    validate_phase_manifest(manifest)
    manifest.guard_operation(Phase.BOOTSTRAP, seed, PhaseOperation.WRITE_MEMORY)
    policy = manifest.policy_for(Phase.BOOTSTRAP)
    trace = Path(trace_path).resolve()
    turn_count = _trace_turn_count(trace)
    if turn_count > policy.budget:
        raise ValueError("bootstrap trace exceeds phase budget")

    episode_result = Path(episode_result_path).resolve()
    episode = json.loads(episode_result.read_text(encoding="utf-8"))
    episode_seed = episode.get("seed")
    if episode_seed is not None and episode_seed != seed:
        raise ValueError("episode seed does not match bootstrap seed")

    task_decision = promote_task_memory(
        trace, episode_result, Path(task_memory_dir).resolve()
    )
    ledger_path = Path(global_ledger_path).resolve()
    ledger = GlobalMemoryLedger.load(ledger_path)
    global_audit = ledger.process_trace(
        trace, run_status=run_status, ledger_path=ledger_path
    )
    audit = {
        "schema_version": 1,
        "phase": Phase.BOOTSTRAP.value,
        "manifest": manifest.to_dict(),
        "seed": seed,
        "budget": policy.budget,
        "turn_count": turn_count,
        "source": {
            "trace_path": str(trace),
            "trace_sha256": _sha256(trace.read_bytes()),
            "episode_result_path": str(episode_result),
            "episode_result_sha256": _sha256(episode_result.read_bytes()),
        },
        "task_memory": {
            "promoted": task_decision.accepted,
            "reasons": list(task_decision.reasons),
            "audit": task_decision.audit,
        },
        "global_memory": global_audit,
    }
    json.dumps(audit, sort_keys=True)
    return BootstrapMemoryResult(task_decision, audit)


def promote_global_memory(
    global_ledger_path: str,
    *,
    candidate_id: str,
    semantic_interpretation: str,
) -> Dict[str, Any]:
    """Persist and reload-validate one explicit semantic promotion."""
    ledger = GlobalMemoryLedger.load(global_ledger_path)
    decision = ledger.promote(
        candidate_id, semantic_interpretation=semantic_interpretation
    )
    ledger.save(global_ledger_path)
    reloaded = GlobalMemoryLedger.load(global_ledger_path, read_only=True)
    persisted = next(
        item for item in reloaded.decisions
        if item.candidate.identity == decision.candidate.identity
    )
    return persisted.to_dict()


def _render_context(commands: Sequence[Mapping[str, Any]], memory: GlobalMemory) -> str:
    task_payload = json.dumps(
        list(commands), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    context = "\n".join((
        "Task Memory (symbolic task structure):",
        task_payload,
        "",
        memory.render(),
    ))
    if "xyz" in context.casefold():
        raise ValueError("deployment memory context contains forbidden xyz")
    return context


@dataclass
class DeploymentMemorySession:
    """Loaded deployment context with write, budget, and integrity guards."""

    manifest: PhaseManifest
    seed: int
    task_memory_dir: Path
    global_ledger_path: Path
    task_audit: Mapping[str, Any]
    task_commands: Tuple[Mapping[str, Any], ...]
    global_memory: GlobalMemory
    context: str
    hashes_before: Dict[str, str]

    def guard_write(self, writer: Optional[Callable[[], Any]] = None) -> None:
        """Reject a deployment write before an optional writer can run."""
        self.manifest.guard_operation(
            Phase.DEPLOYMENT, self.seed, PhaseOperation.WRITE_MEMORY
        )
        if writer is not None:
            writer()

    def guard_budget(self, consumed_steps: int) -> None:
        if isinstance(consumed_steps, bool) or not isinstance(consumed_steps, int):
            raise ValueError("consumed_steps must be an integer")
        if consumed_steps < 0:
            raise ValueError("consumed_steps must not be negative")
        if consumed_steps > self.manifest.deployment_budget:
            raise ValueError("deployment steps exceed phase budget")

    def _current_hashes(self) -> Dict[str, str]:
        audit_path = self.task_memory_dir / "audit.json"
        commands_path = self.task_memory_dir / "commands.jsonl"
        return {
            "task_audit_sha256": _sha256(audit_path.read_bytes()),
            "task_commands_sha256": _sha256(commands_path.read_bytes()),
            "global_ledger_sha256": _sha256(self.global_ledger_path.read_bytes()),
            "global_rendered_sha256": _sha256(
                self.global_memory.render().encode("utf-8")
            ),
        }

    def finalize(self) -> Dict[str, Any]:
        """Prove that every deployment memory artifact stayed unchanged."""
        hashes_after = self._current_hashes()
        if hashes_after != self.hashes_before:
            raise RuntimeError("deployment memory changed")
        return {
            "schema_version": 1,
            "phase": Phase.DEPLOYMENT.value,
            "manifest": self.manifest.to_dict(),
            "seed": self.seed,
            "hashes_before": dict(self.hashes_before),
            "hashes_after": hashes_after,
            "unchanged": True,
        }


def prepare_deployment(
    manifest: PhaseManifest,
    *,
    seed: int,
    task_memory_dir: str,
    global_ledger_path: str,
) -> DeploymentMemorySession:
    """Load held-out, immutable Task and promoted Global Memory context."""
    validate_phase_manifest(manifest)
    manifest.guard_operation(Phase.DEPLOYMENT, seed, PhaseOperation.READ_MEMORY)
    memory_dir = Path(task_memory_dir).resolve()
    ledger_path = Path(global_ledger_path).resolve()
    if not ledger_path.is_file():
        raise ValueError("Global Memory ledger does not exist")
    audit, commands = load_task_memory(memory_dir)
    GlobalMemoryLedger.load(ledger_path, read_only=True)
    global_memory = GlobalMemory.from_ledger(
        str(ledger_path), include_seed=False
    )
    context = _render_context(commands, global_memory)
    session = DeploymentMemorySession(
        manifest=manifest,
        seed=seed,
        task_memory_dir=memory_dir,
        global_ledger_path=ledger_path,
        task_audit=audit,
        task_commands=tuple(commands),
        global_memory=global_memory,
        context=context,
        hashes_before={},
    )
    session.hashes_before = session._current_hashes()
    return session


__all__ = [
    "BootstrapMemoryResult",
    "DeploymentMemorySession",
    "bootstrap_memories",
    "prepare_deployment",
    "promote_global_memory",
]