"""Pure phase separation policy for Harness VLA bootstrap and deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple, Union


class Phase(str, Enum):
    """Protocol phases with distinct seed and memory permissions."""

    BOOTSTRAP = "bootstrap"
    DEPLOYMENT = "deployment"


class PhaseOperation(str, Enum):
    """Operations whose legality depends on the active protocol phase."""

    RESET = "reset"
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    REPORT_METRIC = "report_metric"


PhaseLike = Union[Phase, str]
OperationLike = Union[PhaseOperation, str]


def _phase(value: PhaseLike) -> Phase:
    try:
        return Phase(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown phase: {value!r}") from error


def _operation(value: OperationLike) -> PhaseOperation:
    try:
        return PhaseOperation(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown phase operation: {value!r}") from error


def _seed(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _budget(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero")
    return value


@dataclass(frozen=True)
class PhasePolicy:
    """Resolved, non-configurable permissions for one phase and its seeds."""

    phase: Phase
    seeds: Tuple[int, ...]
    budget: int
    reset_allowed: bool
    memory_write_allowed: bool
    reportable: bool

    def __post_init__(self) -> None:
        phase = _phase(self.phase)
        seeds = tuple(self.seeds)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "seeds", seeds)
        expected = {
            Phase.BOOTSTRAP: (True, True, False),
            Phase.DEPLOYMENT: (False, False, True),
        }[phase]
        actual = (
            self.reset_allowed,
            self.memory_write_allowed,
            self.reportable,
        )
        if actual != expected:
            raise ValueError(f"permissions do not match {phase.value} phase")
        if not seeds:
            raise ValueError("phase seeds must not be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("phase seeds must be unique")
        for seed in seeds:
            _seed(seed, "phase seed")
        _budget(self.budget, "phase budget")


@dataclass(frozen=True)
class PhaseManifest:
    """Validated seed partition, budgets, and derived phase policies."""

    bootstrap_seed: int
    evaluation_seeds: Tuple[int, ...]
    bootstrap_budget: int
    deployment_budget: int

    def __post_init__(self) -> None:
        evaluation_seeds = tuple(self.evaluation_seeds)
        object.__setattr__(self, "evaluation_seeds", evaluation_seeds)
        _seed(self.bootstrap_seed, "bootstrap_seed")
        if not evaluation_seeds:
            raise ValueError("evaluation_seeds must not be empty")
        for seed in evaluation_seeds:
            _seed(seed, "evaluation seed")
        if len(set(evaluation_seeds)) != len(evaluation_seeds):
            raise ValueError("evaluation_seeds must be unique")
        if self.bootstrap_seed in evaluation_seeds:
            raise ValueError("bootstrap_seed must not appear in evaluation_seeds")
        bootstrap_budget = _budget(self.bootstrap_budget, "bootstrap_budget")
        deployment_budget = _budget(self.deployment_budget, "deployment_budget")
        if bootstrap_budget < deployment_budget:
            raise ValueError("bootstrap_budget must be greater than or equal to deployment_budget")

    def policy_for(self, phase: PhaseLike) -> PhasePolicy:
        """Return permissions derived from the requested phase."""
        resolved = _phase(phase)
        if resolved is Phase.BOOTSTRAP:
            return PhasePolicy(resolved, (self.bootstrap_seed,), self.bootstrap_budget, True, True, False)
        return PhasePolicy(
            resolved,
            self.evaluation_seeds,
            self.deployment_budget,
            False,
            False,
            True,
        )

    def guard_operation(
        self,
        phase: PhaseLike,
        seed: int,
        operation: OperationLike,
    ) -> None:
        """Raise ``ValueError`` unless an operation is legal for the phase and seed."""
        policy = self.policy_for(phase)
        checked_seed = _seed(seed, "seed")
        if checked_seed not in policy.seeds:
            raise ValueError(f"seed {checked_seed} does not belong to {policy.phase.value} phase")

        resolved_operation = _operation(operation)
        if resolved_operation is PhaseOperation.RESET and not policy.reset_allowed:
            raise ValueError("reset is forbidden during deployment")
        if resolved_operation is PhaseOperation.WRITE_MEMORY and not policy.memory_write_allowed:
            raise ValueError("memory writes are forbidden during deployment")
        if resolved_operation is PhaseOperation.REPORT_METRIC and not policy.reportable:
            raise ValueError("bootstrap seed must not be included in reported metrics")

    def to_dict(self) -> Dict[str, object]:
        """Return a stable manifest representation containing no configurable permissions."""
        return {
            "bootstrap_budget": self.bootstrap_budget,
            "bootstrap_seed": self.bootstrap_seed,
            "deployment_budget": self.deployment_budget,
            "evaluation_seeds": list(self.evaluation_seeds),
        }

    def to_json(self) -> str:
        """Serialize the manifest deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def build_phase_manifest(
    bootstrap_seed: int,
    evaluation_seeds: Iterable[int],
    bootstrap_budget: int,
    deployment_budget: int,
) -> PhaseManifest:
    """Build and validate a phase manifest from ordinary Python values."""
    try:
        seeds = tuple(evaluation_seeds)
    except TypeError as error:
        raise ValueError("evaluation_seeds must be an iterable of integers") from error
    return PhaseManifest(
        bootstrap_seed=bootstrap_seed,
        evaluation_seeds=seeds,
        bootstrap_budget=bootstrap_budget,
        deployment_budget=deployment_budget,
    )


def validate_phase_manifest(manifest: PhaseManifest) -> None:
    """Validate an existing manifest, including instances restored unsafely."""
    if not isinstance(manifest, PhaseManifest):
        raise TypeError("manifest must be a PhaseManifest")
    manifest.__post_init__()


__all__ = [
    "Phase",
    "PhaseManifest",
    "PhaseOperation",
    "PhasePolicy",
    "build_phase_manifest",
    "validate_phase_manifest",
]