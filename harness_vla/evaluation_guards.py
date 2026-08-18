"""Pure pre-execution guards used by the Harness VLA evaluator."""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Tuple


def _vla_mode(invocation: Dict) -> str:
    mode = str(invocation.get("mode", invocation.get("prompt", "grasp"))).strip().lower()
    if "place" in mode or "put" in mode or "drop" in mode:
        return "place"
    if "push" in mode or "wipe" in mode:
        return "push"
    return "grasp"


def _candidates(
    coords: Dict[str, Sequence[float]],
    roles: Dict[str, Sequence[str]],
    labels: Optional[Dict[str, str]],
    role: str,
) -> str:
    labels = labels or {}
    values = []
    for object_id in coords:
        if role in roles.get(object_id, []):
            label = labels.get(object_id)
            values.append(f"{object_id} ({label})" if label else object_id)
    return ", ".join(values) if values else "none"


def validate_vla_semantics(
    invocation: Dict,
    object_coords: Dict[str, Sequence[float]],
    object_roles: Dict[str, Sequence[str]],
    held_object_id: Optional[str] = None,
    object_labels: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, str]]:
    """Return ``(status, feedback)`` when a VLA invocation must not execute."""
    if invocation.get("action") != "vla_act":
        return None

    mode = _vla_mode(invocation)
    manipulable = _candidates(object_coords, object_roles, object_labels, "manipulable")
    destinations = _candidates(object_coords, object_roles, object_labels, "destination")
    if mode == "grasp":
        object_id = invocation.get("object", invocation.get("target"))
        if object_id not in object_coords or "manipulable" not in object_roles.get(object_id, []):
            return (
                "semantic_error",
                "Grasp rejected before execution: 'object' (or legacy 'target') must be a "
                f"visible manipulable ID. Valid visible example(s): {manipulable}.",
            )
        return None

    if mode != "place":
        return None

    object_id = invocation.get("object")
    destination_id = invocation.get("destination")
    if object_id is None or destination_id is None:
        return (
            "semantic_error",
            "Place rejected before execution: mode='place' requires both 'object' and "
            "'destination'; legacy 'target' is not allowed. Valid visible examples: "
            f"object={manipulable}; destination={destinations}.",
        )
    if object_id == destination_id:
        return (
            "semantic_error",
            "Place rejected before execution: object and destination must be different. "
            f"Valid visible examples: object={manipulable}; destination={destinations}.",
        )
    if object_id not in object_coords or "manipulable" not in object_roles.get(object_id, []):
        return (
            "semantic_error",
            "Place rejected before execution: 'object' must be a visible manipulable ID. "
            f"Valid visible example(s): {manipulable}.",
        )
    if destination_id not in object_coords or "destination" not in object_roles.get(destination_id, []):
        return (
            "semantic_error",
            "Place rejected before execution: 'destination' must be a visible destination ID. "
            f"Valid visible example(s): {destinations}.",
        )
    if held_object_id != object_id:
        return (
            "place_rejected",
            "Place rejected before execution: the specified object must match the verified held "
            f"object. held_object_id={held_object_id!r}; first grasp {object_id!r} successfully.",
        )
    return None


class NoProgressGuard:
    """Reject a fourth consecutive identical zero-progress execution."""

    def __init__(self, limit: int = 3):
        self.limit = int(limit)
        self._signature = None
        self._count = 0

    @staticmethod
    def signature(invocation: Dict) -> str:
        return json.dumps(invocation, sort_keys=True, separators=(",", ":"), default=str)

    def should_reject(self, invocation: Dict) -> bool:
        return self._signature == self.signature(invocation) and self._count >= self.limit

    def observe_execution(self, invocation: Dict, step_results: Sequence[Dict]) -> None:
        signature = self.signature(invocation)
        no_progress = bool(step_results) and all(
            float(step.get("reward", 0)) == 0.0
            and float(step.get("task_success", 0)) == 0.0
            for step in step_results
        )
        if no_progress and signature == self._signature:
            self._count += 1
        elif no_progress:
            self._signature = signature
            self._count = 1
        else:
            self._signature = None
            self._count = 0