"""Prompt construction for the Harness VLA beta planner.

The planner is asked to emit exactly **one** JSON primitive invocation per turn,
observe the resulting feedback, and iterate (a closed-loop REPL, simplified to an
in-process loop for the beta). The prompt is assembled from:

* a role + success-signal preamble (Appendix E "Shared Prompt Core");
* the fixed primitive vocabulary and JSON schema;
* the Global Memory (success rules + failure models);
* the current object coordinate table (language-only perception);
* the running action/feedback history for the episode.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from harness_vla.primitives.primitives import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ROT_MAX,
    VOXEL_SIZE,
)

HARNESS_ROLE = (
    "You are a memory-guided hybrid manipulation agent operating the "
    "EB-Manipulation tabletop environment. A benchmark driver is already running "
    "and waiting for your commands. Your job is to complete the task by reading "
    "the task state, localizing objects from the provided coordinate table, and "
    "choosing ONE primitive to execute per turn. You invoke the contact-rich "
    "primitive `vla_act` only for local grasping/placement/pushing, and use the "
    "analytic primitives for grounding, staging, transport, and release."
)

# Rendered once; describes the fixed primitive library and the JSON contract.
PRIMITIVE_REFERENCE = f"""\
Coordinate system: positions X, Y, Z are voxel indices in [0, {VOXEL_SIZE}].
Orientations Roll, Pitch, Yaw are discrete bins in [0, {ROT_MAX}].
Gripper: {GRIPPER_OPEN} = open, {GRIPPER_CLOSED} = closed.

Fixed primitive library (you may ONLY use these; you cannot invent new ones):

Analytic primitives:
  move_to      -> move the end-effector to a voxel target.
                  {{"action": "move_to", "xyz": [x, y, z], "gripper": "open"|"close"(optional)}}
                  xyz may instead be an object name via "target": "object 2".
  rotate_wrist -> set wrist yaw while holding position.
                  {{"action": "rotate_wrist", "target_yaw": <int in [0,{ROT_MAX}]>}}
  rotate_pitch -> set wrist pitch while holding position.
                  {{"action": "rotate_pitch", "target_pitch": <int in [0,{ROT_MAX}]>}}
  set_gripper  -> drive the gripper open or closed in place.
                  {{"action": "set_gripper", "gripper": "open"|"close"}}
  release      -> open the gripper to release a held object.
                  {{"action": "release", "lift": true|false(optional)}}

Contact-rich primitive (a retryable frozen-VLA stand-in):
    vla_act grasp -> {{"action": "vla_act", "object": "object 1", "mode": "grasp"}}
    vla_act place -> {{"action": "vla_act", "object": "object 1", "destination": "object 2", "mode": "place"}}
                                    The object being held and destination MUST be distinct.
    legacy grasp/push -> {{"action": "vla_act", "target": "object 1", "mode": "grasp"|"push"}}
                  Legacy "target" is NEVER allowed for place; place always requires object and destination.
                  For push you may add "direction": [dx, dy, dz].
                  vla_act is retryable: if a grasp comes up empty, re-stage and call it again.

Grounding rule: the visible planner IDs, labels, and roles supplied each turn are
authoritative. Grasp only a manipulable ID. For place, use a manipulable object
ID and a distinct destination ID; never infer roles from ID numbering or wording.

Safety rule: call canonical place only after history explicitly reports
grasp_verified for that same object. action_success means only that a simulator
command executed; it is NOT proof that an object was grasped. On empty_grasp or
grasp_unverified, do not place: inspect feedback, re-stage, and retry grasp.

Division of labor: use vla_act for the moment of contact (grasp/place/push);
use analytic primitives for everything else (staging, transport, release)."""

OUTPUT_CONTRACT = """\
Respond with a SINGLE JSON object and nothing else, in this exact shape:
{
  "reasoning": "<one or two sentences: current state, why this primitive now>",
  "action": { ... one primitive invocation from the library ... }
}
Do not output multiple actions. Do not wrap the JSON in markdown fences.
Output only the JSON object."""


def build_system_prompt(global_memory_text: str) -> str:
    """Assemble the static portion of the prompt (role, primitives, memory)."""
    return "\n\n".join(
        [
            HARNESS_ROLE,
            PRIMITIVE_REFERENCE,
            global_memory_text,
            OUTPUT_CONTRACT,
        ]
    )


def _format_candidates(
    object_coords: Dict[str, Sequence[float]],
    object_roles: Optional[Dict[str, Sequence[str]]],
    object_labels: Optional[Dict[str, str]],
    role: str,
) -> str:
    roles = object_roles or {}
    labels = object_labels or {}
    candidates = []
    for object_id, coord in object_coords.items():
        if roles and role not in roles.get(object_id, []):
            continue
        label = labels.get(object_id)
        suffix = f", label={label}" if label else ""
        candidates.append(f"  {object_id}: coords={list(coord)}{suffix}")
    return "\n".join(candidates) if candidates else "  (none localized this turn)"


def build_turn_prompt(
    user_instruction: str,
    object_coords: Dict[str, Sequence[float]],
    pose_action: Sequence[int],
    history: List[Dict],
    max_history: int = 8,
    object_roles: Optional[Dict[str, Sequence[str]]] = None,
    object_labels: Optional[Dict[str, str]] = None,
    held_object_id: Optional[str] = None,
    attachment_evidence_available: bool = False,
    resolved_task_memory: Optional[Sequence[Dict]] = None,
) -> str:
    """Assemble the per-turn user message.

    ``history`` is a list of ``{"action": <invocation>, "feedback": <str>,
    "status": <str>}`` records from earlier turns of the current episode.
    """
    lines: List[str] = []
    lines.append(f"Task instruction: {user_instruction.rstrip('.')}")
    lines.append("")
    lines.append("Current end-effector pose [X, Y, Z, Roll, Pitch, Yaw, Gripper]:")
    lines.append(f"  {list(pose_action)}")
    if attachment_evidence_available:
        if held_object_id:
            lines.append(
                f"Currently attached object (authoritative simulator state): {held_object_id}. "
                "Keep the gripper closed during transport; opening it detaches the object."
            )
        else:
            lines.append(
                "Currently attached object (authoritative simulator state): none. "
                "No object is held; a verified grasp is required before any place."
            )
    lines.append("")
    lines.append("Manipulable candidates (planner ID, voxel coords, optional label):")
    lines.append(_format_candidates(object_coords, object_roles, object_labels, "manipulable"))
    lines.append("")
    lines.append("Destination candidates (planner ID, voxel coords, optional label):")
    lines.append(_format_candidates(object_coords, object_roles, object_labels, "destination"))
    lines.append("For place, choose a destination ID distinct from the manipulable object ID.")
    lines.append("")

    if resolved_task_memory is not None:
        lines.append("Task Specific Memory structural prior (not an execution script):")
        lines.append(
            "  All planner IDs and any coordinates below were re-grounded from the "
            "CURRENT scene; no seed coordinates are present."
        )
        lines.append(
            "  Use this only as a structural prior. Current observation, simulator "
            "state, and current feedback are authoritative and take precedence."
        )
        lines.append(
            json.dumps(list(resolved_task_memory), ensure_ascii=True, sort_keys=True)
        )
        lines.append("")

    if history:
        lines.append("Recent primitive history (most recent last):")
        for i, record in enumerate(history[-max_history:]):
            act = record.get("action")
            status = record.get("status", "")
            feedback = record.get("feedback", "")
            lines.append(f"  turn {i}: action={act} status={status}")
            if feedback:
                lines.append(f"          feedback: {feedback}")
        lines.append("")
    else:
        lines.append("No primitives executed yet this episode.")
        lines.append("")

    lines.append(
        "Choose the single best next primitive to make progress toward the task. "
        "Respond with the JSON object described above."
    )
    return "\n".join(lines)
