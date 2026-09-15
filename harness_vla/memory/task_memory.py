"""Offline, symbolic Task Specific Memory generation for verified rollouts.

The module-level defaults (``FORBIDDEN_OUTPUT_KEYS``, ``symbolize_record``)
encode the LIBERO/EB-Manipulation trace schema; environment adapters may pass
their own ``forbidden_output_keys=`` / ``symbolize=`` without changing them.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from harness_vla.trace_io import (
    load_complete_jsonl,
    write_json_atomic,
    write_text_atomic,
)


FORBIDDEN_OUTPUT_KEYS = {
    "compiled_actions",
    "frame_id",
    "grounding_objects",
    "id_to_sim_name",
    "object_coords",
    "planner_coords",
    "pose_after",
    "pose_before",
    "voxel",
    "xyz",
}
COMMAND_KEYS = {
    "sequence",
    "source_turn",
    "action",
    "target",
    "object",
    "destination",
    "mode",
    "gripper",
    "lift",
    "target_pitch",
    "target_yaw",
}
BINDING_KEYS = {"label", "roles"}


@dataclass(frozen=True)
class TaskMemoryDecision:
    accepted: bool
    reasons: Sequence[str]
    audit: Optional[Dict] = None
    commands: Sequence[Dict] = ()


def _canonical_commands_payload(commands):
    return "".join(
        json.dumps(command, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for command in commands
    )


def _binding(record, object_id):
    if not isinstance(object_id, str):
        raise ValueError("symbolic object ID must be a string")
    labels = record.get("object_labels", {})
    roles = record.get("object_roles", {})
    label = labels.get(object_id)
    object_roles = roles.get(object_id)
    if label is None or not object_roles:
        return None
    return {
        "label": str(label),
        "roles": sorted(str(role) for role in object_roles),
    }


def symbolize_record(record):
    """Convert one verified trace record to a coordinate-free command."""
    invocation = record.get("invocation")
    primitive = record.get("primitive")
    if not isinstance(invocation, dict) or not primitive:
        raise ValueError("record has no executed primitive invocation")
    if "xyz" in invocation:
        raise ValueError("literal xyz cannot be promoted to symbolic memory")

    command = {
        "sequence": 0,
        "source_turn": int(record["turn"]),
        "action": str(primitive),
    }
    target_id = invocation.get("target")
    object_id = invocation.get("object")
    destination_id = invocation.get("destination")

    if target_id is not None:
        binding = _binding(record, target_id)
        if binding is None:
            raise ValueError("target has no symbolic binding")
        command["target"] = binding
    if object_id is not None:
        binding = _binding(record, object_id)
        if binding is None:
            raise ValueError("object has no symbolic binding")
        command["object"] = binding
    if destination_id is not None:
        binding = _binding(record, destination_id)
        if binding is None:
            raise ValueError("destination has no symbolic binding")
        command["destination"] = binding

    for key in ("mode", "gripper", "lift", "target_pitch", "target_yaw"):
        if key in invocation:
            command[key] = invocation[key]
    return command


def _contains_forbidden_key(value, forbidden_output_keys=None):
    forbidden = (
        FORBIDDEN_OUTPUT_KEYS if forbidden_output_keys is None else forbidden_output_keys
    )
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_forbidden_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


def validate_task_memory(audit, commands, forbidden_output_keys=None):
    """Reject malformed or spatially contaminated memory payloads."""
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ValueError("unsupported audit schema")
    if audit.get("classification") != "paper-compatible":
        raise ValueError("invalid memory classification")
    if audit.get("command_count") != len(commands):
        raise ValueError("command count does not match payload")
    if _contains_forbidden_key(audit, forbidden_output_keys) or _contains_forbidden_key(
        commands, forbidden_output_keys
    ):
        raise ValueError("forbidden spatial data")
    for expected_sequence, command in enumerate(commands, 1):
        if not isinstance(command, dict) or set(command) - COMMAND_KEYS:
            raise ValueError("command contains unknown fields")
        if command.get("sequence") != expected_sequence:
            raise ValueError("commands are not in canonical sequence")
        if not isinstance(command.get("source_turn"), int):
            raise ValueError("command source turn is missing")
        if not isinstance(command.get("action"), str):
            raise ValueError("command action is missing")
        for key in ("target", "object", "destination"):
            binding = command.get(key)
            if binding is None:
                continue
            if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
                raise ValueError("invalid symbolic binding")
            if not isinstance(binding["label"], str) or not binding["label"]:
                raise ValueError("binding label is missing")
            if not isinstance(binding["roles"], list) or not binding["roles"]:
                raise ValueError("binding roles are missing")
            if not all(isinstance(role, str) and role for role in binding["roles"]):
                raise ValueError("binding roles must be non-empty strings")
    source = audit.get("source")
    expected_hash = source.get("commands_sha256") if isinstance(source, dict) else None
    actual_hash = hashlib.sha256(
        _canonical_commands_payload(commands).encode("utf-8")
    ).hexdigest()
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError("command hash mismatch")


def load_task_memory(memory_dir):
    """Load and validate one deterministic ``audit.json``/``commands.jsonl`` pair."""
    memory_path = Path(memory_dir)
    audit = json.loads((memory_path / "audit.json").read_text(encoding="utf-8"))
    commands_path = memory_path / "commands.jsonl"
    payload = commands_path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("commands JSONL is incomplete")
    try:
        commands = tuple(
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid commands JSONL") from error
    validate_task_memory(audit, commands)
    return audit, commands


def _normalize_symbol(value):
    return " ".join(str(value).strip().lower().split())


def _resolve_binding(binding, object_coords, object_labels, object_roles):
    required_label = _normalize_symbol(binding["label"])
    required_roles = {_normalize_symbol(role) for role in binding["roles"]}
    matches = []
    for object_id in object_coords:
        if object_id not in object_labels or object_id not in object_roles:
            continue
        label_matches = _normalize_symbol(object_labels[object_id]) == required_label
        current_roles = {_normalize_symbol(role) for role in object_roles[object_id]}
        if label_matches and required_roles.issubset(current_roles):
            matches.append(object_id)
    if not matches:
        raise ValueError(
            "no current grounding match for label {!r} with roles {}".format(
                binding["label"], sorted(binding["roles"])
            )
        )
    if len(matches) > 1:
        raise ValueError(
            "ambiguous current grounding for label {!r} with roles {}: {}".format(
                binding["label"], sorted(binding["roles"]), sorted(matches)
            )
        )
    return matches[0]


def resolve_task_memory_commands(
    commands: Sequence[Mapping],
    object_coords: Mapping[str, Sequence[float]],
    object_labels: Mapping[str, str],
    object_roles: Mapping[str, Sequence[str]],
    forbidden_output_keys=None,
) -> Tuple[Dict, ...]:
    """Resolve symbolic bindings against current grounding without mutating memory."""
    if _contains_forbidden_key(commands, forbidden_output_keys):
        raise ValueError("seed commands contain forbidden spatial data")
    resolved_commands = []
    for command in commands:
        resolved = dict(command)
        resolved_ids = {}
        for field in ("target", "object", "destination"):
            binding = command.get(field)
            if binding is None:
                continue
            object_id = _resolve_binding(
                binding, object_coords, object_labels, object_roles
            )
            resolved[field] = object_id
            resolved_ids[field] = object_id
        if command.get("action") == "move_to" and "target" in resolved_ids:
            xyz = object_coords[resolved_ids["target"]]
            if (
                not isinstance(xyz, (list, tuple))
                or len(xyz) != 3
                or not all(isinstance(value, (int, float)) for value in xyz)
            ):
                raise ValueError("current target coordinates must be numeric [x, y, z]")
            resolved["xyz"] = list(xyz)
        resolved_commands.append(resolved)
    return tuple(resolved_commands)


def build_task_memory(
    records,
    episode_result,
    trace_complete=True,
    *,
    symbolize: Callable[[Dict], Dict] = None,
    forbidden_output_keys=None,
):
    """Build memory only from a fully verified successful rollout.

    ``symbolize`` maps one executed record to a coordinate-free command
    (defaults to the LIBERO ``symbolize_record``).
    """
    symbolize = symbolize_record if symbolize is None else symbolize
    action_records = [
        record for record in records
        if record.get("status") != "initialization_reset"
    ]
    reasons: List[str] = []
    if float(episode_result.get("task_success", 0.0)) != 1.0:
        reasons.append("task_not_successful")
    if not trace_complete:
        reasons.append("trace_incomplete")
    expected_turns = episode_result.get("num_turns")
    if expected_turns is not None and int(expected_turns) != len(action_records):
        reasons.append("trace_turn_count_mismatch")
    structural_errors = {
        "parse_error",
        "compile_error",
        "semantic_error",
        "semantic_reject",
        "place_rejected",
        "no_progress_rejected",
    }
    if any(record.get("status") in structural_errors for record in action_records):
        reasons.append("structural_error_in_trace")

    executed = [record for record in action_records if record.get("primitive")]
    if not executed:
        reasons.append("no_executed_primitives")
    if any(record.get("primitive_postcondition_met") is not True for record in executed):
        reasons.append("unverified_primitive_postcondition")
    if not any(record.get("is_contact") is True for record in executed):
        reasons.append("no_verified_contact_primitive")

    commands = []
    if not reasons:
        try:
            for sequence, record in enumerate(executed, 1):
                command = symbolize(record)
                command["sequence"] = sequence
                commands.append(command)
        except ValueError as error:
            reasons.append(str(error))

    if reasons:
        return TaskMemoryDecision(False, tuple(sorted(set(reasons))))

    canonical_commands = _canonical_commands_payload(commands)
    canonical_trace = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    audit = {
        "schema_version": 1,
        "classification": "paper-compatible",
        "instruction": str(episode_result.get("instruction", "")),
        "source": {
            "turn_count": len(action_records),
            "env_step_count": sum(
                len(record.get("step_results", [])) for record in action_records
            ),
            "trace_sha256": hashlib.sha256(canonical_trace.encode("utf-8")).hexdigest(),
            "commands_sha256": hashlib.sha256(canonical_commands.encode("utf-8")).hexdigest(),
        },
        "promotion": {
            "task_success": True,
            "all_executed_postconditions_verified": True,
            "has_contact_primitive": True,
        },
        "command_count": len(commands),
    }
    try:
        validate_task_memory(audit, commands, forbidden_output_keys)
    except ValueError as error:
        return TaskMemoryDecision(False, (str(error),))
    return TaskMemoryDecision(True, (), audit=audit, commands=tuple(commands))


def promote_task_memory(trace_path, episode_result_path, output_dir):
    """Evaluate and, when accepted, write deterministic offline memory files."""
    trace_path = Path(trace_path)
    records = load_complete_jsonl(trace_path)
    episode_result = json.loads(Path(episode_result_path).read_text(encoding="utf-8"))
    decision = build_task_memory(
        records,
        episode_result,
        trace_complete=trace_path.read_bytes().endswith(b"\n"),
    )
    if not decision.accepted:
        return decision

    memory_path = Path(output_dir)
    memory_path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(memory_path / "audit.json", decision.audit)
    commands_payload = _canonical_commands_payload(decision.commands)
    write_text_atomic(memory_path / "commands.jsonl", commands_payload)
    return decision


def evaluate_memory_candidates(results_dir, output_path=None):
    """Evaluate every complete episode in a results directory without promotion."""
    results_path = Path(results_dir)
    episodes = []
    for result_path in sorted(
        results_path.glob("episode_*_res.json"),
        key=lambda path: int(path.name.split("_")[1]),
    ):
        episode = int(result_path.name.split("_")[1])
        trace_path = results_path / f"trace_episode_{episode}.jsonl"
        if not trace_path.exists():
            episodes.append({
                "episode": episode,
                "accepted": False,
                "reasons": ["missing_trace"],
            })
            continue
        records = load_complete_jsonl(trace_path)
        episode_result = json.loads(result_path.read_text(encoding="utf-8"))
        decision = build_task_memory(
            records,
            episode_result,
            trace_complete=trace_path.read_bytes().endswith(b"\n"),
        )
        episodes.append({
            "episode": episode,
            "accepted": decision.accepted,
            "reasons": list(decision.reasons),
        })
    report = {
        "schema_version": 1,
        "classification": "paper-compatible",
        "episode_count": len(episodes),
        "accepted_count": sum(item["accepted"] for item in episodes),
        "rejected_count": sum(not item["accepted"] for item in episodes),
        "episodes": episodes,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report