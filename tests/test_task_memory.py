"""Simulator-free tests for offline Task Specific Memory generation."""

import json

import pytest

from harness_vla.memory.task_memory import (
    FORBIDDEN_OUTPUT_KEYS,
    build_task_memory,
    evaluate_memory_candidates,
    load_task_memory,
    promote_task_memory,
    resolve_task_memory_commands,
    validate_task_memory,
)
from harness_vla.planner.prompts import build_turn_prompt


def _record(turn, invocation, primitive, contact=False):
    return {
        "turn": turn,
        "invocation": invocation,
        "primitive": primitive,
        "is_contact": contact,
        "primitive_postcondition_met": True,
        "status": "postcondition_met",
        "object_roles": {
            "object 1": ["manipulable"],
            "object 2": ["destination"],
        },
        "object_labels": {
            "object 1": "red cube",
            "object 2": "silver container",
        },
        "step_results": [{"task_success": 0.0}],
    }


def _verified_rollout():
    return [
        _record(1, {"action": "move_to", "target": "object 1"}, "move_to"),
        _record(
            2,
            {"action": "vla_act", "object": "object 1", "mode": "grasp"},
            "vla_act",
            contact=True,
        ),
        _record(
            3,
            {
                "action": "vla_act",
                "object": "object 1",
                "destination": "object 2",
                "mode": "place",
            },
            "vla_act",
            contact=True,
        ),
    ]


def _successful_result():
    return {"task_success": 1.0, "instruction": "put the red cube in the container"}


def _all_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(child) for child in value.values()), set())
    if isinstance(value, list) or isinstance(value, tuple):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()


def _current_grounding(red_xyz=(11, 12, 13), container_xyz=(21, 22, 23)):
    object_coords = {
        "current red": list(red_xyz),
        "current container": list(container_xyz),
    }
    object_labels = {
        "current red": " Red   Cube ",
        "current container": "silver container",
    }
    object_roles = {
        "current red": ["manipulable", "visible"],
        "current container": ["destination", "visible"],
    }
    return object_coords, object_labels, object_roles


def test_verified_rollout_generates_ordered_symbolic_memory():
    decision = build_task_memory(_verified_rollout(), _successful_result())
    assert decision.accepted is True
    assert [command["sequence"] for command in decision.commands] == [1, 2, 3]
    assert decision.commands[0]["target"] == {
        "label": "red cube",
        "roles": ["manipulable"],
    }
    assert decision.commands[2]["destination"]["label"] == "silver container"
    assert not (_all_keys(decision.commands) & FORBIDDEN_OUTPUT_KEYS)


def test_success_signal_does_not_promote_unverified_contact():
    records = _verified_rollout()
    records[1]["primitive_postcondition_met"] = False
    records[1]["status"] = "grasp_unverified"
    decision = build_task_memory(records, _successful_result())
    assert decision.accepted is False
    assert "unverified_primitive_postcondition" in decision.reasons


def test_literal_xyz_rejects_entire_rollout():
    records = _verified_rollout()
    records[0]["invocation"] = {"action": "move_to", "xyz": [10, 20, 30]}
    decision = build_task_memory(records, _successful_result())
    assert decision.accepted is False
    assert "literal xyz cannot be promoted to symbolic memory" in decision.reasons


def test_failed_task_is_never_promoted():
    decision = build_task_memory(_verified_rollout(), {"task_success": 0.0})
    assert decision.accepted is False
    assert decision.reasons == ("task_not_successful",)


def test_generation_is_deterministic_and_writes_one_command_per_line(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result_path = tmp_path / "episode.json"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _verified_rollout()),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(_successful_result()), encoding="utf-8")

    first = promote_task_memory(trace_path, result_path, tmp_path / "first")
    second = promote_task_memory(trace_path, result_path, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first" / "audit.json").read_bytes() == (
        tmp_path / "second" / "audit.json"
    ).read_bytes()
    lines = (tmp_path / "first" / "commands.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_schema_rejects_unknown_or_spatial_command_fields():
    decision = build_task_memory(_verified_rollout(), _successful_result())
    contaminated = [dict(command) for command in decision.commands]
    contaminated[0]["xyz"] = [1, 2, 3]
    try:
        validate_task_memory(decision.audit, contaminated)
    except ValueError as error:
        assert str(error) == "forbidden spatial data"
    else:
        raise AssertionError("spatial command fields must be rejected")


def test_candidate_report_preserves_rejection_reasons(tmp_path):
    results_path = tmp_path / "results"
    results_path.mkdir()
    (results_path / "episode_1_res.json").write_text(
        json.dumps(_successful_result()), encoding="utf-8"
    )
    (results_path / "trace_episode_1.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in _verified_rollout()),
        encoding="utf-8",
    )
    (results_path / "episode_2_res.json").write_text(
        json.dumps({"task_success": 0.0}), encoding="utf-8"
    )
    output_path = tmp_path / "promotion_report.json"

    report = evaluate_memory_candidates(results_path, output_path)
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 1
    assert report["episodes"][1] == {
        "episode": 2,
        "accepted": False,
        "reasons": ["missing_trace"],
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_semantic_guard_status_rejects_otherwise_successful_rollout():
    records = _verified_rollout()
    records.insert(1, {"turn": 2, "status": "semantic_error"})
    result = dict(_successful_result(), num_turns=4)
    decision = build_task_memory(records, result)
    assert decision.accepted is False
    assert "structural_error_in_trace" in decision.reasons


def test_truncated_trace_with_success_result_is_rejected(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result_path = tmp_path / "episode.json"
    records = _verified_rollout()
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
        + '{"turn":4,"status":',
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(dict(_successful_result(), num_turns=4)), encoding="utf-8"
    )
    decision = promote_task_memory(trace_path, result_path, tmp_path / "memory")
    assert decision.accepted is False
    assert "trace_incomplete" in decision.reasons
    assert "trace_turn_count_mismatch" in decision.reasons
    assert not (tmp_path / "memory" / "audit.json").exists()


def test_binding_roles_must_be_non_empty_strings():
    decision = build_task_memory(_verified_rollout(), _successful_result())
    contaminated = [dict(command) for command in decision.commands]
    contaminated[0] = dict(contaminated[0])
    contaminated[0]["target"] = {"label": "red cube", "roles": [1]}
    try:
        validate_task_memory(decision.audit, contaminated)
    except ValueError as error:
        assert str(error) == "binding roles must be non-empty strings"
    else:
        raise AssertionError("non-string binding roles must be rejected")


def test_position_swap_uses_current_xyz_without_mutating_seed_commands():
    decision = build_task_memory(_verified_rollout(), _successful_result())
    original = tuple(json.loads(json.dumps(command)) for command in decision.commands)
    first = resolve_task_memory_commands(
        decision.commands, *_current_grounding(red_xyz=(1, 2, 3))
    )
    swapped = resolve_task_memory_commands(
        decision.commands, *_current_grounding(red_xyz=(21, 22, 23), container_xyz=(1, 2, 3))
    )

    assert first[0]["xyz"] == [1, 2, 3]
    assert swapped[0]["xyz"] == [21, 22, 23]
    assert decision.commands == original
    assert first[1]["object"] == "current red"
    assert first[2]["destination"] == "current container"
    assert all("xyz" not in command for command in first[1:])


def test_resolver_rejects_missing_object():
    object_coords, object_labels, object_roles = _current_grounding()
    del object_coords["current red"]
    del object_labels["current red"]
    del object_roles["current red"]
    commands = build_task_memory(_verified_rollout(), _successful_result()).commands

    with pytest.raises(ValueError, match="no current grounding match"):
        resolve_task_memory_commands(
            commands, object_coords, object_labels, object_roles
        )


def test_resolver_rejects_ambiguous_binding():
    object_coords, object_labels, object_roles = _current_grounding()
    object_coords["other red"] = [31, 32, 33]
    object_labels["other red"] = "red cube"
    object_roles["other red"] = ["manipulable"]
    commands = build_task_memory(_verified_rollout(), _successful_result()).commands

    with pytest.raises(ValueError, match="ambiguous current grounding"):
        resolve_task_memory_commands(
            commands, object_coords, object_labels, object_roles
        )


def test_resolver_requires_all_binding_roles():
    object_coords, object_labels, object_roles = _current_grounding()
    object_roles["current red"] = ["visible"]
    commands = build_task_memory(_verified_rollout(), _successful_result()).commands

    with pytest.raises(ValueError, match="no current grounding match"):
        resolve_task_memory_commands(
            commands, object_coords, object_labels, object_roles
        )


def test_loader_rejects_tampered_command_hash(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result_path = tmp_path / "episode.json"
    memory_path = tmp_path / "memory"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _verified_rollout()),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(_successful_result()), encoding="utf-8")
    promote_task_memory(trace_path, result_path, memory_path)
    commands_path = memory_path / "commands.jsonl"
    lines = commands_path.read_text(encoding="utf-8").splitlines()
    command = json.loads(lines[0])
    command["action"] = "release"
    lines[0] = json.dumps(command, sort_keys=True, separators=(",", ":"))
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="command hash mismatch"):
        load_task_memory(memory_path)


def test_loader_and_resolver_preserve_command_order(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result_path = tmp_path / "episode.json"
    memory_path = tmp_path / "memory"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _verified_rollout()),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(_successful_result()), encoding="utf-8")
    promote_task_memory(trace_path, result_path, memory_path)

    audit, commands = load_task_memory(memory_path)
    resolved = resolve_task_memory_commands(commands, *_current_grounding())

    assert audit["command_count"] == 3
    assert [command["sequence"] for command in commands] == [1, 2, 3]
    assert [command["sequence"] for command in resolved] == [1, 2, 3]
    assert [command["action"] for command in resolved] == [
        "move_to",
        "vla_act",
        "vla_act",
    ]


def test_loaded_memory_prompt_contains_only_current_scene_grounding(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result_path = tmp_path / "episode.json"
    memory_path = tmp_path / "memory"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _verified_rollout()),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(_successful_result()), encoding="utf-8")
    promote_task_memory(trace_path, result_path, memory_path)
    _, commands = load_task_memory(memory_path)
    current_coords, current_labels, current_roles = _current_grounding(
        red_xyz=(71, 72, 73), container_xyz=(81, 82, 83)
    )

    resolved = resolve_task_memory_commands(
        commands, current_coords, current_labels, current_roles
    )
    prompt = build_turn_prompt(
        "put the red cube in the container",
        current_coords,
        [0, 0, 0, 0, 0, 0, 1],
        [],
        object_roles=current_roles,
        object_labels=current_labels,
        resolved_task_memory=resolved,
    )

    assert '"target": "current red"' in prompt
    assert '"xyz": [71, 72, 73]' in prompt
    assert "object 1" not in prompt
    assert "object 2" not in prompt


def test_resolver_rejects_seed_coordinates():
    commands = [
        {
            "sequence": 1,
            "source_turn": 1,
            "action": "move_to",
            "target": {"label": "red cube", "roles": ["manipulable"]},
            "xyz": [99, 99, 99],
        }
    ]

    with pytest.raises(ValueError, match="seed commands contain forbidden spatial data"):
        resolve_task_memory_commands(commands, *_current_grounding())


def test_loader_rejects_invalid_utf8_commands(tmp_path):
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    (memory_path / "audit.json").write_text("{}", encoding="utf-8")
    (memory_path / "commands.jsonl").write_bytes(b"\xff\xfe\n")

    with pytest.raises(ValueError, match="invalid commands JSONL"):
        load_task_memory(memory_path)


def test_resolver_rejects_invalid_current_coordinates():
    commands = [{
        "sequence": 1,
        "source_turn": 1,
        "action": "move_to",
        "target": {"label": "red cube", "roles": ["manipulable"]},
    }]

    with pytest.raises(ValueError, match="current target coordinates"):
        resolve_task_memory_commands(
            commands,
            {"current cube": [1, 2]},
            {"current cube": "red cube"},
            {"current cube": ["manipulable"]},
        )