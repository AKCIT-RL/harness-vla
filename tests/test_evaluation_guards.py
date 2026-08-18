"""Simulator-free tests for Harness evaluator pre-execution guards."""

from harness_vla.evaluation_guards import (
    NoProgressGuard,
    validate_vla_semantics,
)


COORDS = {"object 1": [10, 20, 30], "object 2": [40, 50, 60], "object 3": [1, 2, 3]}
ROLES = {
    "object 1": ["destination"],
    "object 2": ["destination"],
    "object 3": ["manipulable"],
}
LABELS = {"object 2": "container", "object 3": "cube"}


def test_grasp_requires_visible_manipulable_role_with_valid_example():
    rejection = validate_vla_semantics(
        {"action": "vla_act", "target": "object 2", "mode": "grasp"},
        COORDS, ROLES, None, LABELS,
    )
    assert rejection[0] == "semantic_error"
    assert "object 3 (cube)" in rejection[1]


def test_place_requires_roles_distinct_ids_and_matching_held_object():
    valid = {"action": "vla_act", "object": "object 3", "destination": "object 2", "mode": "place"}
    assert validate_vla_semantics(valid, COORDS, ROLES, "object 3", LABELS) is None

    wrong_role = dict(valid, object="object 1")
    assert validate_vla_semantics(wrong_role, COORDS, ROLES, "object 1", LABELS)[0] == "semantic_error"

    same = dict(valid, destination="object 3")
    assert validate_vla_semantics(same, COORDS, ROLES, "object 3", LABELS)[0] == "semantic_error"

    assert validate_vla_semantics(valid, COORDS, ROLES, None, LABELS)[0] == "place_rejected"


def test_legacy_place_is_rejected_but_legacy_grasp_and_push_remain_valid():
    legacy_place = {"action": "vla_act", "target": "object 2", "mode": "place"}
    rejection = validate_vla_semantics(legacy_place, COORDS, ROLES, None, LABELS)
    assert rejection[0] == "semantic_error"
    assert "legacy 'target' is not allowed" in rejection[1]
    assert validate_vla_semantics(
        {"action": "vla_act", "target": "object 3", "mode": "grasp"},
        COORDS, ROLES,
    ) is None
    assert validate_vla_semantics(
        {"action": "vla_act", "target": "object 3", "mode": "push"},
        COORDS, ROLES,
    ) is None


def test_no_progress_guard_rejects_fourth_identical_action_without_ending_episode():
    guard = NoProgressGuard(limit=3)
    invocation = {"action": "move_to", "target": "object 3"}
    zero_progress = [{"reward": 0, "task_success": 0}]
    for _ in range(3):
        assert not guard.should_reject(invocation)
        guard.observe_execution(invocation, zero_progress)
    assert guard.should_reject(invocation)
    assert not guard.should_reject({"action": "rotate_wrist", "target_yaw": 30})


def test_no_progress_guard_resets_on_progress_or_different_executed_action():
    guard = NoProgressGuard(limit=3)
    first = {"action": "move_to", "target": "object 3"}
    other = {"action": "rotate_wrist", "target_yaw": 30}
    for _ in range(3):
        guard.observe_execution(first, [{"reward": 0, "task_success": 0}])
    guard.observe_execution(other, [{"reward": 0, "task_success": 0}])
    assert not guard.should_reject(first)
    guard.observe_execution(first, [{"reward": 1, "task_success": 0}])
    assert not guard.should_reject(first)