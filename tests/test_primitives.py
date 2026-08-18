"""Unit tests for the fixed primitive library (no simulator required)."""

import pytest

from harness_vla.primitives.primitives import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PoseState,
    PrimitiveError,
    PrimitiveLibrary,
    VOXEL_SIZE,
    classify_grasp_outcome,
    classify_spatial_postcondition,
    primitive_termination,
    reconcile_held_object,
    summarize_physical_state,
)


@pytest.fixture
def lib():
    return PrimitiveLibrary(approach_dz=8, lift_dz=6)


@pytest.fixture
def coords():
    return {"object 1": [50, 60, 20], "object 2": [30, 40, 25]}


def test_primitive_set_is_fixed():
    assert PrimitiveLibrary.CONTACT_PRIMITIVE == "vla_act"
    assert set(PrimitiveLibrary.PRIMITIVES) == {
        "move_to",
        "rotate_wrist",
        "rotate_pitch",
        "set_gripper",
        "release",
        "vla_act",
    }


def test_move_to_explicit_xyz(lib):
    pose = PoseState(x=10, y=10, z=10, gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "move_to", "xyz": [50, 60, 20]}, pose)
    assert res.name == "move_to"
    assert res.actions == [[50, 60, 20, pose.roll, pose.pitch, pose.yaw, GRIPPER_OPEN]]
    assert not res.is_contact


def test_move_to_object_name(lib, coords):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "target": "object 2"}, pose, coords)
    assert res.actions[0][:3] == [30, 40, 25]


def test_move_to_tolerant_name(lib, coords):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "target": "Object2"}, pose, coords)
    assert res.actions[0][:3] == [30, 40, 25]


def test_move_to_gripper_override(lib):
    pose = PoseState(gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "move_to", "xyz": [1, 2, 3], "gripper": "close"}, pose)
    assert res.actions[0][6] == GRIPPER_CLOSED


def test_rotate_wrist_holds_position(lib):
    pose = PoseState(x=5, y=6, z=7, yaw=10)
    res = lib.compile({"action": "rotate_wrist", "target_yaw": 40}, pose)
    assert res.actions[0][:3] == [5, 6, 7]
    assert res.actions[0][5] == 40


def test_rotate_pitch_holds_position(lib):
    pose = PoseState(x=5, y=6, z=7, pitch=10)
    res = lib.compile({"action": "rotate_pitch", "target_pitch": 33}, pose)
    assert res.actions[0][4] == 33


def test_set_gripper(lib):
    pose = PoseState(gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "set_gripper", "gripper": "close"}, pose)
    assert res.actions[0][6] == GRIPPER_CLOSED


def test_release_opens_gripper(lib):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    res = lib.compile({"action": "release"}, pose)
    assert res.actions[-1][6] == GRIPPER_OPEN


def test_release_with_lift(lib):
    pose = PoseState(x=1, y=2, z=50, gripper=GRIPPER_CLOSED)
    res = lib.compile({"action": "release", "lift": True}, pose)
    assert len(res.actions) == 2
    assert res.actions[1][2] == 56  # lifted by lift_dz


def test_vla_act_grasp_sequence(lib, coords):
    pose = PoseState(x=0, y=0, z=0, gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "vla_act", "target": "object 1", "mode": "grasp"}, pose, coords)
    assert res.is_contact
    assert res.meta["mode"] == "grasp"
    # 4 sub-actions: approach(open), descend(open), close, lift
    assert len(res.actions) == 4
    approach, descend, close, lift = res.actions
    assert approach[2] == 20 + 8 and approach[6] == GRIPPER_OPEN
    assert descend[2] == 20 and descend[6] == GRIPPER_OPEN
    assert close[6] == GRIPPER_CLOSED
    assert lift[2] == 20 + 6 and lift[6] == GRIPPER_CLOSED


def test_vla_act_canonical_grasp_uses_object_and_meta(lib, coords):
    res = lib.compile(
        {"action": "vla_act", "object": "object 1", "mode": "grasp"},
        PoseState(),
        coords,
    )
    assert res.meta["object_id"] == "object 1"
    assert res.meta["destination_id"] is None


def test_vla_act_place_sequence(lib, coords):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    res = lib.compile(
        {"action": "vla_act", "object": "object 1", "destination": "object 2", "mode": "place"},
        pose,
        coords,
    )
    assert res.meta["mode"] == "place"
    assert len(res.actions) == 3
    assert res.actions[-1][6] == GRIPPER_OPEN  # release at the end


def test_vla_act_canonical_place_uses_distinct_destination(lib, coords):
    res = lib.compile(
        {
            "action": "vla_act",
            "object": "object 1",
            "destination": "object 2",
            "mode": "place",
        },
        PoseState(gripper=GRIPPER_CLOSED),
        coords,
    )
    assert res.meta["object_id"] == "object 1"
    assert res.meta["destination_id"] == "object 2"
    assert res.actions[1][:3] == coords["object 2"]


def test_vla_act_canonical_place_rejects_same_object(lib, coords):
    with pytest.raises(PrimitiveError, match="must be different"):
        lib.compile(
            {
                "action": "vla_act",
                "object": "object 1",
                "destination": "object 1",
                "mode": "place",
            },
            PoseState(gripper=GRIPPER_CLOSED),
            coords,
        )


def test_vla_act_prompt_implies_place(lib, coords):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    with pytest.raises(PrimitiveError, match="legacy 'target' is not allowed"):
        lib.compile(
            {"action": "vla_act", "target": "object 1", "prompt": "place the cube in the box"},
            pose,
            coords,
        )


def test_vla_act_legacy_place_target_is_never_accepted(lib, coords):
    with pytest.raises(PrimitiveError, match="requires both 'object' and 'destination'"):
        lib.compile(
            {"action": "vla_act", "target": "object 2", "mode": "place"},
            PoseState(gripper=GRIPPER_CLOSED),
            coords,
        )


def test_vla_act_push(lib, coords):
    pose = PoseState()
    res = lib.compile(
        {"action": "vla_act", "target": "object 1", "mode": "push", "direction": [5, 0, 0]},
        pose,
        coords,
    )
    assert res.meta["mode"] == "push"
    assert res.actions[-1][0] == 55


def test_unknown_primitive_raises(lib):
    with pytest.raises(PrimitiveError):
        lib.compile({"action": "teleport", "xyz": [1, 2, 3]}, PoseState())


def test_missing_action_raises(lib):
    with pytest.raises(PrimitiveError):
        lib.compile({"xyz": [1, 2, 3]}, PoseState())


def test_unknown_target_raises(lib, coords):
    with pytest.raises(PrimitiveError):
        lib.compile({"action": "move_to", "target": "object 9"}, PoseState(), coords)


def test_clamping_within_bounds(lib):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "xyz": [999, -5, 50]}, pose)
    assert res.actions[0][0] == VOXEL_SIZE
    assert res.actions[0][1] == 0


def test_actions_are_plain_ints(lib, coords):
    res = lib.compile({"action": "vla_act", "target": "object 1"}, PoseState(), coords)
    for action in res.actions:
        assert all(isinstance(v, int) for v in action)
        assert len(action) == 7


def test_classify_grasp_attachment_target_has_priority():
    result = classify_grasp_outcome(
        "object 1", ["cube_basic0"], [0, 0, 0], [0, 0, 0], [0, 0, 0], [50, 50, 50],
        target_sim_name="cube_basic0",
    )
    assert result["outcome"] == "grasp_verified"
    assert result["classification_source"] == "attachment"
    assert result["object_lift"] == 0
    assert result["gripper_lift"] == 50
    assert result["geometry_consistent_with_attachment"] is False


def test_classify_grasp_attachment_keeps_consistent_geometry_metrics():
    result = classify_grasp_outcome(
        "object 1", ["cube_basic0"], [10, 10, 10], [10, 10, 16],
        [10, 10, 10], [10, 10, 16], target_sim_name="cube_basic0",
    )
    assert result["outcome"] == "grasp_verified"
    assert result["classification_source"] == "attachment"
    assert result["object_lift"] == 6
    assert result["comotion_residual"] == 0
    assert result["geometry_consistent_with_attachment"] is True


def test_classify_grasp_wrong_attachment():
    result = classify_grasp_outcome(
        "object 1", ["cube_basic1"], target_sim_name="cube_basic0"
    )
    assert result["outcome"] == "grasp_unverified"
    assert result["reason"] == "wrong_object_attached"


def test_classify_grasp_attachment_matches_sim_name_mapping():
    result = classify_grasp_outcome(
        "object 7", ["cube_basic0"], target_sim_name="cube_basic0"
    )
    assert result["outcome"] == "grasp_verified"
    assert result["classification_source"] == "attachment"


def test_classify_grasp_attachment_matches_visual_shape_to_physical_body():
    result = classify_grasp_outcome(
        "object 3", ["star_normal0"], target_sim_name="star_normal_visual0"
    )
    assert result["outcome"] == "grasp_verified"
    assert result["reason"] == "target_attached"
    assert result["matched_grasped_object_name"] == "star_normal0"


def test_unmapped_attachment_falls_back_to_geometry():
    result = classify_grasp_outcome(
        "object 1", ["cube_basic0"], [10, 10, 10], [10, 10, 16],
        [10, 10, 10], [10, 10, 16],
    )
    assert result["outcome"] == "grasp_verified"
    assert result["classification_source"] == "geometry"


def test_geometry_rejects_large_comotion_residual():
    result = classify_grasp_outcome(
        "object 1", [], [10, 10, 10], [15, 10, 16],
        [10, 10, 10], [10, 10, 16], max_comotion_residual=2.0,
    )
    assert result["outcome"] == "grasp_unverified"
    assert result["comotion_residual"] == 5


def test_classify_grasp_geometry_verified():
    result = classify_grasp_outcome(
        "object 1", [], [10, 10, 10], [10, 10, 16], [10, 10, 10], [10, 10, 16]
    )
    assert result["outcome"] == "grasp_verified"
    assert result["object_lift"] == 6


def test_classify_empty_grasp_geometry():
    result = classify_grasp_outcome(
        "object 1", [], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 16]
    )
    assert result["outcome"] == "empty_grasp"


def test_classify_grasp_missing_evidence_is_unverified():
    result = classify_grasp_outcome("object 1")
    assert result["outcome"] == "grasp_unverified"
    assert result["geometry_consistent_with_attachment"] is None


def test_spatial_postcondition_reports_distance_and_tolerance():
    result = classify_spatial_postcondition([10, 10, 10], [11, 12, 10], 3.0)
    assert result["postcondition_met"] is True
    assert result["distance"] == pytest.approx(5 ** 0.5)
    assert result["reason"] == "within_tolerance"


def test_spatial_postcondition_rejects_distant_or_missing_observation():
    distant = classify_spatial_postcondition([10, 10, 10], [20, 10, 10], 2.0)
    missing = classify_spatial_postcondition([10, 10, 10], None, 2.0)
    assert distant["postcondition_met"] is False
    assert distant["reason"] == "outside_tolerance"
    assert missing["postcondition_met"] is None
    assert missing["reason"] == "missing_spatial_evidence"


@pytest.mark.parametrize(
    "spatial_met, expected",
    [
        (True, ("postcondition_met", True)),
        (False, ("target_pose_not_reached", False)),
        (None, ("unverified", False)),
    ],
)
def test_move_to_termination_requires_spatial_postcondition(spatial_met, expected):
    assert primitive_termination(
        None,
        primitive_name="move_to",
        spatial_postcondition_met=spatial_met,
    ) == expected


def test_physical_state_tracks_only_verified_manipulable_objects():
    state = summarize_physical_state(
        {
            "object 1": ["manipulable"],
            "object 2": ["manipulable"],
            "object 3": ["destination"],
        },
        held_object_id="object 2",
        placed_object_ids=["object 1", "object 3"],
    )
    assert state == {
        "held": "object 2",
        "placed": ["object 1"],
        "remaining": [],
    }


ID_TO_SIM = {"object 1": "star_visual", "object 2": "cube_visual"}


def test_reconcile_clears_stale_held_object_after_detach():
    held, available = reconcile_held_object("object 1", [], True, ID_TO_SIM)
    assert held is None
    assert available is True


def test_reconcile_keeps_held_object_while_attached():
    held, available = reconcile_held_object("object 1", ["star"], True, ID_TO_SIM)
    assert held == "object 1"
    assert available is True


def test_reconcile_adopts_attached_object_when_untracked():
    held, available = reconcile_held_object(None, ["cube"], True, ID_TO_SIM)
    assert held == "object 2"
    assert available is True


def test_reconcile_without_evidence_preserves_tracking():
    held, available = reconcile_held_object("object 1", [], False, ID_TO_SIM)
    assert held == "object 1"
    assert available is False


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("grasp_verified", ("postcondition_met", True)),
        ("empty_grasp", ("empty_grasp", False)),
        ("grasp_unverified", ("unverified", False)),
    ],
)
def test_grasp_termination_follows_outcome(outcome, expected):
    assert primitive_termination("grasp", grasp_outcome=outcome) == expected


def test_place_terminated_before_release_fails_postcondition():
    assert primitive_termination(
        "place", env_done=True, release_executed=False,
    ) == ("environment_terminated_before_release", False)


def test_place_release_with_attachment_api_proves_postcondition():
    assert primitive_termination(
        "place",
        env_done=True,
        release_executed=True,
        attachment_evidence_available=True,
        grasped_object_names=[],
        spatial_postcondition_met=True,
    ) == ("postcondition_met", True)


def test_place_release_outside_destination_fails_postcondition():
    assert primitive_termination(
        "place",
        release_executed=True,
        attachment_evidence_available=True,
        grasped_object_names=[],
        spatial_postcondition_met=False,
    ) == ("released_outside_destination_tolerance", False)


def test_place_release_without_attachment_api_is_unverified():
    assert primitive_termination(
        "place", release_executed=True, attachment_evidence_available=False,
    ) == ("unverified", False)


def test_release_requires_confirmed_detachment():
    assert primitive_termination(
        None,
        primitive_name="release",
        release_executed=True,
        attachment_evidence_available=True,
        grasped_object_names=[],
    ) == ("postcondition_met", True)
    assert primitive_termination(
        None,
        primitive_name="release",
        release_executed=True,
        attachment_evidence_available=True,
        grasped_object_names=["cube_basic0"],
    ) == ("unverified", False)
