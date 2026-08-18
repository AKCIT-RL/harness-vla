import math

import numpy as np
import pytest

from harness_vla.primitives.libero_primitives import (
    LiberoPrimitiveError,
    compile_gripper_action,
    compile_move_action,
    orientation_setpoint,
    pose_postcondition,
    validate_pose_target,
)


def _observation(xyz=(0.0, 0.0, 0.5), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        "robot0_eef_pos": np.asarray(xyz, dtype=float),
        "robot0_eef_quat": np.asarray(quaternion, dtype=float),
    }


def test_move_to_uses_installed_osc_scale_and_preserves_orientation():
    action = compile_move_action(
        _observation(), [0.025, -0.1, 0.5], gripper="close"
    )

    np.testing.assert_allclose(action, [0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_move_pose_uses_world_frame_rotation_error_and_shortest_quaternion():
    half_angle = math.pi / 8.0
    target = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

    action = compile_move_action(
        _observation(), [0.0, 0.0, 0.5],
        target_quaternion=target, gripper="open"
    )

    np.testing.assert_allclose(action[:3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 1.0], atol=1e-9)
    assert action[6] == -1.0


def test_move_pose_clips_large_rotation_delta():
    half_angle = math.pi / 2.0
    target = [math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)]

    action = compile_move_action(
        _observation(), [0.0, 0.0, 0.5],
        target_quaternion=target, gripper="close"
    )

    np.testing.assert_allclose(action[3:6], [1.0, 0.0, 0.0], atol=1e-9)


def test_gripper_commands_match_panda_open_close_convention():
    np.testing.assert_allclose(compile_gripper_action("open"), [0.0] * 6 + [-1.0])
    np.testing.assert_allclose(compile_gripper_action("close"), [0.0] * 6 + [1.0])


def test_rotation_setpoints_preserve_other_orientation_components():
    yaw = orientation_setpoint(_observation(), axis="yaw", angle=math.pi / 2.0)
    pitch = orientation_setpoint(_observation(), axis="pitch", angle=-math.pi / 4.0)

    np.testing.assert_allclose(
        yaw, [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    )
    np.testing.assert_allclose(
        pitch, [0.0, math.sin(-math.pi / 8.0), 0.0, math.cos(math.pi / 8.0)]
    )


def test_move_pose_contract_requires_bounded_xyz_and_unit_quaternion():
    xyz, pose = validate_pose_target([0.1, -0.2, 1.05], [0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(xyz, [0.1, -0.2, 1.05])
    np.testing.assert_allclose(pose, [0.0, 0.0, 0.0, 1.0])

    with pytest.raises(LiberoPrimitiveError):
        validate_pose_target([2.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(LiberoPrimitiveError):
        validate_pose_target([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 2.0])


def test_pose_postcondition_reports_position_and_rotation_separately():
    result = pose_postcondition(
        _observation(xyz=(0.001, 0.0, 0.5)),
        [0.0, 0.0, 0.5],
        position_tolerance=0.002,
        target_quaternion=[0.0, 0.0, 0.0, -1.0],
        rotation_tolerance=0.01,
    )

    assert result["postcondition_met"] is True
    assert result["position_error_m"] == pytest.approx(0.001)
    assert result["rotation_error_rad"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: compile_move_action(_observation(), [0.0, float("nan"), 0.5], gripper="close"),
        lambda: compile_move_action(_observation(), [0.0, 0.0, 0.5], gripper="hold"),
        lambda: pose_postcondition(_observation(), [0.0, 0.0, 0.5], position_tolerance=0.0),
    ],
)
def test_invalid_primitive_inputs_fail_closed(call):
    with pytest.raises(LiberoPrimitiveError):
        call()