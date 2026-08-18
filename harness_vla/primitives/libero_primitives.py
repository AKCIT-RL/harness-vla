"""Analytic primitive compilation for native LIBERO OSC_POSE actions.

The installed controller consumes normalized
``[dx, dy, dz, ax, ay, az, gripper]`` deltas with 0.05 meter position and
0.5 radian rotation scales. Panda gripper commands are -1 open and +1 closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np


LIBERO_POSITION_SCALE = np.asarray([0.05, 0.05, 0.05], dtype=float)
LIBERO_ROTATION_SCALE = np.asarray([0.5, 0.5, 0.5], dtype=float)
LIBERO_GRIPPER_OPEN = -1.0
LIBERO_GRIPPER_CLOSED = 1.0
# Local safety guards; the paper specifies world-frame goals but not ranges.
LIBERO_XYZ_BOUNDS = ((-1.5, 1.5), (-1.5, 1.5), (0.0, 2.0))
LIBERO_YAW_BOUNDS = (-math.pi, math.pi)
LIBERO_PITCH_BOUNDS = (-math.pi / 2.0, math.pi / 2.0)


class LiberoPrimitiveError(ValueError):
    """Raised when a native LIBERO primitive cannot be compiled safely."""


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LiberoPrimitiveError(
            "%s must contain %d finite values" % (name, size)
        ) from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise LiberoPrimitiveError("%s must contain %d finite values" % (name, size))
    return vector


def _normalized_quaternion(value: Sequence[float], name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise LiberoPrimitiveError("%s must not be a zero quaternion" % name)
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left
    x0, y0, z0, w0 = right
    return np.asarray(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dtype=float,
    )


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _normalized_quaternion(quaternion, "relative quaternion")
    sine_half_angle = float(np.linalg.norm(quaternion[:3]))
    if sine_half_angle <= 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(sine_half_angle, float(quaternion[3]))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return quaternion[:3] * (angle / sine_half_angle)


def _quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = _normalized_quaternion(quaternion, "quaternion")
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalized_quaternion(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        "set-point quaternion",
    )


def validate_pose_target(
    xyz: Sequence[float], pose: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the local bounded world-frame quaternion pose contract."""
    target_xyz = _finite_vector(xyz, 3, "xyz")
    for value, bounds in zip(target_xyz, LIBERO_XYZ_BOUNDS):
        if not bounds[0] <= value <= bounds[1]:
            raise LiberoPrimitiveError("xyz is outside the guarded LIBERO workspace")
    quaternion = _finite_vector(pose, 4, "pose")
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-3):
        raise LiberoPrimitiveError("pose must be a unit quaternion [x, y, z, w]")
    return target_xyz, _normalized_quaternion(quaternion, "pose")


def validate_rotation_setpoint(axis: str, angle: float) -> float:
    """Validate a bounded absolute yaw or pitch set-point in radians."""
    if axis not in ("yaw", "pitch"):
        raise LiberoPrimitiveError("rotation axis must be yaw or pitch")
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise LiberoPrimitiveError("rotation set-point must be a finite number")
    angle = float(angle)
    bounds = LIBERO_YAW_BOUNDS if axis == "yaw" else LIBERO_PITCH_BOUNDS
    if not math.isfinite(angle) or not bounds[0] <= angle <= bounds[1]:
        raise LiberoPrimitiveError("rotation set-point is outside guarded bounds")
    return angle


def orientation_setpoint(
    observation: Dict[str, Any], *, axis: str, angle: float
) -> np.ndarray:
    """Replace one world-frame Euler component while preserving the others."""
    angle = validate_rotation_setpoint(axis, angle)
    roll, pitch, yaw = _quaternion_to_euler(
        LiberoPoseState.from_observation(observation).quaternion
    )
    if axis == "yaw":
        yaw = angle
    else:
        pitch = angle
    return _euler_to_quaternion(roll, pitch, yaw)


def _gripper_command(value: str) -> float:
    if value == "open":
        return LIBERO_GRIPPER_OPEN
    if value == "close":
        return LIBERO_GRIPPER_CLOSED
    raise LiberoPrimitiveError("gripper must be 'open' or 'close'")


@dataclass(frozen=True)
class LiberoPoseState:
    xyz: np.ndarray
    quaternion: np.ndarray

    @classmethod
    def from_observation(cls, observation: Dict[str, Any]) -> "LiberoPoseState":
        return cls(
            xyz=_finite_vector(observation["robot0_eef_pos"], 3, "robot0_eef_pos"),
            quaternion=_normalized_quaternion(
                observation["robot0_eef_quat"], "robot0_eef_quat"
            ),
        )


def compile_move_action(
    observation: Dict[str, Any],
    target_xyz: Sequence[float],
    *,
    gripper: str,
    target_quaternion: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compile one closed-loop OSC command toward an absolute world pose."""
    current = LiberoPoseState.from_observation(observation)
    target = _finite_vector(target_xyz, 3, "xyz")
    position_delta = np.clip(
        (target - current.xyz) / LIBERO_POSITION_SCALE, -1.0, 1.0
    )

    rotation_delta = np.zeros(3, dtype=float)
    if target_quaternion is not None:
        target_rotation = _normalized_quaternion(target_quaternion, "pose")
        current_inverse = np.asarray(
            [-current.quaternion[0], -current.quaternion[1],
             -current.quaternion[2], current.quaternion[3]],
            dtype=float,
        )
        relative = _quaternion_multiply(target_rotation, current_inverse)
        rotation_delta = np.clip(
            _quaternion_to_axis_angle(relative) / LIBERO_ROTATION_SCALE,
            -1.0,
            1.0,
        )

    return np.concatenate(
        (position_delta, rotation_delta, [_gripper_command(gripper)])
    )


def compile_gripper_action(gripper: str) -> np.ndarray:
    """Compile a stationary OSC action that drives the gripper set-point."""
    return np.asarray([0.0] * 6 + [_gripper_command(gripper)], dtype=float)


def pose_postcondition(
    observation: Dict[str, Any],
    target_xyz: Sequence[float],
    *,
    position_tolerance: float,
    target_quaternion: Optional[Sequence[float]] = None,
    rotation_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure native LIBERO pose error without conflating it with task success."""
    if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise LiberoPrimitiveError("position_tolerance must be positive and finite")
    current = LiberoPoseState.from_observation(observation)
    target = _finite_vector(target_xyz, 3, "xyz")
    position_error = float(np.linalg.norm(target - current.xyz))
    rotation_error = None
    rotation_met = True
    if target_quaternion is not None:
        if (
            rotation_tolerance is None
            or not math.isfinite(rotation_tolerance)
            or rotation_tolerance <= 0.0
        ):
            raise LiberoPrimitiveError(
                "rotation_tolerance must be positive and finite for move_pose"
            )
        target_rotation = _normalized_quaternion(target_quaternion, "pose")
        dot = float(abs(np.dot(target_rotation, current.quaternion)))
        rotation_error = 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
        rotation_met = rotation_error <= rotation_tolerance
    return {
        "position_error_m": position_error,
        "position_tolerance_m": float(position_tolerance),
        "rotation_error_rad": rotation_error,
        "rotation_tolerance_rad": rotation_tolerance,
        "postcondition_met": position_error <= position_tolerance and rotation_met,
    }


__all__ = [
    "LIBERO_GRIPPER_CLOSED",
    "LIBERO_GRIPPER_OPEN",
    "LIBERO_POSITION_SCALE",
    "LIBERO_PITCH_BOUNDS",
    "LIBERO_ROTATION_SCALE",
    "LIBERO_XYZ_BOUNDS",
    "LIBERO_YAW_BOUNDS",
    "LiberoPoseState",
    "LiberoPrimitiveError",
    "compile_gripper_action",
    "compile_move_action",
    "orientation_setpoint",
    "pose_postcondition",
    "validate_pose_target",
    "validate_rotation_setpoint",
]