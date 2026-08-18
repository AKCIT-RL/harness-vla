"""Injectable native LIBERO adapters for the multi-turn evaluator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from harness_vla.eval.libero_analytic_executor import (
    LiberoPrimitiveExecution,
    execute_gripper_primitive,
    execute_pose_primitive,
    execute_release_primitive,
)
from harness_vla.eval.libero_multi_turn_evaluator import LiberoVLAExecution
from harness_vla.eval.libero_vla_smoke import (
    _NativeLiftAndGraspMonitor,
    prepare_pirlinf_observation,
)
from harness_vla.primitives.libero_primitives import (
    LiberoPoseState,
    LiberoPrimitiveError,
    orientation_setpoint,
    validate_pose_target,
)
from harness_vla.primitives.libero_tau import read_bilateral_contact


Grounder = Callable[[Mapping[str, Any], str], Mapping[str, Any]]
FrameCallback = Callable[[Mapping[str, Any]], None]
LIBERO_ROTATION_TOLERANCE_RAD = 0.05


def _positive_finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LiberoPrimitiveError("%s must be finite and greater than zero" % name) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise LiberoPrimitiveError("%s must be finite and greater than zero" % name)
    return number


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


@dataclass(frozen=True)
class LiberoNativeOffsets:
    above_m: float
    release_pose_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "above_m", _positive_finite(self.above_m, "above_m"))
        object.__setattr__(
            self,
            "release_pose_m",
            _positive_finite(self.release_pose_m, "release_pose_m"),
        )


@dataclass(frozen=True)
class LiberoResolvedTarget:
    target: str
    mode: str
    xyz: tuple[float, float, float]
    provenance: Dict[str, Any]


@dataclass
class LiberoNativeExecutionState:
    holding: Optional[str] = None


class VisualLiftAndGraspMonitor:
    """Infer lift-and-grasp from repeated visual RGB-D target grounding."""

    def __init__(
        self,
        grounder: Grounder,
        observation: Mapping[str, Any],
        target: str,
        minimum_lift_m: float = 0.03,
    ) -> None:
        self.grounder = grounder
        self.target = target
        self.minimum_lift_m = _positive_finite(minimum_lift_m, "minimum_lift_m")
        self.baseline_z = float(grounder(observation, target)["world_xyz"][2])

    def evaluate(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        current_z = float(self.grounder(observation, self.target)["world_xyz"][2])
        delta_z = current_z - self.baseline_z
        return {
            "predicate": "visual_lift_and_grasp",
            "tau_satisfied": delta_z >= self.minimum_lift_m,
            "target": self.target,
            "lift": {
                "baseline_target_z_m": self.baseline_z,
                "current_target_z_m": current_z,
                "delta_z_m": delta_z,
                "minimum_lift_m": self.minimum_lift_m,
                "threshold_met": delta_z >= self.minimum_lift_m,
                "coordinate_source": "visual_rgbd_projection",
            },
            "privileged_contact_state": False,
            "task_success_evaluated": False,
        }


def resolve_target_xyz(
    observation: Mapping[str, Any],
    target: str,
    mode: str,
    *,
    grounder: Grounder,
    offsets: LiberoNativeOffsets,
) -> LiberoResolvedTarget:
    """Ground one target and apply the configured vertical mode offset."""
    if mode not in ("above", "release_pose"):
        raise LiberoPrimitiveError("unsupported move mode %r" % mode)
    grounded = grounder(observation, target)
    try:
        base_xyz = np.asarray(grounded["world_xyz"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise LiberoPrimitiveError("grounder must return finite world_xyz") from exc
    if base_xyz.shape != (3,) or not np.all(np.isfinite(base_xyz)):
        raise LiberoPrimitiveError("grounder must return finite world_xyz")
    offset = offsets.above_m if mode == "above" else offsets.release_pose_m
    resolved = base_xyz + np.asarray([0.0, 0.0, offset])
    provenance = _json_safe(grounded.get("provenance", {}))
    provenance.update(
        {
            "target": target,
            "mode": mode,
            "base_world_xyz": base_xyz.tolist(),
            "vertical_offset_m": offset,
            "resolved_world_xyz": resolved.tolist(),
        }
    )
    return LiberoResolvedTarget(
        target=target,
        mode=mode,
        xyz=tuple(float(value) for value in resolved),
        provenance=provenance,
    )


class _FrameCaptureEnv:
    def __init__(self, env, frame_callback: Optional[FrameCallback]):
        self._env = env
        self._frame_callback = frame_callback

    def step(self, action):
        result = self._env.step(action)
        if self._frame_callback is not None:
            self._frame_callback(result[0])
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_native_move_executor(
    env,
    grounder: Grounder,
    *,
    offsets: LiberoNativeOffsets,
    position_tolerance: float,
    rotation_tolerance: float = LIBERO_ROTATION_TOLERANCE_RAD,
    execution_state: Optional[LiberoNativeExecutionState] = None,
    grasp_monitor=None,
    frame_callback: Optional[FrameCallback] = None,
):
    """Build a move adapter that re-grounds immediately before every call."""
    tolerance = _positive_finite(position_tolerance, "position_tolerance")
    angular_tolerance = _positive_finite(rotation_tolerance, "rotation_tolerance")
    captured_env = _FrameCaptureEnv(env, frame_callback)

    def preserve_grasp(current_observation):
        if execution_state is None or execution_state.holding is None:
            return None
        if grasp_monitor is False:
            return None
        if grasp_monitor is None:
            evidence = read_bilateral_contact(env, execution_state.holding)
            preserved = bool(evidence["bilateral_contact"])
        else:
            preserved = bool(
                grasp_monitor(env, current_observation, execution_state.holding)
            )
        return None if preserved else "grasp_lost"

    def execute(invocation, observation, *, max_steps):
        action = invocation.get("action")
        if action == "set_gripper":
            execution = execute_gripper_primitive(
                captured_env,
                observation,
                gripper=invocation["gripper"],
                max_steps=max_steps,
            )
            if (
                execution.primitive_success
                and invocation["gripper"] == "open"
                and execution_state is not None
            ):
                execution_state.holding = None
            return execution

        target_quaternion = None
        rotation_tolerance_for_call = None
        if action == "move_to":
            if invocation.get("gripper") != "close":
                raise LiberoPrimitiveError("native move requires gripper close")
            resolved = resolve_target_xyz(
                observation,
                invocation["target"],
                invocation["mode"],
                grounder=grounder,
                offsets=offsets,
            )
            target_xyz = resolved.xyz
            gripper = "close"
        elif action == "move_pose":
            target_xyz, target_quaternion = validate_pose_target(
                invocation["xyz"], invocation["pose"]
            )
            gripper = invocation["gripper"]
            rotation_tolerance_for_call = angular_tolerance
            resolved = None
        elif action in ("rotate_wrist", "rotate_pitch"):
            axis = "yaw" if action == "rotate_wrist" else "pitch"
            field = "target_yaw" if axis == "yaw" else "target_pitch"
            pose_state = LiberoPoseState.from_observation(observation)
            target_xyz = pose_state.xyz
            target_quaternion = orientation_setpoint(
                observation, axis=axis, angle=invocation[field]
            )
            gripper = (
                "close"
                if execution_state is not None and execution_state.holding is not None
                else "open"
            )
            rotation_tolerance_for_call = angular_tolerance
            resolved = None
        else:
            raise LiberoPrimitiveError("unsupported native analytic primitive")
        execution = execute_pose_primitive(
            captured_env,
            observation,
            target_xyz,
            gripper=gripper,
            max_steps=max_steps,
            position_tolerance=tolerance,
            target_quaternion=target_quaternion,
            rotation_tolerance=rotation_tolerance_for_call,
            post_step_guard=preserve_grasp,
        )
        if resolved is None:
            return execution
        grounding_trace = {
            "event": "target_resolved",
            "target": resolved.target,
            "mode": resolved.mode,
            "target_xyz": list(resolved.xyz),
            "provenance": resolved.provenance,
        }
        return replace(execution, trace=[grounding_trace] + execution.trace)

    return execute


def make_native_release_executor(
    env,
    *,
    execution_state: Optional[LiberoNativeExecutionState] = None,
    frame_callback: Optional[FrameCallback] = None,
):
    """Build a release adapter using the existing native open primitive."""
    captured_env = _FrameCaptureEnv(env, frame_callback)

    def execute(invocation, observation, *, max_steps):
        execution = execute_release_primitive(
            captured_env, observation, max_steps=max_steps
        )
        if execution.primitive_success and execution_state is not None:
            execution_state.holding = None
        return execution

    return execute


def make_native_vla_executor(
    env,
    backend,
    *,
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
    execution_state: Optional[LiberoNativeExecutionState] = None,
    tau_monitor_factory=None,
    frame_callback: Optional[FrameCallback] = None,
):
    """Build a bounded raw-delta VLA adapter with per-action tau checks."""
    monitor_factory = tau_monitor_factory or _NativeLiftAndGraspMonitor

    def execute(invocation, observation, *, max_steps):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise LiberoPrimitiveError("max_steps must be an integer greater than zero")
        max_chunks = invocation["max_chunks"]
        current = observation
        traces = []
        steps_executed = 0
        checker = getattr(env, "check_success", None)
        task_success_mode = invocation.get("tau") == "task_success"
        monitor = None
        if not task_success_mode:
            try:
                monitor = monitor_factory(env, current, invocation["target"])
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                trace = [{"event": "tau_setup_error", "error": str(exc), "tau_satisfied": False}]
                primitive = LiberoPrimitiveExecution(
                    current, False, False, "tau_setup_error", 0, trace
                )
                return LiberoVLAExecution(primitive, False, None)

        tau_satisfied = False
        task_success = False
        env_done = False
        for chunk_index in range(1, max_chunks + 1):
            if steps_executed >= max_steps:
                break
            prepared = prepare_pirlinf_observation(
                current, resize_with_pad, convert_to_uint8
            )
            chunk = backend.infer_chunk(prepared, invocation["prompt"])
            chunk_trace = {
                "chunk_index": chunk_index,
                "inference_duration_s": float(chunk.inference_duration_s),
                "full_chunk_length": int(chunk.full_chunk_length),
                "executed_actions": [],
                "rewards": [],
                "dones": [],
                "task_successes": [],
                "tau_evidence": [],
                "tau_evaluation_errors": [],
            }
            traces.append(chunk_trace)
            for raw_action in chunk.raw_deltas:
                if steps_executed >= max_steps:
                    break
                action = np.asarray(raw_action, dtype=float)
                step_result = env.step(action.tolist())
                if len(step_result) == 4:
                    current, reward, env_done, _ = step_result
                elif len(step_result) == 5:
                    current, reward, terminated, truncated, _ = step_result
                    env_done = bool(terminated or truncated)
                else:
                    raise LiberoPrimitiveError("env.step must return four or five values")
                steps_executed += 1
                if frame_callback is not None:
                    frame_callback(current)
                task_success = bool(checker() if callable(checker) else False)
                chunk_trace["executed_actions"].append(action.tolist())
                chunk_trace["rewards"].append(float(reward))
                chunk_trace["dones"].append(bool(env_done))
                chunk_trace["task_successes"].append(task_success)
                if task_success_mode:
                    tau_satisfied = task_success
                    evidence = {
                        "predicate": "task_success",
                        "source": "env.check_success",
                        "tau_satisfied": task_success,
                        "task_success_evaluated": True,
                    }
                else:
                    try:
                        evidence = _json_safe(monitor.evaluate(current))
                        tau_satisfied = bool(evidence.get("tau_satisfied", False))
                    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                        error = {"action_index": steps_executed, "error": str(exc)}
                        chunk_trace["tau_evaluation_errors"].append(error)
                        evidence = {
                            "predicate": "lift_and_grasp",
                            "tau_satisfied": False,
                            "evaluation_error": error,
                            "task_success_evaluated": False,
                        }
                        tau_satisfied = False
                chunk_trace["tau_evidence"].append(evidence)
                if task_success or env_done or tau_satisfied:
                    break
            chunk_trace["tau_satisfied"] = tau_satisfied
            if task_success or env_done or tau_satisfied:
                break

        if task_success:
            reason = "task_success"
        elif env_done:
            reason = "env_done"
        elif tau_satisfied:
            reason = "lift_and_grasp_satisfied"
        elif steps_executed >= max_steps:
            reason = "step_budget_exhausted"
        else:
            reason = "chunk_budget_exhausted"
        primitive = LiberoPrimitiveExecution(
            observation=current,
            primitive_success=(
                task_success if task_success_mode else tau_satisfied or task_success
            ),
            task_success=task_success,
            termination_reason=reason,
            steps_executed=steps_executed,
            trace=_json_safe(traces),
        )
        if task_success_mode:
            holding = None if task_success else invocation["target"]
            if execution_state is not None:
                execution_state.holding = holding
        else:
            holding = invocation["target"] if tau_satisfied else None
            if tau_satisfied and execution_state is not None:
                execution_state.holding = holding
        return LiberoVLAExecution(primitive, tau_satisfied, holding)

    return execute


__all__ = [
    "LiberoNativeOffsets",
    "LiberoNativeExecutionState",
    "LiberoResolvedTarget",
    "VisualLiftAndGraspMonitor",
    "make_native_move_executor",
    "make_native_release_executor",
    "make_native_vla_executor",
    "resolve_target_xyz",
]