"""Artifact-backed orchestration for one native LIBERO multi-turn episode."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from harness_vla.eval.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoMultiTurnEvaluator,
)
from harness_vla.eval.libero_file_repl_bridge import (
    LiberoFileREPLBridge,
    LiberoProtocolMetadata,
)
from harness_vla.eval.libero_memory_lifecycle import DeploymentMemorySession
from harness_vla.eval.libero_privileged_diagnostics import (
    LiberoPrivilegedDiagnostics,
)
from harness_vla.eval.libero_native_multi_turn import (
    LIBERO_ROTATION_TOLERANCE_RAD,
    LiberoNativeExecutionState,
    LiberoNativeOffsets,
    make_native_move_executor,
    make_native_release_executor,
    make_native_vla_executor,
    VisualLiftAndGraspMonitor,
)
from harness_vla.eval.libero_vla_smoke import LIBERO_DUMMY_ACTION
from harness_vla.perception.libero_grounding import (
    calibration_from_sim,
    depth_to_meters,
    ground_instance,
)
from harness_vla.perception.libero_visual_grounding import (
    VisualPixelObservation,
    ground_visual_instance,
)
from harness_vla.planner.phase_policy import (
    Phase,
    PhaseManifest,
    PhaseOperation,
    validate_phase_manifest,
)
from harness_vla.trace_io import (
    resolve_git_commit,
    write_json_atomic,
)


SETTLING_STEPS = 10


def _default_video_writer(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=10)


def _positive_finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be finite and greater than zero" % name) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s must be finite and greater than zero" % name)
    return number


def _unpack_step(step_result):
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, reward, bool(done), info
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, reward, bool(terminated or truncated), info
    raise ValueError("env.step must return four or five values")


def _video_frame(observation: Mapping[str, Any], camera: str) -> np.ndarray:
    image = np.asarray(observation["%s_image" % camera])
    return np.ascontiguousarray(image[::-1, ::-1])


def _native_grounder(env, observation, camera: str, frame_id: int):
    depth = np.asarray(observation["%s_depth" % camera])
    height, width = depth.shape[:2]
    sim = env.env.sim
    calibration = calibration_from_sim(sim, camera, height, width, frame_id)
    grounding_frame = frame_id

    def grounder(current_observation, target):
        nonlocal grounding_frame
        grounding_frame += 1
        current_calibration = replace(calibration, frame_id=grounding_frame)
        metric_depth = depth_to_meters(
            sim, current_observation["%s_depth" % camera]
        )
        return ground_instance(
            current_observation,
            env.instance_to_id,
            target,
            current_calibration,
            metric_depth,
        )

    return grounder


def _visual_grounder(
    env,
    observation,
    camera: str,
    frame_id: int,
    locator,
    target_descriptions: Mapping[str, str],
):
    depth = np.asarray(observation["%s_depth" % camera])
    height, width = depth.shape[:2]
    sim = env.env.sim
    calibration = calibration_from_sim(sim, camera, height, width, frame_id)
    grounding_frame = frame_id

    def grounder(current_observation, target):
        nonlocal grounding_frame
        grounding_frame += 1
        current_calibration = replace(calibration, frame_id=grounding_frame)
        metric_depth = depth_to_meters(
            sim, current_observation["%s_depth" % camera]
        )
        return ground_visual_instance(
            VisualPixelObservation(current_observation["%s_image" % camera]),
            target_descriptions[target],
            current_calibration,
            metric_depth,
            locator,
        )

    return grounder


def _semantic_maps(targets, object_labels, object_roles):
    labels = {
        target: str((object_labels or {}).get(target) or target.replace("_", " "))
        for target in targets
    }
    if object_roles is None:
        roles = {
            target: ["manipulable" if index == 0 else "destination"]
            for index, target in enumerate(targets)
        }
    else:
        roles = {target: list(object_roles.get(target, ())) for target in targets}
    return labels, roles


def run_libero_multi_turn_episode(
    *,
    env,
    backend,
    planner,
    initial_state,
    instruction: str,
    run_root,
    task_suite: str,
    task_id: int,
    initial_state_index: int,
    seed: int,
    budgets: LiberoMultiTurnBudgets,
    offsets: LiberoNativeOffsets,
    position_tolerance: float,
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
    grounder=None,
    vla_executor=None,
    move_executor=None,
    release_executor=None,
    available_targets: Optional[Sequence[str]] = None,
    camera: str = "agentview",
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] = _default_video_writer,
    backend_name: str = "pirlinf_websocket",
    phase_manifest: Optional[PhaseManifest] = None,
    phase: Optional[str] = None,
    protocol_seed: Optional[int] = None,
    deployment_memory_session: Optional[DeploymentMemorySession] = None,
    file_repl_dir=None,
    visual_locator=None,
    target_descriptions: Optional[Mapping[str, str]] = None,
    object_labels: Optional[Mapping[str, str]] = None,
    object_roles: Optional[Mapping[str, Sequence[str]]] = None,
    reset_environment: Optional[bool] = None,
    privileged_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Run one episode and durably persist its manifest, trace, video and result."""
    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    if not isinstance(budgets, LiberoMultiTurnBudgets):
        raise ValueError("budgets must be a LiberoMultiTurnBudgets")
    if not isinstance(offsets, LiberoNativeOffsets):
        raise ValueError("offsets must be a LiberoNativeOffsets")
    tolerance = _positive_finite(position_tolerance, "position_tolerance")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if not isinstance(camera, str) or not camera:
        raise ValueError("camera must be a non-empty string")

    resolved_phase = None
    phase_policy = None
    if phase_manifest is not None:
        validate_phase_manifest(phase_manifest)
        if phase is None:
            raise ValueError("phase is required with phase_manifest")
        resolved_phase = Phase(phase)
        phase_policy = phase_manifest.policy_for(resolved_phase)
        phase_manifest.guard_operation(resolved_phase, seed, PhaseOperation.READ_MEMORY)
        budgets = replace(
            budgets,
            max_turns=min(budgets.max_turns, phase_policy.budget),
            horizon=min(budgets.horizon, phase_policy.budget),
        )
        if resolved_phase is Phase.DEPLOYMENT:
            if not isinstance(deployment_memory_session, DeploymentMemorySession):
                raise ValueError("deployment requires DeploymentMemorySession")
            if deployment_memory_session.manifest != phase_manifest:
                raise ValueError("deployment memory session manifest mismatch")
            if deployment_memory_session.seed != seed:
                raise ValueError("deployment memory session seed mismatch")
    elif phase is not None or deployment_memory_session is not None:
        raise ValueError("phase and deployment memory require phase_manifest")

    targets = list(
        getattr(env, "obj_of_interest", ())
        if available_targets is None
        else available_targets
    )
    if not targets or any(not isinstance(target, str) or not target for target in targets):
        raise ValueError("available_targets must contain non-empty strings")
    labels, roles = _semantic_maps(targets, object_labels, object_roles)
    descriptions = dict(labels)
    descriptions.update(target_descriptions or {})
    if any(target not in descriptions or not descriptions[target] for target in targets):
        raise ValueError("target descriptions must cover every available target")

    should_reset = (
        resolved_phase is not Phase.DEPLOYMENT
        if reset_environment is None
        else bool(reset_environment)
    )
    if should_reset and phase_manifest is not None:
        phase_manifest.guard_operation(resolved_phase, seed, PhaseOperation.RESET)

    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    manifest_path = root / "run_manifest.json"
    manifest = {
        "run_type": "libero_multi_turn",
        "status": "in_progress",
        "harness_complete": bool(
            phase_manifest is not None
            and file_repl_dir is not None
            and visual_locator is not None
            and resolved_phase is Phase.DEPLOYMENT
            and deployment_memory_session is not None
        ),
        "implementation_scope": "published_libero_primitive_vocabulary",
        "vla_tau_modes": ["lift_and_grasp", "task_success"],
        "scientific_classification": {
            "paper_confirmed": [
                "one_primitive_per_turn",
                "execute_observe_feedback_replan",
                "seven_primitive_libero_vocabulary_and_roles",
                "official_task_success_termination",
            ]
            + (["bootstrap_deployment_separation"] if phase_manifest is not None else [])
            + (["file_mediated_repl"] if file_repl_dir is not None else [])
            + (["visual_rgbd_world_grounding"] if visual_locator is not None else [])
            + (
                ["task_specific_memory", "global_memory"]
                if deployment_memory_session is not None
                else []
            ),
            "paper_compatible": [
                "in_process_loop",
                "rgbd_target_mode_resolution",
                "configured_offsets_tolerances_budgets",
                "quaternion_xyzw_pose_and_radian_setpoints",
                "native_osc_pose_execution",
            ]
            + (
                ["closed_gripper_transport_without_continuous_grasp_guard"]
                if visual_locator is not None
                else []
            ),
            "beta_only": (
                [
                    "visual_pixel_locator",
                    "visual_rgbd_lift_tau",
                    "guarded_libero_workspace_and_rotation_ranges",
                ]
                if visual_locator is not None
                else [
                    "privileged_instance_segmentation",
                    "privileged_contact_state",
                    "guarded_libero_workspace_and_rotation_ranges",
                ]
            )
            + (["privileged_diagnostics"] if privileged_diagnostics else []),
        },
        "task_memory": deployment_memory_session is not None,
        "global_memory": deployment_memory_session is not None,
        "perception": (
            "visual_rgbd_locator"
            if visual_locator is not None
            else "rgbd_with_privileged_instance_segmentation"
        ),
        "privileged_segmentation": visual_locator is None,
        "privileged_contact_state": visual_locator is None,
        "privileged_diagnostics": bool(privileged_diagnostics),
        "planner_receives_oracle_coordinates": False,
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "planner_model": getattr(planner, "model_name", None),
        "think": bool(getattr(planner, "think", False)),
        "backend": backend_name,
        "backend_host": getattr(backend, "host", None),
        "backend_port": getattr(backend, "port", None),
        "available_targets": targets,
        "config": {
            "settling_steps": SETTLING_STEPS,
            "budgets": asdict(budgets),
            "offsets_m": asdict(offsets),
            "position_tolerance_m": tolerance,
            "rotation_tolerance_rad": LIBERO_ROTATION_TOLERANCE_RAD,
            "camera": camera,
        },
        "git_commit": resolve_git_commit(Path(__file__)),
    }
    if phase_manifest is not None:
        resolved_protocol_seed = seed if protocol_seed is None else protocol_seed
        manifest.update(
            {
                "phase_manifest": phase_manifest.to_dict(),
                "phase": resolved_phase.value,
                "reportable": phase_policy.reportable,
                "protocol_seed": resolved_protocol_seed,
                "memory_hashes": (
                    dict(deployment_memory_session.hashes_before)
                    if deployment_memory_session is not None
                    else None
                ),
                "file_repl": (
                    {"directory": str(Path(file_repl_dir))}
                    if file_repl_dir is not None
                    else None
                ),
            }
        )
    write_json_atomic(manifest_path, manifest)

    frames = []
    video_path = None
    try:
        if should_reset:
            env.reset()
        observation = env.set_init_state(initial_state)
        frames.append(_video_frame(observation, camera))
        for _ in range(SETTLING_STEPS):
            observation, _, _, _ = _unpack_step(
                env.step(list(LIBERO_DUMMY_ACTION))
            )
            frames.append(_video_frame(observation, camera))

        frame_callback = lambda current: frames.append(_video_frame(current, camera))
        execution_state = LiberoNativeExecutionState()
        active_vla_executor = vla_executor or make_native_vla_executor(
            env,
            backend,
            resize_with_pad=resize_with_pad,
            convert_to_uint8=convert_to_uint8,
            execution_state=execution_state,
            frame_callback=frame_callback,
        )
        active_grounder = grounder
        if active_grounder is None and visual_locator is not None:
            active_grounder = _visual_grounder(
                env,
                observation,
                camera,
                SETTLING_STEPS,
                visual_locator,
                descriptions,
            )
        if move_executor is None:
            if active_grounder is None:
                active_grounder = _native_grounder(
                    env, observation, camera, SETTLING_STEPS
                )
            active_move_executor = make_native_move_executor(
                env,
                active_grounder,
                offsets=offsets,
                position_tolerance=tolerance,
                execution_state=execution_state,
                grasp_monitor=False if visual_locator is not None else None,
                frame_callback=frame_callback,
            )
        else:
            active_move_executor = move_executor
        if vla_executor is None and visual_locator is not None:
            active_vla_executor = make_native_vla_executor(
                env,
                backend,
                resize_with_pad=resize_with_pad,
                convert_to_uint8=convert_to_uint8,
                execution_state=execution_state,
                tau_monitor_factory=(
                    lambda unused_env, baseline, target: VisualLiftAndGraspMonitor(
                        active_grounder, baseline, target
                    )
                ),
                frame_callback=frame_callback,
            )
        active_release_executor = release_executor or make_native_release_executor(
            env, execution_state=execution_state, frame_callback=frame_callback
        )
        protocol_metadata = None
        file_repl_bridge = None
        if file_repl_dir is not None:
            resolved_protocol_seed = seed if protocol_seed is None else protocol_seed
            metadata = LiberoProtocolMetadata(
                Path(file_repl_dir), resolved_protocol_seed
            )
            protocol_metadata = {
                "protocol_seed": metadata.protocol_seed,
                "directory": str(metadata.directory),
            }
            file_repl_bridge = LiberoFileREPLBridge(
                metadata,
                vla_executor=active_vla_executor,
                move_executor=active_move_executor,
                release_executor=active_release_executor,
            )
        evaluator = LiberoMultiTurnEvaluator(
            planner,
            vla_executor=active_vla_executor,
            move_executor=active_move_executor,
            release_executor=active_release_executor,
            trace_path=root / "trace.jsonl",
            file_repl_bridge=file_repl_bridge,
            diagnostics_callback=(
                LiberoPrivilegedDiagnostics(
                    env,
                    available_targets=targets,
                    object_roles=roles,
                )
                if privileged_diagnostics
                else None
            ),
        )
        result = evaluator.run(
            instruction.strip(),
            observation,
            available_targets=targets,
            budgets=budgets,
            object_labels=labels,
            object_roles=roles,
            memory_context=(
                deployment_memory_session.context
                if deployment_memory_session is not None
                else None
            ),
            protocol_metadata=protocol_metadata,
        )
        if deployment_memory_session is not None:
            deployment_memory_session.guard_budget(result.steps_executed)
            memory_integrity = deployment_memory_session.finalize()
            manifest["memory_hashes"] = memory_integrity

        video_status = "success" if result.task_success else "failure"
        video_path = video_root / (
            "task_%03d_state_%03d_%s.mp4"
            % (task_id, initial_state_index, video_status)
        )
        video_writer(video_path, frames)
        episode = {
            "run_type": manifest["run_type"],
            "task_suite": task_suite,
            "task_id": task_id,
            "initial_state_index": initial_state_index,
            "seed": seed,
            "instruction": instruction.strip(),
            "task_success": result.task_success,
            "primitive_success": result.primitive_success,
            "env_done": result.env_done,
            "termination_reason": result.termination_reason,
            "turns_executed": result.turns_executed,
            "actions_executed": result.steps_executed,
            "horizon": budgets.horizon,
            "holding": result.holding,
            "video": str(video_path.relative_to(root)),
        }
        summary = {
            "episodes": 1,
            "successes": int(result.task_success),
            "task_success_rate": float(result.task_success),
            "turns_executed": result.turns_executed,
            "actions_executed": result.steps_executed,
            "budget_exhausted": result.termination_reason
            in ("max_turns_exhausted", "horizon_exhausted"),
            "termination_reason": result.termination_reason,
        }
        write_json_atomic(root / "episode.json", episode)
        write_json_atomic(root / "summary.json", summary)
        manifest.update(
            {
                "status": "completed",
                "task_success": result.task_success,
                "termination_reason": result.termination_reason,
                "video": str(video_path.relative_to(root)),
            }
        )
        write_json_atomic(manifest_path, manifest)
        return {"episode": episode, "summary": summary, "manifest": manifest}
    except Exception as exc:
        if frames:
            video_path = video_root / (
                "task_%03d_state_%03d_incomplete.mp4"
                % (task_id, initial_state_index)
            )
            try:
                video_writer(video_path, frames)
            except Exception:
                video_path = None
        manifest.update(
            {
                "status": "incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if video_path is not None:
            manifest["video"] = str(video_path.relative_to(root))
        write_json_atomic(manifest_path, manifest)
        raise


__all__ = ["SETTLING_STEPS", "run_libero_multi_turn_episode"]