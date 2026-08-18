"""Minimal native LIBERO smoke for a frozen, chunk-producing VLA."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from harness_vla.backends.pirlinf_backend import PiRLinfObservation
from harness_vla.perception.libero_grounding import (
    calibration_from_sim,
    depth_to_meters,
    ground_instance,
)
from harness_vla.primitives.libero_tau import (
    LiftAndGraspTau,
    read_bilateral_contact,
)
from harness_vla.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    resolve_git_commit,
    write_json_atomic,
)
from harness_vla.backends.vla_runtime import VLARuntime


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class _NativeLiftAndGraspMonitor:
    def __init__(self, env, observation, target, minimum_lift_m=0.03):
        self.env = env
        self.target = target
        self.camera = "agentview"
        height, width = observation["agentview_depth"].shape[:2]
        self.calibration = calibration_from_sim(
            env.env.sim, self.camera, height, width, 0
        )
        self.frame_id = 0
        baseline = self._ground(observation)
        self.tau = LiftAndGraspTau(
            baseline_target_z_m=baseline["world_xyz"][2],
            minimum_lift_m=minimum_lift_m,
        )

    def _ground(self, observation):
        metric_depth = depth_to_meters(
            self.env.env.sim, observation["agentview_depth"]
        )
        return ground_instance(
            observation,
            self.env.instance_to_id,
            self.target,
            self.calibration,
            metric_depth,
        )

    def evaluate(self, observation):
        self.frame_id += 1
        self.calibration = replace(self.calibration, frame_id=self.frame_id)
        grounding = self._ground(observation)
        contact = read_bilateral_contact(self.env, self.target)
        evidence = self.tau.evaluate(
            current_target_z_m=grounding["world_xyz"][2],
            left_finger_contact=contact["left_finger_contact"],
            right_finger_contact=contact["right_finger_contact"],
        )
        evidence["target_instance"] = self.target
        evidence["grounding_provenance"] = grounding["provenance"]
        evidence["contact_source"] = contact["source"]
        evidence["privileged_contact_state"] = True
        return evidence


def prepare_pirlinf_observation(
    observation: Dict[str, Any],
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
) -> PiRLinfObservation:
    """Apply the OpenPI LIBERO image and state preprocessing contract."""
    front = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    front = convert_to_uint8(resize_with_pad(front, 224, 224))
    wrist = convert_to_uint8(resize_with_pad(wrist, 224, 224))
    gripper_pose = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            np.asarray(observation["robot0_eef_quat"]),
        )
    )
    return PiRLinfObservation(
        front_rgb=np.asarray(front, dtype=np.uint8),
        wrist_rgb=np.asarray(wrist, dtype=np.uint8),
        gripper_pose=gripper_pose,
        gripper_qpos=np.asarray(observation["robot0_gripper_qpos"]),
        mode="task",
    )


def _unpack_step(step_result):
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, reward, bool(done), info
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, reward, bool(terminated or truncated), info
    raise ValueError("env.step must return four or five values")


def _default_video_writer(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=10)


def run_libero_vla_smoke(
    *,
    env,
    backend,
    initial_state,
    prompt: str,
    run_root,
    task_suite: str,
    task_id: int,
    initial_state_index: int,
    seed: int,
    replan_steps: int,
    max_chunks: int,
    horizon: int,
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
    planner=None,
    lift_tau_factory=None,
    host: str = "127.0.0.1",
    port: int = 8000,
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] = _default_video_writer,
) -> Dict[str, Any]:
    """Run one native LIBERO task/state smoke and persist its audit artifacts."""
    for name, value in (
        ("replan_steps", replan_steps),
        ("max_chunks", max_chunks),
        ("horizon", horizon),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("%s must be an integer greater than zero" % name)

    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    trace_path = root / "trace.jsonl"
    initialize_jsonl(trace_path)

    run_type = "harness_vla_only_smoke" if planner is not None else "vla_only_smoke"
    manifest = {
        "run_type": run_type,
        "harness_complete": False,
        "analytic_primitives_available": False,
        "task_memory": False,
        "global_memory": False,
        "perception": "text_only_task_instruction",
        "planner_enabled": planner is not None,
        "planner_model": getattr(planner, "model_name", None),
        "planner_raw_output": None,
        "planner_thinking": None,
        "planner_invocation": None,
        "backend": "pirlinf_websocket",
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "host": host,
        "port": port,
        "replan_steps": replan_steps,
        "max_chunks": max_chunks,
        "horizon": horizon,
        "git_commit": resolve_git_commit(Path(__file__)),
    }

    env.reset()
    raw_observation = env.set_init_state(initial_state)
    for _ in range(10):
        raw_observation, _, _, _ = _unpack_step(env.step(list(LIBERO_DUMMY_ACTION)))

    live_observation = prepare_pirlinf_observation(
        raw_observation, resize_with_pad, convert_to_uint8
    )
    video_frames = [live_observation.front_rgb]
    executed_action_count = 0
    task_success = False
    chunk_traces = []
    runtime_chunk_limit = min(max_chunks, int(math.ceil(float(horizon) / replan_steps)))
    effective_prompt = prompt
    planner_parse_error = False
    tau_setup_error = None
    lift_monitor = None
    latest_tau_evidence = None
    if planner is not None:
        invocation, raw_output = planner.act(
            prompt,
            runtime_chunk_limit,
            available_targets=getattr(env, "obj_of_interest", ()),
        )
        thinking = getattr(planner, "last_thinking", None)
        manifest.update(
            {
                "planner_raw_output": raw_output,
                "planner_thinking": thinking,
                "planner_invocation": invocation,
            }
        )
        if invocation is None:
            planner_parse_error = True
        else:
            effective_prompt = invocation["prompt"]
            runtime_chunk_limit = invocation["max_chunks"]
            if invocation["tau"] == "lift_and_grasp":
                manifest.update(
                    {
                        "perception": "rgbd_with_privileged_instance_segmentation",
                        "tau": "lift_and_grasp",
                        "minimum_lift_m": 0.03,
                        "privileged_segmentation": True,
                        "privileged_contact_state": True,
                        "tau_monitor_ready": False,
                        "tau_setup_error": None,
                    }
                )
                try:
                    factory = lift_tau_factory or _NativeLiftAndGraspMonitor
                    lift_monitor = factory(
                        env, raw_observation, invocation["target"]
                    )
                    manifest["tau_monitor_ready"] = True
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                    tau_setup_error = str(exc)
                    manifest["tau_setup_error"] = tau_setup_error
    write_json_atomic(root / "run_manifest.json", manifest)

    def execute_chunk(chunk):
        nonlocal executed_action_count, live_observation, raw_observation
        nonlocal task_success, latest_tau_evidence
        executed_actions = []
        rewards = []
        dones = []
        successes = []
        tau_evaluation_errors = []
        for raw_action in chunk.raw_deltas:
            if (
                executed_action_count >= horizon
                or task_success
                or bool(latest_tau_evidence and latest_tau_evidence["tau_satisfied"])
            ):
                break
            action = np.asarray(raw_action, dtype=float)
            raw_observation, reward, done, _ = _unpack_step(
                env.step(action.tolist())
            )
            executed_action_count += 1
            checker = getattr(env, "check_success", None)
            task_success = bool(done) or bool(checker() if callable(checker) else False)
            executed_actions.append(action.tolist())
            rewards.append(float(reward))
            dones.append(bool(done))
            successes.append(task_success)
            live_observation = prepare_pirlinf_observation(
                raw_observation, resize_with_pad, convert_to_uint8
            )
            video_frames.append(live_observation.front_rgb)
            if lift_monitor is not None:
                try:
                    latest_tau_evidence = lift_monitor.evaluate(raw_observation)
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                    error = {
                        "action_index": executed_action_count,
                        "error": str(exc),
                    }
                    tau_evaluation_errors.append(error)
                    latest_tau_evidence = {
                        "predicate": "lift_and_grasp",
                        "tau_satisfied": False,
                        "evaluation_error": error,
                        "task_success_evaluated": False,
                    }
            if task_success or bool(
                latest_tau_evidence and latest_tau_evidence["tau_satisfied"]
            ):
                break

        trace = {
            "prompt": effective_prompt,
            "chunk_index": len(chunk_traces) + 1,
            "inference_duration_s": float(chunk.inference_duration_s),
            "full_chunk_length": int(chunk.full_chunk_length),
            "executed_actions": executed_actions,
            "rewards": rewards,
            "dones": dones,
            "task_successes": successes,
            "tau_satisfied": (
                latest_tau_evidence["tau_satisfied"]
                if lift_monitor is not None and latest_tau_evidence is not None
                else task_success
            ),
            "tau_evidence": latest_tau_evidence,
            "tau_evaluation_errors": tau_evaluation_errors,
            "planner_raw_output": manifest["planner_raw_output"],
            "planner_thinking": manifest["planner_thinking"],
            "planner_invocation": manifest["planner_invocation"],
        }
        chunk_traces.append(trace)
        append_jsonl_record(trace_path, trace)
        return live_observation

    runtime_result = None
    if planner_parse_error or tau_setup_error is not None:
        append_jsonl_record(
            trace_path,
            {
                "event": (
                    "planner_parse_error" if planner_parse_error else "tau_setup_error"
                ),
                "prompt": prompt,
                "planner_raw_output": manifest["planner_raw_output"],
                "planner_thinking": manifest["planner_thinking"],
                "planner_invocation": None,
                "error": tau_setup_error,
                "tau_satisfied": False,
            },
        )
    else:
        runtime_result = VLARuntime(backend).run(
            live_observation,
            effective_prompt,
            runtime_chunk_limit,
            lambda _: (
                bool(latest_tau_evidence and latest_tau_evidence["tau_satisfied"])
                if lift_monitor is not None
                else task_success
            ),
            execute_chunk,
        )

    tau_satisfied = False if runtime_result is None else runtime_result.tau_satisfied
    status = (
        "success"
        if task_success
        else "tau_success_task_incomplete"
        if tau_satisfied
        else "failure"
    )
    video_name = "task_%03d_state_%03d_%s.mp4" % (
        task_id,
        initial_state_index,
        status,
    )
    video_path = video_root / video_name
    video_writer(video_path, video_frames)

    episode = {
        "run_type": run_type,
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "prompt": effective_prompt,
        "planner_raw_output": manifest["planner_raw_output"],
        "planner_thinking": manifest["planner_thinking"],
        "planner_invocation": manifest["planner_invocation"],
        "task_success": task_success,
        "tau_satisfied": tau_satisfied,
        "termination_reason": (
            ("planner_parse_error" if planner_parse_error else "tau_setup_error")
            if runtime_result is None
            else runtime_result.termination_reason
        ),
        "chunks_executed": 0 if runtime_result is None else runtime_result.chunks_executed,
        "actions_executed": executed_action_count,
        "horizon": horizon,
        "video": str(video_path.relative_to(root)),
    }
    summary = {
        "episodes": 1,
        "successes": int(task_success),
        "task_success_rate": float(task_success),
        "tau_satisfied": tau_satisfied,
        "tau_success_rate": float(tau_satisfied),
        "budget_exhausted": (
            runtime_result is not None
            and runtime_result.termination_reason == "budget_exhausted"
        ),
        "termination_reason": episode["termination_reason"],
    }
    write_json_atomic(root / "episode.json", episode)
    write_json_atomic(root / "summary.json", summary)
    return {"episode": episode, "summary": summary, "manifest": manifest}


__all__ = [
    "LIBERO_DUMMY_ACTION",
    "prepare_pirlinf_observation",
    "run_libero_vla_smoke",
]