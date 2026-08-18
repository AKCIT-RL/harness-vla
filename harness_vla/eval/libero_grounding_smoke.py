"""Isolated native LIBERO RGB-D grounding smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from harness_vla.perception.libero_grounding import (
    calibration_from_sim,
    depth_to_meters,
    ground_instance,
)
from harness_vla.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    resolve_git_commit,
    write_json_atomic,
)


LIBERO_GROUNDING_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _default_video_writer(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=10)


def _body_oracle(sim, instance_name: str):
    matches = []
    for body_id, body_name in enumerate(sim.model.body_names):
        decoded = body_name.decode() if isinstance(body_name, bytes) else str(body_name)
        if decoded in (instance_name, "%s_main" % instance_name):
            matches.append(np.asarray(sim.data.body_xpos[body_id], dtype=float))
    return matches[0] if len(matches) == 1 else None


def run_libero_grounding_smoke(
    *,
    env,
    initial_state,
    run_root,
    task_suite: str,
    task_id: int,
    initial_state_index: int,
    seed: int,
    camera: str,
    height: int,
    width: int,
    settle_steps: int = 10,
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] = _default_video_writer,
) -> Dict[str, Any]:
    """Ground benchmark instances and persist planner/diagnostic data separately."""
    if settle_steps < 1:
        raise ValueError("settle_steps must be greater than zero")
    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    trace_path = root / "trace.jsonl"
    initialize_jsonl(trace_path)

    env.reset()
    observation = env.set_init_state(initial_state)
    frames = []
    for _ in range(settle_steps):
        observation, _, _, _ = env.step(list(LIBERO_GROUNDING_DUMMY_ACTION))
        frames.append(np.ascontiguousarray(observation["%s_image" % camera][::-1]))

    sim = env.env.sim
    calibration = calibration_from_sim(sim, camera, height, width, settle_steps)
    metric_depth = depth_to_meters(sim, observation["%s_depth" % camera])
    estimates = {}
    oracle_metrics = []
    for instance_name in env.obj_of_interest:
        estimate = ground_instance(
            observation,
            env.instance_to_id,
            instance_name,
            calibration,
            metric_depth,
        )
        estimates[instance_name] = estimate
        oracle = _body_oracle(sim, instance_name)
        oracle_metric = None
        if oracle is not None:
            oracle_metric = {
                "instance_name": instance_name,
                "body_center_xyz": oracle.tolist(),
                "surface_to_body_center_error_m": float(
                    np.linalg.norm(np.asarray(estimate["world_xyz"]) - oracle)
                ),
            }
            oracle_metrics.append(oracle_metric)
        append_jsonl_record(
            trace_path,
            {"event": "ground_instance", "estimate": estimate, "oracle": oracle_metric},
        )

    errors = [item["surface_to_body_center_error_m"] for item in oracle_metrics]
    manifest = {
        "run_type": "libero_grounding_smoke",
        "scientific_classification": "beta-only",
        "harness_complete": False,
        "task_success_evaluated": False,
        "perception": "rgbd_with_privileged_instance_segmentation",
        "privileged_segmentation": True,
        "planner_receives_oracle_coordinates": False,
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "camera": camera,
        "height": height,
        "width": width,
        "settle_steps": settle_steps,
        "git_commit": resolve_git_commit(Path(__file__)),
    }
    grounding = {"objects": estimates, "oracle_metrics": oracle_metrics}
    summary = {
        "objects_requested": len(env.obj_of_interest),
        "objects_grounded": len(estimates),
        "oracle_body_centers_found": len(errors),
        "mean_surface_to_body_center_error_m": float(np.mean(errors)) if errors else None,
        "max_surface_to_body_center_error_m": float(np.max(errors)) if errors else None,
        "task_success": None,
    }
    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    video_path = video_root / (
        "task_%03d_state_%03d_grounding.mp4" % (task_id, initial_state_index)
    )
    video_writer(video_path, frames)
    manifest["video"] = str(video_path.relative_to(root))
    write_json_atomic(root / "run_manifest.json", manifest)
    write_json_atomic(root / "grounding.json", grounding)
    write_json_atomic(root / "summary.json", summary)
    return {"manifest": manifest, "grounding": grounding, "summary": summary}


__all__ = ["LIBERO_GROUNDING_DUMMY_ACTION", "run_libero_grounding_smoke"]